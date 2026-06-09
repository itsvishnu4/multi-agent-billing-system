from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing import Literal, Optional
import json
import re

llm = init_chat_model("anthropic:claude-sonnet-4-6")


class EligibilityResult(BaseModel):
    verdict: Literal["ELIGIBLE", "INELIGIBLE"] = Field(
        ..., description="Whether the patient's insurance covers the requested service")
    reason: str = Field(...,
                        description="Clear explanation of the eligibility decision")
    prior_auth_required: bool = Field(...,
                                      description="Whether prior authorization is needed")
    coverage_gaps: list[str] = Field(
        default_factory=list, description="Any coverage risks or gaps found")


class AuditResult(BaseModel):
    verdict: Literal["APPROVED", "NEEDS CORRECTION"] = Field(
        ..., description="Whether the claim passes audit or requires fixes")
    issues: list[str] = Field(default_factory=list,
                              description="List of issues found, if any")
    corrected_claim: Optional[dict] = Field(
        None, description="The corrected claim JSON if fixes were applied")


# replace with real payer API (e.g. Availity, Change Healthcare)
def check_eligibility(member_id: str, plan: str, service: str) -> dict:
    return {
        "status": "active",
        "deductible_met": True,
        "copay": 30,
        "prior_auth_required": False,
        "covered": True
    }


def validate_icd10_code(code: str) -> dict:
    valid = bool(re.match(r'^[A-Z]\d{2}(\.\d{1,4})?$', code))
    return {"code": code, "valid": valid}


def eligibility_agent(state: dict) -> dict:
    patient = state["patient_data"]
    service = state["service_data"]

    tool_result = check_eligibility(
        member_id=patient.get("member_id", ""),
        plan=patient.get("insurance_plan", ""),
        service=service.get("service", "")
    )

    result = llm.with_structured_output(EligibilityResult).invoke([
        {
            "role": "system",
            "content": (
                "You are an insurance eligibility verifier. Given patient data and live eligibility results, "
                "determine if the requested service is covered. Flag any coverage gaps or prior auth requirements. "
                "Return ELIGIBLE or INELIGIBLE with a clear reason. Do not create the claim."
            )
        },
        {
            "role": "user",
            "content": f"Patient: {json.dumps(patient)}\nService: {json.dumps(service)}\nEligibility data: {json.dumps(tool_result)}"
        }
    ])

    return {
        "eligibility_verdict": result.verdict,
        "eligibility_reason": result.reason,
        "coverage_gaps": result.coverage_gaps,
        "messages": [{"role": "assistant", "content": f"[Eligibility] {result.verdict}: {result.reason}"}]
    }


def eligibility_router(state: dict) -> str:
    return "create_claim" if state.get("eligibility_verdict") == "ELIGIBLE" else "denied"


def claim_creator_agent(state: dict) -> dict:
    patient = state["patient_data"]
    service = state["service_data"]
    icd_validation = validate_icd10_code(service.get("diagnosis", ""))

    reply = llm.invoke([
        {
            "role": "system",
            "content": (
                "You are a medical claim creator. Build a complete insurance claim JSON from the provided data. "
                "Required fields: patient (name, dob, member_id, insurance_plan, address), provider (name, npi, address), "
                "service (date, place_of_service_code, cpt_code), diagnosis (icd10_code), billing (billed_amount, units). "
                "Infer the CPT code from the service description if not provided. "
                "Output only valid JSON, then on a new line write MISSING: followed by any absent fields."
            )
        },
        {
            "role": "user",
            "content": (
                f"Patient: {json.dumps(patient)}\nService: {json.dumps(service)}\n"
                f"ICD-10 validation: {json.dumps(icd_validation)}\n"
                f"Prior audit issues to fix: {state.get('audit_issues', [])}"
            )
        }
    ])

    raw = reply.content
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    claim = json.loads(match.group()) if match else {}

    missing = []
    if "MISSING:" in raw:
        missing = [m.strip() for m in raw.split("MISSING:")
                   [-1].strip().split(",") if m.strip()]

    return {
        "claim_json": claim,
        "missing_fields": missing,
        "messages": [{"role": "assistant", "content": f"[Claim Creator] Built. Missing: {missing or 'None'}"}]
    }


def claim_auditor_agent(state: dict) -> dict:
    claim = state.get("claim_json", {})

    result = llm.with_structured_output(AuditResult).invoke([
        {
            "role": "system",
            "content": (
                "You are an insurance claim auditor. Check that all required fields are present, "
                "CPT and ICD-10 codes are clinically consistent, NPI is exactly 10 digits, "
                "date of service is not in the future, and billed amount is reasonable. "
                "Return APPROVED or NEEDS CORRECTION. If correcting, populate corrected_claim."
            )
        },
        {
            "role": "user",
            "content": f"Claim to audit:\n{json.dumps(claim, indent=2)}"
        }
    ])

    return {
        "audit_verdict": result.verdict,
        "audit_issues": result.issues,
        "claim_json": result.corrected_claim or claim,
        "messages": [{"role": "assistant", "content": f"[Auditor] {result.verdict}. Issues: {result.issues or 'None'}"}]
    }


def audit_router(state: dict) -> str:
    if state.get("audit_verdict") == "APPROVED" or state.get("retry_count", 0) >= 3:
        return "approved"
    state["retry_count"] = state.get("retry_count", 0) + 1
    return "create_claim"
