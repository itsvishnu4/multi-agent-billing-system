from dotenv import load_dotenv
from typing import Annotated, Literal, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
import json
import re
import uuid

load_dotenv()

llm = init_chat_model("anthropic:claude-sonnet-4-6")


# structured output
class EligibilityResult(BaseModel):
    verdict: Literal["ELIGIBLE", "INELIGIBLE"] = Field(
        ...,
        description="Whether the patient's insurance covers the requested service"
    )
    reason: str = Field(...,
                        description="Clear explanation of the eligibility decision")
    prior_auth_required: bool = Field(...,
                                      description="Whether prior authorization is needed")
    coverage_gaps: list[str] = Field(
        default_factory=list, description="Any coverage risks or gaps found")


class AuditResult(BaseModel):
    verdict: Literal["APPROVED", "NEEDS CORRECTION"] = Field(
        ...,
        description="Whether the claim passes audit or requires fixes"
    )
    issues: list[str] = Field(default_factory=list,
                              description="List of issues found, if any")
    corrected_claim: Optional[dict] = Field(
        None, description="The corrected claim JSON if fixes were applied")


# all share the same state
class State(TypedDict):
    messages: Annotated[list, add_messages]

    # Raw input data
    patient_data: dict
    service_data: dict

    # Agent 1 output
    eligibility_verdict: str | None
    eligibility_reason: str | None
    coverage_gaps: list[str] | None

    # Agent 2 output
    claim_json: dict | None
    missing_fields: list[str] | None

    # Agent 3 output
    audit_verdict: str | None
    audit_issues: list[str] | None

    # Routing (same pattern as your "next" field)
    next: str | None

    # Retry tracking
    retry_count: int


# fake tools that will be replaced with API calls (APIs are unable here)
def check_eligibility(member_id: str, plan: str, service: str) -> dict:
    return {
        "status": "active",
        "deductible_met": True,
        "copay": 30,
        "prior_auth_required": False,
        "covered": True
    }


def lookup_cpt_code(service_description: str) -> str:
    cpt_map = {"office visit": "99213",
               "blood panel": "80053", "x-ray": "71046"}
    return cpt_map.get(service_description.lower(), "99213")


def validate_icd10_code(code: str) -> dict:
    valid = bool(re.match(r'^[A-Z]\d{2}(\.\d{1,4})?$', code))
    return {"code": code, "valid": valid}


def save_claim_to_file(claim: dict) -> str:
    fname = f"claim_{uuid.uuid4().hex[:8]}.json"
    with open(fname, "w") as f:
        json.dump(claim, f, indent=2)
    return fname


# agent node 1: eligibility verifier
def eligibility_agent(state: State):
    patient = state["patient_data"]
    service = state["service_data"]

    # call the eligibility tool with real data
    tool_result = check_eligibility(
        member_id=patient.get("member_id", ""),
        plan=patient.get("insurance_plan", ""),
        service=service.get("service", "")
    )

    # structured output
    eligibility_llm = llm.with_structured_output(EligibilityResult)

    result = eligibility_llm.invoke([
        {
            "role": "system",
            "content": """You are an expert Insurance Eligibility Verifier. Your sole job is to determine 
            whether a patient's insurance will cover a requested service BEFORE the claim is filed.
            Steps to follow:
            1) Review the eligibility data provided from the insurance system
            2) Report: active/inactive status, deductible info, copay, and whether prior auth is required
            3) Flag any coverage gaps or risks that could cause the claim to be denied
            4) Return verdict as either ELIGIBLE or INELIGIBLE with clear reasons.
            Do NOT create the claim. Simply verify eligibility."""
        },
        {
            "role": "user",
            "content": f"""
            Patient: {json.dumps(patient, indent=2)}
            Service requested: {json.dumps(service, indent=2)}
            Eligibility data from insurance system: {json.dumps(tool_result, indent=2)}
            """
        }
    ])

    return {
        "eligibility_verdict": result.verdict,
        "eligibility_reason": result.reason,
        "coverage_gaps": result.coverage_gaps,
        "messages": [{"role": "assistant", "content": f"[Eligibility] {result.verdict}: {result.reason}"}]
    }


# router 1: decide between create claim or deny
def eligibility_router(state: State):
    if state.get("eligibility_verdict") == "ELIGIBLE":
        return {"next": "create_claim"}
    return {"next": "denied"}


# agent node 2: claim creator
def claim_creator_agent(state: State):
    patient = state["patient_data"]
    service = state["service_data"]

    # look up CPT code if not provided
    cpt_code = service.get("cpt_code") or lookup_cpt_code(
        service.get("service", ""))

    # validate ICD-10 code
    icd_validation = validate_icd10_code(service.get("diagnosis", ""))

    reply = llm.invoke([
        {
            "role": "system",
            "content": """You are an expert Medical Claim Creator. Take raw patient and service data 
            and produce a structured, complete insurance claim.
            A valid claim must include:
            - Patient info: name, date of birth, member ID, insurance plan
            - Provider info: name, NPI number, address
            - Service info: date of service, place of service code, CPT procedure code(s)
            - Diagnosis info: ICD-10 code(s)
            - Billing info: billed amount, units
            Steps:
            1) Extract all data from the input provided
            2) Use the CPT code and ICD-10 validation results already provided
            3) Output ONLY a valid JSON object with all required fields
            4) After the JSON, list any MISSING fields on a new line starting with MISSING:
            Do NOT verify eligibility. That is not your job."""
        },
        {
            "role": "user",
            "content": f"""
            Patient data: {json.dumps(patient, indent=2)}
            Service data: {json.dumps(service, indent=2)}
            CPT code resolved: {cpt_code}
            ICD-10 validation: {json.dumps(icd_validation, indent=2)}
            Previous audit issues to fix (if any): {state.get("audit_issues", [])}
            """
        }
    ])

    # extract JSON from the LLM response
    raw = reply.content
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    claim = json.loads(match.group()) if match else {}

    # extract missing fields note
    missing = []
    if "MISSING:" in raw:
        missing_line = raw.split("MISSING:")[-1].strip()
        missing = [m.strip() for m in missing_line.split(",") if m.strip()]

    return {
        "claim_json": claim,
        "missing_fields": missing,
        "messages": [{"role": "assistant", "content": f"[Claim Creator] Claim built. Missing fields: {missing or 'None'}"}]
    }


# agent node 3: claim auditor
def claim_auditor_agent(state: State):
    claim = state.get("claim_json", {})

    audit_llm = llm.with_structured_output(AuditResult)

    result = audit_llm.invoke([
        {
            "role": "system",
            "content": """You are an expert Insurance Claim Auditor and Verifier. Review completed 
            insurance claims to ensure they are accurate, complete, and unlikely to be denied.
            Checklist:
            - All required fields are present (patient, provider, service, diagnosis, billing)
            - CPT and ICD-10 codes are clinically consistent with each other
            - Date of service is not in the future
            - NPI number is present (must be exactly 10 digits)
            - Billed amount is reasonable for the CPT code
            - Place of service code is present
            - No obvious red flags like duplicate codes
            Return verdict as APPROVED or NEEDS CORRECTION.
            If NEEDS CORRECTION, populate corrected_claim with the fixed version.
            Be specific and actionable in your issues list."""
        },
        {
            "role": "user",
            "content": f"Claim to audit:\n{json.dumps(claim, indent=2)}"
        }
    ])

    # save file if approved
    saved_file = None
    if result.verdict == "APPROVED":
        saved_file = save_claim_to_file(claim)

    return {
        "audit_verdict": result.verdict,
        "audit_issues": result.issues,
        # use corrected version if provided
        "claim_json": result.corrected_claim or claim,
        "messages": [{"role": "assistant", "content": f"[Auditor] {result.verdict}. {'Saved to: ' + saved_file if saved_file else 'Issues: ' + str(result.issues)}"}]
    }


# router 2: approve or retry claim
def audit_router(state: State):
    retry_count = state.get("retry_count", 0)

    if state.get("audit_verdict") == "APPROVED":
        return {"next": "approved"}

    # Safety cap: stop retrying after 3 attempts
    if retry_count >= 3:
        return {"next": "approved"}  # accept best effort after max retries

    return {"next": "create_claim", "retry_count": retry_count + 1}


# building the graph
graph_builder = StateGraph(State)

graph_builder.add_node("eligibility_agent",  eligibility_agent)
graph_builder.add_node("eligibility_router", eligibility_router)
graph_builder.add_node("claim_creator",      claim_creator_agent)
graph_builder.add_node("claim_auditor",      claim_auditor_agent)
graph_builder.add_node("audit_router",       audit_router)

graph_builder.add_edge(START,                "eligibility_agent")
graph_builder.add_edge("eligibility_agent",  "eligibility_router")

graph_builder.add_conditional_edges(
    "eligibility_router",
    lambda state: state.get("next"),
    {"create_claim": "claim_creator", "denied": END}
)

graph_builder.add_edge("claim_creator", "claim_auditor")
graph_builder.add_edge("claim_auditor", "audit_router")

graph_builder.add_conditional_edges(
    "audit_router",
    lambda state: state.get("next"),
    {"approved": END, "create_claim": "claim_creator"}
)

graph = graph_builder.compile()

# runner to run all of the agents


def run_billing_pipeline():
    print("=== Health Insurance Billing Pipeline ===\n")

    # collect patient data
    print("Patient Information")
    patient_data = {
        "name":           input("Patient full name: ").strip(),
        "dob":            input("Date of birth (YYYY-MM-DD): ").strip(),
        "member_id":      input("Insurance member ID: ").strip(),
        "insurance_plan": input("Insurance plan name: ").strip(),
        "address":        input("Patient address: ").strip(),
    }

    # collect service data
    print("\nService Information")
    service_data = {
        "service":              input("Service description (e.g. 'office visit'): ").strip(),
        "date_of_service":      input("Date of service (YYYY-MM-DD): ").strip(),
        "provider_name":        input("Provider full name: ").strip(),
        "npi":                  input("Provider NPI number (10 digits): ").strip(),
        "provider_address":     input("Provider address: ").strip(),
        "place_of_service_code": input("Place of service code (e.g. '11' for office): ").strip(),
        "diagnosis":            input("ICD-10 diagnosis code (e.g. 'J06.9'): ").strip(),
        "billed_amount":        float(input("Billed amount ($): ").strip()),
        "units":                int(input("Units: ").strip()),
    }

    # cpt code is optional; if left blank, Agent 2 will look it up
    cpt_input = input("CPT code (press Enter to auto-lookup): ").strip()
    if cpt_input:
        service_data["cpt_code"] = cpt_input

    # run pipeline
    initial_state = {
        "messages": [],
        "patient_data": patient_data,
        "service_data": service_data,
        "eligibility_verdict": None,
        "eligibility_reason": None,
        "coverage_gaps": None,
        "claim_json": None,
        "missing_fields": None,
        "audit_verdict": None,
        "audit_issues": None,
        "next": None,
        "retry_count": 0
    }

    print("\nProcessing claim...\n")
    final_state = graph.invoke(initial_state)

    # print results
    print("agent trail")
    for msg in final_state.get("messages", []):
        print(f"  {msg.content}")

    print("\nFinal Result")
    print(f"  Eligibility : {final_state.get('eligibility_verdict')}")
    print(f"  Reason      : {final_state.get('eligibility_reason')}")
    print(f"  Audit       : {final_state.get('audit_verdict')}")

    if final_state.get("coverage_gaps"):
        print(f"  Coverage gaps: {final_state.get('coverage_gaps')}")

    if final_state.get("audit_issues"):
        print(f"  Issues      : {final_state.get('audit_issues')}")

    if final_state.get("claim_json"):
        print(f"\nFinal Claim JSON")
        print(json.dumps(final_state.get("claim_json"), indent=2))


if __name__ == "__main__":
    run_billing_pipeline()
