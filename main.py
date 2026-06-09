from dotenv import load_dotenv
import json
from db import init_db, get_patient, save_patient, save_claim, get_claims_for_patient
from graph import graph

load_dotenv()


def prompt_patient_data() -> dict:
    print("\nPatient Information")
    member_id = input("  Member ID: ").strip()

    existing = get_patient(member_id)
    if existing:
        print(
            f"  Found existing patient: {existing['name']} ({existing['insurance_plan']})")
        use_existing = input("  Use saved data? (y/n): ").strip().lower()
        if use_existing == "y":
            return existing

    patient = {
        "member_id":      member_id,
        "name":           input("  Full name: ").strip(),
        "dob":            input("  Date of birth (YYYY-MM-DD): ").strip(),
        "insurance_plan": input("  Insurance plan: ").strip(),
        "address":        input("  Address: ").strip(),
    }
    save_patient(patient)
    print("  Patient saved.")
    return patient


def prompt_service_data() -> dict:
    print("\nService Information")
    return {
        "service":               input("  Service description: ").strip(),
        "date_of_service":       input("  Date of service (YYYY-MM-DD): ").strip(),
        "provider_name":         input("  Provider name: ").strip(),
        "npi":                   input("  Provider NPI (10 digits): ").strip(),
        "provider_address":      input("  Provider address: ").strip(),
        "place_of_service_code": input("  Place of service code (e.g. 11): ").strip(),
        "diagnosis":             input("  ICD-10 code (e.g. J06.9): ").strip(),
        "billed_amount":         float(input("  Billed amount ($): ").strip()),
        "units":                 int(input("  Units: ").strip()),
    }


def show_claim_history(member_id: str):
    claims = get_claims_for_patient(member_id)
    if not claims:
        print("  No prior claims found.")
        return
    print(f"\n  Prior claims for member {member_id}:")
    for c in claims:
        print(
            f"    [{c['created_at'][:10]}] {c['service']} — {c['audit_verdict']} (id: {c['id']})")


def run():
    init_db()
    print("Health Insurance Billing Pipeline")

    patient_data = prompt_patient_data()

    show_history = input(
        "\n  View prior claims for this patient? (y/n): ").strip().lower()
    if show_history == "y":
        show_claim_history(patient_data["member_id"])

    service_data = prompt_service_data()

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
        "retry_count": 0
    }

    print("\nProcessing...\n")
    final = graph.invoke(initial_state)

    print("Agent trail:")
    for msg in final.get("messages", []):
        print(f"  {msg.content}")

    print(f"\nEligibility : {final.get('eligibility_verdict')}")
    print(f"Reason      : {final.get('eligibility_reason')}")
    print(f"Audit       : {final.get('audit_verdict')}")

    if final.get("coverage_gaps"):
        print(f"Gaps        : {final.get('coverage_gaps')}")
    if final.get("audit_issues"):
        print(f"Issues      : {final.get('audit_issues')}")

    # persist approved claim to DB
    if final.get("audit_verdict") == "APPROVED" and final.get("claim_json"):
        claim_id = save_claim(
            member_id=patient_data["member_id"],
            service=service_data["service"],
            claim=final["claim_json"],
            verdict=final["audit_verdict"],
            issues=final.get("audit_issues", [])
        )
        print(f"\nClaim saved to database (id: {claim_id})")
        print(f"Final Claim:\n{json.dumps(final['claim_json'], indent=2)}")


if __name__ == "__main__":
    run()
