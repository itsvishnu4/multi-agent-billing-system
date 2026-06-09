import sqlite3
import json
import uuid
from datetime import datetime

DB_PATH = "billing.db"


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                member_id     TEXT PRIMARY KEY,
                name          TEXT NOT NULL,
                dob           TEXT NOT NULL,
                insurance_plan TEXT NOT NULL,
                address       TEXT NOT NULL,
                created_at    TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS claims (
                id            TEXT PRIMARY KEY,
                member_id     TEXT NOT NULL,
                service       TEXT,
                claim_json    TEXT,
                audit_verdict TEXT,
                audit_issues  TEXT,
                created_at    TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (member_id) REFERENCES patients(member_id)
            )
        """)


def get_patient(member_id: str) -> dict | None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM patients WHERE member_id = ?", (member_id,)
        ).fetchone()
    return dict(row) if row else None


def save_patient(patient: dict):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO patients (member_id, name, dob, insurance_plan, address)
            VALUES (:member_id, :name, :dob, :insurance_plan, :address)
        """, patient)


def save_claim(member_id: str, service: str, claim: dict, verdict: str, issues: list) -> str:
    claim_id = uuid.uuid4().hex[:8]
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO claims (id, member_id, service, claim_json, audit_verdict, audit_issues)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (claim_id, member_id, service, json.dumps(claim), verdict, json.dumps(issues)))
    return claim_id


def get_claims_for_patient(member_id: str) -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM claims WHERE member_id = ? ORDER BY created_at DESC",
            (member_id,)
        ).fetchall()
    return [dict(r) for r in rows]
