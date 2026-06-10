# Multi Agent Billing System
A multi-agent system built with LangGraph that automates health insurance claims processing. Covers eligibility verification, claim creation, and audit.

## Details
Agent 1: Eligibility Verifier - checks whether the patient's insurance covers the requested service before any claim is filed

If the patient is ineligible, the pipeline stops early.

Agent 2: Claim Creator - builds a structured insruance claim with CPT and ICD codes

Agent 3: Claim Auditor - reviews the claim for accuracy and completeness; retries if errors are found

If the claim is approved, it is saved as a JSON file.

## Tech Stack
LangGraph - agent orchestration and state graph

LangChain - LLM interface

Anthropic Claude (Sonnet 4-6) - underlying model for all agents

Pydantic - structured output validation

SQLite - local data storage for patients and claims; easy to use since no setup is required

## Setup
1 Clone this repo

2 Create the virtual environment: 

python -m venv .venv 

source .venv/bin/activate on Mac

3 Install dependencies: pip install -r requirements.txt on Mac

4 Create a .env file with your own API key: API_KEY=your_key_here

5 Run python main.py

6 NEVER commit .env file. It contains your personal API key.

## How to Use
On the first run, enter patient and service information. This info will be saved to a local billing.db file by member ID. On future runs, entering the same member ID will allow you to reuse the saved data instead of re-entering everything.

You can also view prior claim history for any patient at the start of every run.

Approved claims will be stored in the database. The multi-agent system will retry claim creation up to 3 times if the auditor finds any errors.
