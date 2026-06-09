# Multi Agent Billing System
A multi-agent system built with LangGraph that automates health insurance claims processing. Cover eligibility verification, claim creation, and audit.

## Details
Agent 1: Eligibility Verifier - checks whether the patient's insurance covers the requested service before any claim is filed

If the patient is ineligible, the piple stops early.

Agent 2: Claim Creator - builds a structured insruance claim with CPT and ICD codes

Agent 3: Claim Auditor - reviews the claim for accuracy and completeness; retries if errors are found

If the claim is approved, it is saved as a JSON file.

## Tech Stack
LangGraph - agent orchestration and state graph
LangChain - LLM interface
Anthropic Claude (Sonnet 4-6) - underlying model for all agents
Pydantic - structured output validation

## Setup
1 Clone this repo
2 Create a .env file wiht your own API key: API_KEY=your_key_here
3 Install dependencies: pip install -r requirements.txt
4 Run python main.py
5 NEVER commit .env file. It contains your personal API key.
