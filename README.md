# 🤖 AI Technical Interview Agent Backend

A production-grade, adaptive technical interview backend designed for evaluating AI engineering cohort candidates. The agent personalizes each interview based on candidate curriculum history, past missions, pass/fail attempts, job role, and years of experience, producing actionable structured feedback upon completion.

---

## 🌟 Key Features

1. **Exact API Contract Compliance**: Implements the unified `POST /api/interview` endpoint adhering strictly to `technical-spec.md`.
2. **Deterministic & Adaptive LLM Modes**:
   - **Real Claude Mode**: Uses the official Anthropic SDK (`claude-3-5-sonnet-20241022` or configurable model) with structured tool calling for feedback.
   - **Mock Mode**: Fully operational, zero-configuration deterministic LLM mode that allows 100% test coverage and local development without requiring an API key.
3. **Deep Candidate & Curriculum Intelligence**:
   - Joins candidate missions with 31-day, 8-module curriculum metadata.
   - Categorizes topics into **MASTERED**, **STRUGGLED**, **FAILED**, and **SKIPPED**.
   - Dynamically calibrates interview depth and pacing to candidate role (Intern, Mid-Level, Senior, Distinguished).
4. **Stateful In-Memory Session Management**:
   - Tracks turn counts, conversation history, candidate briefs, and final feedback across HTTP requests via `sessionId`.
5. **Robust Error Resilience & Guardrails**:
   - Handles empty/whitespace input without crashing.
   - Retries structured feedback generation with automated fallback.
   - Configurable hard maximum turn limit (`MAX_TURNS`).

---

## 🏗️ Architecture & Project Structure

```
interview-agent/
│
├── app/
│   ├── main.py                  # FastAPI application entrypoint with lifespan handlers
│   ├── config.py                # Environment configuration (Anthropic, MAX_TURNS, etc.)
│   │
│   ├── models/
│   │   ├── schemas.py           # Pydantic request/response schemas matching technical-spec.md
│   │   └── domain.py            # Internal typed models (InterviewBrief, MissionAnalysis, etc.)
│   │
│   ├── data/
│   │   ├── candidates.json      # Cohort candidates dataset
│   │   ├── curriculum.json      # 31-day AI engineering curriculum dataset
│   │   └── loader.py            # Startup dataset validation and indexed lookup
│   │
│   ├── services/
│   │   ├── session_manager.py   # In-memory session store lifecycle management
│   │   ├── candidate_profile.py # Candidate profile analysis, bucketing, and difficulty calibration
│   │   ├── prompt_builder.py    # LLM system prompts and structured feedback prompt formatting
│   │   ├── llm_client.py        # Anthropic SDK client + Mock engine + fallback recovery
│   │   └── interview_engine.py  # Core orchestration: start, turn, and finish
│   │
│   ├── api/
│   │   └── interview_routes.py  # POST /api/interview route handler
│   │
│   └── utils/
│       └── logging.py           # Structured logging utility
│
├── tests/
│   ├── test_interview_flow.py   # End-to-end API lifecycle tests for multiple candidate profiles
│   ├── test_session_manager.py  # Session state and concurrency tests
│   ├── test_candidate_profile.py# Candidate bucketing (mastered/struggled/failed/skipped) tests
│   ├── test_prompt_builder.py   # Prompt generation tests
│   └── test_edge_cases.py       # Duplicate sessions, unknown sessions, whitespace input tests
│
├── .env.example                 # Example environment variables
├── requirements.txt             # Python dependencies
├── README.md                    # Documentation
└── run_manual_test.py           # Interactive end-to-end HTTP verification script
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+ (Tested on Python 3.14)
- `pip`

### 2. Installation

Navigate to the `interview-agent` directory:

```bash
cd interview-agent
```

Create and activate a virtual environment:

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Environment variables:

| Variable | Description | Default |
|---|---|---|
| `ANTHROPIC_API_KEY` | Anthropic API Key (leave empty for Mock mode) | `""` |
| `ANTHROPIC_MODEL` | Claude model name | `claude-3-5-sonnet-20241022` |
| `MAX_TURNS` | Max candidate-answer turns before concluding | `10` |
| `HOST` | Server host binding | `0.0.0.0` |
| `PORT` | Server port binding | `8000` |

---

## 🖥️ Running the Application

### Start the FastAPI Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The server will start at: `http://127.0.0.1:8000`
- Interactive API Docs: `http://127.0.0.1:8000/docs`
- Health Check: `http://127.0.0.1:8000/health`

---

## 🧪 Running Tests

Execute the comprehensive test suite with `pytest`:

```bash
python -m pytest
```

Run tests with verbose output:

```bash
python -m pytest -v
```

---

## 🔬 Running Manual HTTP Verification

With the server running, execute the automated manual test client:

```bash
# Test default candidate (CAND-003 Emily Chen)
python run_manual_test.py

# Test a struggling candidate (CAND-010 Gerald Combs)
python run_manual_test.py http://127.0.0.1:8000 CAND-010

# Test a candidate with skipped missions (CAND-011 Mia Alvarez)
python run_manual_test.py http://127.0.0.1:8000 CAND-011
```

---

## 📡 API Specification & Sample Payloads

### 1. Start Interview
**Request:**
```http
POST /api/interview HTTP/1.1
Content-Type: application/json

{
  "sessionId": "interview-session-001",
  "candidate": {
    "member": {
      "id": "CAND-003",
      "name": "Emily Chen",
      "jobRole": "AI Engineer",
      "yearsExperience": 6,
      "education": "MS Artificial Intelligence",
      "status": "COMPLETED"
    },
    "missions": [
      { "day": 7, "title": "Embeddings Explained", "passed": true, "attempts": 1 },
      { "day": 8, "title": "Vector Databases Overview", "passed": true, "attempts": 1 },
      { "day": 10, "title": "Retrieval & Matching Engine", "passed": true, "attempts": 1 },
      { "day": 11, "title": "RAG End-to-End & LLM API Basics", "passed": true, "attempts": 1 }
    ],
    "signals": {
      "commitDays": 31,
      "missionsCompleted": 31,
      "missionsFirstTry": 30
    }
  }
}
```

**Response (HTTP 200):**
```json
{
  "reply": "Welcome Emily Chen! It's a pleasure to conduct your technical interview. Based on your impressive track record as a AI Engineer, let's start with Embeddings Explained. Could you walk me through the end-to-end data pipeline from document chunking to semantic vector retrieval, and how you optimize for search latency?",
  "done": false
}
```

---

### 2. Conversation Turn
**Request:**
```http
POST /api/interview HTTP/1.1
Content-Type: application/json

{
  "sessionId": "interview-session-001",
  "message": "We normalize 1536-dimensional embeddings, index using HNSW in Qdrant with an ef_search of 64, and combine sparse BM25 scores with dense vectors via Reciprocal Rank Fusion."
}
```

**Response (HTTP 200):**
```json
{
  "reply": "That makes sense. In a production environment with high throughput, how do you handle vector database index updates, and what strategies do you employ for hybrid search?",
  "done": false
}
```

---

### 3. Concluding Turn & Final Structured Feedback
**Response on completion (HTTP 200):**
```json
{
  "reply": "Thank you for completing this technical interview. We have concluded all interview questions.",
  "done": true,
  "feedback": {
    "summary": "Candidate Emily Chen (AI Engineer, 6.0 years experience) demonstrated strong technical proficiency and architectural maturity consistent with an AI Engineer. During the interview, the candidate engaged thoughtfully across curriculum modules, exhibiting clear strengths alongside specific areas identified for continued refinement before production deployment.",
    "strengths": [
      "Demonstrated solid mastery in core curriculum areas including Day 7 (Embeddings Explained), Day 8 (Vector Databases Overview), Day 10 (Retrieval & Matching Engine).",
      "High initial problem-solving velocity with a 97% first-try pass rate across cohort missions.",
      "Applied practical engineering perspective suited for AI Engineer role, considering system boundaries and integration."
    ],
    "gaps": [
      "Further optimization needed on edge-case error recovery and high-concurrency production latency."
    ],
    "next": [
      "Build a production benchmark suite measuring RAG retrieval recall, hallucination rate, and p99 latency.",
      "Study advanced agent orchestration patterns and standardized Model Context Protocol (MCP) integrations."
    ]
  }
}
```

---

## 🧩 Candidate Profile Bucketing Matrix

| Category | Criteria | Interview Strategy |
|---|---|---|
| **MASTERED** | `passed == true` AND `attempts <= 2` | High-level sanity check & architectural trade-offs |
| **STRUGGLED** | `passed == true` AND `attempts >= 3` | In-depth probing on edge cases and friction points |
| **FAILED** | `passed == false` | Top Priority: directly test whether knowledge was remediated |
| **SKIPPED** | `skipped == true` | Treat as unverified; test foundational understanding |

---

## 🔒 Error Handling

| Status Code | Scenario | Description |
|---|---|---|
| `400 Bad Request` | Missing/empty `sessionId` or invalid candidate data | Returns clear error message |
| `404 Not Found` | Unknown `sessionId` on turn request | Indicates session does not exist |
| `409 Conflict` | Attempting to initialize with an already active `sessionId` | Prevents accidental session overwrites |
| `200 OK` (Handled) | Empty/whitespace candidate response | Politely prompts user to answer without penalizing turn count |
| `200 OK` (Fallback) | Provider API failure during feedback generation | Retries once, then applies deterministic fallback feedback |
