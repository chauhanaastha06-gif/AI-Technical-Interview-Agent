import sys
import json
import uuid
import httpx
from pathlib import Path

# Set UTF-8 output encoding if supported
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_manual_test(base_url: str = "http://127.0.0.1:8000", candidate_id: str = "CAND-003"):
    print("=" * 70)
    print("AI INTERVIEW AGENT - END-TO-END MANUAL VERIFICATION")
    print("=" * 70)
    print(f"Target Base URL: {base_url}")
    print(f"Selected Candidate ID: {candidate_id}")

    # Load candidate data
    candidates_file = PROJECT_ROOT / "app" / "data" / "candidates.json"
    with open(candidates_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    candidates = {c["member"]["id"]: c for c in data.get("candidates", [])}
    candidate = candidates.get(candidate_id)
    if not candidate:
        print(f"[!] Candidate '{candidate_id}' not found! Using first available candidate.")
        candidate = data["candidates"][0]

    session_id = f"manual-session-{uuid.uuid4().hex[:8]}"
    print(f"Candidate: {candidate['member']['name']} ({candidate['member']['jobRole']}, {candidate['member']['yearsExperience']} yrs exp)")
    print(f"Session ID: {session_id}\n")

    # Connect to server
    client = httpx.Client(base_url=base_url, timeout=30.0)

    try:
        # 1. Check Health
        try:
            health = client.get("/health").json()
            print(f"[+] Server Health: {health['status']} | Mode: {health.get('model', 'N/A')}")
        except Exception as e:
            print("[!] Server not responding at base_url. If the server is not running, start it using:")
            print("    uvicorn app.main:app --reload\n")

        # 2. Start Interview
        print("-" * 70)
        print("STEP 1: START INTERVIEW")
        print("-" * 70)
        start_payload = {
            "sessionId": session_id,
            "candidate": candidate,
        }
        res = client.post("/api/interview", json=start_payload)
        if res.status_code != 200:
            print(f"[!] Start request failed with HTTP {res.status_code}: {res.text}")
            return

        res_data = res.json()
        print(f"Interviewer [Start]:\n{res_data.get('reply')}\n")
        print(f"Done status: {res_data.get('done')}")

        # Realistic candidate answers for up to MAX_TURNS
        realistic_answers = [
            "We used Cosine Similarity on 1536-dimensional OpenAI text-embedding-3-small vectors, indexing them in Qdrant with HNSW. Cosine similarity worked best for normalized document chunks.",
            "For hybrid search, we combine sparse BM25 scores with dense vector similarity using Reciprocal Rank Fusion (RRF), weighting keyword matches at 0.3 and semantic vectors at 0.7.",
            "To enforce structured outputs, we define Pydantic models with Field constraints and pass them via Anthropic tool definitions or OpenAI JSON schema mode, with automatic retry validation.",
            "For agent orchestration, we implement LangGraph state machines with checkpoints. Each agent has bounded tool execution limits and explicit timeouts to prevent runaway loops.",
            "In our Capstone project, we deployed the FastAPI microservice on Kubernetes with an HPA scaling on request latency, Redis caching for embedding lookups, and Prometheus observability.",
            "For rate limiting we use Redis Token Bucket algorithms, and for caching we cache prompt embeddings and deterministic LLM responses using semantic similarity thresholds.",
            "We conduct evaluations using Ragas to measure faithfulness, answer relevancy, and context recall against curated ground-truth datasets.",
            "For security guardrails, we use NeMo Guardrails and regex filters to scrub PII and prevent jailbreak attempts before queries hit the foundation model.",
            "We containerize all worker nodes using slim multi-stage Docker builds and manage deployments via ArgoCD with automated canary rollouts.",
            "In summary, we prioritize resilient architecture, comprehensive monitoring with OpenTelemetry, and strict schema validation across the full pipeline.",
        ]

        turn = 1
        done = res_data.get("done", False)

        for answer in realistic_answers:
            if done:
                break

            print("-" * 70)
            print(f"STEP {turn + 1}: CANDIDATE TURN {turn}")
            print("-" * 70)
            print(f"Candidate: \n{answer}\n")

            turn_payload = {
                "sessionId": session_id,
                "message": answer,
            }
            res = client.post("/api/interview", json=turn_payload)
            if res.status_code != 200:
                print(f"[!] Turn request failed with HTTP {res.status_code}: {res.text}")
                return

            res_data = res.json()
            done = res_data.get("done", False)
            print(f"Interviewer:\n{res_data.get('reply')}\n")
            print(f"Done status: {done}")
            turn += 1

        if done and "feedback" in res_data:
            fb = res_data["feedback"]
            print("=" * 70)
            print("INTERVIEW COMPLETED - FINAL STRUCTURED FEEDBACK")
            print("=" * 70)
            print(f"SUMMARY:\n{fb.get('summary')}\n")
            print("STRENGTHS:")
            for s in fb.get("strengths", []):
                print(f"  + {s}")
            print("\nGAPS / AREAS FOR GROWTH:")
            for g in fb.get("gaps", []):
                print(f"  ! {g}")
            print("\nRECOMMENDED NEXT STEPS:")
            for n in fb.get("next", []):
                print(f"  > {n}")
            print("=" * 70)
            print("[+] Manual test passed successfully with valid API contract!")
        else:
            print(f"[*] Interview did not complete within the test turns (Done: {done}).")

    except httpx.ConnectError:
        print(f"[!] Could not connect to {base_url}. Please ensure the server is running.")
    finally:
        client.close()


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    cid = sys.argv[2] if len(sys.argv) > 2 else "CAND-003"
    run_manual_test(base_url=url, candidate_id=cid)
