import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.data.loader import data_loader
from app.services.session_manager import session_manager
from app.services.interview_engine import interview_engine

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_sessions():
    session_manager.clear()
    yield
    session_manager.clear()


def test_scenario_a_strong_interview_yields_strong_fit():
    """Scenario A: Candidate gives strong technical answers -> Strong Fit"""
    original_turns = interview_engine.max_turns
    interview_engine.max_turns = 2

    try:
        cand = data_loader.get_candidate_by_id("CAND-003")
        session_id = "test-strong-interview"

        client.post("/api/interview", json={"sessionId": session_id, "candidate": cand})
        
        # Turn 1: Strong answer with concepts (cosine, HNSW, vector)
        client.post(
            "/api/interview",
            json={
                "sessionId": session_id,
                "message": "We use cosine similarity metrics with HNSW index configuration in Qdrant vector database for sub-50ms latency."
            }
        )
        
        # Turn 2: Strong answer with concepts (Pydantic, JSON mode, schema)
        res = client.post(
            "/api/interview",
            json={
                "sessionId": session_id,
                "message": "We enforce strict Pydantic model schemas and JSON mode formatting with automated tool validation handlers."
            }
        )
        
        data = res.json()
        assert data["done"] is True
        assert "feedback" in data
        assert data["feedback"]["disposition"] == "Strong Fit"
    finally:
        interview_engine.max_turns = original_turns


def test_scenario_b_mixed_interview_yields_consider():
    """Scenario B: Candidate gives a mixture of adequate/weak answers -> Consider"""
    original_turns = interview_engine.max_turns
    interview_engine.max_turns = 3

    try:
        cand = data_loader.get_candidate_by_id("CAND-003")
        session_id = "test-mixed-interview"

        client.post("/api/interview", json={"sessionId": session_id, "candidate": cand})
        
        # Turn 1: Strong
        client.post(
            "/api/interview",
            json={
                "sessionId": session_id,
                "message": "We use cosine similarity metrics with HNSW index configuration in vector database."
            }
        )
        # Turn 2: Weak
        client.post(
            "/api/interview",
            json={"sessionId": session_id, "message": "I don't know much about Pydantic schemas."}
        )
        # Turn 3: Adequate
        res = client.post(
            "/api/interview",
            json={"sessionId": session_id, "message": "We use standard LangChain agents with simple tools."}
        )
        
        data = res.json()
        assert data["done"] is True
        assert "feedback" in data
        assert data["feedback"]["disposition"] == "Consider"
    finally:
        interview_engine.max_turns = original_turns


def test_scenario_c_weak_interview_yields_needs_development():
    """Scenario C: Candidate gives consistently weak answers -> Needs Development"""
    original_turns = interview_engine.max_turns
    interview_engine.max_turns = 3

    try:
        cand = data_loader.get_candidate_by_id("CAND-003")
        session_id = "test-weak-interview"

        client.post("/api/interview", json={"sessionId": session_id, "candidate": cand})
        
        client.post("/api/interview", json={"sessionId": session_id, "message": "I don't know."})
        client.post("/api/interview", json={"sessionId": session_id, "message": "No idea."})
        res = client.post("/api/interview", json={"sessionId": session_id, "message": "Pass on this question."})
        
        data = res.json()
        assert data["done"] is True
        assert "feedback" in data
        assert data["feedback"]["disposition"] == "Needs Development"
    finally:
        interview_engine.max_turns = original_turns


def test_scenario_d_strong_profile_with_weak_interview_is_not_strong_fit():
    """Scenario D: Strong historical candidate profile (Emily Chen 97% pass rate) + weak interview -> NOT Strong Fit"""
    original_turns = interview_engine.max_turns
    interview_engine.max_turns = 3

    try:
        cand = data_loader.get_candidate_by_id("CAND-003") # Emily Chen (97% first-try)
        session_id = "test-emily-weak"

        client.post("/api/interview", json={"sessionId": session_id, "candidate": cand})
        
        client.post("/api/interview", json={"sessionId": session_id, "message": "idk"})
        client.post("/api/interview", json={"sessionId": session_id, "message": "no experience"})
        res = client.post("/api/interview", json={"sessionId": session_id, "message": "cannot answer"})
        
        data = res.json()
        assert data["done"] is True
        assert "feedback" in data
        disposition = data["feedback"]["disposition"]
        assert disposition != "Strong Fit", f"Expected non-Strong Fit, got {disposition}"
        assert disposition == "Needs Development"
    finally:
        interview_engine.max_turns = original_turns


def test_scenario_e_weak_profile_with_strong_interview_improves_disposition():
    """Scenario E: Candidate with weak historical profile (Mia Alvarez 36% pass rate) + strong interview -> Disposition improves"""
    original_turns = interview_engine.max_turns
    interview_engine.max_turns = 2

    try:
        cand = data_loader.get_candidate_by_id("CAND-011") # Mia Alvarez (36% pass rate)
        session_id = "test-mia-strong"

        client.post("/api/interview", json={"sessionId": session_id, "candidate": cand})
        
        client.post(
            "/api/interview",
            json={
                "sessionId": session_id,
                "message": "We use cosine similarity metrics with HNSW index configuration for high recall and vector search."
            }
        )
        res = client.post(
            "/api/interview",
            json={
                "sessionId": session_id,
                "message": "We enforce strict Pydantic schemas and JSON mode formatting with automated tool validation handlers."
            }
        )
        
        data = res.json()
        assert data["done"] is True
        assert "feedback" in data
        disposition = data["feedback"]["disposition"]
        assert disposition in ["Strong Fit", "Consider"], f"Expected improved disposition, got {disposition}"
    finally:
        interview_engine.max_turns = original_turns
