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


def test_assessment_focus_returned():
    cand = data_loader.get_candidate_by_id("CAND-003")
    res = client.post("/api/interview", json={"sessionId": "test-focus", "candidate": cand})
    assert res.status_code == 200
    data = res.json()

    assert "assessmentFocus" in data
    focus = data["assessmentFocus"]
    assert "topic" in focus and isinstance(focus["topic"], str)
    assert "reason" in focus and isinstance(focus["reason"], str)


def test_skill_profile_in_feedback():
    original_turns = interview_engine.max_turns
    interview_engine.max_turns = 1

    try:
        cand = data_loader.get_candidate_by_id("CAND-003")
        session_id = "test-skill-profile"

        client.post("/api/interview", json={"sessionId": session_id, "candidate": cand})
        turn_res = client.post("/api/interview", json={"sessionId": session_id, "message": "My answer for technical turn."})
        assert turn_res.status_code == 200
        data = turn_res.json()
        assert data["done"] is True
        assert "feedback" in data
        fb = data["feedback"]
        assert "skillProfile" in fb
        sp = fb["skillProfile"]
        assert isinstance(sp, dict)
        assert "RAG & Retrieval Architecture" in sp
    finally:
        interview_engine.max_turns = original_turns


def test_multi_candidate_flexibility():
    candidate_ids = ["CAND-003", "CAND-010", "CAND-011"]

    for cid in candidate_ids:
        cand = data_loader.get_candidate_by_id(cid)
        assert cand is not None, f"Candidate {cid} not found"
        session_id = f"test-cand-{cid}"

        res = client.post("/api/interview", json={"sessionId": session_id, "candidate": cand})
        assert res.status_code == 200
        data = res.json()
        assert data["done"] is False
        assert "reply" in data
        assert "assessmentFocus" in data
        assert len(data["assessmentFocus"]["topic"]) > 0
