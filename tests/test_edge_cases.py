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


def test_reused_session_id_error():
    cand = data_loader.get_candidate_by_id("CAND-001")
    session_id = "reused-session-test"

    # First start succeeds
    res1 = client.post("/api/interview", json={"sessionId": session_id, "candidate": cand})
    assert res1.status_code == 200

    # Second start with same session_id fails with 409 Conflict
    res2 = client.post("/api/interview", json={"sessionId": session_id, "candidate": cand})
    assert res2.status_code == 409
    assert "already exists" in res2.json()["detail"]


def test_unknown_session_id_error():
    res = client.post("/api/interview", json={"sessionId": "non-existent-session-id-999", "message": "Hello"})
    assert res.status_code == 404
    assert "does not exist" in res.json()["detail"]


def test_missing_or_empty_session_id():
    res = client.post("/api/interview", json={"sessionId": "   ", "message": "Hello"})
    assert res.status_code in [400, 422]


def test_empty_or_whitespace_message():
    cand = data_loader.get_candidate_by_id("CAND-001")
    session_id = "whitespace-msg-test"

    # Start
    client.post("/api/interview", json={"sessionId": session_id, "candidate": cand})

    # Send whitespace message
    res = client.post("/api/interview", json={"sessionId": session_id, "message": "    \n\t  "})
    assert res.status_code == 200
    data = res.json()
    assert data["done"] is False
    assert "Please provide an answer" in data["reply"]

    # Ensure turn count was not incremented
    state = session_manager.get(session_id)
    assert state.turn_count == 0


def test_candidate_with_minimal_missions():
    minimal_cand = {
        "member": {
            "id": "CAND-MINIMAL",
            "name": "Alex Minimal",
            "jobRole": "Junior Engineer",
            "yearsExperience": 1,
            "education": "BS CS",
            "status": "COMPLETED",
        },
        "missions": [
            {"day": 7, "title": "Embeddings Explained", "passed": True, "attempts": 1}
        ],
        "signals": {"commitDays": 5, "missionsCompleted": 1, "missionsFirstTry": 1},
    }

    session_id = "minimal-cand-session"
    res = client.post("/api/interview", json={"sessionId": session_id, "candidate": minimal_cand})
    assert res.status_code == 200
    assert res.json()["done"] is False


def test_turn_on_already_completed_interview():
    original_turns = interview_engine.max_turns
    interview_engine.max_turns = 1

    try:
        cand = data_loader.get_candidate_by_id("CAND-001")
        session_id = "already-done-session"

        # Start
        client.post("/api/interview", json={"sessionId": session_id, "candidate": cand})

        # Turn 1 -> completes
        res1 = client.post("/api/interview", json={"sessionId": session_id, "message": "My answer"})
        assert res1.status_code == 200
        assert res1.json()["done"] is True

        # Turn 2 -> returns completed
        res2 = client.post("/api/interview", json={"sessionId": session_id, "message": "Another message"})
        assert res2.status_code == 200
        assert res2.json()["done"] is True
        assert "feedback" in res2.json()
    finally:
        interview_engine.max_turns = original_turns
