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


def test_start_interview_contract():
    cand = data_loader.get_candidate_by_id("CAND-003")
    payload = {
        "sessionId": "test-session-start",
        "candidate": cand,
    }

    res = client.post("/api/interview", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert "reply" in data
    assert isinstance(data["reply"], str)
    assert len(data["reply"]) > 0
    assert data["done"] is False
    assert "feedback" not in data or data.get("feedback") is None


def test_full_interview_flow_with_feedback():
    # Configure 3 turns for fast test execution
    original_turns = interview_engine.max_turns
    interview_engine.max_turns = 3

    try:
        cand = data_loader.get_candidate_by_id("CAND-003")
        session_id = "test-session-full"

        # 1. Start Interview
        start_res = client.post("/api/interview", json={"sessionId": session_id, "candidate": cand})
        assert start_res.status_code == 200
        assert start_res.json()["done"] is False

        # 2. Turn 1
        turn1_res = client.post(
            "/api/interview",
            json={"sessionId": session_id, "message": "I use cosine similarity with HNSW indexing for high recall and sub-50ms latency."},
        )
        assert turn1_res.status_code == 200
        assert turn1_res.json()["done"] is False
        assert "reply" in turn1_res.json()

        # 3. Turn 2
        turn2_res = client.post(
            "/api/interview",
            json={"sessionId": session_id, "message": "We enforce strict Pydantic schemas and enable JSON mode to guarantee type-safe tool calls."},
        )
        assert turn2_res.status_code == 200
        assert turn2_res.json()["done"] is False

        # 4. Turn 3 (Final turn reaching max_turns=3)
        turn3_res = client.post(
            "/api/interview",
            json={"sessionId": session_id, "message": "For agents, we implement circuit breakers, max iteration limits, and Redis-backed session memory."},
        )
        assert turn3_res.status_code == 200
        final_data = turn3_res.json()

        assert final_data["done"] is True
        assert "reply" in final_data
        assert "feedback" in final_data

        fb = final_data["feedback"]
        assert "summary" in fb and isinstance(fb["summary"], str) and len(fb["summary"]) > 0
        assert "strengths" in fb and isinstance(fb["strengths"], list) and len(fb["strengths"]) > 0
        assert "gaps" in fb and isinstance(fb["gaps"], list) and len(fb["gaps"]) > 0
        assert "next" in fb and isinstance(fb["next"], list) and len(fb["next"]) > 0

    finally:
        interview_engine.max_turns = original_turns


def test_interview_for_struggling_candidate():
    original_turns = interview_engine.max_turns
    interview_engine.max_turns = 2

    try:
        cand = data_loader.get_candidate_by_id("CAND-010")
        session_id = "test-session-struggling"

        # Start
        start_res = client.post("/api/interview", json={"sessionId": session_id, "candidate": cand})
        assert start_res.status_code == 200
        assert start_res.json()["done"] is False

        # Opening question should mention candidate or relevant topics
        reply = start_res.json()["reply"]
        assert "Gerald" in reply or "Vector" in reply or "Embeddings" in reply or "Day 8" in reply or "Day 10" in reply

        # Turn 1
        t1 = client.post("/api/interview", json={"sessionId": session_id, "message": "I struggled with vector indexes in Day 8 but learned how cosine distance works."})
        assert t1.status_code == 200

        # Turn 2 -> Complete
        t2 = client.post("/api/interview", json={"sessionId": session_id, "message": "I would improve logging and error handling for multi-agent loops."})
        assert t2.status_code == 200
        final_data = t2.json()
        assert final_data["done"] is True
        assert "feedback" in final_data
        assert len(final_data["feedback"]["gaps"]) > 0
    finally:
        interview_engine.max_turns = original_turns


def test_interview_for_skipped_candidate():
    original_turns = interview_engine.max_turns
    interview_engine.max_turns = 2

    try:
        cand = data_loader.get_candidate_by_id("CAND-011")
        session_id = "test-session-skipped"

        # Start
        start_res = client.post("/api/interview", json={"sessionId": session_id, "candidate": cand})
        assert start_res.status_code == 200
        assert start_res.json()["done"] is False

        # Turn 1 & 2
        client.post("/api/interview", json={"sessionId": session_id, "message": "I understand the concepts from a UX perspective."})
        t2 = client.post("/api/interview", json={"sessionId": session_id, "message": "I focus on user feedback loops and evaluation benchmarks."})
        assert t2.status_code == 200
        final_data = t2.json()
        assert final_data["done"] is True
        assert "feedback" in final_data
        assert any("skipped" in g.lower() or "unverified" in g.lower() or "Day" in g for g in final_data["feedback"]["gaps"] + final_data["feedback"]["next"])
    finally:
        interview_engine.max_turns = original_turns


def test_mock_10_turn_sequence_uniqueness():
    """
    Verifies that a full 10-turn mock interview generates distinct, non-repeating questions
    for all candidate answer turns 0 through 9, terminates on turn 10 with done=true,
    and returns valid structured feedback.
    """
    cand = data_loader.get_candidate_by_id("CAND-003")
    session_id = "test-session-10-turns-unique"

    # Start Interview (Turn 0 question returned)
    start_res = client.post("/api/interview", json={"sessionId": session_id, "candidate": cand})
    assert start_res.status_code == 200
    start_data = start_res.json()
    assert start_data["done"] is False

    questions = [start_data["reply"]]

    # Execute candidate answer turns 1 through 9
    for i in range(1, 10):
        res = client.post(
            "/api/interview",
            json={
                "sessionId": session_id,
                "message": f"Candidate answer for technical topic iteration {i}.",
            },
        )
        assert res.status_code == 200
        data = res.json()
        assert data["done"] is False
        questions.append(data["reply"])

    # Verify that all 10 questions (turns 0-9) are distinct
    assert len(questions) == 10
    unique_questions = set(questions)
    assert len(unique_questions) == 10, f"Expected 10 unique questions, but found {len(unique_questions)} unique"

    # Turn 10 (Final candidate answer turn reaching MAX_TURNS=10)
    final_res = client.post(
        "/api/interview",
        json={"sessionId": session_id, "message": "Final candidate turn concluding interview."},
    )
    assert final_res.status_code == 200
    final_data = final_res.json()

    assert final_data["done"] is True
    assert "feedback" in final_data
    fb = final_data["feedback"]
    assert "summary" in fb and isinstance(fb["summary"], str) and len(fb["summary"]) > 0
    assert "strengths" in fb and isinstance(fb["strengths"], list) and len(fb["strengths"]) > 0
    assert "gaps" in fb and isinstance(fb["gaps"], list) and len(fb["gaps"]) > 0
    assert "next" in fb and isinstance(fb["next"], list) and len(fb["next"]) > 0

