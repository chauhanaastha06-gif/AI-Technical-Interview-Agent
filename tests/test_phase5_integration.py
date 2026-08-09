import pytest
from app.services.interview_engine import InterviewEngine
from app.services.session_manager import session_manager
from app.data.loader import data_loader
from app.models.schemas import CandidateData


@pytest.fixture(autouse=True)
def cleanup_sessions():
    session_manager.clear()
    yield
    session_manager.clear()


def test_start_interview_integration():
    engine = InterviewEngine(max_turns=5)
    raw_cand = data_loader.get_candidate_by_id("CAND-001")
    cand_data = CandidateData(**raw_cand)

    session_id = "test-session-phase5-start"
    res = engine.start_interview(session_id, cand_data)

    assert res.done is False
    assert res.reply is not None and len(res.reply) > 5
    assert res.currentTopicModule is not None
    assert res.assessmentFocus is not None

    state = session_manager.get(session_id)
    assert state.turn_count == 0
    assert len(state.asked_questions) == 1


def test_unknown_answer_no_positive_validation():
    engine = InterviewEngine(max_turns=5)
    raw_cand = data_loader.get_candidate_by_id("CAND-001")
    cand_data = CandidateData(**raw_cand)

    session_id = "test-session-phase5-unknown"
    start_res = engine.start_interview(session_id, cand_data)

    # Candidate responds "i dont know"
    turn_res = engine.handle_turn(session_id, "i dont know")

    assert turn_res.done is False
    assert turn_res.skillEvaluation is not None
    assert turn_res.skillEvaluation.status == "Needs Attention"

    # Crucial: Reply must NEVER contain positive validation phrases
    POSITIVE_PHRASES = [
        "that makes sense", "great answer", "excellent", "good point",
        "exactly", "absolutely", "thats correct", "that's correct", "well said", "nice"
    ]
    reply_lower = turn_res.reply.lower()
    for phrase in POSITIVE_PHRASES:
        assert phrase not in reply_lower, f"Positive validation phrase '{phrase}' found in reply after 'i dont know'"


def test_strong_answer_evaluation_and_progression():
    engine = InterviewEngine(max_turns=5)
    raw_cand = data_loader.get_candidate_by_id("CAND-001")
    cand_data = CandidateData(**raw_cand)

    session_id = "test-session-phase5-strong"
    engine.start_interview(session_id, cand_data)

    strong_answer = "Vector embeddings transform high-dimensional text into dense floating point vectors using Cosine similarity and HNSW index structures."
    turn_res = engine.handle_turn(session_id, strong_answer)

    assert turn_res.done is False
    assert turn_res.skillEvaluation is not None
    assert turn_res.skillEvaluation.status in ["Strong", "Good"]
    assert turn_res.currentTopicModule is not None


def test_max_probes_per_module():
    engine = InterviewEngine(max_turns=10)
    raw_cand = data_loader.get_candidate_by_id("CAND-001")
    cand_data = CandidateData(**raw_cand)

    session_id = "test-session-phase5-probes"
    engine.start_interview(session_id, cand_data)

    state = session_manager.get(session_id)
    # Simulate probing module 2 times
    mod = state.current_module
    state.module_probe_counts[mod] = 2

    # Next turn should switch module
    turn_res = engine.handle_turn(session_id, "We use RAG pipelines for contextual retrieval.")
    assert turn_res.currentTopicModule is not None


def test_full_interview_completion():
    engine = InterviewEngine(max_turns=3)
    raw_cand = data_loader.get_candidate_by_id("CAND-001")
    cand_data = CandidateData(**raw_cand)

    session_id = "test-session-phase5-complete"
    engine.start_interview(session_id, cand_data)

    engine.handle_turn(session_id, "Answer 1 with embeddings details.")
    engine.handle_turn(session_id, "Answer 2 with RAG details.")
    final_res = engine.handle_turn(session_id, "Answer 3 with LLM prompt details.")

    assert final_res.done is True
    assert final_res.feedback is not None
    assert final_res.feedback.summary is not None
    assert final_res.feedback.skillProfile is not None
    assert len(final_res.feedback.skillProfile) == 7
