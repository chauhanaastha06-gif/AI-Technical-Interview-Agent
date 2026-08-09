import pytest
from app.services.interview_engine import (
    InterviewEngine,
    update_cumulative_module_assessment,
    calculate_deterministic_disposition_and_feedback,
)
from app.services.session_manager import session_manager
from app.data.loader import data_loader
from app.models.schemas import CandidateData


@pytest.fixture(autouse=True)
def cleanup_sessions():
    session_manager.clear()
    yield
    session_manager.clear()


def test_scenario_1_unknown_answer_needs_attention_no_positive_validation():
    """TEST 1: Candidate says 'i don't know' -> Needs Attention, score <= 0.10, no positive validation."""
    engine = InterviewEngine(max_turns=5)
    raw_cand = data_loader.get_candidate_by_id("CAND-001")
    cand_data = CandidateData(**raw_cand)
    session_id = "test-q-1"

    engine.start_interview(session_id, cand_data)
    res = engine.handle_turn(session_id, "i don't know")

    assert res.skillEvaluation.status == "Needs Attention"
    assert res.skillEvaluation.cumulative_status == "Needs Attention"

    forbidden_phrases = ["that makes sense", "great answer", "excellent", "good point", "exactly", "nice"]
    for phrase in forbidden_phrases:
        assert phrase not in res.reply.lower()


def test_scenario_2_shallow_keyword_heavy_answer_not_strong():
    """TEST 2: Candidate gives a shallow keyword-heavy answer -> Must NOT automatically become Strong."""
    engine = InterviewEngine(max_turns=5)
    raw_cand = data_loader.get_candidate_by_id("CAND-001")
    cand_data = CandidateData(**raw_cand)
    session_id = "test-q-2"

    engine.start_interview(session_id, cand_data)
    # Shallow keyword dump without mechanism explanation
    shallow_ans = "embeddings cosine hnsw rag"
    res = engine.handle_turn(session_id, shallow_ans)

    assert res.skillEvaluation.status in ["Developing", "Good", "Needs Attention"]
    assert res.skillEvaluation.status != "Strong", "Shallow keyword dump must NOT be rated Strong!"


def test_scenario_3_partially_correct_answer_not_strong():
    """TEST 3: Candidate gives a partially correct answer with missing concepts -> Developing or Good, NOT Strong."""
    engine = InterviewEngine(max_turns=5)
    raw_cand = data_loader.get_candidate_by_id("CAND-001")
    cand_data = CandidateData(**raw_cand)
    session_id = "test-q-3"

    engine.start_interview(session_id, cand_data)
    partially_correct_ans = "We use embeddings to compare text vectors in a database."
    res = engine.handle_turn(session_id, partially_correct_ans)

    assert res.skillEvaluation.status in ["Developing", "Good"]
    assert res.skillEvaluation.status != "Strong"


def test_scenario_4_correct_but_shallow_answer_not_strong():
    """TEST 4: Candidate gives a technically correct but shallow answer -> Good or Developing, NOT Strong."""
    engine = InterviewEngine(max_turns=5)
    raw_cand = data_loader.get_candidate_by_id("CAND-001")
    cand_data = CandidateData(**raw_cand)
    session_id = "test-q-4"

    engine.start_interview(session_id, cand_data)
    correct_shallow = "Vector databases index embeddings for similarity search."
    res = engine.handle_turn(session_id, correct_shallow)

    assert res.skillEvaluation.status in ["Good", "Developing"]
    assert res.skillEvaluation.status != "Strong"


def test_scenario_5_deep_correct_complete_technical_answer_is_strong():
    """TEST 5: Candidate gives a deep, correct, complete technical answer -> Strong."""
    engine = InterviewEngine(max_turns=5)
    raw_cand = data_loader.get_candidate_by_id("CAND-001")
    cand_data = CandidateData(**raw_cand)
    session_id = "test-q-5"

    engine.start_interview(session_id, cand_data)
    deep_answer = (
        "We chunk documents into 500-token semantic windows with 10% overlap, generate 1536-dimensional embeddings "
        "using OpenAI text-embedding-3-small, and index them in Qdrant using an HNSW graph with Cosine distance metric "
        "to balance memory usage and achieve sub-50ms retrieval latency."
    )
    res = engine.handle_turn(session_id, deep_answer)

    assert res.skillEvaluation.status == "Strong"


def test_scenario_6_needs_attention_then_strong_does_not_erase_history():
    """TEST 6: Needs Attention followed by Strong on same module -> cumulative status improves to Developing/Good, not Strong."""
    engine = InterviewEngine(max_turns=5)
    raw_cand = data_loader.get_candidate_by_id("CAND-001")
    cand_data = CandidateData(**raw_cand)
    session_id = "test-q-6"

    engine.start_interview(session_id, cand_data)
    state = session_manager.get(session_id)

    # Turn 1: i dont know (Needs Attention, score 0.10)
    engine.handle_turn(session_id, "i dont know")

    # Turn 2: strong answer on same module
    state.current_module = "Embeddings & Vector Search"
    deep_answer = (
        "We chunk documents into 500-token semantic windows with 10% overlap, generate 1536-dimensional embeddings "
        "and index them in Qdrant using HNSW graph with Cosine distance to achieve sub-50ms retrieval latency."
    )
    res2 = engine.handle_turn(session_id, deep_answer)

    rec = state.module_assessments["Embeddings & Vector Search"]
    assert rec["evaluations"] == ["Needs Attention", "Strong"]
    assert rec["cumulative_status"] in ["Developing", "Good"]
    assert rec["cumulative_status"] != "Strong", "Initial failure must NOT be erased to Strong immediately!"


def test_scenario_7_mediocre_answers_cannot_be_strong_fit():
    """TEST 7: Candidate gives mostly mediocre answers across interview -> final disposition MUST NOT be Strong Fit."""
    engine = InterviewEngine(max_turns=3)
    raw_cand = data_loader.get_candidate_by_id("CAND-001")
    cand_data = CandidateData(**raw_cand)
    session_id = "test-q-7"

    engine.start_interview(session_id, cand_data)
    # Mediocre / shallow answers
    engine.handle_turn(session_id, "We use vector databases for similarity.")
    engine.handle_turn(session_id, "Pydantic validates structured outputs.")
    res = engine.handle_turn(session_id, "Agents execute tools in a loop.")

    assert res.done is True
    assert res.feedback.disposition != "Strong Fit"
    assert res.feedback.disposition in ["Consider", "Needs Development"]


def test_scenario_8_consistently_strong_answers_can_be_strong_fit():
    """TEST 8: Candidate gives consistently deep, strong answers -> final disposition CAN be Strong Fit."""
    engine = InterviewEngine(max_turns=3)
    raw_cand = data_loader.get_candidate_by_id("CAND-001")
    cand_data = CandidateData(**raw_cand)
    session_id = "test-q-8"

    engine.start_interview(session_id, cand_data)
    strong_1 = (
        "We chunk documents into 500-token semantic windows with 10% overlap, generate embeddings "
        "and index them in Qdrant using an HNSW graph with Cosine distance to balance memory and sub-50ms latency."
    )
    strong_2 = (
        "We enforce structured JSON output by defining strict Pydantic schemas, setting system prompt parameters, "
        "and applying automated JSON validation with exponential backoff retries in FastAPI handlers."
    )
    strong_3 = (
        "Our multi-agent system uses LangGraph state graphs where agents communicate via standardized Model Context Protocol (MCP) "
        "tools, maintaining isolated state keys and using context distillation to manage token window budgets."
    )

    engine.handle_turn(session_id, strong_1)
    engine.handle_turn(session_id, strong_2)
    res = engine.handle_turn(session_id, strong_3)

    assert res.done is True
    assert res.feedback.disposition == "Strong Fit"


def test_scenario_9_low_answer_quality_prevents_strong_fit():
    """TEST 9: Low overall answer-quality scores (< 0.65) must prevent Strong Fit disposition."""
    engine = InterviewEngine(max_turns=3)
    raw_cand = data_loader.get_candidate_by_id("CAND-001")
    cand_data = CandidateData(**raw_cand)
    session_id = "test-q-9"

    engine.start_interview(session_id, cand_data)
    engine.handle_turn(session_id, "embeddings cosine hnsw rag")
    engine.handle_turn(session_id, "i dont know")
    res = engine.handle_turn(session_id, "pydantic json mode")

    assert res.done is True
    assert res.feedback.disposition != "Strong Fit"


def test_scenario_10_poor_hard_question_performance_negatively_affects_disposition():
    """TEST 10: Poor performance on hard questions prevents Strong Fit."""
    engine = InterviewEngine(max_turns=3)
    raw_cand = data_loader.get_candidate_by_id("CAND-001")
    cand_data = CandidateData(**raw_cand)
    session_id = "test-q-10"

    engine.start_interview(session_id, cand_data)
    state = session_manager.get(session_id)

    # Simulate hard question turn with weak score
    state.turn_skill_evals.append({
        "module": "Production Engineering",
        "status": "Needs Attention",
        "score": 0.10,
        "difficulty": "hard",
    })
    state.turn_skill_evals.append({
        "module": "LLM Core & Prompting",
        "status": "Good",
        "score": 0.70,
        "difficulty": "medium",
    })

    fb = calculate_deterministic_disposition_and_feedback(state)
    assert fb.disposition != "Strong Fit"


def test_scenario_11_mixed_performance_yields_consider():
    """TEST 11: Mixed performance (Strong + Developing + Needs Attention) yields Consider."""
    engine = InterviewEngine(max_turns=3)
    raw_cand = data_loader.get_candidate_by_id("CAND-001")
    cand_data = CandidateData(**raw_cand)
    session_id = "test-q-11"

    engine.start_interview(session_id, cand_data)
    strong_ans = (
        "We chunk documents into 500-token semantic windows with 10% overlap, generate embeddings "
        "and index them in Qdrant using HNSW graph with Cosine distance to achieve sub-50ms latency."
    )
    engine.handle_turn(session_id, strong_ans)
    engine.handle_turn(session_id, "i dont know")
    res = engine.handle_turn(session_id, "We use basic prompt caching in redis.")

    assert res.done is True
    assert res.feedback.disposition in ["Consider", "Needs Development"]


def test_scenario_12_feedback_text_consistent_with_evidence():
    """TEST 12: Feedback executive summary must NOT claim outstanding performance when disposition is Needs Development."""
    engine = InterviewEngine(max_turns=2)
    raw_cand = data_loader.get_candidate_by_id("CAND-001")
    cand_data = CandidateData(**raw_cand)
    session_id = "test-q-12"

    engine.start_interview(session_id, cand_data)
    engine.handle_turn(session_id, "i dont know")
    res = engine.handle_turn(session_id, "not sure")

    assert res.done is True
    fb = res.feedback
    assert fb.disposition == "Needs Development"
    assert "outstanding" not in fb.summary.lower()
    assert "strong technical mastery" not in fb.summary.lower()
    assert "knowledge gaps" in fb.summary.lower()
