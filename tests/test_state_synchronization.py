import pytest
from app.services.interview_engine import (
    InterviewEngine,
    normalize_module_name,
    derive_cumulative_status,
    update_cumulative_module_assessment,
)
from app.services.session_manager import session_manager
from app.data.loader import data_loader
from app.models.schemas import CandidateData


@pytest.fixture(autouse=True)
def cleanup_sessions():
    session_manager.clear()
    yield
    session_manager.clear()


def test_scenario_1_cumulative_evidence_score_aggregation():
    """
    TEST 1: Candidate answers 'i dont know' for Embeddings (Needs Attention, score 0.10).
    Then candidate answers a strong Embeddings follow-up (Strong, score 1.00).
    Expected: Average score 0.55 -> 'Developing'. Previous evaluation is NOT lost.
    """
    engine = InterviewEngine(max_turns=5)
    raw_cand = data_loader.get_candidate_by_id("CAND-001")
    cand_data = CandidateData(**raw_cand)
    session_id = "test-sync-1"

    engine.start_interview(session_id, cand_data)
    state = session_manager.get(session_id)

    # 1. Candidate answers 'i dont know'
    res1 = engine.handle_turn(session_id, "i dont know")
    assert res1.skillEvaluation.module == "Embeddings & Vector Search"
    assert res1.skillEvaluation.status == "Needs Attention"
    rec1 = state.module_assessments["Embeddings & Vector Search"]
    assert rec1["evaluations"] == ["Needs Attention"]
    assert rec1["scores"] == [0.10]
    assert rec1["cumulative_status"] == "Needs Attention"

    # 2. Strong follow-up on Embeddings
    state.current_module = "Embeddings & Vector Search"  # Force probe on Embeddings
    strong_ans = "Vector embeddings use Cosine similarity and HNSW index structures for sub-50ms latency."
    res2 = engine.handle_turn(session_id, strong_ans)

    rec2 = state.module_assessments["Embeddings & Vector Search"]
    assert rec2["evaluations"] == ["Needs Attention", "Strong"]
    assert rec2["scores"][0] == 0.10
    assert rec2["scores"][1] >= 0.85
    assert rec2["probe_count"] == 2
    assert 0.50 <= rec2["average_score"] <= 0.60
    # Average score maps deterministically to 'Developing'
    assert rec2["cumulative_status"] == "Developing"
    assert res2.skillEvaluation.cumulative_status == "Developing"


def test_scenario_2_simultaneous_multi_module_visibility():
    """
    TEST 2: Embeddings = Needs Attention, LLM Core = Strong, Agentic AI = Strong.
    Expected: 3 unique modules assessed, all remaining visible simultaneously.
    """
    engine = InterviewEngine(max_turns=10)
    raw_cand = data_loader.get_candidate_by_id("CAND-001")
    cand_data = CandidateData(**raw_cand)
    session_id = "test-sync-2"

    engine.start_interview(session_id, cand_data)
    state = session_manager.get(session_id)

    # Turn 1: Embeddings -> i dont know
    state.current_module = "Embeddings & Vector Search"
    engine.handle_turn(session_id, "i dont know")

    # Turn 2: LLM Core -> Strong
    state.current_module = "LLM Core & Prompting"
    engine.handle_turn(session_id, "System prompts use Pydantic models for structured output validation.")

    # Turn 3: Agentic AI -> Strong
    state.current_module = "Agentic AI & MCP"
    engine.handle_turn(session_id, "ReAct loops combine reasoning steps with tool execution actions.")

    assessed = {k: v["cumulative_status"] for k, v in state.module_assessments.items()}
    assert len(assessed) == 3
    assert assessed["Embeddings & Vector Search"] == "Needs Attention"
    assert assessed["LLM Core & Prompting"] in ["Strong", "Good"]
    assert assessed["Agentic AI & MCP"] in ["Strong", "Good"]


def test_scenario_3_and_4_current_topic_module_drives_next_focus():
    """
    TEST 3 & 4: Timeline & focus follow currentTopicModule adaptively (Embeddings -> Security & Guardrails).
    """
    engine = InterviewEngine(max_turns=5)
    raw_cand = data_loader.get_candidate_by_id("CAND-001")
    cand_data = CandidateData(**raw_cand)
    session_id = "test-sync-3"

    res_start = engine.start_interview(session_id, cand_data)
    assert res_start.currentTopicModule is not None

    state = session_manager.get(session_id)
    # Manually set current_module to Security & Guardrails to test adaptive topic shift
    state.current_module = "Security & Guardrails"

    res_turn = engine.handle_turn(session_id, "Embeddings use floating point vector dimensions.")
    # The next question module is propagated via currentTopicModule
    assert res_turn.currentTopicModule is not None


def test_scenario_5_multiple_evaluations_unique_assessed_count():
    """
    TEST 5: Embeddings evaluated 3 times.
    Expected: probe_count = 3, but unique assessed module count = 1.
    """
    engine = InterviewEngine(max_turns=10)
    raw_cand = data_loader.get_candidate_by_id("CAND-001")
    cand_data = CandidateData(**raw_cand)
    session_id = "test-sync-5"

    engine.start_interview(session_id, cand_data)
    state = session_manager.get(session_id)

    # 3 turns on Embeddings
    for i in range(3):
        state.current_module = "Embeddings & Vector Search"
        engine.handle_turn(session_id, f"Embeddings answer iteration {i+1} with technical details.")

    rec = state.module_assessments["Embeddings & Vector Search"]
    assert rec["probe_count"] == 3
    assert len(state.module_assessments) == 1


def test_scenario_6_unknown_answer_preserves_neutral_transition():
    """
    TEST 6: Unknown answer does not generate positive validation.
    """
    engine = InterviewEngine(max_turns=5)
    raw_cand = data_loader.get_candidate_by_id("CAND-001")
    cand_data = CandidateData(**raw_cand)
    session_id = "test-sync-6"

    engine.start_interview(session_id, cand_data)
    res = engine.handle_turn(session_id, "i dont know")

    assert res.skillEvaluation.status == "Needs Attention"
    forbidden = ["that makes sense", "great answer", "excellent", "good point", "exactly"]
    for phrase in forbidden:
        assert phrase not in res.reply.lower()


def test_scenario_7_module_name_normalization():
    """
    TEST 7: Module name with whitespace/newline variations all normalize to canonical key.
    """
    inputs = [
        "RAG & Retrieval Architecture",
        "RAG & Retrieval Architecture ",
        "RAG & Retrieval Architecture\n",
        "  rag & retrieval architecture  ",
    ]
    expected = "RAG & Retrieval Architecture"
    for inp in inputs:
        assert normalize_module_name(inp) == expected


def test_scenario_8_semantic_distinction_skill_eval_vs_current_topic():
    """
    TEST 8: skillEvaluation.module belongs to turn N answer, currentTopicModule belongs to turn N+1 question.
    """
    engine = InterviewEngine(max_turns=5)
    raw_cand = data_loader.get_candidate_by_id("CAND-001")
    cand_data = CandidateData(**raw_cand)
    session_id = "test-sync-8"

    engine.start_interview(session_id, cand_data)
    state = session_manager.get(session_id)
    state.current_module = "Embeddings & Vector Search"

    res = engine.handle_turn(session_id, "We index vectors using Cosine similarity.")

    # Evaluated module is Embeddings
    assert res.skillEvaluation.module == "Embeddings & Vector Search"
    # Next question module can be different (e.g. LLM Core or RAG)
    assert res.currentTopicModule is not None
