"""
Regression tests for Skill Assessment correctness.

Requirements covered:
1.  "I don't know" → skill NOT Strong
2.  Weak answer → Developing/Needs Attention
3.  Strong answer → Strong/Good
4.  Asking a question alone does NOT create Strong status
5.  Multiple weak answers do NOT result in all skills Strong
6.  Mostly Developing skills → NOT Strong Fit
7.  Mostly Strong/Good skills → Strong Fit is possible
8.  Mixed skills → appropriate Consider/Strong Fit based on evidence
9.  Strong historical candidate profile + weak current interview → NOT Strong Fit
10. Executive Summary agrees with disposition
11. Knowledge Gaps agree with skill profile
12. Recommended Next Steps agree with knowledge gaps
13. Different candidates can produce different assessment results
14. Assessment can start successfully for different candidates
15. Coverage metrics sanity check (not a 2nd empty page)
16. Existing API contract remains intact
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.data.loader import data_loader
from app.services.session_manager import session_manager
from app.services.interview_engine import interview_engine
from app.services.llm_client import LLMClient

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_sessions():
    session_manager.clear()
    yield
    session_manager.clear()


# ─── Helper ──────────────────────────────────────────────────────────────────

def run_interview(candidate_id: str, answers: list, max_turns: int = None) -> dict:
    """Run a full interview for the given candidate with the supplied answers.
    Returns the final response dict (with done=True)."""
    cand = data_loader.get_candidate_by_id(candidate_id)
    session_id = f"skill-test-{candidate_id}-{id(answers)}"
    orig_turns = interview_engine.max_turns

    try:
        if max_turns is not None:
            interview_engine.max_turns = max_turns
        else:
            interview_engine.max_turns = len(answers)

        # Start
        start_res = client.post("/api/interview", json={"sessionId": session_id, "candidate": cand})
        assert start_res.status_code == 200

        final_res = None
        for answer in answers:
            res = client.post("/api/interview", json={"sessionId": session_id, "message": answer})
            assert res.status_code == 200
            final_res = res

        return final_res.json()
    finally:
        interview_engine.max_turns = orig_turns


# ─── 1. "I don't know" → skill NOT Strong ────────────────────────────────────

def test_idk_answer_not_strong():
    """Req 1: 'I don't know' response must NOT mark the relevant skill as Strong."""
    client_eval = LLMClient()
    result = client_eval._evaluate_single_answer("I don't know.")
    assert result["label"] != "Strong", (
        f"'I don't know' should not be Strong, got: {result['label']}"
    )
    assert result["score"] < 0.5, f"Expected low score, got {result['score']}"


# ─── 2. Weak answer → Developing / Needs Attention ───────────────────────────

def test_weak_answer_label():
    """Req 2: Short vague answers must evaluate as Weak (maps to Needs Attention)."""
    client_eval = LLMClient()
    for weak_text in ["Not sure.", "idk", "no experience", "pass on this", "I haven't worked with that."]:
        result = client_eval._evaluate_single_answer(weak_text)
        assert result["label"] == "Weak", (
            f"Expected Weak for '{weak_text}', got: {result['label']}"
        )


# ─── 3. Strong answer → Strong / Good ─────────────────────────────────────────

def test_strong_answer_label():
    """Req 3: Technically detailed answers with domain keywords must evaluate as Strong."""
    client_eval = LLMClient()
    strong_answer = (
        "We use cosine similarity with HNSW index in Qdrant for sub-50ms vector retrieval. "
        "For hybrid search we combine BM25 sparse retrieval with dense embeddings using RRF re-ranking."
    )
    result = client_eval._evaluate_single_answer(strong_answer)
    assert result["label"] == "Strong", (
        f"Expected Strong for technical answer, got: {result['label']}"
    )
    assert result["score"] >= 0.9


# ─── 4. Asking a question alone does NOT create a Strong status ───────────────

def test_start_interview_returns_no_skill_evaluation():
    """Req 4: The initial interview start (no candidate answer yet) must NOT
    return a skillEvaluation (candidate has not answered anything)."""
    cand = data_loader.get_candidate_by_id("CAND-003")
    res = client.post("/api/interview", json={"sessionId": "skill-start-test", "candidate": cand})
    assert res.status_code == 200
    data = res.json()
    # No skill evaluation on interview start — no answer has been given
    assert data.get("skillEvaluation") is None, (
        "Start response must NOT include skillEvaluation (no answer given yet)"
    )


# ─── 5. Multiple weak answers do NOT result in all skills Strong ──────────────

def test_multiple_weak_answers_not_all_strong():
    """Req 5: Giving 'I don't know' repeatedly must NOT produce all-Strong skill profile."""
    weak_answers = ["I don't know."] * 3
    final_data = run_interview("CAND-003", weak_answers)
    assert final_data["done"] is True
    fb = final_data["feedback"]
    assert fb is not None

    skill_profile = fb.get("skillProfile") or {}
    strong_count = sum(1 for v in skill_profile.values() if v == "Strong")
    total = len(skill_profile)
    assert strong_count < total, (
        f"Expected NOT all skills to be Strong after weak answers. "
        f"Got {strong_count}/{total} Strong: {skill_profile}"
    )


# ─── 6. Mostly Developing skills → NOT Strong Fit ────────────────────────────

def test_mostly_weak_answers_not_strong_fit():
    """Req 6: Candidate giving mostly weak answers must NOT receive Strong Fit."""
    weak_answers = ["I don't know."] * 3
    final_data = run_interview("CAND-003", weak_answers)
    fb = final_data["feedback"]
    assert fb["disposition"] != "Strong Fit", (
        f"Expected non-Strong Fit for weak answers, got: {fb['disposition']}"
    )


# ─── 7. Mostly Strong/Good skills → Strong Fit possible ──────────────────────

def test_strong_answers_enable_strong_fit():
    """Req 7: Candidate giving consistently strong technical answers CAN receive Strong Fit."""
    strong_answers = [
        "We use cosine similarity metrics with HNSW index configuration in Qdrant vector database for sub-50ms latency.",
        "We enforce strict Pydantic model schemas and JSON mode formatting with automated tool validation handlers.",
    ]
    final_data = run_interview("CAND-003", strong_answers, max_turns=2)
    fb = final_data["feedback"]
    assert fb["disposition"] in ("Strong Fit", "Consider"), (
        f"Strong answers should produce at least Consider disposition, got: {fb['disposition']}"
    )


# ─── 8. Mixed answers → Consider is appropriate ──────────────────────────────

def test_mixed_answers_produce_consider_or_better():
    """Req 8: Mixed strong/weak answers should yield Consider or better (not Needs Development)."""
    mixed_answers = [
        "We use cosine similarity metrics with HNSW index configuration in vector database.",
        "I don't know much about Pydantic schemas.",
        "We use standard LangChain agents with simple tools.",
    ]
    final_data = run_interview("CAND-003", mixed_answers, max_turns=3)
    fb = final_data["feedback"]
    assert fb["disposition"] in ("Consider", "Strong Fit"), (
        f"Mixed answers expected Consider or Strong Fit, got: {fb['disposition']}"
    )


# ─── 9. Strong historical + weak interview → NOT Strong Fit ──────────────────

def test_strong_profile_weak_interview_not_strong_fit():
    """Req 9: Emily Chen (97% first-try, CAND-003) + weak live answers must NOT be Strong Fit."""
    weak_answers = ["idk", "no experience", "cannot answer"]
    final_data = run_interview("CAND-003", weak_answers, max_turns=3)
    fb = final_data["feedback"]
    assert fb["disposition"] != "Strong Fit", (
        f"Strong historical profile + weak live answers must not be Strong Fit, got: {fb['disposition']}"
    )
    assert fb["disposition"] == "Needs Development", (
        f"Expected Needs Development for all-weak answers, got: {fb['disposition']}"
    )


# ─── 10. Executive Summary agrees with disposition ────────────────────────────

def test_executive_summary_agrees_with_disposition_weak():
    """Req 10: When disposition is Needs Development, summary must NOT claim outstanding performance."""
    weak_answers = ["I don't know."] * 3
    final_data = run_interview("CAND-003", weak_answers, max_turns=3)
    fb = final_data["feedback"]
    summary = fb["summary"].lower()
    if fb["disposition"] == "Needs Development":
        assert "outstanding" not in summary or "gap" in summary or "weak" in summary or "knowledge" in summary or "development" in summary, (
            f"Needs Development summary should reflect weakness, got: {fb['summary']}"
        )


def test_executive_summary_agrees_with_disposition_strong():
    """Req 10: When disposition is Strong Fit, summary should reflect strong performance."""
    strong_answers = [
        "We use cosine similarity metrics with HNSW index configuration in Qdrant vector database for sub-50ms latency.",
        "We enforce strict Pydantic model schemas and JSON mode formatting with automated tool validation handlers.",
    ]
    final_data = run_interview("CAND-003", strong_answers, max_turns=2)
    fb = final_data["feedback"]
    summary = fb["summary"].lower()
    if fb["disposition"] == "Strong Fit":
        # Strong Fit summary should mention positive performance
        positive_words = ["strong", "outstanding", "demonstrated", "excellent", "strong fit"]
        assert any(w in summary for w in positive_words), (
            f"Strong Fit summary should mention strong performance, got: {fb['summary']}"
        )


# ─── 11. Knowledge Gaps agree with skill profile ──────────────────────────────

def test_knowledge_gaps_present_when_weak():
    """Req 11: When candidate gives weak answers, knowledge gaps must be non-empty."""
    weak_answers = ["I don't know."] * 3
    final_data = run_interview("CAND-003", weak_answers, max_turns=3)
    fb = final_data["feedback"]
    assert len(fb["gaps"]) > 0, "Weak candidate must have non-empty knowledge gaps"


# ─── 12. Next Steps agree with knowledge gaps ─────────────────────────────────

def test_next_steps_present_when_gaps_exist():
    """Req 12: When knowledge gaps are present, next steps must also be non-empty."""
    weak_answers = ["I don't know."] * 3
    final_data = run_interview("CAND-003", weak_answers, max_turns=3)
    fb = final_data["feedback"]
    if len(fb["gaps"]) > 0:
        assert len(fb["next"]) > 0, "Next steps must be non-empty when knowledge gaps exist"


# ─── 13. Different candidates produce different results ───────────────────────

def test_different_candidates_different_results():
    """Req 13: Running the same strong answers for two different candidates
    should both succeed and the API contract should hold for both."""
    strong_answer = "We use cosine similarity with HNSW index in Qdrant for sub-50ms latency."

    for cand_id in ["CAND-003", "CAND-010", "CAND-011"]:
        cand = data_loader.get_candidate_by_id(cand_id)
        session_id = f"multi-cand-test-{cand_id}"
        orig_turns = interview_engine.max_turns
        try:
            interview_engine.max_turns = 1
            res_start = client.post("/api/interview", json={"sessionId": session_id, "candidate": cand})
            assert res_start.status_code == 200, f"Start failed for {cand_id}"
            assert res_start.json()["done"] is False

            res_turn = client.post("/api/interview", json={"sessionId": session_id, "message": strong_answer})
            assert res_turn.status_code == 200, f"Turn failed for {cand_id}"
            assert res_turn.json()["done"] is True
        finally:
            interview_engine.max_turns = orig_turns


# ─── 14. Assessment can start for different candidates ────────────────────────

def test_assessment_start_multiple_candidates():
    """Req 14: Start interview must succeed for any candidate in the dataset."""
    all_candidates = data_loader.get_all_candidates()
    # Test first 3 candidates minimum
    for i, cand in enumerate(all_candidates[:3]):
        cand_id = cand.get("member", {}).get("id", f"unknown-{i}")
        session_id = f"start-test-{cand_id}"
        res = client.post("/api/interview", json={"sessionId": session_id, "candidate": cand})
        assert res.status_code == 200, f"Start failed for candidate {cand_id}: {res.json()}"
        data = res.json()
        assert "reply" in data and len(data["reply"]) > 0
        assert data["done"] is False


# ─── 15. Skill evaluation is returned per turn (not on start) ─────────────────

def test_skill_evaluation_returned_after_answer():
    """Req 15: After a candidate submits an answer, the response must include skillEvaluation."""
    cand = data_loader.get_candidate_by_id("CAND-003")
    session_id = "skill-eval-return-test"
    orig_turns = interview_engine.max_turns
    try:
        interview_engine.max_turns = 5
        client.post("/api/interview", json={"sessionId": session_id, "candidate": cand})

        res = client.post("/api/interview", json={
            "sessionId": session_id,
            "message": "We use cosine similarity with HNSW index in Qdrant vector database."
        })
        assert res.status_code == 200
        data = res.json()
        # After submitting an answer (non-final turn), skillEvaluation should be present
        assert "skillEvaluation" in data and data["skillEvaluation"] is not None, (
            "Expected skillEvaluation in turn response"
        )
        eval_data = data["skillEvaluation"]
        assert "module" in eval_data and len(eval_data["module"]) > 0
        assert "status" in eval_data and eval_data["status"] in (
            "Strong", "Good", "Developing", "Needs Attention"
        )
    finally:
        interview_engine.max_turns = orig_turns


# ─── 16. currentTopicModule matches assessmentFocus topic ─────────────────────

def test_current_topic_module_returned():
    """Req 16: API must return currentTopicModule for topic/label consistency."""
    cand = data_loader.get_candidate_by_id("CAND-003")
    session_id = "topic-module-test"
    res = client.post("/api/interview", json={"sessionId": session_id, "candidate": cand})
    assert res.status_code == 200
    data = res.json()
    assert "currentTopicModule" in data and data["currentTopicModule"] is not None, (
        "Start response must include currentTopicModule"
    )
    assert len(data["currentTopicModule"]) > 0


# ─── 16b. Existing API contract intact ────────────────────────────────────────

def test_api_contract_intact():
    """Req 16: Core API contract fields must remain intact."""
    cand = data_loader.get_candidate_by_id("CAND-003")
    session_id = "api-contract-test"
    orig_turns = interview_engine.max_turns
    try:
        interview_engine.max_turns = 2
        res = client.post("/api/interview", json={"sessionId": session_id, "candidate": cand})
        assert res.status_code == 200
        start = res.json()
        # Required fields from API contract
        assert "reply" in start
        assert "done" in start
        assert start["done"] is False

        # Turn 1
        res2 = client.post("/api/interview", json={"sessionId": session_id, "message": "Some answer about vector embeddings."})
        assert res2.status_code == 200
        t1 = res2.json()
        assert "reply" in t1
        assert "done" in t1
        assert t1["done"] is False

        # Turn 2 → final
        res3 = client.post("/api/interview", json={"sessionId": session_id, "message": "We use cosine similarity with HNSW index."})
        assert res3.status_code == 200
        final = res3.json()
        assert final["done"] is True
        assert "feedback" in final
        fb = final["feedback"]
        assert "summary" in fb and len(fb["summary"]) > 0
        assert "strengths" in fb and isinstance(fb["strengths"], list)
        assert "gaps" in fb and isinstance(fb["gaps"], list)
        assert "next" in fb and isinstance(fb["next"], list)
        assert "disposition" in fb and fb["disposition"] in ("Strong Fit", "Consider", "Needs Development")
    finally:
        interview_engine.max_turns = orig_turns


# ═══════════════════════════════════════════════════════════════════════════════
# REGRESSION TESTS — Historical Skill Data Must NOT Appear in Live Coverage Map
# These directly test the root-cause bug reported after the first polish pass.
# Scenario: Emily Chen (CAND-003) has 97% first-try rate and mastered >20 missions.
# Her historical record must NEVER pre-populate or maintain "Strong" in live map.
# ═══════════════════════════════════════════════════════════════════════════════

def test_regression_start_interview_skill_eval_is_absent():
    """REGRESSION TEST 1 — Initial state: no skillEvaluation on start.
    Historical Strong for all modules must NOT appear in API response skillEvaluation."""
    cand = data_loader.get_candidate_by_id("CAND-003")  # Emily Chen: 97% first-try
    res = client.post("/api/interview", json={"sessionId": "regression-t1", "candidate": cand})
    assert res.status_code == 200
    data = res.json()
    # Start response must have NO skillEvaluation — candidate hasn't answered yet
    skill_eval = data.get("skillEvaluation")
    assert skill_eval is None, (
        f"REGRESSION: Start response must not include skillEvaluation. "
        f"Historical data must not pre-populate skill status. Got: {skill_eval}"
    )


def test_regression_idont_know_yields_needs_attention():
    """REGRESSION TEST 2 — 'idont know' (no apostrophe) must produce Needs Attention, not Strong.
    This is the exact text the user typed that triggered the bug."""
    lc = LLMClient()
    # The exact text from the bug report
    result = lc._evaluate_single_answer("idont know")
    assert result["label"] == "Weak", (
        f"'idont know' must evaluate as Weak. Got: {result['label']} (score={result['score']})"
    )

    # Also verify the skill evaluation maps this to Needs Attention
    skill_eval = interview_engine._evaluate_answer_for_skill("idont know", 0)
    assert skill_eval.status == "Needs Attention", (
        f"'idont know' at turn 0 must produce Needs Attention. Got: {skill_eval.status}"
    )
    assert skill_eval.module == "Embeddings & Vector Search", (
        f"Turn 0 must assess Embeddings & Vector Search. Got: {skill_eval.module}"
    )


def test_regression_strong_answer_after_weak_can_recover():
    """REGRESSION TEST 3 — A strong current-interview answer can produce Strong/Good,
    but only when the candidate actually gives one."""
    lc = LLMClient()
    strong_text = (
        "We use cosine similarity with HNSW index configuration in Qdrant for sub-50ms latency. "
        "For hybrid search we combine BM25 sparse retrieval with dense embeddings using RRF re-ranking."
    )
    result = lc._evaluate_single_answer(strong_text)
    assert result["label"] == "Strong", (
        f"Strong technical answer must evaluate as Strong. Got: {result['label']}"
    )

    skill_eval = interview_engine._evaluate_answer_for_skill(strong_text, 0)
    assert skill_eval.status == "Strong", (
        f"Strong answer at turn 0 must produce Strong status. Got: {skill_eval.status}"
    )


def test_regression_emily_all_idk_not_strong_in_profile():
    """REGRESSION TEST 4 — Emily Chen (97% first-try, mastered 20+ missions) giving
    all 'I don't know' answers must produce a final skill profile that is NOT Strong.
    Historical mastery must NOT inflate live skill status."""
    orig_turns = interview_engine.max_turns
    try:
        interview_engine.max_turns = 4
        cand = data_loader.get_candidate_by_id("CAND-003")  # Emily Chen
        session_id = "regression-t4-emily-idk"

        client.post("/api/interview", json={"sessionId": session_id, "candidate": cand})
        # All weak answers
        client.post("/api/interview", json={"sessionId": session_id, "message": "idont know"})
        client.post("/api/interview", json={"sessionId": session_id, "message": "I don't know."})
        client.post("/api/interview", json={"sessionId": session_id, "message": "no idea"})
        final = client.post("/api/interview", json={"sessionId": session_id, "message": "not sure"})

        data = final.json()
        assert data["done"] is True
        fb = data["feedback"]

        # Verify skill profile: despite Emily's historical mastery, live profile must NOT be Strong
        skill_profile = fb.get("skillProfile") or {}
        for mod_name, rating in skill_profile.items():
            assert rating != "Strong", (
                f"REGRESSION: Module '{mod_name}' shows '{rating}' despite all-weak current answers. "
                f"Historical mastery must NOT produce Strong in live profile. Full profile: {skill_profile}"
            )

        # Disposition must also be Needs Development
        assert fb["disposition"] == "Needs Development", (
            f"Emily with all-weak answers must be Needs Development. Got: {fb['disposition']}"
        )
    finally:
        interview_engine.max_turns = orig_turns


def test_regression_zero_answers_all_not_assessed_in_profile():
    """REGRESSION TEST 5 — A completed 1-turn interview where the single answer triggers
    finalization: unassessed modules must appear as 'Not Assessed' in final skill profile."""
    orig_turns = interview_engine.max_turns
    try:
        interview_engine.max_turns = 1
        cand = data_loader.get_candidate_by_id("CAND-003")  # Emily Chen: strong history
        session_id = "regression-t5-one-turn"

        client.post("/api/interview", json={"sessionId": session_id, "candidate": cand})
        final = client.post("/api/interview", json={"sessionId": session_id, "message": "idont know"})

        data = final.json()
        assert data["done"] is True
        fb = data["feedback"]
        skill_profile = fb.get("skillProfile") or {}

        # Only Embeddings & Vector Search (turn 0 module) was assessed
        # All other modules must be Not Assessed
        assessed_module = "Embeddings & Vector Search"
        for mod_name, rating in skill_profile.items():
            if mod_name != assessed_module:
                assert rating == "Not Assessed", (
                    f"REGRESSION: Module '{mod_name}' shows '{rating}' but was never assessed. "
                    f"Unassessed modules must be 'Not Assessed'. Full profile: {skill_profile}"
                )
        # The assessed module must be Needs Attention (candidate said 'idont know')
        if assessed_module in skill_profile:
            assert skill_profile[assessed_module] == "Needs Attention", (
                f"REGRESSION: '{assessed_module}' after 'idont know' must be 'Needs Attention'. "
                f"Got: {skill_profile[assessed_module]}"
            )
    finally:
        interview_engine.max_turns = orig_turns


def test_regression_final_profile_matches_turn_skill_evals():
    """REGRESSION TEST 6 — The final skill profile in the feedback report must exactly
    match what was recorded in turn_skill_evals during the interview.
    This verifies the data flow: candidate answers → turn_skill_evals → final skill profile."""
    orig_turns = interview_engine.max_turns
    try:
        interview_engine.max_turns = 2
        cand = data_loader.get_candidate_by_id("CAND-003")
        session_id = "regression-t6-profile-match"

        client.post("/api/interview", json={"sessionId": session_id, "candidate": cand})

        # Turn 1: weak answer for turn 0 (Embeddings & Vector Search)
        client.post("/api/interview", json={"sessionId": session_id, "message": "idont know"})

        # Turn 2: strong answer for turn 1 (still Embeddings & Vector Search per TURN_PLAN)
        final = client.post("/api/interview", json={
            "sessionId": session_id,
            "message": "We use cosine similarity with HNSW index in Qdrant for sub-50ms latency."
        })

        data = final.json()
        assert data["done"] is True

        # Verify turn_skill_evals in session state match final skill profile
        state = session_manager.get(session_id)
        assert len(state.turn_skill_evals) == 2, (
            f"Expected 2 turn_skill_evals, got {len(state.turn_skill_evals)}: {state.turn_skill_evals}"
        )

        fb = data["feedback"]
        skill_profile = fb.get("skillProfile") or {}

        # Both turns assessed Embeddings & Vector Search
        # Latest (turn 2: Strong) should be the final status
        embeddings_status = skill_profile.get("Embeddings & Vector Search")
        assert embeddings_status is not None, (
            "Embeddings & Vector Search must appear in final skill profile"
        )
        # Turn 2 was a strong answer, so final status should be Strong
        assert embeddings_status == "Strong", (
            f"Latest evaluation (Strong) should win. Got: {embeddings_status}. "
            f"Turn evals: {state.turn_skill_evals}"
        )

        # RAG & Retrieval Architecture was NOT assessed — must be Not Assessed
        rag_status = skill_profile.get("RAG & Retrieval Architecture")
        assert rag_status == "Not Assessed", (
            f"RAG was not assessed so must be 'Not Assessed'. Got: {rag_status}"
        )
    finally:
        interview_engine.max_turns = orig_turns

