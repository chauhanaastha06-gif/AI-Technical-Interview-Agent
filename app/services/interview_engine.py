from typing import Optional, List, Dict
from app.config import settings
from app.models.schemas import CandidateData, InterviewResponse, FeedbackResponse, AssessmentFocus, SkillEvaluation
from app.models.domain import InterviewState, InterviewBrief
from app.services.candidate_profile import build_candidate_brief
from app.services.prompt_builder import build_system_prompt, build_feedback_prompt
from app.services.session_manager import session_manager, SessionAlreadyExistsError, SessionNotFoundError
from app.services.llm_client import llm_client, LLMClient
from app.utils.logging import logger


# =============================================================================
# TURN PLAN — Single Source of Truth for topic/module/assessment-focus labels.
# All frontend labels (Current Focus, Assessment Focus, Timeline, Skill Map) are
# derived from this table via the `currentTopicModule` field in InterviewResponse.
# =============================================================================
TURN_PLAN: List[Dict] = [
    # turn_idx: the 0-based index matching turn_count when question is asked
    # module: canonical curriculum module name (matches CURRICULUM_MODULES in frontend)
    # focus_topic: recruiter-facing Assessment Focus label
    {
        "turn_idx": 0,
        "module": "Embeddings & Vector Search",
        "focus_topic": "Vector Embeddings & Retrieval",
        "focus_detail": "Semantic similarity, distance metrics & index configuration",
    },
    {
        "turn_idx": 1,
        "module": "Embeddings & Vector Search",
        "focus_topic": "Vector Embeddings & Retrieval",
        "focus_detail": "Chunking trade-offs & hybrid search ranking",
    },
    {
        "turn_idx": 2,
        "module": "LLM Core & Prompting",
        "focus_topic": "LLM Integration & Structured Output",
        "focus_detail": "Structured output schemas & tool validation",
    },
    {
        "turn_idx": 3,
        "module": "Agentic AI & MCP",
        "focus_topic": "Agentic Workflow & MCP Integration",
        "focus_detail": "Multi-agent state, reasoning loops & MCP",
    },
    {
        "turn_idx": 4,
        "module": "Security & Guardrails",
        "focus_topic": "Prompt Injection Defense & Security",
        "focus_detail": "Prompt injection defense & container security",
    },
    {
        "turn_idx": 5,
        "module": "Evaluation & Benchmarks",
        "focus_topic": "Evaluation Recall & Benchmarking",
        "focus_detail": "Retrieval recall & LLM-as-a-judge benchmarks",
    },
    {
        "turn_idx": 6,
        "module": "Production Engineering",
        "focus_topic": "Async Token Streaming & Moderation",
        "focus_detail": "Async token streaming & guardrail moderation",
    },
    {
        "turn_idx": 7,
        "module": "Production Engineering",
        "focus_topic": "Semantic Prompt Caching & Model Routing",
        "focus_detail": "Semantic prompt caching & dynamic model routing",
    },
    {
        "turn_idx": 8,
        "module": "RAG & Retrieval Architecture",
        "focus_topic": "RAG vs Fine-Tuning Trade-offs",
        "focus_detail": "RAG context injection vs LoRA fine-tuning",
    },
    {
        "turn_idx": 9,
        "module": "Production Engineering",
        "focus_topic": "Production Capstone & Concurrency",
        "focus_detail": "Enterprise capstone & 10k concurrency scale",
    },
]

# Map TURN_PLAN module names → canonical curriculum module names used in frontend CURRICULUM_MODULES
MODULE_DISPLAY_NAMES = {
    "Embeddings & Vector Search": "Embeddings & Vector Search",
    "RAG & Retrieval Architecture": "RAG & Retrieval Architecture",
    "LLM Core & Prompting": "LLM Core & Prompting",
    "Agentic AI & MCP": "Agentic AI & MCP",
    "Security & Guardrails": "Security & Guardrails",
    "Evaluation & Benchmarks": "Evaluation & Benchmarks",
    "Production Engineering": "Production Engineering",
}


def _get_turn_plan(turn_idx: int) -> Dict:
    """Return the TURN_PLAN entry for a given turn index (clamped to last entry)."""
    idx = min(turn_idx, len(TURN_PLAN) - 1)
    return TURN_PLAN[idx]


class InterviewEngine:
    def __init__(self, llm: Optional[LLMClient] = None, max_turns: Optional[int] = None):
        self.llm = llm or llm_client
        self.max_turns = max_turns if max_turns is not None else settings.MAX_TURNS

    def _build_assessment_focus(self, brief: InterviewBrief, turn_idx: int) -> AssessmentFocus:
        """Build AssessmentFocus from TURN_PLAN — consistent with currentTopicModule."""
        plan = _get_turn_plan(turn_idx)
        topic = plan["focus_topic"]

        # Build recruiter-safe reason from candidate context
        if brief.failed and turn_idx == 0:
            reason = f"Focusing on identified knowledge gap (Day {brief.failed[0].day}: {brief.failed[0].title})."
        elif brief.struggled and turn_idx <= 2:
            reason = f"Probing struggle area (Day {brief.struggled[0].day}: {brief.struggled[0].title})."
        elif brief.years_experience >= 5:
            reason = f"Calibrated {plan['focus_detail']} for {brief.job_role} level ({brief.years_experience} yrs experience)."
        else:
            reason = f"Evaluating {plan['focus_detail']} for {brief.job_role} candidate."

        return AssessmentFocus(topic=topic, reason=reason)

    def _evaluate_answer_for_skill(self, candidate_message: str, turn_idx: int) -> SkillEvaluation:
        """
        Evaluate the candidate's actual answer and return a SkillEvaluation for the current turn.
        Uses the TURN_PLAN to determine which module is being assessed, and
        `_evaluate_single_answer` (from llm_client) to score the answer quality.

        Status mapping:
          Strong  (score=1.0)  → "Strong"
          Adequate (score=0.6) → "Good" or "Developing" depending on length/depth
          Weak    (score=0.1/0.3) → "Needs Attention"
        """
        plan = _get_turn_plan(turn_idx)
        module_name = plan["module"]

        eval_result = self.llm._evaluate_single_answer(candidate_message)
        label = eval_result["label"]
        score = eval_result["score"]

        if label == "Strong":
            status = "Strong"
        elif label == "Adequate":
            # Distinguish Good (reasonable depth) from Developing (just adequate)
            status = "Good" if score >= 0.6 and len(candidate_message.strip()) >= 50 else "Developing"
        else:
            # Weak / empty / "I don't know"
            status = "Needs Attention"

        return SkillEvaluation(module=module_name, status=status)

    def start_interview(self, session_id: str, candidate_data: Optional[CandidateData]) -> InterviewResponse:
        """
        Initializes a new interview session for the given candidate.
        """
        if not session_id or not session_id.strip():
            raise ValueError("sessionId must not be empty.")

        if candidate_data is None:
            raise ValueError("Candidate data is required to start an interview.")

        if session_manager.session_exists(session_id):
            raise SessionAlreadyExistsError(f"Session '{session_id}' already exists and cannot be re-initialized.")

        # 1. Build profile brief
        brief = build_candidate_brief(candidate_data)
        logger.info(f"Built brief for candidate {brief.candidate_name} ({brief.candidate_id}) with {len(brief.mastered)} mastered, {len(brief.struggled)} struggled, {len(brief.failed)} failed, {len(brief.skipped)} skipped.")

        # 2. Build system prompt
        system_prompt = build_system_prompt(brief, max_turns=self.max_turns)

        # 3. Create session state
        session_manager.create(
            session_id=session_id,
            candidate_data=candidate_data,
            brief=brief,
            system_prompt=system_prompt,
        )

        # 4. Generate first question
        opening_reply = self.llm.generate_interview_reply(
            system_prompt=system_prompt,
            conversation_history=[],
            brief=brief,
            turn_count=0,
        )

        # 5. Store opening reply in conversation history
        session_manager.append_turn(session_id=session_id, role="assistant", content=opening_reply)

        # 6. Build assessment focus from TURN_PLAN (turn 0)
        focus = self._build_assessment_focus(brief, turn_idx=0)
        plan = _get_turn_plan(0)

        return InterviewResponse(
            reply=opening_reply,
            done=False,
            assessmentFocus=focus,
            currentTopicModule=plan["module"],
            # No skillEvaluation on start — candidate hasn't answered yet
        )

    def handle_turn(self, session_id: str, message: Optional[str]) -> InterviewResponse:
        """
        Handles an ongoing conversation turn for an active interview session.
        """
        if not session_id or not session_id.strip():
            raise ValueError("sessionId must not be empty.")

        state = session_manager.get(session_id)

        # If already done, return existing state
        if state.done:
            return InterviewResponse(
                reply="Interview has already concluded.",
                done=True,
                feedback=state.feedback,
            )

        # Validate message
        if message is None or not message.strip():
            return InterviewResponse(
                reply="Please provide an answer to continue the interview.",
                done=False,
            )

        clean_message = message.strip()

        # Append candidate message
        session_manager.append_turn(session_id=session_id, role="user", content=clean_message)
        current_turns = session_manager.increment_turn_count(session_id)
        logger.info(f"Session {session_id} - Turn {current_turns}/{self.max_turns}")

        # Evaluate the candidate's answer for the module that was just asked
        # The question asked was for turn_idx = current_turns - 1 (the previous question)
        question_turn_idx = current_turns - 1
        skill_eval = self._evaluate_answer_for_skill(clean_message, question_turn_idx)
        session_manager.append_skill_eval(session_id, {"module": skill_eval.module, "status": skill_eval.status})
        logger.info(f"Session {session_id} - Turn {current_turns} skill eval: {skill_eval.module} = {skill_eval.status}")

        # Check if max turns reached
        if current_turns >= self.max_turns:
            return self._finalize_interview(
                state,
                skill_eval=skill_eval,
                concluding_text="Thank you for completing this technical interview. We have concluded all interview questions."
            )

        # Generate next question from LLM
        reply = self.llm.generate_interview_reply(
            system_prompt=state.system_prompt,
            conversation_history=state.conversation_history,
            brief=state.brief,
            turn_count=current_turns,
        )

        # Append assistant reply
        session_manager.append_turn(session_id=session_id, role="assistant", content=reply)

        # Assessment focus and topic for the NEXT question (current_turns = next question's turn_idx)
        focus = self._build_assessment_focus(state.brief, turn_idx=current_turns)
        next_plan = _get_turn_plan(current_turns)

        return InterviewResponse(
            reply=reply,
            done=False,
            assessmentFocus=focus,
            skillEvaluation=skill_eval,
            currentTopicModule=next_plan["module"],
        )

    def _finalize_interview(
        self,
        state: InterviewState,
        skill_eval: Optional[SkillEvaluation] = None,
        concluding_text: str = "Interview completed."
    ) -> InterviewResponse:
        """
        Finalizes the interview and generates structured feedback.
        The final skill profile is built from state.turn_skill_evals (current interview evidence)
        so it exactly matches the live Skill Coverage Map shown during the interview.
        """
        feedback_prompt = build_feedback_prompt(state.brief, state.conversation_history)
        feedback = self.llm.generate_feedback(state.brief, state.conversation_history, feedback_prompt)

        # ─── Override skillProfile with current interview evidence ──────────────────
        # state.turn_skill_evals is the authoritative record of actual candidate performance.
        # It must replace any historically-derived skill profile from the LLM/mock feedback.
        # This ensures the final report's Skill Profile == the live Skill Coverage Map.
        ALL_MODULES = [
            "Embeddings & Vector Search",
            "RAG & Retrieval Architecture",
            "LLM Core & Prompting",
            "Agentic AI & MCP",
            "Security & Guardrails",
            "Evaluation & Benchmarks",
            "Production Engineering",
        ]

        # Build skill profile from per-turn evaluations (latest answer for each module wins)
        interview_skill_profile: Dict[str, str] = {}
        for eval_dict in state.turn_skill_evals:
            mod = eval_dict.get("module", "")
            status = eval_dict.get("status", "")
            if mod and status:
                # Latest evaluation for a module overwrites earlier ones
                # (allows both recovery from weak answers AND degradation from strong→weak)
                interview_skill_profile[mod] = status

        # Fill unassessed modules with "Not Assessed" — they were never tested this session
        for mod_name in ALL_MODULES:
            if mod_name not in interview_skill_profile:
                interview_skill_profile[mod_name] = "Not Assessed"

        feedback.skillProfile = interview_skill_profile
        # ────────────────────────────────────────────────────────────────────────────

        session_manager.mark_done(state.session_id, feedback=feedback)
        logger.info(f"Interview {state.session_id} finalized. Skill profile: {interview_skill_profile}")

        return InterviewResponse(
            reply=concluding_text,
            done=True,
            feedback=feedback,
            skillEvaluation=skill_eval,
        )


interview_engine = InterviewEngine()

