from typing import Optional, List, Dict, Any
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
        "difficulty": "easy",
    },
    {
        "turn_idx": 1,
        "module": "Embeddings & Vector Search",
        "focus_topic": "Vector Embeddings & Retrieval",
        "focus_detail": "Chunking trade-offs & hybrid search ranking",
        "difficulty": "medium",
    },
    {
        "turn_idx": 2,
        "module": "LLM Core & Prompting",
        "focus_topic": "LLM Integration & Structured Output",
        "focus_detail": "Structured output schemas & tool validation",
        "difficulty": "medium",
    },
    {
        "turn_idx": 3,
        "module": "Agentic AI & MCP",
        "focus_topic": "Agentic Workflow & MCP Integration",
        "focus_detail": "Multi-agent state, reasoning loops & MCP",
        "difficulty": "medium",
    },
    {
        "turn_idx": 4,
        "module": "Security & Guardrails",
        "focus_topic": "Prompt Injection Defense & Security",
        "focus_detail": "Prompt injection defense & container security",
        "difficulty": "medium",
    },
    {
        "turn_idx": 5,
        "module": "Evaluation & Benchmarks",
        "focus_topic": "Evaluation Recall & Benchmarking",
        "focus_detail": "Retrieval recall & LLM-as-a-judge benchmarks",
        "difficulty": "hard",
    },
    {
        "turn_idx": 6,
        "module": "Production Engineering",
        "focus_topic": "Async Token Streaming & Moderation",
        "focus_detail": "Async token streaming & guardrail moderation",
        "difficulty": "hard",
    },
    {
        "turn_idx": 7,
        "module": "Production Engineering",
        "focus_topic": "Semantic Prompt Caching & Model Routing",
        "focus_detail": "Semantic prompt caching & dynamic model routing",
        "difficulty": "hard",
    },
    {
        "turn_idx": 8,
        "module": "RAG & Retrieval Architecture",
        "focus_topic": "RAG vs Fine-Tuning Trade-offs",
        "focus_detail": "RAG context injection vs LoRA fine-tuning",
        "difficulty": "hard",
    },
    {
        "turn_idx": 9,
        "module": "Production Engineering",
        "focus_topic": "Production Capstone & Concurrency",
        "focus_detail": "Enterprise capstone & 10k concurrency scale",
        "difficulty": "hard",
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


# All 7 canonical curriculum modules
ALL_CURRICULUM_MODULES = [
    "Embeddings & Vector Search",
    "RAG & Retrieval Architecture",
    "LLM Core & Prompting",
    "Agentic AI & MCP",
    "Security & Guardrails",
    "Evaluation & Benchmarks",
    "Production Engineering",
]


def normalize_module_name(name: Optional[str]) -> str:
    """Canonical module normalization function handling whitespace, newlines, and case mismatches."""
    if not name:
        return ""
    cleaned = " ".join(name.strip().split())
    for canonical in ALL_CURRICULUM_MODULES:
        if cleaned.lower() == canonical.lower():
            return canonical
    return cleaned


STATUS_SCORE_MAP = {
    "Strong": 1.0,
    "Good": 0.75,
    "Developing": 0.50,
    "Needs Attention": 0.10,
}


def derive_cumulative_status(average_score: float) -> str:
    """
    Deterministic rule to derive cumulative module status from accumulated numerical evidence:
      average_score >= 0.85 -> Strong
      average_score >= 0.60 -> Good
      average_score >= 0.35 -> Developing
      average_score < 0.35  -> Needs Attention
    """
    if average_score >= 0.85:
        return "Strong"
    elif average_score >= 0.60:
        return "Good"
    elif average_score >= 0.35:
        return "Developing"
    else:
        return "Needs Attention"


DIFFICULTY_MULTIPLIER = {
    "easy": 0.8,
    "medium": 1.0,
    "hard": 1.2,
}


def update_cumulative_module_assessment(
    state: InterviewState,
    raw_module: str,
    status: str,
    raw_score: Optional[float] = None,
    difficulty: str = "medium",
) -> SkillEvaluation:
    """
    Updates or initializes the cumulative evidence record for a module on InterviewState.
    Retains all historical evaluations, scores, difficulties, and weighted evidence without losing historical context.
    """
    module = normalize_module_name(raw_module)
    record = state.module_assessments.get(module)

    score = raw_score if raw_score is not None else STATUS_SCORE_MAP.get(status, 0.50)
    mult = DIFFICULTY_MULTIPLIER.get((difficulty or "medium").lower(), 1.0)
    weighted_score = min(1.0, score * mult)

    if record is None:
        evaluations = [status]
        scores = [score]
        weighted_scores = [weighted_score]
        difficulties = [difficulty]
    else:
        evaluations = list(record.get("evaluations", [])) + [status]
        scores = list(record.get("scores", [])) + [score]
        weighted_scores = list(record.get("weighted_scores", [])) + [weighted_score]
        difficulties = list(record.get("difficulties", [])) + [difficulty]

    avg_score = sum(weighted_scores) / len(weighted_scores)
    cum_status = derive_cumulative_status(avg_score)
    probe_cnt = len(evaluations)

    state.module_assessments[module] = {
        "module": module,
        "evaluations": evaluations,
        "scores": scores,
        "weighted_scores": weighted_scores,
        "difficulties": difficulties,
        "probe_count": probe_cnt,
        "latest_status": status,
        "cumulative_score": round(sum(weighted_scores), 4),
        "average_score": round(avg_score, 4),
        "cumulative_status": cum_status,
    }

    return SkillEvaluation(
        module=module,
        status=cum_status,
        cumulative_status=cum_status,
        probe_count=probe_cnt,
    )


def _get_turn_plan(turn_idx: int) -> Dict:
    """Return the TURN_PLAN entry for a given turn index (clamped to last entry)."""
    idx = min(turn_idx, len(TURN_PLAN) - 1)
    return TURN_PLAN[idx]


class InterviewEngine:
    def __init__(self, llm: Optional[LLMClient] = None, max_turns: Optional[int] = None):
        self.llm = llm or llm_client
        self.max_turns = max_turns if max_turns is not None else settings.MAX_TURNS

    def _build_assessment_focus(self, brief: InterviewBrief, turn_idx: int) -> AssessmentFocus:
        """Build AssessmentFocus from TURN_PLAN fallback — consistent with currentTopicModule."""
        plan = _get_turn_plan(turn_idx)
        topic = plan["focus_topic"]

        if brief.failed and turn_idx == 0:
            reason = f"Focusing on identified knowledge gap (Day {brief.failed[0].day}: {brief.failed[0].title})."
        elif brief.struggled and turn_idx <= 2:
            reason = f"Probing struggle area (Day {brief.struggled[0].day}: {brief.struggled[0].title})."
        elif brief.years_experience >= 5:
            reason = f"Calibrated {plan['focus_detail']} for {brief.job_role} level ({brief.years_experience} yrs experience)."
        else:
            reason = f"Evaluating {plan['focus_detail']} for {brief.job_role} candidate."

        return AssessmentFocus(topic=topic, reason=reason)

    def _evaluate_answer_for_skill(
        self,
        candidate_message: str,
        question_module: Any,
        question_asked: str = "",
        job_role: str = "Software Engineer",
        years_experience: float = 0.0,
        state: Optional[InterviewState] = None,
        difficulty: str = "medium",
    ) -> SkillEvaluation:
        """
        Evaluates the candidate's actual answer using structured LLM evaluation or deterministic unknown detection.
        Supports both module string name (e.g., 'Embeddings & Vector Search') and integer turn_idx (for backward compatibility).
        """
        if isinstance(question_module, int):
            module_name = _get_turn_plan(question_module)["module"]
        else:
            module_name = str(question_module)

        module_name = normalize_module_name(module_name)

        eval_out = self.llm.evaluate_answer_structured(
            module=module_name,
            question_asked=question_asked,
            candidate_answer=candidate_message,
            job_role=job_role,
            years_experience=years_experience,
            difficulty=difficulty,
        )

        turn_status = eval_out.status
        turn_score = eval_out.score

        if state is not None:
            state.turn_skill_evals.append({
                "module": module_name,
                "status": turn_status,
                "score": turn_score,
                "difficulty": difficulty,
            })
            return update_cumulative_module_assessment(
                state=state,
                raw_module=module_name,
                status=turn_status,
                raw_score=turn_score,
                difficulty=difficulty,
            )

        return SkillEvaluation(
            module=module_name,
            status=turn_status,
            cumulative_status=turn_status,
            probe_count=1,
        )


    def start_interview(self, session_id: str, candidate_data: Optional[CandidateData]) -> InterviewResponse:
        """
        Initializes a new interview session for the given candidate using adaptive LLM question generation with TURN_PLAN fallback.
        """
        if not session_id or not session_id.strip():
            raise ValueError("sessionId must not be empty.")

        if candidate_data is None:
            raise ValueError("Candidate data is required to start an interview.")

        if session_manager.session_exists(session_id):
            raise SessionAlreadyExistsError(f"Session '{session_id}' already exists and cannot be re-initialized.")

        # 1. Build profile brief
        brief = build_candidate_brief(candidate_data)
        logger.info(f"Built brief for candidate {brief.candidate_name} ({brief.candidate_id})")

        # 2. Build system prompt
        system_prompt = build_system_prompt(brief, max_turns=self.max_turns)

        # 3. Create session state
        state = session_manager.create(
            session_id=session_id,
            candidate_data=candidate_data,
            brief=brief,
            system_prompt=system_prompt,
        )

        # 4. Generate first question (try adaptive LLM first, fallback to mock/TURN_PLAN)
        opening_reply = ""
        first_module = "Embeddings & Vector Search"

        try:
            adaptive_out = self.llm.generate_adaptive_question_structured(
                brief=brief,
                allowed_modules=ALL_CURRICULUM_MODULES,
                current_module=first_module,
                current_skill_map={m: "Not Assessed" for m in ALL_CURRICULUM_MODULES},
                asked_questions=[],
                turn_number=1,
                max_turns=self.max_turns,
                current_difficulty="medium",
            )
            opening_reply = adaptive_out.question
            first_module = adaptive_out.module
            state.current_module = adaptive_out.module
            state.current_difficulty = adaptive_out.difficulty
        except Exception as e:
            logger.info(f"Start interview adaptive question fallback ({e}). Using deterministic reply.")
            opening_reply = self.llm.generate_interview_reply(
                system_prompt=system_prompt,
                conversation_history=[],
                brief=brief,
                turn_count=0,
            )
            plan = _get_turn_plan(0)
            first_module = plan["module"]
            state.current_module = first_module

        # Record opening question in state
        state.asked_questions.append(opening_reply)
        state.module_probe_counts[first_module] = 1

        # 5. Store opening reply in conversation history
        session_manager.append_turn(session_id=session_id, role="assistant", content=opening_reply)

        # 6. Build assessment focus
        focus = AssessmentFocus(
            topic=first_module,
            reason=f"Opening technical assessment module calibrated for {brief.job_role}."
        )

        return InterviewResponse(
            reply=opening_reply,
            done=False,
            assessmentFocus=focus,
            currentTopicModule=first_module,
        )

    def handle_turn(self, session_id: str, message: Optional[str]) -> InterviewResponse:
        """
        Handles an ongoing conversation turn using structured LLM evaluation and adaptive next-question planning.
        """
        if not session_id or not session_id.strip():
            raise ValueError("sessionId must not be empty.")

        state = session_manager.get(session_id)

        if state.done:
            return InterviewResponse(
                reply="Interview has already concluded.",
                done=True,
                feedback=state.feedback,
            )

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

        # 1. Evaluate candidate's answer for the module that was just asked
        asked_module = state.current_module
        previous_question = state.asked_questions[-1] if state.asked_questions else ""

        skill_eval = self._evaluate_answer_for_skill(
            candidate_message=clean_message,
            question_module=asked_module,
            question_asked=previous_question,
            job_role=state.brief.job_role,
            years_experience=state.brief.years_experience,
            state=state,
        )
        state.last_evaluation = {"module": skill_eval.module, "status": skill_eval.status}
        logger.info(f"Session {session_id} - Turn {current_turns} eval: {skill_eval.module} = {skill_eval.status}")

        # Check if max turns reached
        if current_turns >= self.max_turns:
            return self._finalize_interview(
                state,
                skill_eval=skill_eval,
                concluding_text="Thank you for completing this technical interview. We have concluded all interview questions."
            )

        # 2. Build current live skill map dictionary for adaptive planner using cumulative assessments
        current_skill_map = {mod: "Not Assessed" for mod in ALL_CURRICULUM_MODULES}
        for mod, record in state.module_assessments.items():
            if mod in current_skill_map:
                current_skill_map[mod] = record.get("cumulative_status", "Not Assessed")

        # Enforce max 2 probes per module
        allowed_modules = [
            m for m in ALL_CURRICULUM_MODULES
            if state.module_probe_counts.get(m, 0) < 2 or m != state.current_module
        ]
        if not allowed_modules:
            allowed_modules = ALL_CURRICULUM_MODULES

        # 3. Generate next question adaptively (with fallback)
        next_reply = ""
        next_module = state.current_module
        focus = None

        try:
            adaptive_out = self.llm.generate_adaptive_question_structured(
                brief=state.brief,
                allowed_modules=allowed_modules,
                current_module=state.current_module,
                current_skill_map=current_skill_map,
                previous_question=previous_question,
                previous_answer=clean_message,
                previous_eval=state.last_evaluation,
                asked_questions=state.asked_questions,
                turn_number=current_turns + 1,
                max_turns=self.max_turns,
                current_difficulty=state.current_difficulty,
            )
            next_reply = f"{adaptive_out.neutral_transition} {adaptive_out.question}".strip()
            next_module = adaptive_out.module
            state.current_module = adaptive_out.module
            state.current_difficulty = adaptive_out.difficulty
            state.asked_questions.append(adaptive_out.question)
            state.module_probe_counts[next_module] = state.module_probe_counts.get(next_module, 0) + 1
            focus = AssessmentFocus(
                topic=next_module,
                reason=f"Adaptive assessment topic: {next_module} ({state.current_difficulty} difficulty)."
            )
        except Exception as e:
            logger.info(f"Adaptive question generation fallback on turn {current_turns} ({e}). Using deterministic TURN_PLAN.")
            next_reply = self.llm.generate_interview_reply(
                system_prompt=state.system_prompt,
                conversation_history=state.conversation_history,
                brief=state.brief,
                turn_count=current_turns,
            )
            plan = _get_turn_plan(current_turns)
            next_module = plan["module"]
            state.current_module = next_module
            state.current_difficulty = plan.get("difficulty", "medium")
            state.asked_questions.append(next_reply)
            state.module_probe_counts[next_module] = state.module_probe_counts.get(next_module, 0) + 1
            focus = self._build_assessment_focus(state.brief, turn_idx=current_turns)

        # 4. Append assistant reply
        session_manager.append_turn(session_id=session_id, role="assistant", content=next_reply)

        cumulative_skill_map = {
            mod: record["cumulative_status"]
            for mod, record in state.module_assessments.items()
        }

        return InterviewResponse(
            reply=next_reply,
            done=False,
            assessmentFocus=focus,
            skillEvaluation=skill_eval,
            currentTopicModule=next_module,
            cumulativeSkillMap=cumulative_skill_map,
        )

    def _finalize_interview(
        self,
        state: InterviewState,
        skill_eval: Optional[SkillEvaluation] = None,
        concluding_text: str = "Interview completed."
    ) -> InterviewResponse:
        """
        Finalizes the interview and generates structured feedback.
        Uses deterministic evidence-based disposition calculation to guarantee consistency.
        """
        feedback = calculate_deterministic_disposition_and_feedback(state)

        session_manager.mark_done(state.session_id, feedback=feedback)
        logger.info(f"Interview {state.session_id} finalized. Skill profile: {feedback.skillProfile}, Disposition: {feedback.disposition}")

        return InterviewResponse(
            reply=concluding_text,
            done=True,
            feedback=feedback,
            skillEvaluation=skill_eval,
            currentTopicModule=state.current_module,
            cumulativeSkillMap=feedback.skillProfile,
        )


def calculate_deterministic_disposition_and_feedback(state: InterviewState) -> FeedbackResponse:
    """
    Deterministically computes final performance score, final disposition, skill profile,
    and consistent executive summary based on all evaluated candidate turn evidence.
    """
    user_turns = [e for e in state.turn_skill_evals]
    total_turns = len(user_turns)

    # 1. Overall Answer Quality Score (45% weight)
    if total_turns > 0:
        overall_answer_quality_score = sum(e.get("score", 0.50) for e in user_turns) / total_turns
    else:
        overall_answer_quality_score = 0.10

    # 2. Cumulative Module Performance (30% weight)
    assessed_modules = [
        r for r in state.module_assessments.values()
        if r.get("probe_count", 0) > 0 or r.get("cumulative_status") != "Not Assessed"
    ]
    if assessed_modules:
        cumulative_module_performance = sum(
            r.get("average_score", 0.10) for r in assessed_modules
        ) / len(assessed_modules)
    else:
        cumulative_module_performance = 0.10

    # 3. Difficulty-Weighted Performance (15% weight)
    if total_turns > 0:
        difficulty_weighted_performance = sum(
            e.get("score", 0.50) * DIFFICULTY_MULTIPLIER.get((e.get("difficulty") or "medium").lower(), 1.0)
            for e in user_turns
        ) / total_turns
        difficulty_weighted_performance = min(1.0, difficulty_weighted_performance)
    else:
        difficulty_weighted_performance = 0.10

    # 4. Consistency Score (10% weight)
    weak_turn_count = sum(
        1 for e in user_turns
        if e.get("status") in ["Needs Attention", "Developing"] or e.get("score", 0.5) < 0.60
    )
    strong_turn_count = sum(
        1 for e in user_turns
        if e.get("status") == "Strong" or e.get("score", 0.5) >= 0.85
    )

    weak_ratio = (weak_turn_count / total_turns) if total_turns > 0 else 1.0
    consistency_score = max(0.0, 1.0 - weak_ratio)

    # Final Performance Score
    final_performance_score = (
        0.45 * overall_answer_quality_score +
        0.30 * cumulative_module_performance +
        0.15 * difficulty_weighted_performance +
        0.10 * consistency_score
    )
    final_performance_score = round(max(0.0, min(1.0, final_performance_score)), 4)

    # Count modules by cumulative status
    module_statuses = {mod: record.get("cumulative_status", "Not Assessed") for mod, record in state.module_assessments.items()}
    needs_attention_modules_count = sum(1 for s in module_statuses.values() if s == "Needs Attention")
    developing_modules_count = sum(1 for s in module_statuses.values() if s == "Developing")
    strong_modules_count = sum(1 for s in module_statuses.values() if s == "Strong")

    # Hard question performance check
    hard_evals = [e for e in user_turns if (e.get("difficulty") or "").lower() == "hard"]
    hard_performance_weak = bool(hard_evals and (sum(e.get("score", 0.5) for e in hard_evals) / len(hard_evals)) < 0.55)

    # Deterministic Disposition Baseline
    if final_performance_score >= 0.78:
        disposition = "Strong Fit"
    elif final_performance_score >= 0.55:
        disposition = "Consider"
    else:
        disposition = "Needs Development"

    # HARD SAFEGUARDS FOR DISPOSITION
    # Safeguard 1: All weak answers -> Needs Development
    if total_turns > 0 and weak_turn_count == total_turns:
        disposition = "Needs Development"

    # Safeguard 2: Multiple Needs Attention modules -> cannot be Strong Fit
    if needs_attention_modules_count >= 2 and disposition == "Strong Fit":
        disposition = "Consider"

    # Safeguard 3: Low overall answer quality (< 0.65) -> cannot be Strong Fit
    if overall_answer_quality_score < 0.65 and disposition == "Strong Fit":
        disposition = "Consider"

    # Safeguard 4: High weak turn ratio (>= 40%) -> cannot be Strong Fit
    if weak_ratio >= 0.40 and disposition == "Strong Fit":
        disposition = "Consider"

    # Safeguard 5: Consistently weak performance on hard questions -> cannot be Strong Fit
    if hard_performance_weak and disposition == "Strong Fit":
        disposition = "Consider"

    # Build Skill Profile
    skill_profile = dict(module_statuses)
    for mod_name in ALL_CURRICULUM_MODULES:
        if mod_name not in skill_profile:
            skill_profile[mod_name] = "Not Assessed"

    # Build consistent Executive Summary, Strengths, Gaps, Next Steps
    cand_name = state.brief.candidate_name
    role = state.brief.job_role
    exp = state.brief.years_experience

    if disposition == "Needs Development":
        summary = (
            f"Candidate {cand_name} ({role}, {exp} years experience) demonstrated technical interest, "
            f"but live probing revealed significant knowledge gaps across {weak_turn_count} turn(s). "
            f"Multiple answers lacked technical depth or missing key architectural concepts, resulting in a 'Needs Development' disposition."
        )
    elif disposition == "Strong Fit":
        summary = (
            f"Candidate {cand_name} ({role}, {exp} years experience) demonstrated outstanding live interview performance "
            f"and strong technical mastery across {strong_turn_count} high-quality responses. "
            f"The candidate exhibited solid architectural reasoning and practical problem-solving aptitude, earning a 'Strong Fit' disposition."
        )
    else:
        summary = (
            f"Candidate {cand_name} ({role}, {exp} years experience) exhibited a balanced technical understanding during the assessment, "
            f"showing clear responses on some topics alongside areas needing development ({weak_turn_count} weak/developing turn(s)). "
            f"Recommended disposition is 'Consider' with targeted review on identified friction points."
        )

    # Strengths based strictly on high scoring turns (score >= 0.65)
    strengths = []
    high_turns = [e for e in user_turns if e.get("score", 0.0) >= 0.65]
    if high_turns:
        for e in high_turns[:3]:
            mod = e.get("module", "technical topic")
            strengths.append(f"Demonstrated solid technical understanding in {mod}.")
    else:
        strengths.append("Demonstrated foundational interest in modern AI engineering concepts.")

    # Gaps based strictly on weak turns (score < 0.60 or Needs Attention / Developing)
    gaps = []
    low_turns = [e for e in user_turns if e.get("score", 1.0) < 0.60 or e.get("status") in ["Needs Attention", "Developing"]]
    for e in low_turns[:3]:
        mod = e.get("module", "technical topic")
        gaps.append(f"Friction in {mod}: Candidate answer lacked depth or omitted key implementation mechanisms.")
    if not gaps and weak_turn_count > 0:
        gaps.append(f"Live Probing Friction: Candidate gave vague or incomplete answers on {weak_turn_count} interview turn(s).")
    if not gaps:
        gaps.append("Further verification needed on high-concurrency production latency and edge-case failure modes.")

    skipped_list = []
    if state.brief and state.brief.skipped:
        skipped_list = [f"Day {m.day}: {m.title}" for m in state.brief.skipped]
    elif state.candidate_data and state.candidate_data.missions:
        skipped_list = [f"Day {m.day}: {m.title or ''}".strip() for m in state.candidate_data.missions if getattr(m, "skipped", False)]

    if skipped_list:
        for skipped in skipped_list[:2]:
            gaps.append(f"Unverified skipped mission in candidate history ({skipped}).")

    next_steps = [
        "Conduct targeted technical review on identified friction areas to verify architectural depth.",
        "Build a production benchmark suite measuring retrieval recall, hallucination detection, and latency.",
        "Study advanced agent orchestration patterns and standardized Model Context Protocol (MCP) integrations."
    ]
    if skipped_list:
        for skipped in skipped_list[:2]:
            next_steps.append(f"Verify candidate capabilities on skipped mission ({skipped}).")

    return FeedbackResponse(
        summary=summary,
        strengths=strengths,
        gaps=gaps,
        next=next_steps,
        skillProfile=skill_profile,
        disposition=disposition,
    )


interview_engine = InterviewEngine()
