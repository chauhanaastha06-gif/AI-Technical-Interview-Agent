from typing import Optional
from app.config import settings
from app.models.schemas import CandidateData, InterviewResponse, FeedbackResponse
from app.models.domain import InterviewState
from app.services.candidate_profile import build_candidate_brief
from app.services.prompt_builder import build_system_prompt, build_feedback_prompt
from app.services.session_manager import session_manager, SessionAlreadyExistsError, SessionNotFoundError
from app.services.llm_client import llm_client, LLMClient
from app.utils.logging import logger


class InterviewEngine:
    def __init__(self, llm: Optional[LLMClient] = None, max_turns: Optional[int] = None):
        self.llm = llm or llm_client
        self.max_turns = max_turns if max_turns is not None else settings.MAX_TURNS

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

        return InterviewResponse(reply=opening_reply, done=False)

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

        # Check if max turns reached
        if current_turns >= self.max_turns:
            return self._finalize_interview(state, concluding_text="Thank you for completing this technical interview. We have concluded all interview questions.")

        # Generate next question from LLM
        reply = self.llm.generate_interview_reply(
            system_prompt=state.system_prompt,
            conversation_history=state.conversation_history,
            brief=state.brief,
            turn_count=current_turns,
        )

        # Append assistant reply
        session_manager.append_turn(session_id=session_id, role="assistant", content=reply)

        return InterviewResponse(reply=reply, done=False)

    def _finalize_interview(self, state: InterviewState, concluding_text: str = "Interview completed.") -> InterviewResponse:
        """
        Finalizes the interview and generates structured feedback.
        """
        feedback_prompt = build_feedback_prompt(state.brief, state.conversation_history)
        feedback = self.llm.generate_feedback(state.brief, state.conversation_history, feedback_prompt)

        session_manager.mark_done(state.session_id, feedback=feedback)
        logger.info(f"Interview {state.session_id} finalized with feedback summary: {feedback.summary[:60]}...")

        return InterviewResponse(
            reply=concluding_text,
            done=True,
            feedback=feedback,
        )


interview_engine = InterviewEngine()
