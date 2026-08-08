import time
from typing import Dict, Optional, List
from app.models.domain import InterviewState, InterviewBrief
from app.models.schemas import CandidateData, FeedbackResponse
from app.utils.logging import logger


class SessionAlreadyExistsError(Exception):
    """Raised when trying to create a session with an already active session ID."""
    pass


class SessionNotFoundError(Exception):
    """Raised when attempting to access a non-existent session ID."""
    pass


class SessionManager:
    def __init__(self):
        self._sessions: Dict[str, InterviewState] = {}

    def session_exists(self, session_id: str) -> bool:
        return session_id in self._sessions

    def create(
        self,
        session_id: str,
        candidate_data: CandidateData,
        brief: InterviewBrief,
        system_prompt: str,
    ) -> InterviewState:
        if not session_id or not session_id.strip():
            raise ValueError("sessionId cannot be empty")

        if session_id in self._sessions:
            logger.warning(f"Attempted to recreate existing session: {session_id}")
            raise SessionAlreadyExistsError(f"Session '{session_id}' already exists.")

        now = time.time()
        state = InterviewState(
            session_id=session_id,
            candidate_data=candidate_data,
            brief=brief,
            system_prompt=system_prompt,
            conversation_history=[],
            turn_count=0,
            done=False,
            feedback=None,
            created_at=now,
            updated_at=now,
        )
        self._sessions[session_id] = state
        logger.info(f"Created new interview session: {session_id} for candidate {brief.candidate_name} ({brief.candidate_id})")
        return state

    def get(self, session_id: str) -> InterviewState:
        if not session_id or session_id not in self._sessions:
            logger.warning(f"Session not found: {session_id}")
            raise SessionNotFoundError(f"Session '{session_id}' was not found. Please initialize the interview first.")
        return self._sessions[session_id]

    def update(self, session_id: str, state: InterviewState) -> InterviewState:
        if session_id not in self._sessions:
            raise SessionNotFoundError(f"Session '{session_id}' was not found.")
        state.updated_at = time.time()
        self._sessions[session_id] = state
        return state

    def append_turn(self, session_id: str, role: str, content: str) -> None:
        state = self.get(session_id)
        state.conversation_history.append({"role": role, "content": content})
        state.updated_at = time.time()

    def increment_turn_count(self, session_id: str) -> int:
        state = self.get(session_id)
        state.turn_count += 1
        state.updated_at = time.time()
        return state.turn_count

    def append_skill_eval(self, session_id: str, eval_dict: dict) -> None:
        """Append a per-turn skill evaluation {module, status} to the session state."""
        state = self.get(session_id)
        state.turn_skill_evals.append(eval_dict)
        state.updated_at = time.time()

    def mark_done(self, session_id: str, feedback: Optional[FeedbackResponse] = None) -> InterviewState:
        state = self.get(session_id)
        state.done = True
        state.feedback = feedback
        state.updated_at = time.time()
        logger.info(f"Marked interview session {session_id} as done")
        return state

    def clear(self) -> None:
        """Utility for test suite cleanup."""
        self._sessions.clear()


# Global singleton session manager
session_manager = SessionManager()
