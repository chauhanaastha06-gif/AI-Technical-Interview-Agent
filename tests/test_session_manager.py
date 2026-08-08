import pytest
from app.services.session_manager import (
    SessionManager,
    SessionAlreadyExistsError,
    SessionNotFoundError,
)
from app.models.schemas import CandidateData, CandidateMember, FeedbackResponse
from app.models.domain import InterviewBrief


@pytest.fixture
def manager():
    return SessionManager()


@pytest.fixture
def sample_brief():
    return InterviewBrief(
        candidate_id="CAND-TEST",
        candidate_name="Test Candidate",
        job_role="Software Engineer",
        years_experience=3.0,
        education="BS Computer Science",
        status="COMPLETED",
        commit_days=20,
        missions_completed=20,
        missions_first_try=15,
        first_try_rate=0.75,
        difficulty_level="Mid-Level",
    )


def test_create_and_get_session(manager, sample_brief):
    cand_data = CandidateData(member=CandidateMember(id="CAND-TEST", name="Test Candidate"))
    state = manager.create("session-1", cand_data, sample_brief, "System prompt")

    assert state.session_id == "session-1"
    assert state.brief.candidate_name == "Test Candidate"
    assert state.done is False
    assert state.turn_count == 0

    retrieved = manager.get("session-1")
    assert retrieved.session_id == "session-1"


def test_duplicate_session_error(manager, sample_brief):
    cand_data = CandidateData(member=CandidateMember(id="CAND-TEST", name="Test Candidate"))
    manager.create("session-dup", cand_data, sample_brief, "System prompt")

    with pytest.raises(SessionAlreadyExistsError):
        manager.create("session-dup", cand_data, sample_brief, "System prompt")


def test_unknown_session_error(manager):
    with pytest.raises(SessionNotFoundError):
        manager.get("unknown-session-xyz")


def test_append_turns_and_increment(manager, sample_brief):
    cand_data = CandidateData(member=CandidateMember(id="CAND-TEST"))
    manager.create("session-turns", cand_data, sample_brief, "Prompt")

    manager.append_turn("session-turns", role="assistant", content="Hello!")
    manager.append_turn("session-turns", role="user", content="Hi, let's start.")

    turns = manager.increment_turn_count("session-turns")
    assert turns == 1

    state = manager.get("session-turns")
    assert len(state.conversation_history) == 2
    assert state.conversation_history[0]["content"] == "Hello!"
    assert state.conversation_history[1]["content"] == "Hi, let's start."


def test_mark_done(manager, sample_brief):
    cand_data = CandidateData(member=CandidateMember(id="CAND-TEST"))
    manager.create("session-done", cand_data, sample_brief, "Prompt")

    fb = FeedbackResponse(
        summary="Great job",
        strengths=["Good coding"],
        gaps=["Needs more vector DB knowledge"],
        next=["Study MCP"],
    )

    state = manager.mark_done("session-done", feedback=fb)
    assert state.done is True
    assert state.feedback is not None
    assert state.feedback.summary == "Great job"
