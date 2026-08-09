from typing import List, Optional, Union, Any, Dict
from pydantic import BaseModel, Field, ConfigDict


class CandidateMember(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    jobRole: Optional[str] = None
    yearsExperience: Optional[Union[int, float]] = None
    education: Optional[str] = None
    status: Optional[str] = None

    model_config = ConfigDict(extra="allow")


class CandidateMission(BaseModel):
    day: int
    title: Optional[str] = None
    passed: Optional[bool] = None
    attempts: Optional[int] = None
    skipped: Optional[bool] = None

    model_config = ConfigDict(extra="allow")


class CandidateSignals(BaseModel):
    commitDays: Optional[int] = None
    missionsCompleted: Optional[int] = None
    missionsFirstTry: Optional[int] = None

    model_config = ConfigDict(extra="allow")


class CandidateData(BaseModel):
    member: Optional[CandidateMember] = None
    missions: Optional[List[CandidateMission]] = Field(default_factory=list)
    signals: Optional[CandidateSignals] = None

    model_config = ConfigDict(extra="allow")


class InterviewRequest(BaseModel):
    sessionId: str = Field(..., description="Unique interview session ID", min_length=1)
    candidate: Optional[CandidateData] = Field(default=None, description="Candidate profile for starting interview")
    message: Optional[str] = Field(default=None, description="Candidate message for continuing interview")

    model_config = ConfigDict(extra="allow", populate_by_name=True)


class AssessmentFocus(BaseModel):
    topic: str = Field(..., description="High-level assessment focus topic")
    reason: str = Field(..., description="Safe, high-level rationale derived from curriculum or profile")

    model_config = ConfigDict(extra="allow")


class SkillEvaluation(BaseModel):
    """Per-turn skill assessment derived from candidate's actual answer."""
    module: str = Field(..., description="Curriculum module name being assessed this turn")
    status: str = Field(..., description="Evidence-based status: Strong, Good, Developing, Needs Attention")
    cumulative_status: Optional[str] = Field(default=None, description="Deterministic cumulative status incorporating historical evidence")
    probe_count: Optional[int] = Field(default=1, description="Total evaluations for this module")

    model_config = ConfigDict(extra="allow")


class FeedbackResponse(BaseModel):
    summary: str = Field(..., description="Concise overall summary of interview performance")
    strengths: List[str] = Field(default_factory=list, description="Key technical strengths demonstrated")
    gaps: List[str] = Field(default_factory=list, description="Knowledge gaps or areas needing improvement")
    next: List[str] = Field(default_factory=list, description="Recommended next steps and study areas")
    skillProfile: Optional[Dict[str, str]] = Field(default=None, description="Qualitative technical skill levels across curriculum modules")
    disposition: Optional[str] = Field(default=None, description="Assessment recommendation: Strong Fit, Consider, or Needs Development")

    model_config = ConfigDict(extra="allow")


class InterviewResponse(BaseModel):
    reply: str
    done: bool
    feedback: Optional[FeedbackResponse] = None
    assessmentFocus: Optional[AssessmentFocus] = None
    skillEvaluation: Optional[SkillEvaluation] = Field(default=None, description="Per-turn skill assessment from current candidate answer")
    currentTopicModule: Optional[str] = Field(default=None, description="The curriculum module name being assessed this turn — single source of truth for sidebar labels")
    cumulativeSkillMap: Optional[Dict[str, str]] = Field(default=None, description="Canonical cumulative skill map for all curriculum modules")

    model_config = ConfigDict(extra="allow", populate_by_name=True)


class AnswerEvaluationOutput(BaseModel):
    """Structured LLM evaluation output for a single candidate answer across technical dimensions."""
    module: str = Field(..., description="Curriculum module evaluated")
    status: str = Field(..., description="Evidence-based rating: Strong, Good, Developing, or Needs Attention")
    score: float = Field(..., description="Overall numerical evaluation score from 0.0 to 1.0")

    # Quality dimensions (0.0 to 1.0)
    correctness_score: float = Field(default=0.5, description="Technical accuracy and correctness")
    relevance_score: float = Field(default=0.5, description="Direct relevance to the technical question asked")
    completeness_score: float = Field(default=0.5, description="Completeness in addressing key aspects of the question")
    depth_score: float = Field(default=0.5, description="Depth of technical understanding (why/how vs buzzwords)")
    specificity_score: float = Field(default=0.5, description="Concrete mechanisms, trade-offs, or implementation details")
    practical_reasoning_score: float = Field(default=0.5, description="Practical engineering and real-world system reasoning")
    clarity_score: float = Field(default=0.5, description="Logical structure and clarity of explanation")
    overall_score: float = Field(default=0.5, description="Calculated overall score")
    difficulty: str = Field(default="medium", description="Question difficulty level: easy, medium, or hard")

    reasoning_summary: str = Field(..., description="Concise non-spoiler technical rationale for rating")
    knowledge_gaps: List[str] = Field(default_factory=list, description="Identified technical knowledge gaps or friction points")

    model_config = ConfigDict(extra="allow")


class NextQuestionOutput(BaseModel):
    """Structured LLM output for adaptive next-question generation."""
    module: str = Field(..., description="Target curriculum module name selected from allowed CURRICULUM_MODULES")
    difficulty: str = Field(..., description="Selected difficulty level: easy, medium, or hard")
    question: str = Field(..., description="The single technical question to ask the candidate")
    neutral_transition: str = Field(default="Understood. Let's move on.", description="Professional neutral transition phrase (never positive validation for weak/unknown answers)")
    should_probe: bool = Field(default=False, description="Whether this turn is a follow-up/probe into the current topic")
    reasoning: str = Field(default="", description="Internal rationale for selecting this module/question")

    model_config = ConfigDict(extra="allow")


