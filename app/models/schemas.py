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


class FeedbackResponse(BaseModel):
    summary: str = Field(..., description="Concise overall summary of interview performance")
    strengths: List[str] = Field(default_factory=list, description="Key technical strengths demonstrated")
    gaps: List[str] = Field(default_factory=list, description="Knowledge gaps or areas needing improvement")
    next: List[str] = Field(default_factory=list, description="Recommended next steps and study areas")

    model_config = ConfigDict(extra="allow")


class InterviewResponse(BaseModel):
    reply: str
    done: bool
    feedback: Optional[FeedbackResponse] = None

    model_config = ConfigDict(extra="allow", populate_by_name=True)
