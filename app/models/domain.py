from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any
from app.models.schemas import CandidateData, FeedbackResponse


class MissionStatus(str, Enum):
    MASTERED = "MASTERED"
    STRUGGLED = "STRUGGLED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


@dataclass
class CurriculumDayInfo:
    day: int
    title: str
    type: str
    tools: List[str] = field(default_factory=list)
    objectives: List[str] = field(default_factory=list)
    module_number: int = 0
    module_title: str = ""


@dataclass
class CurriculumModuleInfo:
    number: int
    title: str
    start_day: int
    end_day: int


@dataclass
class MissionAnalysis:
    day: int
    title: str
    status: MissionStatus
    passed: Optional[bool]
    attempts: int
    skipped: bool
    module_number: int
    module_title: str
    tools: List[str] = field(default_factory=list)
    objectives: List[str] = field(default_factory=list)


@dataclass
class InterviewBrief:
    candidate_id: str
    candidate_name: str
    job_role: str
    years_experience: float
    education: str
    status: str
    commit_days: int
    missions_completed: int
    missions_first_try: int
    first_try_rate: float
    difficulty_level: str
    mastered: List[MissionAnalysis] = field(default_factory=list)
    struggled: List[MissionAnalysis] = field(default_factory=list)
    failed: List[MissionAnalysis] = field(default_factory=list)
    skipped: List[MissionAnalysis] = field(default_factory=list)
    module_performance: Dict[str, Dict[str, int]] = field(default_factory=dict)
    priority_topics: List[str] = field(default_factory=list)


@dataclass
class InterviewState:
    session_id: str
    candidate_data: CandidateData
    brief: InterviewBrief
    system_prompt: str
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    turn_count: int = 0
    done: bool = False
    feedback: Optional[FeedbackResponse] = None
    created_at: float = 0.0
    updated_at: float = 0.0
    # Per-turn skill evaluations: [{module: str, status: str}, ...]
    # Each entry corresponds to one candidate answer turn.
    turn_skill_evals: List[Dict[str, str]] = field(default_factory=list)

