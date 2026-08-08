from typing import Dict, List, Optional
from app.models.schemas import CandidateData, CandidateMission
from app.models.domain import (
    InterviewBrief,
    MissionAnalysis,
    MissionStatus,
    CurriculumDayInfo,
)
from app.data.loader import DataLoader, data_loader


def determine_difficulty_level(job_role: str, years_experience: float) -> str:
    role_lower = (job_role or "").lower()
    exp = years_experience if years_experience is not None else 0.0

    if "intern" in role_lower or "student" in role_lower or (exp <= 1.0 and "senior" not in role_lower and "lead" not in role_lower):
        return "Entry-Level / Fundamentals Focus - Focus on foundational concepts, basic syntax, terminology, step-by-step reasoning, and standard library usage."
    elif "distinguished" in role_lower or "principal" in role_lower or "fellow" in role_lower or exp >= 18.0:
        return "Distinguished / Strategic Architecture Focus - High-level distributed architectures, scalability limits, failure domain isolation, production reliability, governance, and deep technical trade-offs."
    elif "senior" in role_lower or "lead" in role_lower or "staff" in role_lower or exp >= 6.0:
        return "Senior / Deep Implementation & Architecture - End-to-end component design, production-grade error handling, RAG chunking vs retrieval trade-offs, agent loops, async streaming, and deployment bottlenecks."
    else:
        return "Mid-Level / Practical Implementation - Practical API integrations, vector search nuances, prompt structuring, data pipelines, function calling schemas, and unit testing."


def analyze_mission(mission: CandidateMission, loader: DataLoader) -> MissionAnalysis:
    day_num = mission.day
    day_info: Optional[CurriculumDayInfo] = loader.get_curriculum_day(day_num)

    title = mission.title or (day_info.title if day_info else f"Day {day_num}")
    passed = mission.passed
    attempts = mission.attempts if mission.attempts is not None else 0
    skipped = bool(mission.skipped)

    # Classification according to specification
    if skipped:
        status = MissionStatus.SKIPPED
    elif passed is False:
        status = MissionStatus.FAILED
    elif passed is True:
        if attempts <= 2:
            status = MissionStatus.MASTERED
        else:
            status = MissionStatus.STRUGGLED
    else:
        # Fallback if passed is None and not skipped
        status = MissionStatus.SKIPPED if attempts == 0 else MissionStatus.STRUGGLED

    mod_num = day_info.module_number if day_info else 0
    mod_title = day_info.module_title if day_info else "General"
    tools = day_info.tools if day_info else []
    objectives = day_info.objectives if day_info else []

    return MissionAnalysis(
        day=day_num,
        title=title,
        status=status,
        passed=passed,
        attempts=attempts,
        skipped=skipped,
        module_number=mod_num,
        module_title=mod_title,
        tools=tools,
        objectives=objectives,
    )


def build_candidate_brief(candidate: CandidateData, loader: Optional[DataLoader] = None) -> InterviewBrief:
    loader = loader or data_loader
    member = candidate.member
    signals = candidate.signals

    candidate_id = member.id if member and member.id else "UNKNOWN_CANDIDATE"
    candidate_name = member.name if member and member.name else "Candidate"
    job_role = member.jobRole if member and member.jobRole else "Software Engineer"
    years_exp = float(member.yearsExperience) if member and member.yearsExperience is not None else 0.0
    education = member.education if member and member.education else "N/A"
    status_str = member.status if member and member.status else "IN_PROGRESS"

    commit_days = signals.commitDays if signals and signals.commitDays is not None else 0
    missions_completed = signals.missionsCompleted if signals and signals.missionsCompleted is not None else 0
    missions_first_try = signals.missionsFirstTry if signals and signals.missionsFirstTry is not None else 0

    first_try_rate = (missions_first_try / missions_completed) if missions_completed > 0 else 0.0

    difficulty_level = determine_difficulty_level(job_role, years_exp)

    mastered: List[MissionAnalysis] = []
    struggled: List[MissionAnalysis] = []
    failed: List[MissionAnalysis] = []
    skipped: List[MissionAnalysis] = []

    module_perf: Dict[str, Dict[str, int]] = {}

    missions = candidate.missions or []
    for m in missions:
        analysis = analyze_mission(m, loader)
        mod_key = f"Module {analysis.module_number}: {analysis.module_title}" if analysis.module_number else "Module: Unassigned"
        if mod_key not in module_perf:
            module_perf[mod_key] = {"MASTERED": 0, "STRUGGLED": 0, "FAILED": 0, "SKIPPED": 0}

        module_perf[mod_key][analysis.status.value] += 1

        if analysis.status == MissionStatus.MASTERED:
            mastered.append(analysis)
        elif analysis.status == MissionStatus.STRUGGLED:
            struggled.append(analysis)
        elif analysis.status == MissionStatus.FAILED:
            failed.append(analysis)
        elif analysis.status == MissionStatus.SKIPPED:
            skipped.append(analysis)

    # Priority topics for interviewer probing
    priority_topics: List[str] = []
    for f in failed:
        priority_topics.append(f"[FAILED - PRIORITY 1] Day {f.day} ({f.title}): Needs direct testing to check if knowledge gaps were remediated.")
    for s in struggled:
        priority_topics.append(f"[STRUGGLED ({s.attempts} attempts) - PRIORITY 2] Day {s.day} ({s.title}): Probe practical implementation details and edge cases.")
    for sk in skipped:
        priority_topics.append(f"[SKIPPED - PRIORITY 3] Day {sk.day} ({sk.title}): Treat knowledge as unverified, test conceptual understanding.")
    for m in mastered[:3]:  # Top mastered for sanity check
        priority_topics.append(f"[MASTERED - DEPTH CHECK] Day {m.day} ({m.title}): Verify genuine high-level mastery and best practices.")

    return InterviewBrief(
        candidate_id=candidate_id,
        candidate_name=candidate_name,
        job_role=job_role,
        years_experience=years_exp,
        education=education,
        status=status_str,
        commit_days=commit_days,
        missions_completed=missions_completed,
        missions_first_try=missions_first_try,
        first_try_rate=first_try_rate,
        difficulty_level=difficulty_level,
        mastered=mastered,
        struggled=struggled,
        failed=failed,
        skipped=skipped,
        module_performance=module_perf,
        priority_topics=priority_topics,
    )
