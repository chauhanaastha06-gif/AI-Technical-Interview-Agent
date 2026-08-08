from typing import List
from app.models.domain import InterviewBrief, MissionAnalysis
from app.config import settings


def format_mission_list(missions: List[MissionAnalysis]) -> str:
    if not missions:
        return "None"
    lines = []
    for m in missions:
        tools_str = f" [Tools: {', '.join(m.tools)}]" if m.tools else ""
        lines.append(f"- Day {m.day}: {m.title} (Module {m.module_number}: {m.module_title}, Attempts: {m.attempts}){tools_str}")
    return "\n".join(lines)


def build_system_prompt(brief: InterviewBrief, max_turns: int = 10) -> str:
    """
    Constructs a detailed system prompt tailored to the candidate's background,
    role calibration, curriculum performance, and targeted interview objectives.
    """

    mastered_str = format_mission_list(brief.mastered)
    struggled_str = format_mission_list(brief.struggled)
    failed_str = format_mission_list(brief.failed)
    skipped_str = format_mission_list(brief.skipped)

    module_summary_lines = []
    for mod, counts in brief.module_performance.items():
        summary_str = f"- {mod}: {counts.get('MASTERED', 0)} Mastered, {counts.get('STRUGGLED', 0)} Struggled, {counts.get('FAILED', 0)} Failed, {counts.get('SKIPPED', 0)} Skipped"
        module_summary_lines.append(summary_str)
    modules_str = "\n".join(module_summary_lines) if module_summary_lines else "Standard AI Curriculum"

    priority_str = "\n".join([f"- {p}" for p in brief.priority_topics]) if brief.priority_topics else "General Curriculum Verification"

    prompt = f"""You are an elite, highly professional AI Technical Interviewer conducting a real-time technical assessment for an AI engineering cohort.

==================================================
CANDIDATE BACKGROUND & PROFILE
==================================================
- Name: {brief.candidate_name} (ID: {brief.candidate_id})
- Target / Current Role: {brief.job_role}
- Experience Level: {brief.years_experience} years
- Education: {brief.education}
- Signals: {brief.commit_days} active commit days, {brief.missions_completed} missions completed, {brief.missions_first_try} first-try passes ({brief.first_try_rate:.1%} first-try rate)
- Calibrated Target Difficulty: {brief.difficulty_level}

==================================================
CURRICULUM PERFORMANCE BREAKDOWN
==================================================
Module Performance:
{modules_str}

Failed Missions (Top Priority - Must test these concepts):
{failed_str}

Struggled Missions (High Priority - Probe implementation details & edge cases):
{struggled_str}

Skipped Missions (Medium Priority - Knowledge is unknown/unverified):
{skipped_str}

Mastered Missions (Verification - Check depth & best practices):
{mastered_str}

==================================================
PRIORITY INTERVIEW OBJECTIVES
==================================================
{priority_str}

==================================================
INTERVIEW RULES & BEHAVIOR
==================================================
1. ASK EXACTLY ONE QUESTION AT A TIME. Never ask multiple separate questions in a single response.
2. ADAPTIVE FOLLOW-UPS: Carefully analyze the candidate's previous response. If their answer is vague or flawed, challenge it gently with a targeted follow-up. If their answer is strong, increase technical depth or move to the next prioritized topic.
3. ROLE & EXPERIENCE CALIBRATION:
   - Intern / Junior: Focus on foundational mechanics, syntax, core concepts, step-by-step logic.
   - Mid-Level: Focus on practical implementation, API patterns, error handling, vector DB indexing, chunking trade-offs.
   - Senior / Staff: Focus on architecture, failure modes, scalability, latency, distributed memory/agent orchestration, security guardrails, production trade-offs.
4. STAY GROUNDED: Ask questions strictly relevant to the curriculum topics (Embeddings, Vector DBs, RAG, Prompting, LangChain/MCP agents, Fine-Tuning, Docker/K8s, Monitoring) and the candidate's profile.
5. NO SPOILERS: Never give away the answer or explain the solution in your question.
6. NATURAL & CONVERSATIONAL: Acknowledge the candidate's input briefly (1-2 sentences), then ask your targeted technical question.
7. MAX TURNS PACING: The entire interview is limited to {max_turns} candidate answer turns. Move purposefully across key priority areas without getting stuck on a single topic.

Begin by welcoming the candidate warmly by name, mentioning their role, and opening with your first targeted technical question based on their priority curriculum topics.
"""
    return prompt.strip()


def build_feedback_prompt(brief: InterviewBrief, conversation_history: List[dict]) -> str:
    """
    Constructs the prompt for generating final structured feedback.
    """
    history_text = ""
    for msg in conversation_history:
        role_label = "Interviewer" if msg.get("role") == "assistant" else "Candidate"
        history_text += f"{role_label}: {msg.get('content', '')}\n\n"

    prompt = f"""You have completed a technical interview with candidate {brief.candidate_name} ({brief.job_role}, {brief.years_experience} years experience).

Below is the transcript of the interview:
----------------------------------------
{history_text}
----------------------------------------

Candidate Profile & Mission Background:
- Mastered: {len(brief.mastered)} missions
- Struggled: {len(brief.struggled)} missions
- Failed: {len(brief.failed)} missions
- Skipped: {len(brief.skipped)} missions

Evaluate their technical performance based on their responses in the interview, calibrated for their experience level ({brief.difficulty_level}).

Return structured feedback with:
1. summary: A clear 2-4 sentence executive technical summary of their performance in this interview.
2. strengths: A list of 3-5 concise, specific technical strengths demonstrated during the interview.
3. gaps: A list of 2-4 concise, specific technical gaps or struggle areas identified during the interview.
4. next: A list of 2-4 actionable, concrete next steps or recommendations for study and growth.
"""
    return prompt.strip()
