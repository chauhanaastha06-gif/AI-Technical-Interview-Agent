import pytest
from app.data.loader import data_loader
from app.models.schemas import CandidateData
from app.models.domain import MissionStatus
from app.services.candidate_profile import build_candidate_brief, determine_difficulty_level


def test_cand_003_strong_performer():
    raw_cand = data_loader.get_candidate_by_id("CAND-003")
    assert raw_cand is not None
    cand_data = CandidateData(**raw_cand)
    brief = build_candidate_brief(cand_data, data_loader)

    assert brief.candidate_name == "Emily Chen"
    assert brief.job_role == "AI Engineer"
    assert brief.years_experience == 6.0
    assert len(brief.mastered) == 10
    assert len(brief.struggled) == 0
    assert len(brief.failed) == 0
    assert len(brief.skipped) == 0
    assert brief.first_try_rate == 30 / 31


def test_cand_010_struggling_and_failed():
    raw_cand = data_loader.get_candidate_by_id("CAND-010")
    assert raw_cand is not None
    cand_data = CandidateData(**raw_cand)
    brief = build_candidate_brief(cand_data, data_loader)

    assert brief.candidate_name == "Gerald Combs"
    assert len(brief.failed) == 3  # Day 8, 10, 22
    assert len(brief.skipped) == 2  # Day 27, 28
    assert len(brief.struggled) >= 3  # Day 7 (5 att), 12 (5 att), 16 (4 att), 31 (3 att)

    failed_days = [m.day for m in brief.failed]
    assert 8 in failed_days
    assert 10 in failed_days
    assert 22 in failed_days

    # Priority topics should emphasize failed missions first
    assert any("[FAILED - PRIORITY 1]" in pt for pt in brief.priority_topics)


def test_cand_011_skipped_performer():
    raw_cand = data_loader.get_candidate_by_id("CAND-011")
    assert raw_cand is not None
    cand_data = CandidateData(**raw_cand)
    brief = build_candidate_brief(cand_data, data_loader)

    assert brief.candidate_name == "Mia Alvarez"
    assert len(brief.skipped) == 5  # Day 7, 8, 12, 16, 22
    skipped_days = [m.day for m in brief.skipped]
    assert 7 in skipped_days
    assert 8 in skipped_days
    assert 12 in skipped_days
    assert 16 in skipped_days
    assert 22 in skipped_days


def test_difficulty_calibration():
    # Intern / entry-level
    diff_intern = determine_difficulty_level("Computer Science Intern", 0.0)
    assert "Entry-Level" in diff_intern or "Fundamentals" in diff_intern

    # Senior
    diff_senior = determine_difficulty_level("Senior Data Engineer", 9.0)
    assert "Senior" in diff_senior

    # Distinguished
    diff_dist = determine_difficulty_level("Distinguished Engineer", 28.0)
    assert "Distinguished" in diff_dist or "Architecture" in diff_dist
