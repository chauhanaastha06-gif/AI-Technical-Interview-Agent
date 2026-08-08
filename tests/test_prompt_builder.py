import pytest
from app.data.loader import data_loader
from app.models.schemas import CandidateData
from app.services.candidate_profile import build_candidate_brief
from app.services.prompt_builder import build_system_prompt, build_feedback_prompt


def test_prompt_builder_contains_candidate_details():
    raw_cand = data_loader.get_candidate_by_id("CAND-010")
    cand_data = CandidateData(**raw_cand)
    brief = build_candidate_brief(cand_data, data_loader)

    prompt = build_system_prompt(brief, max_turns=8)

    # Verify key context is in the prompt
    assert "Gerald Combs" in prompt
    assert "IT Support Specialist" in prompt
    assert "20" in prompt
    assert "Failed Missions" in prompt
    assert "Day 8" in prompt
    assert "Day 10" in prompt
    assert "Day 22" in prompt
    assert "EXACTLY ONE QUESTION AT A TIME" in prompt
    assert "8 candidate answer turns" in prompt


def test_feedback_prompt_structure():
    raw_cand = data_loader.get_candidate_by_id("CAND-003")
    cand_data = CandidateData(**raw_cand)
    brief = build_candidate_brief(cand_data, data_loader)

    history = [
        {"role": "assistant", "content": "Welcome Emily! Let's talk about embeddings."},
        {"role": "user", "content": "Embeddings map high-dimensional text to dense vectors."},
    ]

    prompt = build_feedback_prompt(brief, history)
    assert "Emily Chen" in prompt
    assert "AI Engineer" in prompt
    assert "Embeddings map high-dimensional text" in prompt
    assert "strengths" in prompt
    assert "gaps" in prompt
    assert "next" in prompt
