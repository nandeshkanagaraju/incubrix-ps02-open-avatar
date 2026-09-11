"""Unit tests: prompt construction, including the no-nationality-leak guarantee."""
from __future__ import annotations

from openavatar.prompt import NEGATIVE_GROUPS, attribute_probes, build_negative_prompt, build_prompt
from openavatar.spec import AvatarSpec


def _spec(**kw):
    d = {"spec_id": "p1"}
    d.update(kw)
    return AvatarSpec.model_validate(d)


def test_prompt_is_deterministic():
    s = _spec()
    assert build_prompt(s) == build_prompt(s)


def test_region_context_never_reaches_the_prompt():
    """The core anti-stereotyping guarantee: a region label changes nothing."""
    plain = _spec()
    labelled = _spec(region_context="south-asia-coverage")
    assert build_prompt(plain) == build_prompt(labelled)
    assert "south" not in build_prompt(labelled).lower()
    assert "asia" not in build_prompt(labelled).lower()


def test_notes_never_reach_the_prompt():
    assert build_prompt(_spec(notes="make him look Nigerian")) == build_prompt(_spec())


def test_attributes_do_reach_the_prompt():
    p = build_prompt(_spec(appearance={"attire": "lab-coat", "eyewear": "thick-frame-glasses"}))
    assert "lab coat" in p
    assert "thick-frame glasses" in p


def test_shaved_head_suppresses_hair_colour_and_texture():
    p = build_prompt(_spec(appearance={"hair_length": "shaved", "hair_texture": "none",
                                       "hair_colour": "none"}))
    assert "shaved head" in p
    assert "none" not in p


def test_extra_prompt_is_appended():
    assert "holding a clipboard" in build_prompt(_spec(extra_prompt="holding a clipboard"))


def test_negative_prompt_covers_every_group():
    neg = build_negative_prompt()
    for group in NEGATIVE_GROUPS.values():
        for term in group:
            assert term in neg


def test_attribute_probes_are_discriminative():
    probes = attribute_probes(_spec(appearance={"attire": "lab-coat"}))
    pos, contrast = probes["attire"]
    assert "lab coat" in pos
    assert pos != contrast
