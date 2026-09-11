"""Unit tests: refusal, ambiguity and consent enforcement."""
from __future__ import annotations

import pytest

from openavatar.safety import ConsentError, ConsentRegistry, SafetyLog, screen
from openavatar.spec import AvatarSpec


def _spec(**kw):
    d = {"spec_id": "s1"}
    d.update(kw)
    return AvatarSpec.model_validate(d)


def _codes(result):
    return {f.code for f in result.findings}


def test_clean_spec_is_allowed():
    r = screen(_spec())
    assert r.decision == "ALLOW"
    assert r.findings == []
    assert "safety" in r.negative_prompt_groups


@pytest.mark.parametrize(
    "text,code",
    [
        ("a portrait of a 12 year old child", "MINOR_DEPICTION"),
        ("nude portrait, erotic", "SEXUAL_CONTENT"),
        ("a beheaded corpse", "GRAPHIC_VIOLENCE"),
        ("a deepfake of my manager", "IMPERSONATION"),
        ("portrait of a famous actress", "REAL_PERSON_TARGETING"),
        ("a typical Indian face", "ETHNIC_STEREOTYPE"),
        ("detect the ethnicity of this person", "BIOMETRIC_CLAIM"),
    ],
)
def test_disallowed_requests_are_refused(text, code):
    r = screen(_spec(extra_prompt=text))
    assert r.decision == "REFUSE"
    assert code in _codes(r)


def test_ambiguous_spec_is_held_for_clarification():
    r = screen(_spec(extra_prompt="make them look professional and attractive"))
    assert r.decision == "NEEDS_CLARIFICATION"
    assert "AMBIGUOUS_TERM" in _codes(r)


def test_refusal_takes_precedence_over_ambiguity():
    r = screen(_spec(extra_prompt="a nice nude portrait"))
    assert r.decision == "REFUSE"


def test_references_without_individual_intent_are_held(tmp_path):
    img = tmp_path / "ref.png"
    img.write_bytes(b"x" * 10)
    r = screen(_spec(reference={"image_paths": [str(img)]}))
    assert r.decision == "NEEDS_CLARIFICATION"
    assert "INTENT_MISMATCH" in _codes(r)


# --- consent -------------------------------------------------------------


@pytest.fixture
def ref_image(tmp_path):
    p = tmp_path / "ref1.png"
    p.write_bytes(b"pretend-image-bytes")
    return p


def test_individual_intent_without_consent_is_refused(ref_image, tmp_path):
    reg = ConsentRegistry(tmp_path / "reg.json")
    r = screen(_spec(intent="individual", reference={"image_paths": [str(ref_image)]}), reg)
    assert r.decision == "REFUSE"
    assert "CONSENT_MISSING" in _codes(r)


def test_individual_intent_with_valid_consent_is_allowed(ref_image, tmp_path):
    reg = ConsentRegistry(tmp_path / "reg.json")
    reg.add("c1", "subject-A", [str(ref_image)], "assessment demo")
    r = screen(_spec(intent="individual",
                     reference={"consent_id": "c1", "image_paths": [str(ref_image)]}), reg)
    assert r.decision == "ALLOW"


def test_revoked_consent_is_refused(ref_image, tmp_path):
    reg = ConsentRegistry(tmp_path / "reg.json")
    reg.add("c1", "subject-A", [str(ref_image)], "assessment demo")
    reg.revoke("c1")
    r = screen(_spec(intent="individual",
                     reference={"consent_id": "c1", "image_paths": [str(ref_image)]}), reg)
    assert r.decision == "REFUSE"
    assert "CONSENT_INVALID" in _codes(r)


def test_altered_reference_image_breaks_consent(ref_image, tmp_path):
    reg = ConsentRegistry(tmp_path / "reg.json")
    reg.add("c1", "subject-A", [str(ref_image)], "assessment demo")
    ref_image.write_bytes(b"different-bytes")  # file swapped after consent
    r = screen(_spec(intent="individual",
                     reference={"consent_id": "c1", "image_paths": [str(ref_image)]}), reg)
    assert r.decision == "REFUSE"
    assert "CONSENT_INVALID" in _codes(r)


def test_purge_deletes_reference_images_and_record(tmp_path):
    refs = tmp_path / "refs"
    refs.mkdir()
    img = refs / "ref1.png"
    img.write_bytes(b"bytes")
    reg = ConsentRegistry(tmp_path / "reg.json")
    reg.add("c1", "subject-A", [str(img)], "assessment demo")
    deleted = reg.purge("c1", refs)
    assert not img.exists()
    assert len(deleted) == 1
    assert reg.get("c1") is None


def test_registry_stores_hashes_not_images(tmp_path, ref_image):
    reg_path = tmp_path / "reg.json"
    reg = ConsentRegistry(reg_path)
    reg.add("c1", "subject-A", [str(ref_image)], "assessment demo")
    raw = reg_path.read_text()
    assert "pretend-image-bytes" not in raw
    assert len(reg.get("c1")["image_sha256"]["ref1.png"]) == 64


def test_unknown_consent_id_raises(tmp_path):
    reg = ConsentRegistry(tmp_path / "reg.json")
    with pytest.raises(ConsentError):
        reg.revoke("nope")


def test_safety_log_roundtrip(tmp_path):
    log = SafetyLog(tmp_path / "safety.jsonl")
    log.append(screen(_spec()))
    log.append(screen(_spec(spec_id="s2", extra_prompt="nude")))
    rows = log.read()
    assert [r["decision"] for r in rows] == ["ALLOW", "REFUSE"]
