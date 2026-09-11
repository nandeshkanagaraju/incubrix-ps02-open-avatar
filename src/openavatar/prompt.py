"""Deterministic prompt construction from a structured spec.

Contract:
  * Only ``spec.appearance``, ``spec.scene`` and ``spec.extra_prompt`` feed the
    positive prompt.
  * ``spec.region_context`` is *never* read here. ``tests/test_prompt.py`` asserts
    this, which is how we keep nationality labels out of generation.
  * The same spec always yields byte-identical prompt strings.
"""

from __future__ import annotations

from . import attributes as A
from .spec import AvatarSpec

# Negative prompt = the "negative/safety controls" required by the brief.
# Split so the safety layer can log which groups were applied.
NEGATIVE_GROUPS: dict[str, tuple[str, ...]] = {
    "quality": (
        "lowres", "blurry", "jpeg artifacts", "watermark", "signature", "text",
        "deformed", "disfigured", "extra fingers", "extra limbs", "bad anatomy",
        "mutated hands", "cropped head", "out of frame",
    ),
    "safety": (
        "nude", "nsfw", "sexual", "lingerie", "underwear", "gore", "blood",
        "violence", "weapon",
    ),
    "minor_exclusion": (
        "child", "kid", "toddler", "baby", "infant", "teenager", "underage",
    ),
    "identity_exclusion": (
        "celebrity", "famous person", "politician", "public figure", "likeness of a real person",
    ),
}


def build_negative_prompt() -> str:
    terms: list[str] = []
    for group in NEGATIVE_GROUPS.values():
        terms.extend(group)
    return ", ".join(terms)


def _phrase(field: str, value: str) -> str:
    return A.PHRASES.get(field, {}).get(value, "")


def build_prompt(spec: AvatarSpec) -> str:
    """Compose the positive prompt from neutral attributes only."""
    ap, sc = spec.appearance, spec.scene

    subject = f"{_phrase('age_band', ap.age_band)}, {_phrase('presentation', ap.presentation)}"

    skin = A.SKIN_TONE_PHRASES[ap.skin_tone]

    if ap.hair_length == "shaved":
        hair = "shaved head"
    else:
        texture = _phrase("hair_texture", ap.hair_texture)
        length = _phrase("hair_length", ap.hair_length)
        colour = "" if ap.hair_colour == "none" else ap.hair_colour.replace("-", " ")
        hair = " ".join(p for p in (colour, texture, length) if p)

    parts = [
        "studio portrait photograph of",
        subject,
        f"with {skin}",
        hair,
        _phrase("facial_hair", ap.facial_hair),
        _phrase("eyewear", ap.eyewear),
        _phrase("expression", ap.expression),
        _phrase("attire", ap.attire),
        _phrase("pose", sc.pose),
        _phrase("background", sc.background),
        f"{sc.lighting.replace('-', ' ')} lighting",
        "sharp focus, natural skin texture, photorealistic, 50mm lens",
    ]
    if spec.extra_prompt:
        parts.append(spec.extra_prompt.strip())

    return ", ".join(p.strip() for p in parts if p and p.strip())


def attribute_probes(spec: AvatarSpec) -> dict[str, tuple[str, str]]:
    """Per-attribute (positive, contrast) text pairs used by the adherence metric.

    For each controlled attribute we produce the phrase that *was* requested and
    a phrase from the same vocabulary that was *not*, so CLIP is asked a
    discriminative question rather than an absolute one.
    """
    ap, sc = spec.appearance, spec.scene
    probes: dict[str, tuple[str, str]] = {}

    def add(field: str, value: str, vocab: tuple[str, ...], phrase_field: str | None = None):
        pf = phrase_field or field
        alt = next((v for v in vocab if v != value and _phrase(pf, v)), None)
        pos = _phrase(pf, value)
        if pos and alt:
            probes[field] = (f"a portrait of {pos}", f"a portrait of {_phrase(pf, alt)}")

    add("age_band", ap.age_band, A.AGE_BANDS)
    add("hair_length", ap.hair_length, A.HAIR_LENGTHS)
    add("attire", ap.attire, A.ATTIRE)
    add("eyewear", ap.eyewear, A.EYEWEAR) if ap.eyewear != "none" else None
    add("expression", ap.expression, A.EXPRESSIONS)
    add("background", sc.background, A.BACKGROUNDS)
    add("pose", sc.pose, A.POSES)

    # Skin tone uses its own phrase table.
    alt_tone = "mst-9" if ap.skin_tone != "mst-9" else "mst-2"
    probes["skin_tone"] = (
        f"a portrait of a person with {A.SKIN_TONE_PHRASES[ap.skin_tone]}",
        f"a portrait of a person with {A.SKIN_TONE_PHRASES[alt_tone]}",
    )
    return probes
