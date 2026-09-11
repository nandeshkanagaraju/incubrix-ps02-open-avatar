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

    For each requested attribute we produce a sentence describing what *was*
    asked for and a sentence describing a value from the same vocabulary that was
    *not*, so CLIP is asked a discriminative question rather than an absolute one.

    Every attribute that reaches the prompt is probed. An earlier version covered
    only 7 of the 11 appearance/scene attributes - it silently omitted
    presentation, hair texture, hair colour and facial hair, which inflated the
    reported adherence because unmeasured attributes cannot miss.
    """
    ap, sc = spec.appearance, spec.scene
    probes: dict[str, tuple[str, str]] = {}

    def add(field: str, positive: str, contrast: str) -> None:
        if positive and contrast and positive != contrast:
            probes[field] = (positive, contrast)

    def other(value: str, vocab: tuple[str, ...]) -> str:
        return next((v for v in vocab if v != value), value)

    # --- subject ---------------------------------------------------------
    add("age_band",
        f"a portrait of {_phrase('age_band', ap.age_band)}",
        f"a portrait of {_phrase('age_band', other(ap.age_band, A.AGE_BANDS))}")

    add("presentation",
        f"a portrait of a {_phrase('presentation', ap.presentation)} person",
        f"a portrait of a {_phrase('presentation', other(ap.presentation, A.PRESENTATIONS))} person")

    alt_tone = "mst-9" if ap.skin_tone != "mst-9" else "mst-2"
    add("skin_tone",
        f"a portrait of a person with {A.SKIN_TONE_PHRASES[ap.skin_tone]}",
        f"a portrait of a person with {A.SKIN_TONE_PHRASES[alt_tone]}")

    # --- hair ------------------------------------------------------------
    add("hair_length",
        f"a portrait of a person with {_phrase('hair_length', ap.hair_length)}",
        f"a portrait of a person with {_phrase('hair_length', other(ap.hair_length, A.HAIR_LENGTHS))}")

    if ap.hair_texture != "none":
        alt_tex = other(ap.hair_texture, tuple(t for t in A.HAIR_TEXTURES if t != "none"))
        add("hair_texture",
            f"a portrait of a person with {_phrase('hair_texture', ap.hair_texture)} hair",
            f"a portrait of a person with {_phrase('hair_texture', alt_tex)} hair")

    if ap.hair_colour != "none":
        alt_col = other(ap.hair_colour, tuple(c for c in A.HAIR_COLOURS if c != "none"))
        add("hair_colour",
            f"a portrait of a person with {ap.hair_colour.replace('-', ' ')} hair",
            f"a portrait of a person with {alt_col.replace('-', ' ')} hair")

    if ap.facial_hair != "none":
        alt_fh = other(ap.facial_hair, tuple(f for f in A.FACIAL_HAIR if f != "none"))
        add("facial_hair",
            f"a portrait of a person with {_phrase('facial_hair', ap.facial_hair)}",
            f"a portrait of a person with {_phrase('facial_hair', alt_fh)}")
    else:
        add("facial_hair",
            "a portrait of a clean-shaven person",
            "a portrait of a person with a full beard")

    # --- worn / expression ----------------------------------------------
    if ap.eyewear != "none":
        alt_ey = other(ap.eyewear, tuple(e for e in A.EYEWEAR if e != "none"))
        add("eyewear",
            f"a portrait of a person {_phrase('eyewear', ap.eyewear)}",
            f"a portrait of a person {_phrase('eyewear', alt_ey)}")
    else:
        add("eyewear",
            "a portrait of a person without glasses",
            "a portrait of a person wearing glasses")

    add("expression",
        f"a portrait of a person with {_phrase('expression', ap.expression)}",
        f"a portrait of a person with {_phrase('expression', other(ap.expression, A.EXPRESSIONS))}")

    add("attire",
        f"a portrait of a person {_phrase('attire', ap.attire)}",
        f"a portrait of a person {_phrase('attire', other(ap.attire, A.ATTIRE))}")

    # --- scene -----------------------------------------------------------
    add("background",
        f"a portrait with a {_phrase('background', sc.background)}",
        f"a portrait with a {_phrase('background', other(sc.background, A.BACKGROUNDS))}")

    add("pose",
        f"a {_phrase('pose', sc.pose)}",
        f"a {_phrase('pose', other(sc.pose, A.POSES))}")

    return probes
