"""Structured avatar specification (the CLI input contract).

A spec is a JSON document. ``AvatarSpec`` is the single source of truth for what
a job may contain; every unknown field is rejected so typos fail loudly instead
of being silently dropped into a prompt.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from . import attributes as A


class Appearance(BaseModel):
    """Neutral appearance attributes. No nationality/ethnicity labels live here."""

    model_config = ConfigDict(extra="forbid")

    age_band: Literal[A.AGE_BANDS] = "adult"  # type: ignore[valid-type]
    presentation: Literal[A.PRESENTATIONS] = "androgynous"  # type: ignore[valid-type]
    skin_tone: str = "mst-5"
    hair_length: Literal[A.HAIR_LENGTHS] = "short"  # type: ignore[valid-type]
    hair_texture: Literal[A.HAIR_TEXTURES] = "straight"  # type: ignore[valid-type]
    hair_colour: Literal[A.HAIR_COLOURS] = "black"  # type: ignore[valid-type]
    facial_hair: Literal[A.FACIAL_HAIR] = "none"  # type: ignore[valid-type]
    eyewear: Literal[A.EYEWEAR] = "none"  # type: ignore[valid-type]
    expression: Literal[A.EXPRESSIONS] = "neutral"  # type: ignore[valid-type]
    attire: Literal[A.ATTIRE] = "plain-crewneck"  # type: ignore[valid-type]

    @field_validator("skin_tone")
    @classmethod
    def _known_tone(cls, v: str) -> str:
        if v not in A.SKIN_TONES:
            raise ValueError(f"skin_tone must be one of {A.SKIN_TONES}, got {v!r}")
        return v

    @model_validator(mode="after")
    def _hair_coherence(self) -> "Appearance":
        if self.hair_length == "shaved" and self.hair_texture != "none":
            raise ValueError("hair_texture must be 'none' when hair_length is 'shaved'")
        return self


class Scene(BaseModel):
    model_config = ConfigDict(extra="forbid")

    background: Literal[A.BACKGROUNDS] = "neutral-grey-studio"  # type: ignore[valid-type]
    pose: Literal[A.POSES] = "front-facing-headshot"  # type: ignore[valid-type]
    lighting: Literal["soft-key", "even-studio", "window-light"] = "even-studio"


class Render(BaseModel):
    model_config = ConfigDict(extra="forbid")

    width: int = Field(default=384, ge=128, le=1024)
    height: int = Field(default=384, ge=128, le=1024)
    seed: int = Field(default=0, ge=0, le=2**31 - 1)
    steps: int = Field(default=20, ge=1, le=100)
    guidance_scale: float = Field(default=7.0, ge=0.0, le=20.0)
    model_key: str = "sd15"

    @model_validator(mode="after")
    def _multiple_of_eight(self) -> "Render":
        if self.width % 8 or self.height % 8:
            raise ValueError("width and height must be multiples of 8 for latent diffusion")
        return self

    @property
    def aspect_ratio(self) -> str:
        from math import gcd

        g = gcd(self.width, self.height)
        return f"{self.width // g}:{self.height // g}"


class ReferenceSet(BaseModel):
    """Consented reference images for individual-avatar work (Exceptional level)."""

    model_config = ConfigDict(extra="forbid")

    consent_id: Optional[str] = None
    image_paths: list[str] = Field(default_factory=list)


class AvatarSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spec_id: str
    identity_id: Optional[str] = Field(
        default=None,
        description="Groups renders that should depict the same fictional person. "
        "Used by the identity-consistency metric.",
    )
    intent: Literal["fictional", "individual"] = "fictional"

    appearance: Appearance = Field(default_factory=Appearance)
    scene: Scene = Field(default_factory=Scene)
    render: Render = Field(default_factory=Render)
    reference: ReferenceSet = Field(default_factory=ReferenceSet)

    # Bookkeeping only. NEVER used to build a prompt - see prompt.build_prompt.
    region_context: Optional[str] = Field(
        default=None,
        description="Free-text geographic/cultural coverage label for the test matrix. "
        "Excluded from prompts by design.",
    )
    notes: Optional[str] = None
    extra_prompt: Optional[str] = Field(
        default=None, description="Free text appended to the prompt; safety-screened."
    )

    @field_validator("spec_id")
    @classmethod
    def _slug(cls, v: str) -> str:
        if not v or any(c in v for c in "/\\ \t\n"):
            raise ValueError("spec_id must be a non-empty slug without whitespace or path separators")
        return v

    @model_validator(mode="after")
    def _individual_requires_reference(self) -> "AvatarSpec":
        if self.intent == "individual" and not self.reference.image_paths:
            raise ValueError("intent='individual' requires reference.image_paths")
        return self

    @classmethod
    def from_file(cls, path: str | Path) -> "AvatarSpec":
        data = json.loads(Path(path).read_text())
        return cls.model_validate(data)


def load_specs(spec_dir: str | Path) -> list[AvatarSpec]:
    """Load every ``*.json`` spec in a directory, sorted by filename."""
    spec_dir = Path(spec_dir)
    if not spec_dir.is_dir():
        raise NotADirectoryError(f"spec directory not found: {spec_dir}")
    files = sorted(spec_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no *.json specs found in {spec_dir}")
    specs: list[AvatarSpec] = []
    seen: set[str] = set()
    for f in files:
        spec = AvatarSpec.from_file(f)
        if spec.spec_id in seen:
            raise ValueError(f"duplicate spec_id {spec.spec_id!r} in {f}")
        seen.add(spec.spec_id)
        specs.append(spec)
    return specs
