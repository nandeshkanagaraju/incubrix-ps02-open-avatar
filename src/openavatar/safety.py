"""Safety screening, ambiguity detection and consent enforcement.

Every spec passes through :func:`screen` before a job is created. The outcome is
one of three decisions, all of which are written to the safety log:

  ALLOW               - job may be prepared
  NEEDS_CLARIFICATION - spec is ambiguous/underspecified; job is held, not run
  REFUSE              - disallowed request; job is never prepared

Refusal reason codes are stable strings so tests and the evidence index can
assert on them.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from .spec import AvatarSpec

Decision = Literal["ALLOW", "NEEDS_CLARIFICATION", "REFUSE"]

# --- disallowed-content patterns -------------------------------------------------

MINOR_TERMS = re.compile(
    r"\b(child|children|kid|kids|toddler|baby|infant|minor|underage|teen|teenage[r]?|"
    r"schoolgirl|schoolboy|\d{1,2}\s*(?:year|yr)s?\s*old)\b",
    re.I,
)
SEXUAL_TERMS = re.compile(
    r"\b(nude|naked|nsfw|sexual|sexy|erotic|porn|lingerie|topless|undress)\b", re.I
)
VIOLENCE_TERMS = re.compile(r"\b(gore|mutilat\w*|beheaded|corpse|blood[y]?\s*wound)\b", re.I)
IMPERSONATION_TERMS = re.compile(
    r"\b(deepfake|face\s*swap|impersonat\w*|lookalike\s+of|likeness\s+of\s+(?!a\s+fictional))\b",
    re.I,
)
# Real-person targeting: "photo of <Capitalised Name> <Capitalised Name>" style,
# plus explicit public-figure framings.
REAL_PERSON_TERMS = re.compile(
    r"\b(celebrity|celebrities|famous\s+(?:actor|actress|singer|person)|"
    r"president|prime\s+minister|politician|public\s+figure)\b",
    re.I,
)
# Ethnic/biometric essentialism we refuse to encode.
STEREOTYPE_TERMS = re.compile(
    r"\b(typical|stereotypical|authentic|pure[- ]blooded|real)\s+"
    r"(indian|chinese|african|arab|asian|european|american|nigerian|japanese|mexican|"
    r"race|racial|ethnic)\w*\b",
    re.I,
)
BIOMETRIC_CLAIM_TERMS = re.compile(
    r"\b(determine|detect|predict|identify)\s+(the\s+)?(race|ethnicity|nationality|religion)\b",
    re.I,
)

REFUSAL_RULES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("MINOR_DEPICTION", MINOR_TERMS, "Request appears to depict a minor."),
    ("SEXUAL_CONTENT", SEXUAL_TERMS, "Request asks for sexual or nude content."),
    ("GRAPHIC_VIOLENCE", VIOLENCE_TERMS, "Request asks for graphic violence or gore."),
    ("IMPERSONATION", IMPERSONATION_TERMS, "Request asks for a deepfake or likeness of a real person."),
    ("REAL_PERSON_TARGETING", REAL_PERSON_TERMS, "Request targets a real or public figure."),
    ("ETHNIC_STEREOTYPE", STEREOTYPE_TERMS, "Request asks for an essentialised ethnic/national stereotype."),
    ("BIOMETRIC_CLAIM", BIOMETRIC_CLAIM_TERMS, "Request asks the system to infer protected attributes."),
)

AMBIGUITY_TERMS = re.compile(
    r"\b(nice|good|better|normal|standard|typical|appropriate|professional[- ]looking|"
    r"attractive|beautiful|handsome|any|whatever|something)\b",
    re.I,
)


@dataclass
class Finding:
    code: str
    message: str
    field: str


@dataclass
class SafetyResult:
    spec_id: str
    decision: Decision
    findings: list[Finding] = field(default_factory=list)
    negative_prompt_groups: list[str] = field(default_factory=list)
    consent_id: str | None = None
    checked_at: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["findings"] = [asdict(f) for f in self.findings]
        return d

    @property
    def allowed(self) -> bool:
        return self.decision == "ALLOW"


# --- consent ---------------------------------------------------------------------


class ConsentError(RuntimeError):
    pass


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ConsentRegistry:
    """File-backed record of consent for identifiable-person reference images.

    A record stores only the *hash* of each reference image, never the image and
    never a name, so the registry itself holds no biometric data. Revoking a
    record is a one-line state change and :meth:`purge` deletes the referenced
    files from disk (privacy/deletion control).
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._data: dict = {"records": {}}
        if self.path.exists():
            self._data = json.loads(self.path.read_text())

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2, sort_keys=True))

    def add(self, consent_id: str, subject_ref: str, image_paths: list[str], scope: str) -> dict:
        hashes = {}
        for p in image_paths:
            if not Path(p).exists():
                raise ConsentError(f"reference image not found: {p}")
            hashes[str(Path(p).name)] = sha256_file(p)
        rec = {
            "consent_id": consent_id,
            "subject_ref": subject_ref,  # pseudonymous label, not a legal name
            "scope": scope,
            "image_sha256": hashes,
            "granted_at": datetime.now(timezone.utc).isoformat(),
            "revoked": False,
            "revoked_at": None,
        }
        self._data["records"][consent_id] = rec
        self._save()
        return rec

    def get(self, consent_id: str) -> dict | None:
        return self._data["records"].get(consent_id)

    def revoke(self, consent_id: str) -> dict:
        rec = self._data["records"].get(consent_id)
        if rec is None:
            raise ConsentError(f"unknown consent_id: {consent_id}")
        rec["revoked"] = True
        rec["revoked_at"] = datetime.now(timezone.utc).isoformat()
        self._save()
        return rec

    def purge(self, consent_id: str, reference_dir: str | Path) -> list[str]:
        """Delete reference images belonging to a record and drop the record."""
        rec = self._data["records"].get(consent_id)
        if rec is None:
            raise ConsentError(f"unknown consent_id: {consent_id}")
        deleted = []
        for name in rec["image_sha256"]:
            p = Path(reference_dir) / name
            if p.exists():
                p.unlink()
                deleted.append(str(p))
        del self._data["records"][consent_id]
        self._save()
        return deleted

    def verify(self, consent_id: str, image_paths: list[str]) -> None:
        """Raise unless the record exists, is live, and the files still match."""
        rec = self.get(consent_id)
        if rec is None:
            raise ConsentError(f"no consent record for consent_id={consent_id!r}")
        if rec["revoked"]:
            raise ConsentError(f"consent {consent_id!r} was revoked at {rec['revoked_at']}")
        for p in image_paths:
            name = Path(p).name
            expected = rec["image_sha256"].get(name)
            if expected is None:
                raise ConsentError(f"{name} is not covered by consent record {consent_id!r}")
            if sha256_file(p) != expected:
                raise ConsentError(f"{name} does not match the hash recorded under {consent_id!r}")


# --- screening -------------------------------------------------------------------


def _free_text(spec: AvatarSpec) -> str:
    return " ".join(filter(None, [spec.extra_prompt or "", spec.notes or "", spec.region_context or ""]))


def screen(spec: AvatarSpec, registry: ConsentRegistry | None = None) -> SafetyResult:
    """Screen a spec. Pure function apart from consent-registry reads."""
    from .prompt import NEGATIVE_GROUPS

    res = SafetyResult(
        spec_id=spec.spec_id,
        decision="ALLOW",
        negative_prompt_groups=sorted(NEGATIVE_GROUPS),
        consent_id=spec.reference.consent_id,
        checked_at=datetime.now(timezone.utc).isoformat(),
    )

    text = _free_text(spec)
    for code, pattern, message in REFUSAL_RULES:
        m = pattern.search(text)
        if m:
            res.decision = "REFUSE"
            res.findings.append(Finding(code, f"{message} (matched {m.group(0)!r})", "extra_prompt/notes"))

    # Structural refusal: age band vocabulary starts at young-adult, but guard
    # against a spec that tries to smuggle a minor in via the free-text field.
    if spec.intent == "individual":
        if not spec.reference.consent_id:
            res.decision = "REFUSE"
            res.findings.append(
                Finding("CONSENT_MISSING", "intent='individual' requires reference.consent_id", "reference.consent_id")
            )
        elif registry is None:
            res.decision = "REFUSE"
            res.findings.append(
                Finding("CONSENT_UNVERIFIABLE", "no consent registry available to verify the record", "reference.consent_id")
            )
        else:
            try:
                registry.verify(spec.reference.consent_id, spec.reference.image_paths)
            except ConsentError as exc:
                res.decision = "REFUSE"
                res.findings.append(Finding("CONSENT_INVALID", str(exc), "reference.consent_id"))

    if res.decision == "REFUSE":
        return res

    # Ambiguity is only reported when nothing above refused.
    amb = AMBIGUITY_TERMS.search(spec.extra_prompt or "")
    if amb:
        res.decision = "NEEDS_CLARIFICATION"
        res.findings.append(
            Finding(
                "AMBIGUOUS_TERM",
                f"subjective/underspecified term {amb.group(0)!r}; replace it with a controlled attribute value",
                "extra_prompt",
            )
        )
    if spec.intent == "fictional" and spec.reference.image_paths:
        res.decision = "NEEDS_CLARIFICATION"
        res.findings.append(
            Finding(
                "INTENT_MISMATCH",
                "reference images supplied but intent='fictional'; set intent='individual' with consent, or drop the references",
                "reference",
            )
        )
    return res


class SafetyLog:
    """Append-only JSONL safety log."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, result: SafetyResult) -> None:
        with open(self.path, "a") as fh:
            fh.write(json.dumps(result.to_dict(), sort_keys=True) + "\n")

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]
