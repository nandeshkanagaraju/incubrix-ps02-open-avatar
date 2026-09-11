"""Quality metrics.

Three families, each deliberately using a model that is *independent of the
generator* so the score is not self-graded:

  prompt/spec adherence  - CLIP ViT-B/32 (OpenAI, MIT-licensed code, open weights).
                           Reported two ways: a raw CLIPScore against the full
                           prompt, and a per-attribute discriminative probe that
                           asks CLIP to choose between the requested attribute
                           value and a contrasting one from the same vocabulary.
  identity consistency   - DINOv2-small (Apache-2.0) embeddings. DINOv2 is a
                           self-supervised vision backbone, not a CLIP variant and
                           not part of the generation stack, so it gives an
                           independent view of "is this the same person".
  diversity / coverage   - mean pairwise DINOv2 distance across a batch, plus
                           literal coverage counts over the requested attribute
                           matrix.

Every model is loaded at a pinned revision from the config.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

from .config import Config
from .jobs import Bundle
from .logging_utils import get_logger
from .prompt import attribute_probes
from .spec import AvatarSpec

log = get_logger(__name__)


@dataclass
class Embedders:
    clip_model: object
    clip_processor: object
    id_model: object
    id_processor: object

    @classmethod
    def load(cls, cfg: Config) -> "Embedders":
        from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

        m = cfg.metrics
        log.info("loading CLIP %s@%s", m["clip_model"], m["clip_revision"][:12])
        clip = CLIPModel.from_pretrained(m["clip_model"], revision=m["clip_revision"]).eval()
        clip_proc = CLIPProcessor.from_pretrained(m["clip_model"], revision=m["clip_revision"])
        log.info("loading identity backbone %s@%s", m["identity_model"], m["identity_revision"][:12])
        idm = AutoModel.from_pretrained(m["identity_model"], revision=m["identity_revision"]).eval()
        idp = AutoImageProcessor.from_pretrained(m["identity_model"], revision=m["identity_revision"])
        return cls(clip, clip_proc, idm, idp)

    # -- embeddings -------------------------------------------------------
    def clip_image(self, paths: list[Path]):
        import torch
        from PIL import Image

        imgs = [Image.open(p).convert("RGB") for p in paths]
        inputs = self.clip_processor(images=imgs, return_tensors="pt")
        with torch.no_grad():
            f = self.clip_model.get_image_features(**inputs)
        return f / f.norm(dim=-1, keepdim=True)

    def clip_text(self, texts: list[str]):
        import torch

        inputs = self.clip_processor(
            text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77
        )
        with torch.no_grad():
            f = self.clip_model.get_text_features(**inputs)
        return f / f.norm(dim=-1, keepdim=True)

    def identity(self, paths: list[Path]):
        import torch
        from PIL import Image

        imgs = [Image.open(p).convert("RGB") for p in paths]
        inputs = self.id_processor(images=imgs, return_tensors="pt")
        with torch.no_grad():
            out = self.id_model(**inputs)
        f = out.last_hidden_state[:, 0]  # CLS token
        return f / f.norm(dim=-1, keepdim=True)


def clip_score(emb: Embedders, image_path: Path, prompt: str) -> float:
    """CLIPScore = max(0, 100 * cos(image, text)). Higher is better."""
    img = emb.clip_image([image_path])
    txt = emb.clip_text([prompt[:300]])
    return round(max(0.0, float((img @ txt.T).item()) * 100), 3)


def attribute_adherence(emb: Embedders, image_path: Path, spec: AvatarSpec) -> dict:
    """Per-attribute discriminative probe.

    For each controlled attribute we compare cos(image, requested-phrase) with
    cos(image, contrast-phrase). ``hit`` is True when the requested phrase wins;
    ``margin`` is the gap. Batch-level ``attribute_accuracy`` is the hit rate.
    """
    probes = attribute_probes(spec)
    if not probes:
        return {"per_attribute": {}, "attribute_accuracy": None}
    fields = list(probes)
    texts = [t for f in fields for t in probes[f]]
    img = emb.clip_image([image_path])
    txt = emb.clip_text(texts)
    sims = (img @ txt.T).squeeze(0).tolist()

    per: dict[str, dict] = {}
    for i, f in enumerate(fields):
        pos, neg = sims[2 * i], sims[2 * i + 1]
        per[f] = {"positive": round(pos, 4), "contrast": round(neg, 4),
                  "margin": round(pos - neg, 4), "hit": bool(pos > neg)}
    hits = sum(1 for v in per.values() if v["hit"])
    return {"per_attribute": per, "attribute_accuracy": round(hits / len(per), 4)}


def _pairwise_cosines(vecs) -> list[float]:
    return [float(vecs[i] @ vecs[j]) for i, j in combinations(range(len(vecs)), 2)]


def evaluate_bundle(bundle_dir: str | Path, cfg: Config) -> dict:
    """Compute every metric for a validated bundle and write ``metrics.json``."""
    bundle_dir = Path(bundle_dir)
    bundle = Bundle.load(bundle_dir)
    validation = json.loads((bundle_dir / "validation.json").read_text())
    ok_jobs = {i["job_id"] for i in validation["items"] if i["ok"]}
    images_dir = bundle_dir / "images"

    emb = Embedders.load(cfg)
    per_image: dict[str, dict] = {}
    id_groups: dict[str, list[Path]] = {}
    model_groups: dict[str, list[str]] = {}
    all_paths: list[Path] = []
    coverage = {"region_context": {}, "skin_tone": {}, "age_band": {}, "attire": {},
                "background": {}, "hair_length": {}, "presentation": {}}

    for job in bundle.jobs:
        if job.job_id not in ok_jobs:
            continue
        path = images_dir / job.output_name
        spec = AvatarSpec.model_validate(json.loads((bundle_dir / "specs" / f"{job.spec_id}.json").read_text()))

        rec = {
            "spec_id": job.spec_id,
            "model_key": job.model_key,
            "clip_score": clip_score(emb, path, job.prompt),
        }
        rec.update(attribute_adherence(emb, path, spec))
        per_image[job.job_id] = rec

        model_groups.setdefault(job.model_key, []).append(job.job_id)
        if job.identity_id:
            id_groups.setdefault(job.identity_id, []).append(path)
        all_paths.append(path)

        for field_name, value in (
            ("region_context", spec.region_context or "unspecified"),
            ("skin_tone", spec.appearance.skin_tone),
            ("age_band", spec.appearance.age_band),
            ("attire", spec.appearance.attire),
            ("background", spec.scene.background),
            ("hair_length", spec.appearance.hair_length),
            ("presentation", spec.appearance.presentation),
        ):
            coverage[field_name][value] = coverage[field_name].get(value, 0) + 1

    # identity consistency: mean pairwise cosine within each identity group
    identity: dict[str, dict] = {}
    for ident, paths in id_groups.items():
        if len(paths) < 2:
            identity[ident] = {"n": len(paths), "mean_cosine": None,
                               "note": "needs >=2 renders to score"}
            continue
        vecs = emb.identity(paths)
        cos = _pairwise_cosines(vecs)
        identity[ident] = {
            "n": len(paths),
            "mean_cosine": round(sum(cos) / len(cos), 4),
            "min_cosine": round(min(cos), 4),
            "pairs": len(cos),
        }

    # diversity: mean pairwise *distance* across distinct specs (lower cosine = more diverse)
    diversity = {"n_images": len(all_paths)}
    if len(all_paths) >= 2:
        vecs = emb.identity(all_paths)
        cos = _pairwise_cosines(vecs)
        diversity.update({
            "mean_pairwise_cosine": round(sum(cos) / len(cos), 4),
            "mean_pairwise_distance": round(1 - sum(cos) / len(cos), 4),
            "max_pairwise_cosine": round(max(cos), 4),
        })

    by_model = {}
    for key, jids in model_groups.items():
        scores = [per_image[j]["clip_score"] for j in jids]
        accs = [per_image[j]["attribute_accuracy"] for j in jids if per_image[j]["attribute_accuracy"] is not None]
        by_model[key] = {
            "n": len(jids),
            "mean_clip_score": round(sum(scores) / len(scores), 3) if scores else None,
            "mean_attribute_accuracy": round(sum(accs) / len(accs), 4) if accs else None,
        }

    out = {
        "batch_id": bundle.batch_id,
        "metric_models": cfg.metrics,
        "n_scored": len(per_image),
        "job_success_rate": validation["success_rate"],
        "by_model": by_model,
        "identity_consistency": identity,
        "diversity": diversity,
        "coverage": coverage,
        "per_image": per_image,
    }
    (bundle_dir / "metrics.json").write_text(json.dumps(out, indent=2, sort_keys=True))
    return out
