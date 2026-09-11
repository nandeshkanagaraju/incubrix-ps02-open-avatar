#!/usr/bin/env python
"""Pre-download every pinned model/checkpoint into the local HF cache.

Usage:  python scripts/download_models.py [--config configs/default.json] [--only sd15]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from huggingface_hub import snapshot_download  # noqa: E402

from openavatar.config import Config  # noqa: E402

# Diffusers reads the subfolder layout only. The SD 1.5 repo also ships
# single-file checkpoints (~12 GB) and fp16 duplicates of every shard; pulling
# either wastes bandwidth and disk. Listing exact paths keeps the footprint at
# ~4.3 GB, and torch casts to float16 at load time where the device wants it.
SD_ALLOW = [
    "model_index.json",
    "scheduler/scheduler_config.json",
    "tokenizer/vocab.json", "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json", "tokenizer/tokenizer_config.json",
    "text_encoder/config.json", "text_encoder/model.safetensors",
    "unet/config.json", "unet/diffusion_pytorch_model.safetensors",
    "vae/config.json", "vae/diffusion_pytorch_model.safetensors",
]
SD_IGNORE = ["*.ckpt", "*.bin", "*.pt", "*.msgpack", "*.onnx*", "*.fp16.*",
             "*nonema*", "v1-5-pruned*", "safety_checker/*", "feature_extractor/*"]

# LoRA repos are a single top-level weights file plus a config.
LORA_ALLOW = ["*.json", "*.safetensors"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--only", default=None, help="download a single model key")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    keys = [args.only] if args.only else cfg.model_keys
    seen: set[tuple[str, str]] = set()

    for key in keys:
        m = cfg.model(key)
        targets = [(m["repo_id"], m["revision"], SD_ALLOW, SD_IGNORE)]
        if m.get("lora"):
            targets.append((m["lora"]["repo_id"], m["lora"]["revision"], LORA_ALLOW, []))
        for repo_id, revision, allow, ignore in targets:
            if (repo_id, revision) in seen:
                continue
            seen.add((repo_id, revision))
            print(f"[download] {repo_id}@{revision[:12]}", flush=True)
            snapshot_download(repo_id=repo_id, revision=revision,
                              allow_patterns=allow, ignore_patterns=ignore)

    met = cfg.metrics
    for repo_id, revision in [
        (met["clip_model"], met["clip_revision"]),
        (met["identity_model"], met["identity_revision"]),
    ]:
        print(f"[download] {repo_id}@{revision[:12]}", flush=True)
        snapshot_download(repo_id=repo_id, revision=revision,
                          allow_patterns=["*.json", "*.txt", "*.safetensors"])
    print("[download] complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
