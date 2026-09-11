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

# Only the diffusers subfolder layout. The repo also ships single-file
# checkpoints (v1-5-pruned*.safetensors, ~12 GB) that from_pretrained never
# reads - excluding them keeps the local footprint near 2 GB.
SD_ALLOW = [
    "model_index.json",
    "*/config.json", "*/*.json", "*/*.txt",
    "text_encoder/model.safetensors",
    "unet/diffusion_pytorch_model.safetensors",
    "vae/diffusion_pytorch_model.safetensors",
    "*.safetensors" ,
]
SD_IGNORE = ["*.ckpt", "*.bin", "*.pt", "*nonema*", "v1-5-pruned*", "*.msgpack", "*.onnx*"]


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
        for repo_id, revision in [(m["repo_id"], m["revision"])] + (
            [(m["lora"]["repo_id"], m["lora"]["revision"])] if m.get("lora") else []
        ):
            if (repo_id, revision) in seen:
                continue
            seen.add((repo_id, revision))
            print(f"[download] {repo_id}@{revision[:12]}", flush=True)
            snapshot_download(
                repo_id=repo_id, revision=revision,
                allow_patterns=SD_ALLOW, ignore_patterns=SD_IGNORE,
            )

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
