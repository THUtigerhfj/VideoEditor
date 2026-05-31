#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
DIFFUSERS = ROOT / "diffusers" / "src"

sys.path.insert(0, str(APP))
sys.path.insert(0, str(DIFFUSERS))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from sketch_reference_workflow import (  # noqa: E402
    build_reference_preview,
    generate_reference_assets,
    load_reference_assets_for_ui,
)


def build_parser():
    parser = argparse.ArgumentParser(description="Generate reference image and mask from a sketch input.")
    parser.add_argument("--sketch_image", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--attrs", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--candidate_count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache_dir", default=str(ROOT / "ckpt" / "sketch_ref"))
    parser.add_argument("--sam2_ckpt", default=str(ROOT / "ckpt" / "sam2_hiera_large.pt"))
    parser.add_argument("--sam2_cfg", default="sam2_hiera_l.yaml")
    parser.add_argument("--device", default="cuda")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    result = generate_reference_assets(
        sketch_image=args.sketch_image,
        label=args.label,
        attrs=args.attrs,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        sam2_cfg=args.sam2_cfg,
        sam2_ckpt=args.sam2_ckpt,
        device=args.device,
        candidate_count=args.candidate_count,
        seed=args.seed,
    )
    payload = {
        "reference_image": str(result["reference_image_path"]),
        "reference_mask": str(result["reference_mask_path"]),
        "reference_meta": str(result["reference_meta_path"]),
        "best_candidate_index": result["metadata"]["best_candidate_index"],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
