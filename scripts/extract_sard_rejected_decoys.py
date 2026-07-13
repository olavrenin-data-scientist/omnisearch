"""Re-purpose the manually-reviewed SARD *rejected* and *not_selected* assets as
hard-negative decoy objects for YOLO training.

Background
----------
When the SARD GrabCut assets were reviewed (see
``configs/cv/sard_grabcut_asset_review.json``), 43 images were marked either
``rejected`` (bad mask quality) or ``not_selected`` (acceptable quality but
excluded from the survivor training set for balance/diversity reasons).  These
are ambiguous, person-shaped cutouts at drone altitude — exactly the kind of
hard negatives that force the detector to learn a genuine human silhouette
rather than "any vaguely human-shaped blob on NAIP terrain".

This script copies those 43 PNGs from ``data/cv_assets/sard_grabcut/`` into
``data/cv_assets/sard_decoys/`` (renaming them ``sard_decoy_XXXX.png``).
No re-processing is needed: the files are already GrabCut-masked RGBA PNGs.

Usage
-----
    python scripts/extract_sard_rejected_decoys.py

    # Custom paths:
    python scripts/extract_sard_rejected_decoys.py \\
        --review-json  configs/cv/sard_grabcut_asset_review.json \\
        --assets-dir   data/cv_assets/sard_grabcut \\
        --out-dir      data/cv_assets/sard_decoys
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--review-json",
        default=str(ROOT / "configs/cv/sard_grabcut_asset_review.json"),
        help="Path to the SARD asset review JSON.",
    )
    ap.add_argument(
        "--assets-dir",
        default=str(ROOT / "data/cv_assets/sard_grabcut"),
        help="Directory that holds ALL extracted SARD PNGs (accepted + rejected).",
    )
    ap.add_argument(
        "--out-dir",
        default=str(ROOT / "data/cv_assets/sard_decoys"),
        help="Output directory for decoy PNGs.",
    )
    ap.add_argument(
        "--include-needs-remask",
        action="store_true",
        help="Also include assets marked 'needs_remask' (masks may be imperfect).",
    )
    args = ap.parse_args()

    review_path = Path(args.review_json)
    if not review_path.exists():
        raise SystemExit(f"Review JSON not found: {review_path}")
    review = json.loads(review_path.read_text(encoding="utf-8"))

    assets_dir = Path(args.assets_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect the asset names that were NOT accepted for survivor training.
    decoy_names: list[str] = []
    decoy_names.extend(review.get("rejected_assets", []))
    decoy_names.extend(review.get("not_selected_assets", []))
    if args.include_needs_remask:
        decoy_names.extend(review.get("needs_remask_assets", []))

    copied = 0
    missing = []
    for idx, name in enumerate(sorted(set(decoy_names)), start=1):
        src = assets_dir / name
        if not src.exists():
            missing.append(name)
            continue
        dst = out_dir / f"sard_decoy_{idx:04d}.png"
        shutil.copy2(src, dst)
        copied += 1

    print(f"Copied {copied} decoy assets to {out_dir}")
    if missing:
        print(f"WARNING: {len(missing)} assets not found in {assets_dir}:")
        for m in missing:
            print(f"  {m}")
        print("Run extract_sard_assets.py --sard-root <path> first to populate the assets directory.")

    if copied == 0:
        raise SystemExit(
            "No decoy assets were copied. Check that --assets-dir contains the SARD PNGs."
        )

    print(
        f"\nNext step: pass --decoy-assets-dir {out_dir} to train_survivor_detector.py"
    )


if __name__ == "__main__":
    main()
