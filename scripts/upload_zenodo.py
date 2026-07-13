"""Package the OmniSearch CV datasets and upload them to Zenodo via the REST API.

This does the whole deposition programmatically:
  1. (optional) package each dataset directory into its own .zip
  2. create a new Zenodo deposition (draft)
  3. upload every file through the deposition's file bucket
  4. write the metadata (title, creators, description, license, keywords, ...)
  5. (optional) publish to mint a DOI

Authentication
--------------
Create a personal access token at:
  https://zenodo.org/account/settings/applications/tokens/new/   (production)
  https://sandbox.zenodo.org/account/settings/applications/tokens/new/  (sandbox)
with the scopes `deposit:write` and `deposit:actions`, then either:
  export ZENODO_TOKEN=...            (production)
  export ZENODO_SANDBOX_TOKEN=...    (sandbox)
or pass --token.

Usage
-----
  # dry run: just build the zips, no network
  python3 scripts/upload_zenodo.py --package --no-upload

  # upload drafts to the SANDBOX first (recommended before the real thing)
  python3 scripts/upload_zenodo.py --package --sandbox

  # real upload, leave as a draft you review + publish in the browser
  python3 scripts/upload_zenodo.py --package

  # real upload AND publish (mints the DOI immediately)
  python3 scripts/upload_zenodo.py --package --publish
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
DOCS = ROOT / "docs"
BUNDLE_DIR = DATA / "zenodo_bundle"

PROD_BASE = "https://zenodo.org/api"
SANDBOX_BASE = "https://sandbox.zenodo.org/api"

# ---------------------------------------------------------------------------
# What goes into the record. Each entry becomes one .zip in the bundle.
#
# Datasets are grouped so we can control redistribution by --scope:
#   GENERATED : our own synthetic output — always safe to publish (CC-BY-4.0).
#   PUBLIC    : third-party but public-domain (NAIP / USDA) — safe to publish.
#   RESTRICTED: raw third-party crops (SARD, VisDrone). VisDrone is
#               research/non-commercial and should NOT be redistributed under
#               CC-BY-4.0; SARD only if its license permits. Excluded by default.
# ---------------------------------------------------------------------------
GENERATED_DATASETS = {
    "omnisearch_drone_survivor": DATA / "cv_train" / "survivor",
    "omnisearch_drone_survivor_naip": DATA / "cv_train" / "survivor_naip",
    "omnisearch_ugv_survivor": DATA / "cv_train" / "ugv",
    "omnisearch_thermal_tir": DATA / "cv_train" / "thermal",
    "omnisearch_ugv_ground_legacy": DATA / "cv_train" / "survivor_ground",
}
PUBLIC_DATASETS = {
    "omnisearch_naip_backgrounds": DATA / "source_cache" / "naip",
}
RESTRICTED_DATASETS = {
    "omnisearch_source_sard_cutouts": DATA / "cv_assets" / "sard_grabcut",
    "omnisearch_source_visdrone_decoys": DATA / "cv_assets" / "visdrone_decoys",
}


def datasets_for_scope(scope: str) -> dict:
    """Return the dataset dirs to package for the requested scope."""
    if scope == "generated":
        return dict(GENERATED_DATASETS)
    if scope == "generated+public":
        return {**GENERATED_DATASETS, **PUBLIC_DATASETS}
    if scope == "all":
        return {**GENERATED_DATASETS, **PUBLIC_DATASETS, **RESTRICTED_DATASETS}
    raise ValueError(f"unknown scope: {scope}")

# Individual files added as-is (not zipped).
EXTRA_FILES = [
    MODELS / "survivor_yolov8s.pt",
    MODELS / "survivor_naip_yolov8s.pt",
    MODELS / "ugv_front_yolov8s.pt",
    MODELS / "ugv_mast_yolov8n.pt",
    MODELS / "thermal_yolov8n.pt",
    DOCS / "cv_dataset_eda.md",
    DOCS / "CV_SURVIVOR_DETECTION.md",
    DOCS / "cv_detection_modality_report.md",
]

# Generation scripts bundled for reproducibility.
# (Disabled — the record ships datasets, weights and the datasheet only.)
SCRIPTS_TO_BUNDLE: list[Path] = []

METADATA = {
    "upload_type": "dataset",
    "title": "OmniSearch: Synthetic Multi-Modal Survivor-Detection Datasets for Wildfire Search & Rescue",
    "description": (
        "<p>Physics-grounded synthetic datasets and trained YOLOv8 detectors for survivor "
        "detection in wildfire search-and-rescue, produced for the UC Berkeley MIDS OmniSearch "
        "capstone. Includes aerial (drone) datasets with altitude-aware sizing and oblique "
        "(side-angle) camera views, ground-robot (UGV) datasets for front and mast cameras, and "
        "a simulated thermal-infrared (TIR) dataset. Real person cutouts (SARD) and hard-negative "
        "vehicle crops (VisDrone) are composited over NAIP aerial imagery with color "
        "harmonization, range-aware blur, and wildfire effects. Bundled with the generation "
        "scripts, a datasheet (EDA), and trained model weights for full reproducibility.</p>"
        "<ul>"
        "<li>Drone survivor (SARD+NAIP composite): 2,000 train + 400 val (25% oblique)</li>"
        "<li>Drone survivor (NAIP-only background): 500 train + 90 val</li>"
        "<li>UGV front + mast cameras: 3,000 train + 600 val</li>"
        "<li>Thermal TIR: 1,000 train + 200 val</li>"
        "<li>Source assets: 54 SARD cutouts, 500 VisDrone decoys, 81 NAIP tiles</li>"
        "</ul>"
        "<p>Labels are in standard YOLO format (class 0 = person) with per-image metadata JSONs.</p>"
    ),
    "creators": [
        {"name": "Schuetz, Ann-Kathrin", "affiliation": "UC Berkeley MIDS"},
        {"name": "Jules, Jefferson-Stanley", "affiliation": "UC Berkeley MIDS"},
        {"name": "Lavrenin, Oleksii", "affiliation": "UC Berkeley MIDS"},
    ],
    "license": "cc-by-4.0",
    "access_right": "open",
    "keywords": [
        "computer vision", "object detection", "YOLOv8", "search and rescue",
        "wildfire", "synthetic data", "thermal infrared", "drone", "UGV",
        "survivor detection", "NAIP", "SARD",
    ],
    "version": "1.0.0",
    "language": "eng",
}


# ---------------------------------------------------------------------------
# Packaging
# ---------------------------------------------------------------------------
def _zip_dir(src: Path, dest_zip: Path) -> None:
    files = [p for p in src.rglob("*") if p.is_file()]
    with zipfile.ZipFile(dest_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in files:
            zf.write(p, arcname=str(p.relative_to(src.parent)))
    print(f"  packaged {len(files):>5} files -> {dest_zip.name} "
          f"({dest_zip.stat().st_size / 1e6:.1f} MB)")


def package(scope: str) -> list[Path]:
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    datasets = datasets_for_scope(scope)
    print(f"Packaging datasets (scope={scope}) into {BUNDLE_DIR} ...")
    for name, src in datasets.items():
        if not src.exists():
            print(f"  [skip] missing: {src}")
            continue
        z = BUNDLE_DIR / f"{name}.zip"
        _zip_dir(src, z)
        out.append(z)

    # scripts bundle
    scripts_zip = BUNDLE_DIR / "omnisearch_generation_scripts.zip"
    present = [s for s in SCRIPTS_TO_BUNDLE if s.exists()]
    if present:
        with zipfile.ZipFile(scripts_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for s in present:
                zf.write(s, arcname=f"scripts/{s.name}")
        print(f"  packaged {len(present):>5} files -> {scripts_zip.name}")
        out.append(scripts_zip)
    return out


def collect_files(packaged: list[Path]) -> list[Path]:
    files = list(packaged)
    for f in EXTRA_FILES:
        if f.exists():
            files.append(f)
        else:
            print(f"  [skip] missing extra file: {f}")
    return files


# ---------------------------------------------------------------------------
# Zenodo REST API
# ---------------------------------------------------------------------------
def upload(files: list[Path], base: str, token: str, publish: bool) -> dict:
    import requests

    params = {"access_token": token}
    headers = {"Content-Type": "application/json"}

    print(f"\nCreating deposition on {base} ...")
    r = requests.post(f"{base}/deposit/depositions", params=params,
                      json={}, headers=headers, timeout=60)
    r.raise_for_status()
    dep = r.json()
    dep_id = dep["id"]
    bucket = dep["links"]["bucket"]
    print(f"  deposition id: {dep_id}")
    print(f"  draft link:    {dep['links'].get('html')}")

    print("\nUploading files ...")
    for f in files:
        size = f.stat().st_size
        print(f"  -> {f.name} ({size / 1e6:.1f} MB)")
        with f.open("rb") as fh:
            up = requests.put(f"{bucket}/{f.name}", data=fh, params=params, timeout=None)
        up.raise_for_status()

    print("\nWriting metadata ...")
    r = requests.put(f"{base}/deposit/depositions/{dep_id}", params=params,
                     json={"metadata": METADATA}, headers=headers, timeout=60)
    r.raise_for_status()

    if publish:
        print("\nPublishing (this mints the DOI and is IRREVERSIBLE) ...")
        r = requests.post(f"{base}/deposit/depositions/{dep_id}/actions/publish",
                          params=params, timeout=120)
        r.raise_for_status()
        pub = r.json()
        print(f"  PUBLISHED. DOI: {pub.get('doi')}")
        print(f"  record: {pub['links'].get('record_html')}")
        return pub

    print(f"\nDraft ready — review & publish in the browser:\n  {dep['links'].get('html')}")
    return dep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", action="store_true", help="build the .zip bundles first")
    ap.add_argument("--scope", default="generated+public",
                    choices=["generated", "generated+public", "all"],
                    help="which datasets to include (default: generated+public; "
                         "'all' adds raw SARD/VisDrone crops — check their licenses first)")
    ap.add_argument("--no-upload", action="store_true", help="package only; skip all network calls")
    ap.add_argument("--sandbox", action="store_true", help="use sandbox.zenodo.org")
    ap.add_argument("--publish", action="store_true", help="publish immediately (mints DOI)")
    ap.add_argument("--token", default=None, help="Zenodo access token (else read from env)")
    args = ap.parse_args()

    packaged: list[Path] = []
    if args.package:
        packaged = package(args.scope)
    else:
        packaged = sorted(BUNDLE_DIR.glob("*.zip")) if BUNDLE_DIR.exists() else []

    files = collect_files(packaged)
    total = sum(f.stat().st_size for f in files)
    print(f"\n{len(files)} files, {total / 1e6:.1f} MB total, ready for upload.")

    if args.no_upload:
        print("--no-upload set: stopping before network calls.")
        return 0

    base = SANDBOX_BASE if args.sandbox else PROD_BASE
    env_key = "ZENODO_SANDBOX_TOKEN" if args.sandbox else "ZENODO_TOKEN"
    token = args.token or os.environ.get(env_key)
    if not token:
        print(f"\nERROR: no token. Set ${env_key} or pass --token.", file=sys.stderr)
        return 1

    upload(files, base, token, args.publish)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
