"""Upload the annotated CV QA-visualization export to Zenodo as a new draft.

This is a separate deposition from the main training-data record
(DOI 10.5281/zenodo.21226010). It ships the same images *with* the size/scale
overlays burned in (bounding boxes, real-world m dimensions, GSD header,
scale bar, decoy-class color legend) so reviewers can visually verify object
sizes without re-running any code.

Usage:
    python3 scripts/upload_annotated_zenodo.py \
        --src /Users/alexlavre/Documents/omnisearch_capstone/cv_annotated_07-11 \
        --sandbox        # test first
    python3 scripts/upload_annotated_zenodo.py \
        --src /Users/alexlavre/Documents/omnisearch_capstone/cv_annotated_07-11
        # real Zenodo draft (no --publish -> stays a draft for review)
"""
from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from upload_zenodo import PROD_BASE, SANDBOX_BASE  # noqa: E402

METADATA = {
    "upload_type": "dataset",
    "title": "OmniSearch: Annotated CV Detection Visualizations (Physical Scale Verification)",
    "description": (
        "<p>Annotated visualization export accompanying the OmniSearch synthetic "
        "survivor-detection datasets (see companion record DOI 10.5281/zenodo.21226010). "
        "Every frame from the drone, UGV (front/mast), synthetic-thermal, and real "
        "thermal (HIT-UAV) training sets is re-rendered with:</p>"
        "<ul>"
        "<li>Bounding boxes for every labeled survivor, with pixel size AND real-world "
        "size in metres (px &times; ground-sample-distance)</li>"
        "<li>Color-coded hard-negative decoy boxes (orange = vehicle, red = animal, "
        "magenta = colorful object) drawn from generation metadata, unlabeled</li>"
        "<li>A header banner with GSD (m/px), altitude, camera mode (nadir/oblique/UGV), "
        "and survivor/decoy counts per frame</li>"
        "<li>A scale bar calibrated to the frame's terrain resolution</li>"
        "</ul>"
        "<p>Purpose: let a reviewer visually confirm that every composited object "
        "(survivor and decoy) is physically plausible relative to the terrain scale, "
        "without re-running any generation code. Frames from the real HIT-UAV thermal "
        "set are flagged 'NO SCALE' since that public dataset does not publish "
        "per-frame camera intrinsics. Packaged as one zip per dataset "
        "(survivor, survivor_6k, survivor_naip, thermal, thermal_real, ugv).</p>"
    ),
    "creators": [
        {"name": "Schuetz, Ann-Kathrin", "affiliation": "UC Berkeley MIDS"},
        {"name": "Jules, Jefferson-Stanley", "affiliation": "UC Berkeley MIDS"},
        {"name": "Lavrenin, Oleksii", "affiliation": "UC Berkeley MIDS"},
    ],
    "license": "cc-by-4.0",
    "access_right": "open",
    "keywords": [
        "computer vision", "object detection", "search and rescue", "wildfire",
        "synthetic data", "data quality", "annotation visualization",
    ],
    "version": "1.0.0",
    "language": "eng",
    "related_identifiers": [
        {"identifier": "10.5281/zenodo.21226010", "relation": "isSupplementTo",
         "resource_type": "dataset"},
    ],
}


def _clean_zip(src: Path, dest_zip: Path) -> Path:
    """Zip a folder, skipping macOS junk (.DS_Store, __MACOSX, AppleDouble)."""
    files = [
        p for p in src.rglob("*")
        if p.is_file()
        and p.name != ".DS_Store"
        and not p.name.startswith("._")
        and "__MACOSX" not in p.parts
    ]
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in files:
            zf.write(p, arcname=str(p.relative_to(src.parent)))
    print(f"  packaged {len(files):>6} files -> {dest_zip.name} "
          f"({dest_zip.stat().st_size / 1e6:.1f} MB)")
    return dest_zip


def package_per_folder(src: Path, out_dir: Path) -> list[Path]:
    """Zip each top-level dataset folder under src separately."""
    out_dir.mkdir(parents=True, exist_ok=True)
    zips = []
    for sub in sorted(p for p in src.iterdir() if p.is_dir()):
        dest = out_dir / f"omnisearch_annotated_{sub.name}.zip"
        zips.append(_clean_zip(sub, dest))
    return zips


def sync_deposition(dep_id: int, base: str, token: str, files: list[Path], publish: bool) -> dict:
    """Replace whatever files a draft deposition has with `files`, then write metadata."""
    import requests

    params = {"access_token": token}
    headers = {"Content-Type": "application/json"}

    r = requests.get(f"{base}/deposit/depositions/{dep_id}", params=params, timeout=60)
    r.raise_for_status()
    dep = r.json()
    bucket = dep["links"]["bucket"]

    print("\nRemoving existing files from the draft ...")
    rf = requests.get(f"{base}/deposit/depositions/{dep_id}/files", params=params, timeout=60)
    rf.raise_for_status()
    for f in rf.json():
        print(f"  deleting {f['filename']}")
        rd = requests.delete(f"{base}/deposit/depositions/{dep_id}/files/{f['id']}", params=params, timeout=60)
        rd.raise_for_status()

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
        return pub

    print(f"\nDraft ready — review & publish in the browser:\n  {dep['links'].get('html')}")
    return dep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="the annotated export folder (contains one subfolder per dataset)")
    ap.add_argument("--zip-dir", default=str(ROOT / "data" / "zenodo_bundle" / "annotated"))
    ap.add_argument("--deposition-id", type=int, default=None,
                    help="existing draft deposition id to update in place (else creates a new one)")
    ap.add_argument("--sandbox", action="store_true")
    ap.add_argument("--publish", action="store_true", help="publish immediately (mints DOI) -- omit to stay a draft")
    ap.add_argument("--token", default=None)
    args = ap.parse_args()

    src = Path(args.src)
    if not src.is_dir():
        print(f"ERROR: {src} is not a directory", file=sys.stderr)
        return 1

    zips = package_per_folder(src, Path(args.zip_dir))

    base = SANDBOX_BASE if args.sandbox else PROD_BASE
    env_key = "ZENODO_SANDBOX_TOKEN" if args.sandbox else "ZENODO_TOKEN"
    token = args.token or os.environ.get(env_key)
    if not token:
        print(f"ERROR: no token. Set ${env_key} or pass --token.", file=sys.stderr)
        return 1

    if args.deposition_id:
        sync_deposition(args.deposition_id, base, token, zips, args.publish)
    else:
        from upload_zenodo import upload
        import upload_zenodo
        upload_zenodo.METADATA = METADATA
        upload(zips, base, token, args.publish)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
