"""Run preliminary detector workflow on rendered drone flight frames."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detection.preliminary_detector import PreliminaryPersonDetector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--flight-metadata",
        default="results/cv_demo_drone_flight_30m_smooth/flight_metadata.json",
        help="Metadata written by render_naip_drone_flight_frames.py.",
    )
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--detection-probability", type=float, default=1.0)
    parser.add_argument("--pixel-noise-std", type=float, default=0.0)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--confidence-jitter", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    metadata_path = Path(args.flight_metadata)
    if not metadata_path.is_absolute():
        metadata_path = ROOT / metadata_path
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    out_json = Path(args.out_json) if args.out_json else metadata_path.parent / "preliminary_detections.json"
    if not out_json.is_absolute():
        out_json = ROOT / out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)

    image_size = int(metadata["yolo_image_size_px"])
    footprint_m = float(metadata["footprint_m"])
    detector = PreliminaryPersonDetector(
        detection_probability=float(args.detection_probability),
        pixel_noise_std=float(args.pixel_noise_std),
        confidence=float(args.confidence),
        confidence_jitter=float(args.confidence_jitter),
        seed=int(args.seed),
    )

    frames = []
    for frame in metadata["frames"]:
        label_path = Path(frame["label_path"])
        if not label_path.is_absolute():
            label_path = ROOT / label_path
        result = detector.detect_yolo_label_file(label_path, image_size=image_size)
        detections = []
        for detection in result.detections:
            center_px = detection.center_xy
            relative_m = _pixel_center_to_relative_m(center_px, image_size=image_size, footprint_m=footprint_m)
            drone_m = tuple(float(value) for value in frame["drone_position_m"])
            estimated_area_m = (drone_m[0] + relative_m[0], drone_m[1] + relative_m[1])
            detections.append(
                {
                    "class_name": detection.class_name,
                    "confidence": detection.confidence,
                    "bbox_xyxy": list(detection.box),
                    "center_px": [center_px[0], center_px[1]],
                    "relative_to_drone_m": [relative_m[0], relative_m[1]],
                    "estimated_area_position_m": [estimated_area_m[0], estimated_area_m[1]],
                }
            )
        frames.append(
            {
                "index": frame["index"],
                "image_path": frame["image_path"],
                "label_path": frame["label_path"],
                "drone_position_m": frame["drone_position_m"],
                "detections": detections,
            }
        )

    output = {
        "mode": "preliminary_ground_truth_detector",
        "source_flight_metadata": str(metadata_path),
        "detection_probability": float(args.detection_probability),
        "pixel_noise_std": float(args.pixel_noise_std),
        "confidence": float(args.confidence),
        "confidence_jitter": float(args.confidence_jitter),
        "frames": frames,
    }
    out_json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    detected_frames = [frame["index"] for frame in frames if frame["detections"]]
    print(f"Wrote preliminary detections: {out_json}")
    print(f"Detected survivor in {len(detected_frames)}/{len(frames)} frames: {detected_frames}")


def _pixel_center_to_relative_m(
    center_px: tuple[float, float],
    *,
    image_size: int,
    footprint_m: float,
) -> tuple[float, float]:
    cx, cy = center_px
    dx_m = ((cx / image_size) - 0.5) * footprint_m
    dy_m = (0.5 - (cy / image_size)) * footprint_m
    return float(dx_m), float(dy_m)


if __name__ == "__main__":
    main()
