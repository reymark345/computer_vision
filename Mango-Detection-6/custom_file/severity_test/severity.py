
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_SOURCE = SCRIPT_DIR.parent / "occluded_test" / "based_path"
DEFAULT_MODEL = PROJECT_ROOT / "Mango-Detection-6" / "models" / "mango_seg" / "weights" / "best.pt"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate mango disease severity from YOLOv8-Seg mango masks."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help="Image file or folder. Defaults to custom_file/occluded_test/based_path.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help="Mango-Detection-6 YOLOv8-Seg best.pt file.")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "severity_results")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--gray-threshold", type=int, default=120,
                        help="Pixels at or above this grayscale value are healthy fruit.")
    parser.add_argument("--healthy-max", type=float, default=0.1,
                        help="Maximum lesion percentage for Healthy.")
    parser.add_argument("--early-max", type=float, default=5.0,
                        help="Maximum lesion percentage for Early.")
    parser.add_argument("--intermediate-max", type=float, default=30.0,
                        help="Maximum lesion percentage for Intermediate.")
    parser.add_argument("--lesion-method", choices=("color", "paper-gray"), default="color",
                        help="Lesion detector. Color reduces healthy green/yellow peel false positives; "
                             "paper-gray reproduces the fixed grayscale threshold method.")
    parser.add_argument("--class-ids", type=int, nargs="+", default=[0, 1],
                        help="Segmentation classes to measure. Defaults to mango and occluded mango.")
    parser.add_argument("--save-crops", action=argparse.BooleanOptionalAction, default=True,
                        help="Save one masked crop and lesion mask for every mango.")
    return parser.parse_args()


def expand_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if source.is_dir():
        return sorted(path for path in source.rglob("*")
                      if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    raise FileNotFoundError(f"Source not found: {source}")


def class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def severity_label(percent: float, args: argparse.Namespace) -> str:
    if percent <= args.healthy_max:
        return "Healthy"
    if percent <= args.early_max:
        return "Early"
    if percent <= args.intermediate_max:
        return "Intermediate"
    return "Final"


def lesion_mask(image_bgr: np.ndarray, fruit_mask: np.ndarray, threshold: int,
                method: str) -> np.ndarray:
    """Estimate lesions inside one fruit while retaining the paper's area formula."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    fruit = fruit_mask > 0
    if method == "paper-gray":
        lesion = fruit & (gray < threshold)
    else:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        brown = (hue >= 3) & (hue <= 30) & (saturation >= 70) & (value <= 175)
        red = ((hue <= 8) | (hue >= 170)) & (saturation >= 75) & (value <= 190)
        dark = (value <= 75) & (saturation >= 25)
        lesion = fruit & (brown | red | dark)

    result = lesion.astype(np.uint8) * 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    return result


def polygon_mask(points: np.ndarray, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(points) >= 3:
        cv2.fillPoly(mask, [points.astype(np.int32)], 255)
    return mask


def annotate(image: np.ndarray, fruit_mask: np.ndarray, disease_mask: np.ndarray,
             text: str, bbox: tuple[int, int, int, int], border_color: tuple[int, int, int],
             text_color: tuple[int, int, int]) -> None:
    overlay = image.copy()
    overlay[disease_mask > 0] = (0, 0, 255)
    image[:] = cv2.addWeighted(image, 0.72, overlay, 0.28, 0)
    contours, _ = cv2.findContours(fruit_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, border_color, 2)
    x1, y1, _, _ = bbox
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.8, min(1.35, image.shape[1] / 1200.0))
    thickness = max(2, round(font_scale * 2))
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness)
    text_x = max(4, min(x1, image.shape[1] - text_width - 8))
    text_y = y1 - 10
    if text_y - text_height - baseline < 4:
        text_y = min(image.shape[0] - 4, bbox[3] + text_height + baseline + 8)
    background_top = max(0, text_y - text_height - baseline - 6)
    background_bottom = min(image.shape[0], text_y + 5)
    cv2.rectangle(image, (text_x - 4, background_top),
                  (text_x + text_width + 4, background_bottom), (0, 0, 0), -1)
    cv2.putText(image, text, (text_x, text_y), font, font_scale,
                text_color, thickness, cv2.LINE_AA)


def save_instance_outputs(image: np.ndarray, fruit_mask: np.ndarray, lesion: np.ndarray,
                          bbox: tuple[int, int, int, int], output_dir: Path,
                          source_stem: str, mango_id: int) -> tuple[Path, Path]:
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
    crop = image[y1:y2, x1:x2].copy()
    crop_mask = fruit_mask[y1:y2, x1:x2]
    crop[crop_mask == 0] = 0
    crop_dir = output_dir / "mango_crops"
    mask_dir = output_dir / "lesion_masks"
    crop_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    crop_path = crop_dir / f"{source_stem}_mango_{mango_id:03d}.png"
    mask_path = mask_dir / f"{source_stem}_mango_{mango_id:03d}.png"
    cv2.imwrite(str(crop_path), crop)
    cv2.imwrite(str(mask_path), lesion[y1:y2, x1:x2])
    return crop_path, mask_path


def process_image(image_path: Path, model: YOLO, args: argparse.Namespace) -> tuple[list[dict[str, Any]], np.ndarray]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    height, width = image.shape[:2]
    result = model.predict(source=str(image_path), conf=args.conf, imgsz=args.imgsz,
                           classes=args.class_ids, max_det=300, verbose=False)[0]
    annotated_image = image.copy()
    rows: list[dict[str, Any]] = []
    boxes = result.boxes
    polygons = result.masks.xy if result.masks is not None else []

    if boxes is not None:
        for index in range(len(boxes)):
            class_id = int(boxes.cls[index].item())
            polygon = np.asarray(polygons[index]) if index < len(polygons) else np.empty((0, 2))
            fruit = polygon_mask(polygon, width, height)
            fruit_area = int(cv2.countNonZero(fruit))
            if fruit_area == 0:
                continue
            disease = lesion_mask(image, fruit, args.gray_threshold, args.lesion_method)
            lesion_area = int(cv2.countNonZero(disease))
            healthy_area = fruit_area - lesion_area
            percent = (lesion_area / fruit_area) * 100.0
            bbox_values = [int(round(value)) for value in boxes.xyxy[index].tolist()]
            label = severity_label(percent, args)
            confidence = float(boxes.conf[index].item())
            border_color = (0, 0, 255) if class_id == 1 else (0, 220, 0)
            label_colors = {
                "Healthy": (0, 220, 0),
                "Early": (0, 220, 255),
                "Intermediate": (0, 165, 255),
                "Final": (0, 80, 255),
            }
            annotate(annotated_image, fruit, disease,
                     f"{label} {percent:.2f}%", tuple(bbox_values), border_color,
                     label_colors[label])
            row: dict[str, Any] = {
                "mango_id": index + 1,
                "detector_class": class_name(result.names, class_id),
                "detector_confidence": round(confidence, 6),
                "bbox_xyxy": bbox_values,
                "fruit_area_pixels": fruit_area,
                "healthy_area_pixels": healthy_area,
                "lesion_area_pixels": lesion_area,
                "severity_percent": round(percent, 4),
                "severity_label": label,
                "gray_threshold": args.gray_threshold,
                "lesion_method": args.lesion_method,
            }
            if args.save_crops:
                crop_path, mask_path = save_instance_outputs(
                    image, fruit, disease, tuple(bbox_values), args.output_dir,
                    image_path.stem, index + 1)
                row["mango_crop"] = str(crop_path)
                row["lesion_mask"] = str(mask_path)
            rows.append(row)

    return rows, annotated_image


def main() -> None:
    args = parse_args()
    args.source = args.source.resolve()
    args.model = args.model.resolve()
    args.output_dir = args.output_dir.resolve()
    images = expand_images(args.source)
    if not images:
        raise FileNotFoundError(f"No images found in {args.source}")
    if not args.model.exists():
        raise FileNotFoundError(f"Segmentation model not found: {args.model}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model))
    all_results: list[dict[str, Any]] = []
    for image_path in images:
        rows, annotated = process_image(image_path, model, args)
        annotated_path = args.output_dir / f"{image_path.stem}_severity.jpg"
        cv2.imwrite(str(annotated_path), annotated)
        for row in rows:
            row["source_image"] = str(image_path)
            row["annotated_image"] = str(annotated_path)
        all_results.extend(rows)
        print(f"{image_path.name}: {len(rows)} mango(es)")
        for row in rows:
            print(f"  {row['detector_class']} {row['mango_id']}: {row['severity_label']} "
                  f"({row['severity_percent']:.2f}%)")

    output_json = args.output_dir / "severity_results.json"
    output_json.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    if all_results:
        output_csv = args.output_dir / "severity_results.csv"
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(all_results[0]))
            writer.writeheader()
            writer.writerows(all_results)
        print(f"CSV results saved to: {output_csv}")
    print(f"\nResults saved to: {output_json}")
    print(f"Annotated images saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

