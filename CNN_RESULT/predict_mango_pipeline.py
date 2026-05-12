from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from train_cnn import build_model, coerce_int_list  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_SEG_MODEL = PROJECT_ROOT / "Mango-Detection-5" / "models" / "mango_seg" / "weights" / "best.pt"
DEFAULT_SOURCE = PROJECT_ROOT / "Mango-Detection-5" / "test" / "images"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLOv8-Seg mango instance detection plus CNN disease classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help="Image file or folder of original mango/crate images.")
    parser.add_argument("--seg-model", type=Path, default=DEFAULT_SEG_MODEL,
                        help="YOLOv8-Seg weights trained with mango/occluded classes.")
    parser.add_argument("--disease-model", type=Path, default=None,
                        help="CNN checkpoint. If omitted, the newest CNN/runs/*/best_model.pt is used.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.50, help="YOLO NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--device", type=str, default=None,
                        help="Use 'cpu', '0', 'cuda:0', etc. Auto-selects when omitted.")
    parser.add_argument("--padding", type=int, default=10, help="Pixels around each mango crop.")
    parser.add_argument("--crop-mode", choices=["bbox", "mask-white", "mask-black"], default="bbox",
                        help="How the crop passed to the CNN is prepared.")
    parser.add_argument("--no-pad-square", action="store_true",
                        help="Do not pad mango crops to a square before CNN classification.")
    parser.add_argument("--skip-occluded", action="store_true",
                        help="Report occluded mangoes but skip disease classification for them.")
    parser.add_argument("--save-crops", action="store_true", help="Save each extracted mango crop.")
    parser.add_argument("--max-images", type=int, default=None, help="Limit images for a quick test run.")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def find_latest_disease_model() -> Path:
    candidates = sorted(
        (SCRIPT_DIR / "runs").glob("*/best_model.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "No CNN disease checkpoint found. Train first with: python CNN\\train_cnn.py"
        )
    return candidates[0]


def load_checkpoint(path: Path, device: torch.device) -> Dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_disease_model(checkpoint_path: Path, device: torch.device):
    checkpoint = load_checkpoint(checkpoint_path, device)
    classes = checkpoint.get("classes")
    if not classes:
        classes_path = checkpoint_path.parent / "classes.json"
        if classes_path.exists():
            classes = json.loads(classes_path.read_text(encoding="utf-8"))
    if not classes:
        raise ValueError(f"Could not read class names from {checkpoint_path}")

    config = checkpoint.get("args", {})
    dropout = float(config.get("dropout", 0.35))
    image_size = int(config.get("image_size", 224))
    conv_channels = coerce_int_list(config.get("conv_channels"), (32, 64, 128, 256), "conv_channels")
    classifier_hidden = coerce_int_list(
        config.get("classifier_hidden"),
        (),
        "classifier_hidden",
        allow_empty=True,
    )
    kernel_size = int(config.get("kernel_size", 3))
    activation = str(config.get("activation", "relu"))
    use_batchnorm = bool(config.get("use_batchnorm", not bool(config.get("no_batchnorm", False))))

    model = build_model(
        num_classes=len(classes),
        dropout=dropout,
        conv_channels=conv_channels,
        kernel_size=kernel_size,
        use_batchnorm=use_batchnorm,
        activation=activation,
        classifier_hidden=classifier_hidden,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, classes, image_size


def build_disease_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def expand_sources(source: Path, max_images: int | None) -> List[Path]:
    source = resolve_path(source)
    if source.is_file():
        images = [source]
    elif source.is_dir():
        images = sorted(
            path for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    else:
        raise FileNotFoundError(f"Source not found: {source}")

    if max_images is not None:
        images = images[:max_images]
    return images


def yolo_names_to_dict(names) -> Dict[int, str]:
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, (list, tuple)):
        return {index: str(value) for index, value in enumerate(names)}
    return {}


def occlusion_status(class_id: int, class_name: str) -> str:
    normalized = class_name.strip().lower().replace(" ", "_")
    if normalized == "occluded" or class_id == 1:
        return "occluded"
    return "non_occluded"


def clamp_bbox(
    bbox_xyxy: Iterable[float],
    image_w: int,
    image_h: int,
    padding: int,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(round(value)) for value in bbox_xyxy]
    return (
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(image_w, x2 + padding),
        min(image_h, y2 + padding),
    )


def polygon_to_mask(points: np.ndarray, image_w: int, image_h: int) -> np.ndarray:
    mask = np.zeros((image_h, image_w), dtype=np.uint8)
    if len(points) >= 3:
        cv2.fillPoly(mask, [points.astype(np.int32)], 255)
    return mask


def prepare_crop(crop_bgr: np.ndarray, crop_mask: np.ndarray | None, crop_mode: str) -> np.ndarray:
    if crop_mode == "bbox" or crop_mask is None:
        return crop_bgr

    background_value = 255 if crop_mode == "mask-white" else 0
    background = np.full_like(crop_bgr, background_value)
    return np.where(crop_mask[:, :, None] > 0, crop_bgr, background)


def pad_to_square(image_bgr: np.ndarray, fill_value: int = 0) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    side = max(h, w)
    top = (side - h) // 2
    bottom = side - h - top
    left = (side - w) // 2
    right = side - w - left
    return cv2.copyMakeBorder(
        image_bgr,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(fill_value, fill_value, fill_value),
    )


def classify_crop(
    model,
    classes: List[str],
    transform: transforms.Compose,
    crop_bgr: np.ndarray,
    device: torch.device,
) -> Tuple[str, float, Dict[str, float]]:
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(crop_rgb)
    tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1).squeeze(0).cpu().numpy()

    best_index = int(probabilities.argmax())
    scores = {class_name: float(probabilities[index]) for index, class_name in enumerate(classes)}
    return classes[best_index], float(probabilities[best_index]), scores


def annotation_text(disease_label: str, disease_confidence: float) -> str:
    if disease_label == "skipped_occluded":
        return "Skipped"
    return f"{disease_label} {disease_confidence * 100:.0f}%"


def disease_color(disease_label: str) -> Tuple[int, int, int]:
    normalized = disease_label.strip().lower()
    if normalized == "healthy":
        return (55, 180, 75)
    if normalized == "anthracnose":
        return (35, 35, 230)
    if normalized == "mango scab":
        return (0, 190, 255)
    if normalized == "stem end rot":
        return (205, 70, 205)
    return (170, 170, 170)


def occlusion_color(occlusion_label: str) -> Tuple[int, int, int]:
    if occlusion_label == "occluded":
        return (35, 35, 230)
    return (55, 180, 75)


def readable_text_color(background_bgr: Tuple[int, int, int]) -> Tuple[int, int, int]:
    b, g, r = background_bgr
    luminance = (0.114 * b) + (0.587 * g) + (0.299 * r)
    return (0, 0, 0) if luminance >= 140 else (255, 255, 255)


def draw_instance(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    polygon: np.ndarray | None,
    label: str,
    box_color: Tuple[int, int, int],
    polygon_color: Tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = bbox
    min_dim = min(image.shape[:2])
    box_thickness = max(4, int(round(min_dim / 420)))
    polygon_thickness = max(3, box_thickness - 1)

    if polygon is not None and len(polygon) >= 3:
        cv2.polylines(
            image,
            [polygon.astype(np.int32)],
            isClosed=True,
            color=polygon_color,
            thickness=polygon_thickness,
            lineType=cv2.LINE_AA,
        )
    cv2.rectangle(image, (x1, y1), (x2, y2), box_color, box_thickness, lineType=cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(1.0, min_dim / 1250)
    text_thickness = max(3, int(round(font_scale * 2.2)))
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
    pad_x = max(10, int(round(text_h * 0.45)))
    pad_y = max(8, int(round(text_h * 0.35)))
    label_w = text_w + (pad_x * 2)
    label_h = text_h + baseline + (pad_y * 2)
    bg_x1 = min(x1, max(0, image.shape[1] - label_w - 2))
    bg_y1 = y1 - label_h - box_thickness
    if bg_y1 < 0:
        bg_y1 = min(image.shape[0] - label_h - 1, y1 + box_thickness)
    bg_x1 = max(0, bg_x1)
    bg_y1 = max(0, bg_y1)
    bg_x2 = min(image.shape[1] - 1, bg_x1 + label_w)
    bg_y2 = min(image.shape[0] - 1, bg_y1 + label_h)

    cv2.rectangle(
        image,
        (bg_x1, bg_y1),
        (bg_x2, bg_y2),
        box_color,
        -1,
        lineType=cv2.LINE_AA,
    )
    cv2.rectangle(
        image,
        (bg_x1, bg_y1),
        (bg_x2, bg_y2),
        (0, 0, 0),
        max(1, box_thickness // 3),
        lineType=cv2.LINE_AA,
    )

    text_origin = (bg_x1 + pad_x, bg_y1 + pad_y + text_h)
    cv2.putText(
        image,
        label,
        text_origin,
        font,
        font_scale,
        readable_text_color(box_color),
        text_thickness,
        cv2.LINE_AA,
    )


def process_image(
    image_path: Path,
    yolo_model: YOLO,
    disease_model,
    disease_classes: List[str],
    disease_transform: transforms.Compose,
    device: torch.device,
    args: argparse.Namespace,
    crops_dir: Path,
    annotated_dir: Path,
) -> List[Dict[str, object]]:
    result = yolo_model.predict(
        source=str(image_path),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        verbose=False,
    )[0]

    image_bgr = result.orig_img.copy()
    annotated = image_bgr.copy()
    image_h, image_w = image_bgr.shape[:2]
    names = yolo_names_to_dict(result.names)
    rows: List[Dict[str, object]] = []

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        annotated_path = annotated_dir / f"{image_path.stem}_annotated.jpg"
        cv2.imwrite(str(annotated_path), annotated)
        return rows

    mask_polygons = result.masks.xy if result.masks is not None else []

    for instance_id in range(len(boxes)):
        class_id = int(boxes.cls[instance_id].item())
        seg_conf = float(boxes.conf[instance_id].item())
        class_name = names.get(class_id, str(class_id))
        status = occlusion_status(class_id, class_name)
        bbox = clamp_bbox(boxes.xyxy[instance_id].tolist(), image_w, image_h, args.padding)
        x1, y1, x2, y2 = bbox

        polygon = None
        crop_mask = None
        if instance_id < len(mask_polygons):
            polygon = np.asarray(mask_polygons[instance_id], dtype=np.float32)
            full_mask = polygon_to_mask(polygon, image_w, image_h)
            crop_mask = full_mask[y1:y2, x1:x2]

        crop_bgr = image_bgr[y1:y2, x1:x2]
        if crop_bgr.size == 0:
            continue

        cnn_crop = prepare_crop(crop_bgr, crop_mask, args.crop_mode)
        if not args.no_pad_square:
            fill_value = 255 if args.crop_mode == "mask-white" else 0
            cnn_crop = pad_to_square(cnn_crop, fill_value=fill_value)

        crop_path = ""
        if args.save_crops:
            crop_path_obj = crops_dir / f"{image_path.stem}_instance_{instance_id:03d}_{status}.png"
            cv2.imwrite(str(crop_path_obj), cnn_crop)
            crop_path = str(crop_path_obj)

        if args.skip_occluded and status == "occluded":
            disease_label = "skipped_occluded"
            disease_conf = 0.0
            disease_scores = {}
        else:
            disease_label, disease_conf, disease_scores = classify_crop(
                disease_model,
                disease_classes,
                disease_transform,
                cnn_crop,
                device,
            )

        label = annotation_text(disease_label, disease_conf)
        draw_instance(
            annotated,
            bbox,
            polygon,
            label,
            box_color=disease_color(disease_label),
            polygon_color=occlusion_color(status),
        )

        row = {
            "source_image": str(image_path),
            "instance_id": instance_id,
            "bbox_x1": x1,
            "bbox_y1": y1,
            "bbox_x2": x2,
            "bbox_y2": y2,
            "seg_class_id": class_id,
            "seg_class_name": class_name,
            "seg_confidence": round(seg_conf, 6),
            "occlusion_status": status,
            "disease_label": disease_label,
            "disease_confidence": round(disease_conf, 6),
            "crop_path": crop_path,
        }
        for disease_class, score in disease_scores.items():
            row[f"disease_score_{disease_class}"] = round(score, 6)
        rows.append(row)

    annotated_path = annotated_dir / f"{image_path.stem}_annotated.jpg"
    cv2.imwrite(str(annotated_path), annotated)
    for row in rows:
        row["annotated_image"] = str(annotated_path)

    return rows


def save_rows_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.source = resolve_path(args.source)
    args.seg_model = resolve_path(args.seg_model)
    args.disease_model = resolve_path(args.disease_model) if args.disease_model else find_latest_disease_model()
    args.output_dir = (
        resolve_path(args.output_dir)
        if args.output_dir
        else SCRIPT_DIR / "pipeline_outputs" / time.strftime("pipeline_%Y%m%d_%H%M%S")
    )

    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    disease_device = torch.device("cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu")

    images = expand_sources(args.source, args.max_images)
    if not images:
        raise FileNotFoundError(f"No images found in {args.source}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = args.output_dir / "crops"
    annotated_dir = args.output_dir / "annotated"
    crops_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading segmentation model: {args.seg_model}")
    yolo_model = YOLO(str(args.seg_model))
    print(f"Loading disease CNN model: {args.disease_model}")
    disease_model, disease_classes, image_size = load_disease_model(args.disease_model, disease_device)
    disease_transform = build_disease_transform(image_size)

    all_rows: List[Dict[str, object]] = []
    for index, image_path in enumerate(images, start=1):
        rows = process_image(
            image_path,
            yolo_model,
            disease_model,
            disease_classes,
            disease_transform,
            disease_device,
            args,
            crops_dir,
            annotated_dir,
        )
        all_rows.extend(rows)
        print(f"[{index}/{len(images)}] {image_path.name}: {len(rows)} mango instances")

    predictions_csv = args.output_dir / "predictions.csv"
    save_rows_csv(predictions_csv, all_rows)

    config = {
        "source": str(args.source),
        "seg_model": str(args.seg_model),
        "disease_model": str(args.disease_model),
        "output_dir": str(args.output_dir),
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "device": args.device,
        "disease_classes": disease_classes,
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    occlusion_counts: Dict[str, int] = {}
    disease_counts: Dict[str, int] = {}
    for row in all_rows:
        occlusion_counts[str(row["occlusion_status"])] = occlusion_counts.get(str(row["occlusion_status"]), 0) + 1
        disease_counts[str(row["disease_label"])] = disease_counts.get(str(row["disease_label"]), 0) + 1

    print("\nPipeline complete")
    print("-" * 52)
    print(f"Images processed:      {len(images)}")
    print(f"Mango instances:       {len(all_rows)}")
    print(f"Occlusion counts:      {occlusion_counts}")
    print(f"Disease counts:        {disease_counts}")
    print(f"Predictions CSV:       {predictions_csv}")
    print(f"Annotated images:      {annotated_dir}")
    if args.save_crops:
        print(f"Saved crops:           {crops_dir}")


if __name__ == "__main__":
    main()
