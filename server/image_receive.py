from __future__ import annotations

import base64
import binascii
import io
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from flask import Flask, jsonify, request, send_file, send_from_directory
from PIL import Image
from ultralytics import YOLO

SERVER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SERVER_DIR.parent
CNN_DIR = PROJECT_ROOT / "CNN_RESULT"

sys.path.insert(0, str(CNN_DIR))

from predict_mango_pipeline import (  # noqa: E402
    DEFAULT_SEG_MODEL,
    build_disease_transform,
    find_latest_disease_model,
    load_disease_model,
    process_image,
)


app = Flask(__name__)

UPLOAD_FOLDER = SERVER_DIR / "upload"
RESULTS_FOLDER = SERVER_DIR / "results"
CROPS_FOLDER = RESULTS_FOLDER / "crops"

IMAGE_SUFFIX = ".png"
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.50
DEFAULT_IMGSZ = 640
DEFAULT_PADDING = 10
DEFAULT_CROP_MODE = "bbox"

for folder in (UPLOAD_FOLDER, RESULTS_FOLDER, CROPS_FOLDER):
    folder.mkdir(parents=True, exist_ok=True)


def clean_filename_part(value: Any, fallback: str) -> str:
    text = str(value or fallback).strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("._") or fallback


def decode_image_from_request() -> tuple[bytes, str | None, str | None]:
    if request.files:
        uploaded_file = request.files.get("image") or next(iter(request.files.values()))
        return uploaded_file.read(), request.form.get("id"), request.form.get("created_at")

    data = request.get_json(silent=True)
    if not data:
        raise ValueError("No JSON or multipart form data received")

    image_base64 = data.get("image")
    if not image_base64:
        raise ValueError("No image data received")

    if "," in image_base64 and image_base64.lstrip().startswith("data:"):
        image_base64 = image_base64.split(",", 1)[1]
    image_base64 = re.sub(r"\s+", "", image_base64)

    try:
        image_bytes = base64.b64decode(image_base64, validate=True)
    except binascii.Error as exc:
        raise ValueError("Image is not valid base64") from exc

    return image_bytes, data.get("id"), data.get("created_at")


def build_pipeline_args() -> SimpleNamespace:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return SimpleNamespace(
        conf=DEFAULT_CONF,
        iou=DEFAULT_IOU,
        imgsz=DEFAULT_IMGSZ,
        device=device,
        padding=DEFAULT_PADDING,
        crop_mode=DEFAULT_CROP_MODE,
        no_pad_square=False,
        skip_occluded=False,
        save_crops=False,
    )


PIPELINE_ARGS = build_pipeline_args()
DISEASE_DEVICE = torch.device(
    "cuda" if PIPELINE_ARGS.device != "cpu" and torch.cuda.is_available() else "cpu"
)

print(f"\nLoading segmentation model: {DEFAULT_SEG_MODEL}")
SEG_MODEL = YOLO(str(DEFAULT_SEG_MODEL))

DISEASE_MODEL_PATH = find_latest_disease_model()
print(f"Loading disease CNN model: {DISEASE_MODEL_PATH}")
DISEASE_MODEL, DISEASE_CLASSES, DISEASE_IMAGE_SIZE = load_disease_model(
    DISEASE_MODEL_PATH,
    DISEASE_DEVICE,
)
DISEASE_TRANSFORM = build_disease_transform(DISEASE_IMAGE_SIZE)
print(f"Model ready on {PIPELINE_ARGS.device}. Classes: {', '.join(DISEASE_CLASSES)}")


def save_request_image(image_bytes: bytes, image_id: str | None) -> Path:
    safe_id = clean_filename_part(image_id, "no_id")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    image_path = UPLOAD_FOLDER / f"mango_{safe_id}_{timestamp}{IMAGE_SUFFIX}"
    image_path.write_bytes(image_bytes)
    return image_path


def result_url(path: Path | str) -> str:
    result_path = Path(path)
    return f"/api/results/{result_path.name}"


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(row.get("disease_label", "unknown")) for row in rows))


def find_result_metadata(image_id: str | None = None) -> dict[str, Any] | None:
    metadata_files = sorted(
        RESULTS_FOLDER.glob("mango_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    for metadata_file in metadata_files:
        try:
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        if image_id is not None and str(metadata.get("id")) != str(image_id):
            continue

        annotated_image = Path(str(metadata.get("annotated_image", "")))
        if annotated_image.exists():
            metadata["metadata_file"] = str(metadata_file)
            return metadata

    return None


def encode_result_image(path: Path, max_size: int = 800, quality: int = 65) -> io.BytesIO:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((max_size, max_size))

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
        buffer.seek(0)
        return buffer


def build_image_result_response(
    image_id: str | None = None,
    extra_data: dict[str, Any] | None = None,
):
    metadata = find_result_metadata(image_id)

    if metadata is None:
        message = "No processed result image found"
        if image_id:
            message = f"No processed result image found for id {image_id}"
        return jsonify({"success": False, "error": message}), 404

    annotated_path = Path(metadata["annotated_image"])
    image_buffer = encode_result_image(annotated_path)

    response = send_file(
        image_buffer,
        mimetype="image/jpeg",
        as_attachment=False,
        download_name=annotated_path.name,
    )
    response.headers["X-Success"] = "true"
    response.headers["X-Image-Id"] = str(metadata.get("id", ""))
    response.headers["X-Result-Filename"] = annotated_path.name

    if extra_data:
        for key, value in extra_data.items():
            if isinstance(value, (str, int, float, bool)):
                header_name = "X-" + key.replace("_", "-").title()
                response.headers[header_name] = str(value)

    return response


@app.route("/api/upload", methods=["POST"])
def upload_image():
    try:
        image_bytes, image_id, created_at = decode_image_from_request()
        if not image_bytes:
            return jsonify({"error": "Image data is empty"}), 400

        image_path = save_request_image(image_bytes, image_id)
        annotated_path = RESULTS_FOLDER / f"{image_path.stem}_annotated.jpg"

        print("\n" + "=" * 50)
        print("Received mango image")
        print(f"  ID: {image_id}")
        print(f"  Created at: {created_at}")
        print(f"  Uploaded image: {image_path}")
        print("  Running mango segmentation + CNN disease classification...")

        rows = process_image(
            image_path=image_path,
            yolo_model=SEG_MODEL,
            disease_model=DISEASE_MODEL,
            disease_classes=DISEASE_CLASSES,
            disease_transform=DISEASE_TRANSFORM,
            device=DISEASE_DEVICE,
            args=PIPELINE_ARGS,
            crops_dir=CROPS_FOLDER,
            annotated_dir=RESULTS_FOLDER,
        )

        for row in rows:
            row["request_id"] = image_id
            row["created_at"] = created_at

        metadata_path = RESULTS_FOLDER / f"{image_path.stem}.json"
        metadata = {
            "id": image_id,
            "created_at": created_at,
            "uploaded_image": str(image_path),
            "annotated_image": str(annotated_path),
            "detections": len(rows),
            "disease_counts": summarize_rows(rows),
            "predictions": rows,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        print(f"  Detections: {len(rows)}")
        print(f"  Annotated result: {annotated_path}")
        print("=" * 50 + "\n")

        return jsonify(
            {
                "success": True,
                "message": "Image uploaded and classified successfully",
                "id": image_id,
                "created_at": created_at,
                "uploaded_filename": image_path.name,
                "result_filename": annotated_path.name,
                "result_url": result_url(annotated_path),
                "image_url": f"/image?id={image_id}",
                "metadata_filename": metadata_path.name,
                "detections": len(rows),
                "disease_counts": metadata["disease_counts"],
            }
        ), 200

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        print(f"Error while processing upload: {exc}")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/results/<path:filename>", methods=["GET"])
def get_result(filename: str):
    return send_from_directory(RESULTS_FOLDER, filename)


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify(
        {
            "status": "ok",
            "segmentation_model": str(DEFAULT_SEG_MODEL),
            "disease_model": str(DISEASE_MODEL_PATH),
            "device": PIPELINE_ARGS.device,
        }
    ), 200


@app.route("/image")
def send_image():
    image_id = request.args.get("id")
    return build_image_result_response(image_id)


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("Flask Server for Mango Disease Classification")
    print("=" * 50)
    print("Upload endpoint: http://0.0.0.0:5000/api/upload")
    print("Health check:    http://0.0.0.0:5000/health")
    print(f"Images saved to: {UPLOAD_FOLDER}")
    print(f"Results saved to: {RESULTS_FOLDER}")
    print("=" * 50 + "\n")

    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
