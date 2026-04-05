from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO


# ====== EDIT THESE PATHS MANUALLY ======
# Path to your trained weights (.pt)
WEIGHTS_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "mango_seg_fresh3"
    / "weights"
    / "best.pt"
)

# Path to the input image you want to run on
IMAGE_PATH = Path(
    r"C:\laragon\www\computer_vision\Mango-Detection-4\based_path\image6.jpg"
)

# Folder + filename for saving the output image
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_IMAGE_NAME = "mango_result.jpg"

# If False, skip cv2.imshow (useful for headless OpenCV environments)
SHOW_WINDOW = False

# Confidence threshold for detections (lower to pick up more mangoes)
CONF_THRESHOLD = 0.6

# NMS IoU threshold (higher = keep more overlapping detections)
IOU_THRESHOLD = 0.8

# Max detections per image
MAX_DET = 300

# If True, apply class-agnostic NMS (can help when classes overlap)
AGNOSTIC_NMS = False

# Detection mode:
# - "non_occluded_only": show only class 0 (mango)
# - "mango_and_occluded": show class 0 (green) and class 1 (red)
# - "merge_occluded_to_mango": show both classes, but draw both as mango (green)
DETECTION_MODE = "mango_and_occluded"
# ======================================


def load_model(weights_path: str) -> YOLO:
    weights = Path(weights_path)
    if not weights.exists():
        raise FileNotFoundError(
            f"Could not find weights at '{weights}'. "
            "Make sure best.pt exists under Mango-Detection-3/models/mango_seg/weights "
            "or pass --weights with the correct path."
        )
    return YOLO(str(weights))


def visualize_segmentation(frame: np.ndarray, result, alpha: float = 0.5) -> np.ndarray:
    """
    Draw segmentation masks and boxes.
    Dataset classes:
    - class 0: mango
    - class 1: occluded
    """
    overlay = frame.copy()

    masks = getattr(result, "masks", None)
    boxes = getattr(result, "boxes", None)
    names = getattr(result, "names", {})

    if masks is not None and masks.data is not None and boxes is not None:
        for idx, mask in enumerate(masks.data):
            # Convert to numpy uint8 mask
            m = mask.cpu().numpy().astype(np.uint8)
            m = (m * 255).astype(np.uint8)

            # Resize mask to match the frame size (H, W)
            mask_resized = cv2.resize(
                m,
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

            cls_id = int(boxes.cls[idx].item())

            if DETECTION_MODE == "non_occluded_only":
                if cls_id != 0:
                    continue
                cls_name = "mango"
                color = (0, 255, 0)
            elif DETECTION_MODE == "mango_and_occluded":
                if cls_id not in (0, 1):
                    continue
                if cls_id == 0:
                    cls_name = "mango"
                    color = (0, 255, 0)
                else:
                    cls_name = "occluded"
                    color = (0, 0, 255)
            elif DETECTION_MODE == "merge_occluded_to_mango":
                if cls_id not in (0, 1):
                    continue
                cls_name = "mango"
                color = (0, 255, 0)
            else:
                raise ValueError(
                    "Invalid DETECTION_MODE. Use: "
                    "'non_occluded_only', 'mango_and_occluded', or 'merge_occluded_to_mango'."
                )

            colored_mask = np.zeros_like(frame, dtype=np.uint8)
            colored_mask[mask_resized == 255] = color

            overlay = cv2.addWeighted(overlay, 1.0, colored_mask, alpha, 0)

            b = boxes.xyxy[idx].cpu().numpy().astype(int)
            x1, y1, x2, y2 = b.tolist()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

            conf = float(boxes.conf[idx].item()) if hasattr(boxes, "conf") else 0.0
            label = f"{cls_name} {conf:.2f}"
            cv2.putText(
                overlay,
                label,
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

    return overlay


def run_on_image(weights: Path, source: Path, conf: float, save_path: Optional[Path]) -> None:
    model = load_model(weights)
    model.overrides["task"] = "segment"

    if not source.exists():
        raise FileNotFoundError(f"Input image not found at '{source}'.")

    frame = cv2.imread(str(source))
    if frame is None:
        raise RuntimeError(f"Failed to read image from '{source}'.")

    if DETECTION_MODE == "non_occluded_only":
        classes = [0]
    else:
        classes = [0, 1]

    results = model.predict(
        source=frame,
        conf=conf,
        iou=IOU_THRESHOLD,
        max_det=MAX_DET,
        agnostic_nms=AGNOSTIC_NMS,
        classes=classes,
        verbose=False,
    )

    if results and results[0].boxes is not None and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        cls_ids = boxes.cls.cpu().numpy().astype(int).tolist()
        mango_count = sum(1 for c in cls_ids if c == 0)
        occluded_count = sum(1 for c in cls_ids if c == 1)
        print(
            f"Detections: total={len(cls_ids)} | mango={mango_count} | occluded={occluded_count}"
        )
    else:
        print(f"No detections found at conf={conf}. Try lowering CONF_THRESHOLD.")

    if results:
        vis_frame = visualize_segmentation(frame, results[0])
    else:
        vis_frame = frame

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), vis_frame)
        print(f"Saved output image to: {save_path}")

    if SHOW_WINDOW:
        window_name = "Mango Segmentation (image)"
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, vis_frame)
            # Wait for a key press before closing the window
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            print("OpenCV GUI is not available. Set SHOW_WINDOW=False to suppress this message.")


def main() -> None:
    # Build final save path from constants
    save_path = OUTPUT_DIR / OUTPUT_IMAGE_NAME if OUTPUT_IMAGE_NAME else None

    run_on_image(
        weights=WEIGHTS_PATH,
        source=IMAGE_PATH,
        conf=CONF_THRESHOLD,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()
