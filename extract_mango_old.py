import os
import glob
import csv
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))

IMAGES_DIR = "Mango-Detection-5/train/images"
LABELS_DIR = "Mango-Detection-5/train/labels"
OUT_DIR = "Mango-Detection-5/mango_crops_train"
REJECT_DIR = "Mango-Detection-5/mango_crops_train_reject"

PADDING = 10
KEEP_TRANSPARENT_BG = True
MAX_SIZE = None

# quality thresholds focused on visibility and shadow
SHADOW_THRESHOLD = 60
MAX_OCCLUDED_RATIO = 0.80
MAX_SHADOW_RATIO = 0.60
MIN_CROP_SIDE = 20
MIN_MANGO_PIXELS = 150
MIN_SHADOW_COMPONENT_AREA = 0.03

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(REJECT_DIR, exist_ok=True)

CSV_PATH = os.path.join(OUT_DIR, "crop_quality_metrics.csv")


def parse_poly_line(line, img_w, img_h):
    parts = line.strip().split()
    cls = int(float(parts[0]))
    coords = list(map(float, parts[1:]))

    if len(coords) < 6 or len(coords) % 2 != 0:
        return cls, None

    pts = []
    for i in range(0, len(coords), 2):
        x = int(round(coords[i] * img_w))
        y = int(round(coords[i + 1] * img_h))
        pts.append([x, y])

    return cls, np.array(pts, dtype=np.int32)


def polygon_to_mask(poly, w, h):
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    return mask


def pad_to_square(img, border_val):
    h, w = img.shape[:2]
    side = max(h, w)
    top = (side - h) // 2
    bottom = side - h - top
    left = (side - w) // 2
    right = side - w - left
    return cv2.copyMakeBorder(
        img, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=border_val
    )


def downscale_only(img, max_size):
    if max_size is None:
        return img
    h, w = img.shape[:2]
    if h <= max_size:
        return img
    return cv2.resize(img, (max_size, max_size), interpolation=cv2.INTER_AREA)


def compute_shadow_ratio(crop_bgr, crop_mask, shadow_threshold=60):
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    mango_region = crop_mask > 0
    mango_pixels = gray[mango_region]

    if len(mango_pixels) == 0:
        return 1.0, 0.0

    dark_mask = np.zeros_like(gray, dtype=np.uint8)
    dark_mask[(gray < shadow_threshold) & mango_region] = 255

    # Remove tiny isolated dark areas so disease spots or speckles are not
    # counted as dominant shadow. We only keep larger connected dark regions.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)
    filtered_dark_mask = np.zeros_like(dark_mask)
    min_component_pixels = max(1, int(len(mango_pixels) * MIN_SHADOW_COMPONENT_AREA))

    for label_idx in range(1, num_labels):
        area = stats[label_idx, cv2.CC_STAT_AREA]
        if area >= min_component_pixels:
            filtered_dark_mask[labels == label_idx] = 255

    shadow_pixels = np.count_nonzero(filtered_dark_mask)
    shadow_ratio = shadow_pixels / len(mango_pixels)
    mean_intensity = float(np.mean(mango_pixels))
    return shadow_ratio, mean_intensity


def classify_quality(visible_ratio, shadow_ratio, crop_w, crop_h, mango_area):
    occluded_ratio = 1.0 - visible_ratio

    # Tiny fragments are not useful for classification.
    if min(crop_w, crop_h) < MIN_CROP_SIDE or mango_area < MIN_MANGO_PIXELS:
        return "REJECT"

    # Reject crops when more than 80% is effectively occluded
    # or when shadow dominates more than 60% of the visible mango.
    if occluded_ratio > MAX_OCCLUDED_RATIO or shadow_ratio > MAX_SHADOW_RATIO:
        return "REJECT"

    return "ACCEPT"


saved = 0
rows = []

img_paths = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.*")))

for img_path in img_paths:
    base = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(LABELS_DIR, base + ".txt")

    if not os.path.exists(label_path):
        continue

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        continue

    h, w = img.shape[:2]

    with open(label_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    inst = 0
    for ln in lines:
        cls, poly = parse_poly_line(ln, w, h)
        if poly is None:
            continue

        mask = polygon_to_mask(poly, w, h)

        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            continue

        x1, x2 = xs.min(), xs.max() + 1
        y1, y2 = ys.min(), ys.max() + 1

        x1 = max(0, x1 - PADDING)
        y1 = max(0, y1 - PADDING)
        x2 = min(w, x2 + PADDING)
        y2 = min(h, y2 + PADDING)

        crop = img[y1:y2, x1:x2]
        crop_mask = mask[y1:y2, x1:x2]

        mango_area = int(np.count_nonzero(crop_mask))
        bbox_area = int(crop_mask.shape[0] * crop_mask.shape[1])
        visible_ratio = mango_area / bbox_area if bbox_area > 0 else 0.0
        occluded_ratio = 1.0 - visible_ratio

        shadow_ratio, mean_intensity = compute_shadow_ratio(
            crop, crop_mask, SHADOW_THRESHOLD
        )

        crop_h, crop_w = crop.shape[:2]
        quality = classify_quality(visible_ratio, shadow_ratio, crop_w, crop_h, mango_area)

        if KEEP_TRANSPARENT_BG:
            crop_out = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
            crop_out[:, :, 3] = crop_mask
            border_val = (0, 0, 0, 0)
            ext = "png"
        else:
            white_bg = np.ones_like(crop, dtype=np.uint8) * 255
            crop_out = np.where(crop_mask[..., None] > 0, crop, white_bg)
            border_val = (255, 255, 255)
            ext = "jpg"

        crop_sq = pad_to_square(crop_out, border_val)
        out_img = downscale_only(crop_sq, MAX_SIZE)

        out_name = f"{base}_cls{cls}_id{inst:03d}.{ext}"
        out_root = REJECT_DIR if quality == "REJECT" else OUT_DIR
        out_path = os.path.join(out_root, out_name)
        cv2.imwrite(out_path, out_img)

        rows.append([
            out_name,
            base,
            cls,
            inst,
            crop_w,
            crop_h,
            mango_area,
            bbox_area,
            round(visible_ratio, 4),
            round(occluded_ratio, 4),
            round(shadow_ratio, 4),
            round(mean_intensity, 2),
            quality
        ])

        inst += 1
        saved += 1

with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "crop_name",
        "source_image",
        "class_id",
        "instance_id",
        "crop_width",
        "crop_height",
        "mango_area",
        "bbox_area",
        "visible_ratio",
        "occluded_ratio",
        "shadow_ratio",
        "mean_intensity",
        "quality"
    ])
    writer.writerows(rows)

print(f"Done. Saved {saved} mango crops into: {OUT_DIR}")
print(f"Saved metrics CSV: {CSV_PATH}")
