import os
import glob
import csv
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))

IMAGES_DIR = "Mango-Detection-5/train/images"
LABELS_DIR = "Mango-Detection-5/train/labels"
OUT_DIR = "Mango-Detection-5/mango_crops_train_by_occlusion"

PADDING = 10
KEEP_TRANSPARENT_BG = True
MAX_SIZE = None

# ======================================================
# EXPERIMENTAL QUALITY THRESHOLDS
# ======================================================
MIN_CROP_SIDE = 40
MIN_MANGO_PIXELS = 800

MAX_OCCLUDED_RATIO = 0.80      # reject only if > 80%
MAX_SHADOW_RATIO = 0.60        # reject if >= 60%

SHADOW_PERCENTILE = 25         # adaptive threshold from mango luminance
MIN_SHADOW_COMPONENT_AREA = 0.05  # only keep larger dark regions

os.makedirs(OUT_DIR, exist_ok=True)

OCCLUSION_DIRS = {
    "0_10": os.path.join(OUT_DIR, "0_10"),
    "10_40": os.path.join(OUT_DIR, "10_40"),
    "40_80": os.path.join(OUT_DIR, "40_80"),
    "80_above": os.path.join(OUT_DIR, "80_above"),
}

for folder in OCCLUSION_DIRS.values():
    os.makedirs(folder, exist_ok=True)

CSV_PATH = os.path.join(OUT_DIR, "crop_quality_metrics_experiment_simple.csv")


def parse_poly_line(line, img_w, img_h):
    """
    YOLOv8-seg polygon format:
    class x1 y1 x2 y2 ... xn yn (normalized)
    """
    parts = line.strip().split()
    cls = int(float(parts[0]))
    coords = list(map(float, parts[1:]))

    if len(coords) < 6 or len(coords) % 2 != 0:
        return cls, None

    pts = []
    for i in range(0, len(coords), 2):
        x = int(round(coords[i] * img_w))
        y = int(round(coords[i + 1] * img_h))
        x = max(0, min(img_w - 1, x))
        y = max(0, min(img_h - 1, y))
        pts.append([x, y])

    return cls, np.array(pts, dtype=np.int32)


def polygon_to_mask(poly, w, h):
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    return mask


def estimate_occlusion_ratio_from_mask(crop_mask):
    """
    Estimate how much of the mango is missing by comparing the visible mask
    area to an ellipse fitted to the mango contour. This is more faithful than
    comparing against the crop rectangle, which overestimates occlusion for
    naturally oval fruit shapes.
    """
    contour_points = cv2.findNonZero(crop_mask)
    visible_area = int(np.count_nonzero(crop_mask))

    if contour_points is None or visible_area == 0:
        return 1.0, visible_area, 0.0

    contour_points = contour_points.reshape(-1, 2)

    if len(contour_points) >= 5:
        ellipse = cv2.fitEllipse(contour_points)
        (_, _), (axis_w, axis_h), _ = ellipse
        estimated_full_area = float(np.pi * (axis_w * 0.5) * (axis_h * 0.5))
    else:
        x, y, w, h = cv2.boundingRect(contour_points)
        estimated_full_area = float(np.pi * (w * 0.5) * (h * 0.5))

    estimated_full_area = max(float(visible_area), estimated_full_area)
    visible_ratio = min(1.0, visible_area / estimated_full_area)
    occluded_ratio = max(0.0, 1.0 - visible_ratio)

    return occluded_ratio, visible_area, estimated_full_area


def compute_shape_completeness_metrics(crop_mask):
    """
    Extra shape checks to keep obviously cut fragments out of the 0%-10% bin.
    """
    contour_points = cv2.findNonZero(crop_mask)
    if contour_points is None:
        return 0.0, 0.0, 4, 0.0

    contour = cv2.convexHull(contour_points)
    hull_area = float(cv2.contourArea(contour))
    visible_area = float(np.count_nonzero(crop_mask))
    bbox_h, bbox_w = crop_mask.shape[:2]
    bbox_area = float(bbox_h * bbox_w) if bbox_h > 0 and bbox_w > 0 else 0.0

    solidity = visible_area / hull_area if hull_area > 0 else 0.0
    extent = visible_area / bbox_area if bbox_area > 0 else 0.0
    aspect_ratio = max(bbox_w, bbox_h) / max(1.0, min(bbox_w, bbox_h))

    touches = 0
    if np.any(crop_mask[0, :] > 0):
        touches += 1
    if np.any(crop_mask[-1, :] > 0):
        touches += 1
    if np.any(crop_mask[:, 0] > 0):
        touches += 1
    if np.any(crop_mask[:, -1] > 0):
        touches += 1

    return solidity, extent, touches, aspect_ratio


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
    if max(h, w) <= max_size:
        return img

    scale = max_size / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def compute_shadow_ratio_lab(crop_bgr, crop_mask, shadow_percentile=25):
    """
    Compute shadow ratio only inside the mango region using LAB L channel.

    This simplified experimental version:
    - uses blurred luminance
    - uses adaptive threshold
    - removes small dark regions
    - returns shadow_ratio only + supporting values for inspection
    """
    mango_region = crop_mask > 0
    mango_pixels_count = int(np.count_nonzero(mango_region))

    if mango_pixels_count == 0:
        return 1.0, 0.0, 0.0, 0.0

    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)

    # Blur to emphasize broad shadows and suppress tiny dark disease spots
    L_blur = cv2.GaussianBlur(L, (21, 21), 0)

    L_fruit = L_blur[mango_region]
    mean_L_fruit = float(np.mean(L_fruit))
    std_L_fruit = float(np.std(L_fruit))

    adaptive_threshold = float(np.percentile(L_fruit, shadow_percentile))

    dark_mask = np.zeros_like(crop_mask, dtype=np.uint8)
    dark_mask[(L_blur < adaptive_threshold) & mango_region] = 255

    # Keep only larger dark connected regions
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        dark_mask, connectivity=8
    )
    filtered_dark_mask = np.zeros_like(dark_mask)

    min_component_pixels = max(1, int(mango_pixels_count * MIN_SHADOW_COMPONENT_AREA))

    for label_idx in range(1, num_labels):
        area = stats[label_idx, cv2.CC_STAT_AREA]
        if area >= min_component_pixels:
            filtered_dark_mask[labels == label_idx] = 255

    shadow_pixels = int(np.count_nonzero(filtered_dark_mask))
    shadow_ratio = shadow_pixels / mango_pixels_count

    return shadow_ratio, mean_L_fruit, std_L_fruit, adaptive_threshold


def classify_quality(visible_ratio, shadow_ratio, crop_w, crop_h, mango_area):
    occluded_ratio = 1.0 - visible_ratio

    # Reject tiny crops or tiny visible fragments
    if min(crop_w, crop_h) < MIN_CROP_SIDE or mango_area < MIN_MANGO_PIXELS:
        return "REJECT"

    # Reject only if occlusion is greater than 80%
    if occluded_ratio > MAX_OCCLUDED_RATIO:
        return "REJECT"

    # Reject if shadow is 60% or more
    if shadow_ratio >= MAX_SHADOW_RATIO:
        return "REJECT"

    return "ACCEPT"


def get_occlusion_bucket(
    occluded_ratio,
    crop_w,
    crop_h,
    mango_area,
    solidity,
    extent,
    border_touches,
    aspect_ratio
):
    """
    Occlusion bands based on the reference:
    - minimal / nearly fully visible: 0%-10%
    - low occlusion: 10%-40%
    - moderate occlusion: 40%-80%
    - strong occlusion: >80%
    """
    if min(crop_w, crop_h) < MIN_CROP_SIDE or mango_area < MIN_MANGO_PIXELS:
        return "80_above"
    if (
        occluded_ratio < 0.05
        and solidity >= 0.94
        and extent >= 0.58
        and border_touches == 0
        and aspect_ratio <= 2.35
    ):
        return "0_10"
    if occluded_ratio < 0.40:
        return "10_40"
    if occluded_ratio < 0.80:
        return "40_80"
    return "80_above"


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

        occluded_ratio, mango_area, estimated_full_area = estimate_occlusion_ratio_from_mask(crop_mask)
        solidity, extent, border_touches, aspect_ratio = compute_shape_completeness_metrics(crop_mask)
        bbox_area = int(crop_mask.shape[0] * crop_mask.shape[1])

        visible_ratio = 1.0 - occluded_ratio

        shadow_ratio, mean_L_fruit, std_L_fruit, adaptive_threshold = compute_shadow_ratio_lab(
            crop,
            crop_mask,
            shadow_percentile=SHADOW_PERCENTILE
        )

        crop_h, crop_w = crop.shape[:2]
        quality = classify_quality(
            visible_ratio,
            shadow_ratio,
            crop_w,
            crop_h,
            mango_area
        )
        occlusion_bucket = get_occlusion_bucket(
            occluded_ratio,
            crop_w,
            crop_h,
            mango_area,
            solidity,
            extent,
            border_touches,
            aspect_ratio
        )

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
        out_root = OCCLUSION_DIRS[occlusion_bucket]
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
            round(estimated_full_area, 2),
            round(visible_ratio, 4),
            round(occluded_ratio, 4),
            round(occluded_ratio * 100.0, 2),
            round(solidity, 4),
            round(extent, 4),
            border_touches,
            round(aspect_ratio, 4),
            occlusion_bucket,
            round(shadow_ratio, 4),
            round(mean_L_fruit, 2),
            round(std_L_fruit, 2),
            round(adaptive_threshold, 2),
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
        "estimated_full_area",
        "visible_ratio",
        "occluded_ratio",
        "occluded_percent",
        "solidity",
        "extent",
        "border_touches",
        "aspect_ratio",
        "occlusion_bucket",
        "shadow_ratio",
        "mean_L_fruit",
        "std_L_fruit",
        "adaptive_threshold",
        "quality"
    ])
    writer.writerows(rows)

print(f"Done. Saved {saved} mango crops.")
for bucket_name, folder_path in OCCLUSION_DIRS.items():
    print(f"{bucket_name} folder: {folder_path}")
print(f"Saved metrics CSV: {CSV_PATH}")
