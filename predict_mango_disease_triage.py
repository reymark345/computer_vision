import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


DISEASE_LABELS = {
    "anthracnose": "Anthracnose",
    "mango_scab": "Mango Scab",
    "stem_end_rot": "Stem end rot",
    "uncertain_or_healthy": "Uncertain / healthy-looking",
}


def image_files(folder):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts)


def resize_for_analysis(img, max_side=640):
    h, w = img.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return img

    scale = max_side / side
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)


def largest_component(mask):
    mask_u8 = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
    if num_labels <= 1:
        return mask

    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return labels == largest


def infer_fruit_mask(img):
    if img.shape[2] == 4:
        alpha = img[:, :, 3]
        mask = alpha > 20
    else:
        bgr = img[:, :, :3]
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Works for both black/transparent-style crop backgrounds and normal photos.
        non_black = gray > 10
        yellow_green_fruit = (
            (hsv[:, :, 1] > 20)
            & (hsv[:, :, 2] > 45)
            & (hsv[:, :, 0] >= 10)
            & (hsv[:, :, 0] <= 95)
        )
        mask = non_black & yellow_green_fruit

    mask = mask.astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return largest_component(mask > 0)


def component_stats(component_mask, fruit_mask):
    fruit_area = max(1, int(np.count_nonzero(fruit_mask)))
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        component_mask.astype(np.uint8), 8
    )

    comps = []
    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[idx]
        comps.append(
            {
                "label": idx,
                "area": area,
                "area_ratio": area / fruit_area,
                "bbox": (x, y, w, h),
                "centroid": (float(cx), float(cy)),
            }
        )
    return comps


def principal_axis_tipness(fruit_mask, point):
    ys, xs = np.where(fruit_mask)
    if len(xs) < 10:
        return 0.0

    coords = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    center = coords.mean(axis=0)
    centered = coords - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    projections = centered @ axis
    half_span = max(abs(float(projections.min())), abs(float(projections.max())), 1.0)
    point_projection = (np.array(point, dtype=np.float32) - center) @ axis
    return min(1.0, abs(float(point_projection)) / half_span)


def principal_axis_component_tip_metrics(fruit_mask, component_mask):
    ys, xs = np.where(fruit_mask)
    comp_ys, comp_xs = np.where(component_mask)
    if len(xs) < 10 or len(comp_xs) == 0:
        return 0.0, 0.0

    fruit_coords = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    center = fruit_coords.mean(axis=0)
    centered = fruit_coords - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    fruit_projections = centered @ axis
    half_span = max(abs(float(fruit_projections.min())), abs(float(fruit_projections.max())), 1.0)

    comp_coords = np.column_stack([comp_xs.astype(np.float32), comp_ys.astype(np.float32)])
    comp_projections = (comp_coords - center) @ axis
    centroid_tipness = min(1.0, abs(float(np.mean(comp_projections))) / half_span)
    tip_overlap = min(1.0, float(np.percentile(np.abs(comp_projections), 95)) / half_span)
    return centroid_tipness, tip_overlap


def analyze_image(path):
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f"Could not read image: {path}")

    img = resize_for_analysis(raw)
    bgr = img[:, :, :3]
    fruit_mask = infer_fruit_mask(img)
    fruit_area = int(np.count_nonzero(fruit_mask))

    if fruit_area < 500:
        return {
            "label_key": "uncertain_or_healthy",
            "confidence": 0.0,
            "reason": "fruit mask too small",
            "scores": {"anthracnose": 0.0, "mango_scab": 0.0, "stem_end_rot": 0.0},
            "features": {},
        }

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    v = hsv[:, :, 2].astype(np.uint8)
    s = hsv[:, :, 1].astype(np.uint8)
    h = hsv[:, :, 0].astype(np.uint8)
    l_chan = lab[:, :, 0].astype(np.uint8)

    fruit_v = v[fruit_mask]
    fruit_l = l_chan[fruit_mask]
    median_v = float(np.median(fruit_v))
    median_l = float(np.median(fruit_l))

    # Local contrast catches spot-like lesions while ignoring broad lighting gradients.
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    large_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (43, 43))
    blackhat_small = cv2.morphologyEx(l_chan, cv2.MORPH_BLACKHAT, small_kernel)
    blackhat_large = cv2.morphologyEx(l_chan, cv2.MORPH_BLACKHAT, large_kernel)

    local_dark = (
        fruit_mask
        & (blackhat_small > 9)
        & (v < median_v - 8)
        & ((s > 18) | (v < median_v - 35))
    )
    deep_dark = fruit_mask & (v < min(120, median_v - 45)) & (l_chan < median_l - 35)
    broad_dark = fruit_mask & (blackhat_large > 13) & (v < median_v - 15)

    # Corky scab is usually brown/tan/black, small, and locally darker than the peel.
    brown_hue = ((h >= 3) & (h <= 32)) | ((h >= 165) & (h <= 179))
    brown_spots = fruit_mask & brown_hue & (s > 28) & (v < median_v - 10) & (
        (blackhat_small > 6) | (blackhat_large > 10)
    )

    lesion_mask = local_dark | deep_dark | broad_dark | brown_spots
    lesion_mask = cv2.morphologyEx(
        lesion_mask.astype(np.uint8),
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    ).astype(bool)

    dark_comps = component_stats(local_dark | deep_dark | broad_dark, fruit_mask)
    brown_comps = component_stats(brown_spots, fruit_mask)
    lesion_comps = component_stats(lesion_mask, fruit_mask)

    dark_area_ratio = float(np.count_nonzero(local_dark | deep_dark | broad_dark) / fruit_area)
    brown_area_ratio = float(np.count_nonzero(brown_spots) / fruit_area)
    lesion_area_ratio = float(np.count_nonzero(lesion_mask) / fruit_area)

    min_small = max(6, int(fruit_area * 0.00003))
    max_small = max(min_small + 1, int(fruit_area * 0.0012))
    max_medium = max(max_small + 1, int(fruit_area * 0.012))

    small_dark = [c for c in dark_comps if min_small <= c["area"] <= max_small]
    medium_dark = [c for c in dark_comps if max_small < c["area"] <= max_medium]
    small_brown = [c for c in brown_comps if min_small <= c["area"] <= max_small]
    largest = max(lesion_comps, key=lambda c: c["area"], default=None)
    largest_ratio = largest["area_ratio"] if largest else 0.0

    fruit_mask_u8 = fruit_mask.astype(np.uint8)
    dist_to_edge = cv2.distanceTransform(fruit_mask_u8, cv2.DIST_L2, 5)
    equivalent_radius = max(1.0, (fruit_area / np.pi) ** 0.5)
    stem_like_largest = 0.0
    edge_norm = 1.0
    tipness = 0.0
    tip_overlap = 0.0
    stem_end_candidate = False

    if largest:
        x, y, w, h_box = largest["bbox"]
        comp_slice = (slice(y, y + h_box), slice(x, x + w))
        comp_mask = lesion_mask[comp_slice]
        comp_dist = dist_to_edge[comp_slice][comp_mask]
        edge_norm = float(np.median(comp_dist) / equivalent_radius) if len(comp_dist) else 1.0
        lesion_labels = cv2.connectedComponentsWithStats(lesion_mask.astype(np.uint8), 8)[1]
        largest_component_mask = lesion_labels == largest["label"]
        tipness, tip_overlap = principal_axis_component_tip_metrics(
            fruit_mask,
            largest_component_mask,
        )
        stem_like_largest = largest_ratio * (1.0 + max(0.0, 0.85 - edge_norm) + tipness)
        stem_end_candidate = (
            largest_ratio >= 0.018
            and edge_norm <= 0.22
            and tipness >= 0.55
            and tip_overlap >= 0.72
        ) or (
            largest_ratio >= 0.05
            and edge_norm <= 0.18
            and tipness >= 0.48
            and tip_overlap >= 0.82
        )

    spot_density = (len(small_dark) + len(medium_dark) * 2) / max(1.0, fruit_area / 10000.0)
    scab_density = len(small_brown) / max(1.0, fruit_area / 10000.0)

    anthracnose_score = (
        min(1.0, dark_area_ratio / 0.045) * 0.55
        + min(1.0, len(medium_dark) / 8.0) * 0.25
        + min(1.0, spot_density / 10.0) * 0.20
    )
    scab_score = (
        min(1.0, brown_area_ratio / 0.025) * 0.45
        + min(1.0, scab_density / 14.0) * 0.40
        + min(1.0, len(small_brown) / 30.0) * 0.15
    )
    stem_score_raw = (
        min(1.0, stem_like_largest / 0.055) * 0.55
        + (0.25 if largest_ratio >= 0.018 and edge_norm <= 0.24 and tipness >= 0.55 else 0.0)
        + (0.20 if largest_ratio >= 0.012 and tip_overlap >= 0.74 else 0.0)
    )
    stem_score = stem_score_raw if stem_end_candidate else min(stem_score_raw, 0.34)

    scores = {
        "anthracnose": float(anthracnose_score),
        "mango_scab": float(scab_score),
        "stem_end_rot": float(stem_score),
    }

    label_key = max(scores, key=scores.get)
    confidence = min(1.0, scores[label_key])

    # Keep low-signal fruit out of disease buckets. This is a triage aid, not a diagnosis.
    if confidence < 0.38 or lesion_area_ratio < 0.001:
        label_key = "uncertain_or_healthy"
        confidence = 1.0 - min(1.0, lesion_area_ratio / 0.015)

    if label_key == "stem_end_rot":
        reason = "largest dark lesion is near a fruit pole/end and border"
    elif label_key == "mango_scab":
        reason = "many small brown/corky spot candidates"
    elif label_key == "anthracnose":
        reason = "dark spot/blotch candidates dominate"
    else:
        reason = "no strong visual signal for the three target diseases"

    return {
        "label_key": label_key,
        "confidence": round(float(confidence), 4),
        "reason": reason,
        "scores": {k: round(v, 4) for k, v in scores.items()},
        "features": {
            "fruit_area": fruit_area,
            "lesion_area_ratio": round(lesion_area_ratio, 5),
            "dark_area_ratio": round(dark_area_ratio, 5),
            "brown_area_ratio": round(brown_area_ratio, 5),
            "small_dark_count": len(small_dark),
            "medium_dark_count": len(medium_dark),
            "small_brown_count": len(small_brown),
            "largest_lesion_ratio": round(float(largest_ratio), 5),
            "largest_edge_norm": round(float(edge_norm), 5),
            "largest_tipness": round(float(tipness), 5),
            "largest_tip_overlap": round(float(tip_overlap), 5),
            "stem_end_candidate": int(stem_end_candidate),
        },
    }


def run(base_dir, buckets, output_csv, output_json):
    rows = []
    totals = Counter()
    bucket_totals = defaultdict(Counter)

    for bucket in buckets:
        folder = base_dir / bucket
        if not folder.exists():
            raise FileNotFoundError(f"Missing folder: {folder}")

        for path in image_files(folder):
            result = analyze_image(path)
            label_key = result["label_key"]
            totals[label_key] += 1
            bucket_totals[bucket][label_key] += 1

            row = {
                "bucket": bucket,
                "image": path.name,
                "path": str(path),
                "prediction": DISEASE_LABELS[label_key],
                "prediction_key": label_key,
                "confidence": result["confidence"],
                "reason": result["reason"],
            }
            row.update({f"score_{k}": v for k, v in result["scores"].items()})
            row.update(result["features"])
            rows.append(row)

    fieldnames = [
        "bucket",
        "image",
        "path",
        "prediction",
        "prediction_key",
        "confidence",
        "reason",
        "score_anthracnose",
        "score_mango_scab",
        "score_stem_end_rot",
        "fruit_area",
        "lesion_area_ratio",
        "dark_area_ratio",
        "brown_area_ratio",
        "small_dark_count",
        "medium_dark_count",
        "small_brown_count",
        "largest_lesion_ratio",
        "largest_edge_norm",
        "largest_tipness",
        "largest_tip_overlap",
        "stem_end_candidate",
    ]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    actual_output_csv = output_csv
    try:
        with actual_output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    except PermissionError:
        actual_output_csv = output_csv.with_name(f"{output_csv.stem}_new{output_csv.suffix}")
        with actual_output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "base_dir": str(base_dir),
        "buckets": list(buckets),
        "total_images": len(rows),
        "totals": {DISEASE_LABELS[k]: totals[k] for k in DISEASE_LABELS},
        "totals_by_key": {k: totals[k] for k in DISEASE_LABELS},
        "by_bucket": {
            bucket: {DISEASE_LABELS[k]: bucket_totals[bucket][k] for k in DISEASE_LABELS}
            for bucket in buckets
        },
        "note": (
            "Pre-expert visual triage from image features only. "
            "Use these counts for early balancing, not final disease diagnosis."
        ),
        "predictions_csv": str(actual_output_csv),
    }

    actual_output_json = output_json
    summary["summary_json"] = str(actual_output_json)
    try:
        with actual_output_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    except PermissionError:
        actual_output_json = output_json.with_name(f"{output_json.stem}_new{output_json.suffix}")
        summary["summary_json"] = str(actual_output_json)
        with actual_output_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Rough visual triage for mango fruit disease crops."
    )
    parser.add_argument(
        "--base-dir",
        default="Mango-Detection-5/mango_crops_train_by_occlusion",
        help="Folder containing the occlusion bucket subfolders.",
    )
    parser.add_argument(
        "--buckets",
        nargs="+",
        default=["0_10", "10_40", "40_80"],
        help="Subfolders to process.",
    )
    parser.add_argument(
        "--output-csv",
        default="Mango-Detection-5/disease_triage_predictions.csv",
        help="Per-image prediction CSV.",
    )
    parser.add_argument(
        "--output-json",
        default="Mango-Detection-5/disease_triage_summary.json",
        help="Summary JSON.",
    )
    args = parser.parse_args()

    summary = run(
        Path(args.base_dir),
        args.buckets,
        Path(args.output_csv),
        Path(args.output_json),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
