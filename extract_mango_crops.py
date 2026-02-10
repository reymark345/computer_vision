import os
import glob
import cv2
import numpy as np

# ======================================================
# PATHS (relative to where THIS FILE lives)
# ======================================================
ROOT = os.path.dirname(os.path.abspath(__file__))

IMAGES_DIR = "Mango-Detection-3/train/images"
LABELS_DIR = "Mango-Detection-3/train/labels"
OUT_DIR = "Mango-Detection-3/mango_crops_train"

# ======================================================
# SETTINGS
# ======================================================
PADDING = 10                 # extra pixels around the mango bbox
KEEP_TRANSPARENT_BG = True   # True => PNG with alpha, False => JPG on white bg

# If you want absolutely NO resizing at all, set MAX_SIZE = None
# If you want to avoid extremely huge outputs, set MAX_SIZE = 1024 (downscale only)
MAX_SIZE = None  # e.g. 1024, or None

os.makedirs(OUT_DIR, exist_ok=True)

# ======================================================
# HELPERS
# ======================================================
def parse_poly_line(line, img_w, img_h):
    """
    YOLOv8-seg polygon format:
    class x1 y1 x2 y2 ... xn yn   (normalized)
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
    """
    Downscale only (never upscale). Assumes square input.
    """
    if max_size is None:
        return img
    h, w = img.shape[:2]
    if h <= max_size:
        return img
    return cv2.resize(img, (max_size, max_size), interpolation=cv2.INTER_AREA)

# ======================================================
# MAIN
# ======================================================
saved = 0
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

        # padding around bbox
        x1 = max(0, x1 - PADDING)
        y1 = max(0, y1 - PADDING)
        x2 = min(w, x2 + PADDING)
        y2 = min(h, y2 + PADDING)

        crop = img[y1:y2, x1:x2]
        crop_mask = mask[y1:y2, x1:x2]

        if KEEP_TRANSPARENT_BG:
            # RGBA with alpha from mask (no background)
            crop_out = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
            crop_out[:, :, 3] = crop_mask
            border_val = (0, 0, 0, 0)
            ext = "png"
        else:
            # White background RGB (often better for experts)
            white_bg = np.ones_like(crop, dtype=np.uint8) * 255
            crop_out = np.where(crop_mask[..., None] > 0, crop, white_bg)
            border_val = (255, 255, 255)
            ext = "jpg"

        # Pad to square (NO resizing yet)
        crop_sq = pad_to_square(crop_out, border_val)

        # Optional downscale only (never upscale)
        out_img = downscale_only(crop_sq, MAX_SIZE)

        out_name = f"{base}_cls{cls}_id{inst:03d}.{ext}"
        cv2.imwrite(os.path.join(OUT_DIR, out_name), out_img)

        inst += 1
        saved += 1

print(f"✅ Done. Saved {saved} mango crops into: {OUT_DIR}")
print("Note: No filtering. No upscaling. Only padding; optional downscale if MAX_SIZE is set.")