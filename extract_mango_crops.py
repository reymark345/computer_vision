import os, glob
import cv2
import numpy as np

IMAGES_DIR = "Mango-Detection-1/train/images"
LABELS_DIR = "Mango-Detection-1/train/labels"
OUT_DIR = "Mango-Detection-1/mango_crops_train"
OUT_SIZE = 224
PADDING = 10
KEEP_TRANSPARENT_BG = True  # saves PNG with alpha

os.makedirs(OUT_DIR, exist_ok=True)

def parse_poly_line(line, img_w, img_h):
    parts = line.strip().split()
    cls = int(float(parts[0]))
    coords = list(map(float, parts[1:]))

    if len(coords) < 6 or len(coords) % 2 != 0:
        return cls, None

    pts = []
    for i in range(0, len(coords), 2):
        x = int(round(coords[i] * img_w))
        y = int(round(coords[i+1] * img_h))
        pts.append([x, y])

    return cls, np.array(pts, dtype=np.int32)

def polygon_to_mask(poly, w, h):
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    return mask

def letterbox_to_square(img, size=224):
    h, w = img.shape[:2]
    side = max(h, w)
    top = (side - h) // 2
    bottom = side - h - top
    left = (side - w) // 2
    right = side - w - left
    border_val = (0, 0, 0, 0) if img.shape[2] == 4 else (0, 0, 0)
    img_sq = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=border_val)
    return cv2.resize(img_sq, (size, size), interpolation=cv2.INTER_AREA)

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
        lines = [ln for ln in f.readlines() if ln.strip()]

    inst = 0
    for ln in lines:
        cls, poly = parse_poly_line(ln, w, h)
        if poly is None:
            continue

        mask = polygon_to_mask(poly, w, h)
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            continue

        x1, x2 = xs.min(), xs.max() + 1
        y1, y2 = ys.min(), ys.max() + 1

        x1 = max(0, x1 - PADDING); y1 = max(0, y1 - PADDING)
        x2 = min(w, x2 + PADDING); y2 = min(h, y2 + PADDING)

        crop = img[y1:y2, x1:x2]
        crop_mask = mask[y1:y2, x1:x2]

        crop_rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
        crop_rgba[:, :, 3] = crop_mask

        out_img = letterbox_to_square(crop_rgba, OUT_SIZE)

        out_name = f"{base}_cls{cls}_id{inst:03d}.png"
        cv2.imwrite(os.path.join(OUT_DIR, out_name), out_img)

        inst += 1
        saved += 1

print(f"✅ Done. Saved {saved} mango crops into: {OUT_DIR}")