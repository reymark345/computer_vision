import json
from pathlib import Path

# ================================
# CONFIG: Set your class mapping
# ================================
# YOLO class IDs MUST start at 0
NAME_TO_YOLO = {
    "mango": 0,
    "occluded": 1,
}

SPLITS = ["train", "valid", "test"]
COCO_FILENAME = "_annotations.coco.json"


def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def build_coco_id_to_yolo(coco: dict) -> dict:
    """
    Convert COCO category IDs -> YOLO class IDs using NAME_TO_YOLO mapping.
    """
    coco_id_to_yolo = {}
    for cat in coco.get("categories", []):
        name = str(cat.get("name", "")).lower().strip()
        if name in NAME_TO_YOLO:
            coco_id_to_yolo[int(cat["id"])] = NAME_TO_YOLO[name]
    return coco_id_to_yolo


def convert_split(split_dir: Path):
    coco_json = split_dir / COCO_FILENAME
    if not coco_json.exists():
        print(f"Skipping {split_dir.name}: {COCO_FILENAME} not found")
        return

    with coco_json.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    coco_id_to_yolo = build_coco_id_to_yolo(coco)
    if not coco_id_to_yolo:
        raise RuntimeError(
            f"No matching categories found in {coco_json}. "
            f"Expected category names: {list(NAME_TO_YOLO.keys())}"
        )

    # Print mapping so you can verify
    print(f"\nConverting {split_dir.name} -> {split_dir / 'labels'}")
    print("YOLO class mapping:", NAME_TO_YOLO)
    print("COCO id -> YOLO id:", coco_id_to_yolo)

    labels_dir = split_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    # image_id -> {file_name, width, height}
    images = {im["id"]: im for im in coco.get("images", [])}

    # Group annotations by image_id
    anns_by_img = {}
    for ann in coco.get("annotations", []):
        img_id = ann.get("image_id")
        if img_id is None:
            continue
        anns_by_img.setdefault(img_id, []).append(ann)

    written_files = 0
    total_instances = 0

    for img_id, im in images.items():
        file_name = im["file_name"]
        w = float(im["width"])
        h = float(im["height"])

        label_path = labels_dir / (Path(file_name).stem + ".txt")
        lines = []

        for ann in anns_by_img.get(img_id, []):
            coco_cat_id = ann.get("category_id")
            if coco_cat_id not in coco_id_to_yolo:
                continue

            yolo_cls = coco_id_to_yolo[coco_cat_id]

            seg = ann.get("segmentation", None)
            if not seg or not isinstance(seg, list):
                continue

            # COCO segmentation is a list of polygons
            # Each polygon is [x1, y1, x2, y2, ...]
            for poly in seg:
                if not poly or len(poly) < 6:
                    continue

                coords = []
                for i in range(0, len(poly), 2):
                    x = clamp01(poly[i] / w)
                    y = clamp01(poly[i + 1] / h)
                    coords.append(f"{x:.6f} {y:.6f}")

                # YOLOv8-seg format: class x1 y1 x2 y2 ...
                lines.append(f"{yolo_cls} " + " ".join(coords))
                total_instances += 1

        if lines:
            label_path.write_text("\n".join(lines), encoding="utf-8")
            written_files += 1

    print(f"Done {split_dir.name}: wrote {written_files} label files, {total_instances} total polygons")


def main():
    base = Path(__file__).parent.resolve()

    for split in SPLITS:
        split_dir = base / split
        if not split_dir.exists():
            print(f"Skipping {split}: folder not found -> {split_dir}")
            continue
        convert_split(split_dir)

    print("\nAll splits processed.")
    print("Next: make sure your data.yaml points to train/images, valid/images, and test/images.")
    print("Then train: python train.py --epochs 100 --imgsz 640 --batch 8")


if __name__ == "__main__":
    main()
