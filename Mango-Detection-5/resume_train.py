from pathlib import Path
import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Resume YOLOv8-Seg training on Mango dataset")
    parser.add_argument("--name", type=str, default="mango_seg")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    checkpoint = base / "models" / args.name / "weights" / "last.pt"

    if not checkpoint.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint}")

    model = YOLO(str(checkpoint))
    model.train(resume=True)


if __name__ == "__main__":
    main()
