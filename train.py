from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(description='Train YOLO model on the mango-2 Roboflow export')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--model', type=str, default='yolov8n-seg.pt', help='pretrained model or yaml')
    parser.add_argument('--name', type=str, default='mango_run')
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    data_yaml = base / 'Mango-Detection-1' / 'data.yaml'

    try:
        from ultralytics import YOLO
    except Exception as e:
        print('Missing dependency: please install requirements first (see README.md)')
        raise

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(base / 'Mango-Detection-1' / 'models'),
        name=args.name,
        pretrained=False,  # Don't use pretrained weights due to class count change
        exist_ok=True,
    )


if __name__ == '__main__':
    main()