from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8-Seg on Mango dataset')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--model', type=str, default='yolov8n-seg.pt',
                        help='pretrained model or model yaml')
    parser.add_argument('--name', type=str, default='mango_seg')
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    data_yaml = base / 'data.yaml'

    from ultralytics import YOLO

    model = YOLO(args.model)

    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(base / 'models'),
        name=args.name,
        pretrained=True,   # ✅ USE pretrained weights
        exist_ok=True,
        task='segment'
    )

    print("Training complete.")
    print("Best model saved at:")
    print(base / 'models' / args.name / 'weights' / 'best.pt')

if __name__ == '__main__':
    main()