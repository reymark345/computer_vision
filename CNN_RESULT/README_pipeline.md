# Mango Segmentation + CNN Disease Pipeline

This folder now supports the combined workflow:

1. Train/use YOLOv8-Seg for mango instance segmentation.
2. Use the YOLO class to report `non_occluded` or `occluded`.
3. Extract each mango crop.
4. Run the CNN disease classifier on each crop.
5. Save per-mango predictions to CSV and annotated images.

## 1. Train Segmentation / Occlusion Model

Your segmentation dataset is in `Mango-Detection-5` and its `data.yaml` contains:

```yaml
names:
  0: mango
  1: occluded
```

That means YOLOv8-Seg is responsible for both mango instance segmentation and
occlusion/non-occlusion detection.

```powershell
cd Mango-Detection-5
python train.py --epochs 100 --imgsz 640 --batch 8 --name mango_seg
cd ..
```

The best segmentation model is saved at:

```text
Mango-Detection-5/models/mango_seg/weights/best.pt
```

## 2. Train Disease CNN

The disease classifier uses the labeled crop folders in `CNN`:

```text
CNN/Anthracnose
CNN/Healthy
CNN/Mango Scab
CNN/Stem end rot
```

Train the custom CNN with adjustable hyperparameters:

```powershell
python CNN\train_cnn.py --epochs 30 --batch-size 16 --lr 0.0005 --run-name custom_cnn_v1 --rebuild-split
```

Try a deeper/wider custom CNN:

```powershell
python CNN\train_cnn.py `
  --epochs 40 `
  --batch-size 16 `
  --lr 0.0003 `
  --conv-channels 32,64,128,256,512 `
  --classifier-hidden 256 `
  --dropout 0.4 `
  --run-name custom_cnn_deeper_v1
```

The best disease model is saved at:

```text
CNN/runs/<run-name>/best_model.pt
```

## 3. Run Combined Pipeline

Run the full system on original images:

```powershell
python CNN\predict_mango_pipeline.py `
  --source Mango-Detection-5\test\images `
  --seg-model Mango-Detection-5\models\mango_seg\weights\best.pt `
  --disease-model CNN\runs\custom_cnn_v1\best_model.pt `
  --save-crops
```

Outputs are saved in:

```text
CNN/pipeline_outputs/<run-name>/
```

Important files:

```text
predictions.csv       per-mango occlusion + disease prediction
annotated/            original images with instance labels
crops/                extracted mango crops when --save-crops is used
config.json           models and settings used for the run
```

Useful options:

```powershell
--conf 0.35              # stricter YOLO detection confidence
--crop-mode mask-white   # remove background using the segmentation mask
--skip-occluded          # report occluded mangoes but do not classify disease
--max-images 5           # quick test run
```
