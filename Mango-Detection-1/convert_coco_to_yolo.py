import json
import os
from pathlib import Path
import numpy as np
from PIL import Image

def convert_coco_to_yolo(coco_json_path, output_dir):
    """Convert COCO segmentation format to YOLOv8 format"""
    
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Build lookup dictionaries
    images_info = {img['id']: img for img in coco_data['images']}
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # Group annotations by image
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    # Convert each image
    for img_id, image_info in images_info.items():
        filename = image_info['file_name']
        width = image_info['width']
        height = image_info['height']
        
        # Create label file
        label_filename = Path(filename).stem + '.txt'
        label_path = os.path.join(output_dir, label_filename)
        
        labels = []
        
        if img_id in annotations_by_image:
            for ann in annotations_by_image[img_id]:
                category_id = ann['category_id']
                
                # Map category IDs: both 0 and 1 are "mango", 2 is "occluded"
                # So: 0->0 (mango), 1->0 (mango), 2->1 (occluded)
                if category_id in [0, 1]:
                    mapped_category_id = 0  # mango
                elif category_id == 2:
                    mapped_category_id = 1  # occluded
                else:
                    continue
                
                # Get segmentation polygon
                if 'segmentation' in ann and ann['segmentation']:
                    seg = ann['segmentation'][0]  # Take first polygon
                    
                    # Normalize coordinates to 0-1
                    normalized_seg = []
                    for i in range(0, len(seg), 2):
                        x_norm = seg[i] / width
                        y_norm = seg[i + 1] / height
                        # Clamp to [0, 1]
                        x_norm = max(0, min(1, x_norm))
                        y_norm = max(0, min(1, y_norm))
                        normalized_seg.append(f"{x_norm:.6f} {y_norm:.6f}")
                    
                    # Format: class_id x1 y1 x2 y2 ... (segmentation coordinates)
                    label_line = f"{mapped_category_id} " + " ".join(normalized_seg)
                    labels.append(label_line)
        
        # Write label file
        if labels:
            with open(label_path, 'w') as f:
                f.write('\n'.join(labels))
            print(f"Created: {label_path}")

def main():
    base_dir = Path(__file__).parent
    
    # Convert train, val, test splits
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        split_dir = base_dir / split
        coco_json = split_dir / '_annotations.coco.json'
        
        if coco_json.exists():
            output_dir = split_dir / 'labels'
            print(f"\nConverting {split} split...")
            convert_coco_to_yolo(str(coco_json), str(output_dir))
            print(f"Completed {split} split!")
        else:
            print(f"Warning: {coco_json} not found")

if __name__ == '__main__':
    main()
