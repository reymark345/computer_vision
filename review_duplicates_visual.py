"""
Visual Duplicate Review Tool
Shows duplicate images side-by-side and lets you decide Y/N for deletion.
Interactive review of flagged duplicates.
"""
import os
import glob
import csv
import cv2
import shutil
from pathlib import Path

OUT_DIR = "Mango-Detection-5/mango_crops_train_by_occlusion"
TARGET_FOLDER = "0_10"  # Change to review different folders
TARGET_DIR = os.path.join(OUT_DIR, TARGET_FOLDER)
BACKUP_DIR = os.path.join(OUT_DIR, "deleted_backup_phash")
DUPLICATES_LOG_CSV = os.path.join(OUT_DIR, f"duplicates_phash_{TARGET_FOLDER}_log.csv")
MAIN_CSV_PATH = os.path.join(OUT_DIR, "crop_quality_metrics_experiment_simple.csv")

DECISIONS_LOG_CSV = os.path.join(OUT_DIR, f"duplicate_review_decisions_{TARGET_FOLDER}.csv")

backup_bucket = os.path.join(BACKUP_DIR, f"{TARGET_FOLDER}_duplicates")


def load_image(img_path, max_width=400, max_height=400):
    """Load and resize image for display."""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None
        
        h, w = img.shape[:2]
        if w > max_width or h > max_height:
            scale = min(max_width / w, max_height / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h))
        
        return img
    except Exception as e:
        print(f"Error loading {img_path}: {e}")
        return None


def display_comparison(dup_path, kept_path, dup_name, kept_name, distance, pair_num, total):
    """Display two images side-by-side and get user decision."""
    img_dup = load_image(dup_path)
    img_kept = load_image(kept_path)
    
    if img_dup is None or img_kept is None:
        print(f"  Could not load images")
        return None
    
    # Create side-by-side comparison
    h_dup, w_dup = img_dup.shape[:2]
    h_kept, w_kept = img_kept.shape[:2]
    
    max_h = max(h_dup, h_kept)
    combined = cv2.hconcat([
        cv2.copyMakeBorder(img_dup, 0, max_h - h_dup, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)),
        cv2.copyMakeBorder(img_kept, 0, max_h - h_kept, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    ])
    
    # Add labels
    cv2.putText(combined, "DUPLICATE (DELETE?)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(combined, f"Distance: {distance}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)
    cv2.putText(combined, f"Pair {pair_num}/{total}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    
    cv2.putText(combined, "KEEP THIS ONE", (w_dup + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display
    cv2.imshow("Duplicate Review: Press Y=DELETE / N=RESTORE / Q=QUIT", combined)
    
    print(f"\n{'='*70}")
    print(f"Pair {pair_num}/{total}")
    print(f"{'='*70}")
    print(f"LEFT  (Duplicate):  {dup_name}")
    print(f"RIGHT (Keep this): {kept_name}")
    print(f"Similarity distance: {distance} (lower = more similar)")
    print(f"\nWindow shown. Press in image window:")
    print(f"  Y = DELETE (move left to backup)")
    print(f"  N = RESTORE (keep both, move back from backup)")
    print(f"  Q = QUIT and save decisions")
    print()
    
    while True:
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('y') or key == ord('Y'):
            cv2.destroyAllWindows()
            return 'DELETE'
        elif key == ord('n') or key == ord('N'):
            cv2.destroyAllWindows()
            return 'RESTORE'
        elif key == ord('q') or key == ord('Q'):
            cv2.destroyAllWindows()
            return 'QUIT'


# Step 1: Check if duplicates log exists
print(f"Step 1: Loading duplicates log...")
if not os.path.exists(DUPLICATES_LOG_CSV):
    print(f"  ERROR: No duplicates log found at {DUPLICATES_LOG_CSV}")
    print(f"  Run deduplicate_single_folder.py first!")
    exit(1)

# Load duplicates
duplicates = []
with open(DUPLICATES_LOG_CSV, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        dup_filename = row[0]
        kept_filename = row[1]
        distance = int(row[2])
        
        dup_path = os.path.join(backup_bucket, dup_filename)
        
        # Find kept file in target folder
        kept_path = None
        for fpath in glob.glob(os.path.join(TARGET_DIR, "*")):
            if os.path.basename(fpath) == kept_filename:
                kept_path = fpath
                break
        
        if os.path.exists(dup_path) and kept_path:
            duplicates.append((dup_path, kept_path, dup_filename, kept_filename, distance))
        else:
            print(f"  WARNING: Could not find {dup_filename} or {kept_filename}")

total_duplicates = len(duplicates)
print(f"  Found {total_duplicates} flagged duplicates to review")

if total_duplicates == 0:
    print(f"  No duplicates to review!")
    exit(0)

# Step 2: Interactive review
print(f"\nStep 2: Starting interactive review...")
decisions = []
deleted_count = 0
restored_count = 0
quit_early = False

for idx, (dup_path, kept_path, dup_name, kept_name, distance) in enumerate(duplicates, 1):
    decision = display_comparison(dup_path, kept_path, dup_name, kept_name, distance, idx, total_duplicates)
    
    if decision == 'QUIT':
        quit_early = True
        decisions.append([dup_name, kept_name, distance, 'SKIPPED'])
        print(f"  Skipping remaining pairs...")
        break
    elif decision == 'DELETE':
        decisions.append([dup_name, kept_name, distance, 'DELETED'])
        deleted_count += 1
        print(f"  ✓ Keeping decision: DELETE {dup_name}")
    elif decision == 'RESTORE':
        # Restore file from backup to original folder
        try:
            dest_path = os.path.join(TARGET_DIR, dup_name)
            shutil.move(dup_path, dest_path)
            decisions.append([dup_name, kept_name, distance, 'RESTORED'])
            restored_count += 1
            print(f"  ✓ RESTORED {dup_name} back to {TARGET_FOLDER}/")
        except Exception as e:
            print(f"  ✗ ERROR restoring: {e}")
            decisions.append([dup_name, kept_name, distance, 'ERROR'])

# Step 3: Save decisions log
print(f"\nStep 3: Saving decisions log...")
with open(DECISIONS_LOG_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["duplicate_filename", "kept_filename", "distance", "decision"])
    writer.writerows(decisions)
print(f"  ✓ Saved: {DECISIONS_LOG_CSV}")

# Step 4: Update main CSV if any deletions
if deleted_count > 0:
    print(f"\nStep 4: Updating main CSV...")
    deleted_names = set(d[0] for d in decisions if d[3] == 'DELETED')
    
    if os.path.exists(MAIN_CSV_PATH):
        rows = []
        with open(MAIN_CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                crop_name = row[0]
                if crop_name not in deleted_names:
                    rows.append(row)
        
        with open(MAIN_CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        
        print(f"  ✓ CSV updated: {len(deleted_names)} entries removed")
        print(f"  ✓ Remaining in CSV: {len(rows)}")

# Step 5: Summary
print(f"\n{'='*70}")
print(f"REVIEW COMPLETE")
print(f"{'='*70}")
print(f"Total pairs reviewed:   {len(decisions)}")
print(f"Deleted:                {deleted_count}")
print(f"Restored:               {restored_count}")
print(f"Skipped:                {total_duplicates - len(decisions)}")
print(f"\nDecisions log: {DECISIONS_LOG_CSV}")
print(f"Backup folder: {backup_bucket}")
print(f"\nRemaining duplicates still in backup (not permanently deleted).")
print(f"You can manually delete them later or restore anytime.")
print(f"{'='*70}")
