import os
import gc
import torch
import numpy as np
import csv
from ultralytics import YOLO

# ------------------------------
# 1. Setup & Paths
# ------------------------------
KFOLD_ROOT = "../data/cilia_io_3_fold_validation_dataset"
PROJECT_NAME = "cilia_io_3_fold_cross_validation"
NUM_FOLDS = 3
EPOCHS = 300
BATCH_SIZE = 4

# Base directory for all CSV data
REBUTTAL_CSV_DIR = "3_fold_cross_validation_csvs"
os.makedirs(REBUTTAL_CSV_DIR, exist_ok=True)

def export_metrics_to_csv(metrics, fold_path, split_type):
    """
    Handles the extraction of curve data and confusion matrices 
    to CSV files for a specific split (val or test).
    """
    # Create directory for this specific fold's split
    target_dir = os.path.join(REBUTTAL_CSV_DIR, fold_path, split_type)
    os.makedirs(target_dir, exist_ok=True)

    # 1. Save PR, F1, and Confidence Curves
    if hasattr(metrics, 'curves') and hasattr(metrics, 'curves_results'):
        for name, arr in zip(metrics.curves, metrics.curves_results):
            x_vals, y_vals, x_label, y_label = arr
            clean_name = name.replace('(', '').replace(')', '').replace('-', '_').replace(' ', '_')
            csv_filename = os.path.join(target_dir, f"{clean_name}.csv")
            
            x_vals = np.array(x_vals).flatten()
            y_vals = np.array(y_vals).flatten()
            
            with open(csv_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([x_label, y_label])
                for x, y in zip(x_vals, y_vals):
                    writer.writerow([x, y])
            print(f"      - Saved curve: {clean_name}.csv")

    # 2. Save Confusion Matrix
    if hasattr(metrics, 'confusion_matrix'):
        cm = metrics.confusion_matrix.matrix
        cm_path = os.path.join(target_dir, "confusion_matrix.csv")
        np.savetxt(cm_path, cm, delimiter=",", fmt='%d')
        print(f"      - Saved confusion matrix")

# ------------------------------
# K-Fold Training & Export Loop
for i in range(1, NUM_FOLDS + 1):
    fold_label = f"fold_{i}"
    data_yaml = os.path.join(KFOLD_ROOT, fold_label, "data.yaml")
    
    print(f"\n{'='*40}")
    print(f" STARTING FOLD {i} OF {NUM_FOLDS}")
    print(f"{'='*40}")

    # Memory cleanup
    gc.collect()
    torch.cuda.empty_cache()

    # Initialize fresh model (prevents weight leakage between folds)
    model = YOLO("yolo11m.pt")

    # --- TRAIN ---
    print(f" --> Training model on {fold_label}...")
    model.train(
        data=data_yaml,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        cache=False,
        project=PROJECT_NAME,
        name=f"{fold_label}_run",
        exist_ok=True
    )

    # Note: After training, it loads the best.pt model for

    # --- val ---
    print(f" --> Running Validation on Fold {i} 'Left-Out' Video...")
    val_metrics = model.val(split='val')
    export_metrics_to_csv(val_metrics, fold_label, "validation")

    # --- test ---
    print(f" --> Running Test on Constant Anchor Video...")
    test_metrics = model.val(split='test')
    export_metrics_to_csv(test_metrics, fold_label, "test")

    print(f" --> Cleaning up Fold {i} memory...")
    del model
    del val_metrics
    del test_metrics
    
    gc.collect()
    torch.cuda.empty_cache()


print(f"\n{'='*40}")
print(" Cross Validation Data Runs Complete :)")
print(f" All CSV data found in: {REBUTTAL_CSV_DIR}")
print(f"{'='*40}")