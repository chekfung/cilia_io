import os
import shutil
import json
import yaml

# ------------------------------
# Setup & Paths
RAW_DATA_DIR = "../../data/labeled_data" 
RAW_TEST_SET_DIR = "../../data/labeled_data_test_set" 
OUTPUT_BASE_DIR = "../../data/cilia_io_3_fold_validation_dataset"

# Identify and separate videos by genotype
all_vids = sorted([f for f in os.listdir(RAW_DATA_DIR) if os.path.isdir(os.path.join(RAW_DATA_DIR, f))])
test_videos = sorted([f for f in os.listdir(RAW_TEST_SET_DIR) if os.path.isdir(os.path.join(RAW_TEST_SET_DIR, f))])

mutants = []
wildtypes = []

for v_folder in all_vids:
    img_path = os.path.join(RAW_DATA_DIR, v_folder, "images")
    if not os.path.exists(img_path):
        continue
    
    # Peek at the first filename to determine genotype
    sample_file = os.listdir(img_path)[0].lower()
    
    if "bbs2" in sample_file or "stl438" in sample_file or '60x' in sample_file:
        mutants.append(v_folder)
    elif "wt" in sample_file:
        wildtypes.append(v_folder)

# Ensure we have a balanced 3/3 split for the 3-fold strategy
print(f"Detected Mutants: {mutants}")
print(f"Detected Wildtypes: {wildtypes}")

# ------------------------------
# Define Stratified Folds
# We pair one mutant with one WT for each of the 3 validation sets (for representative validation sets)
# Fold 1: Mut[0]+WT[0] | Fold 2: Mut[1]+WT[1] | Fold 3: Mut[2]+WT[2]
# Notably, these folds are different from our original split, just because this is a completely separate experiment :)

folds_config = []
for i in range(3):
    val_vids = [mutants[i], wildtypes[i]]
    train_vids = [v for v in all_vids if v not in val_vids]
    folds_config.append({"val": val_vids, "train": train_vids})

all_fold_logs = {}

def process_split(video_list, split_name, fold_dir, source_root):
    """Copies files and returns detailed image/label counts per video."""
    split_images = 0
    split_labels = 0
    video_details = {}

    img_dest = os.path.join(fold_dir, split_name, "images")
    lbl_dest = os.path.join(fold_dir, split_name, "labels")
    os.makedirs(img_dest, exist_ok=True)
    os.makedirs(lbl_dest, exist_ok=True)

    for video in video_list:
        v_path = os.path.join(source_root, video)
        img_src, lbl_src = os.path.join(v_path, "images"), os.path.join(v_path, "labels")

        n_img = len(os.listdir(img_src)) if os.path.exists(img_src) else 0
        n_lbl = len(os.listdir(lbl_src)) if os.path.exists(lbl_src) else 0
        
        video_details[video] = {"images": n_img, "labels": n_lbl}
        split_images += n_img
        split_labels += n_lbl

        if os.path.exists(img_src):
            for f in os.listdir(img_src):
                shutil.copy(os.path.join(img_src, f), os.path.join(img_dest, f))
        if os.path.exists(lbl_src):
            for f in os.listdir(lbl_src):
                shutil.copy(os.path.join(lbl_src, f), os.path.join(lbl_dest, f))

    return video_details, {"total_images": split_images, "total_labels": split_labels}

# ------------------------------
# Execution Loop
print(f"Generating 3 Folds (1 Mutant + 1 WT per Val set)...")

for i, config in enumerate(folds_config):
    fold_name = f"fold_{i + 1}"
    fold_dir = os.path.join(OUTPUT_BASE_DIR, fold_name)
    
    print(f"Creating {fold_name}: Val={config['val']}")

    fold_log = {
        "fold_index": i + 1,
        "train_videos": config["train"],
        "val_videos": config["val"]
    }

    # Logging and Processing
    fold_log["train_stats"], fold_log["train_totals"] = process_split(config["train"], "train", fold_dir, RAW_DATA_DIR)
    fold_log["val_stats"], fold_log["val_totals"] = process_split(config["val"], "val", fold_dir, RAW_DATA_DIR)
    fold_log["test_stats"], fold_log["test_totals"] = process_split(test_videos, "test", fold_dir, RAW_TEST_SET_DIR)

    # YOLO YAML
    yolo_config = {
        'path': os.path.abspath(fold_dir),
        'train': 'train/images', 'val': 'val/images', 'test': 'test/images',
        'names': {0: 'cilia'}
    }
    with open(os.path.join(fold_dir, "data.yaml"), "w") as f:
        yaml.dump(yolo_config, f, default_flow_style=False)

    all_fold_logs[fold_name] = fold_log

# 4. Save Final Summary JSON
summary_file = os.path.join(OUTPUT_BASE_DIR, "kfold_summary.json")
with open(summary_file, "w") as f:
    json.dump(all_fold_logs, f, indent=4)
