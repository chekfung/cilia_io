import os
import shutil
import random
import json

random.seed(42)

'''
From a set of label-studio projects, video-wise train-validation split of the 
project with a separate set of videos reserved for the test set.
'''

def move_files(video_list, split):
    split_images = 0
    split_labels = 0

    for video in video_list:
        video_path = os.path.join(RAW_DATA_DIR, video)

        if split == "test":
            video_path = os.path.join(RAW_TEST_SET_DIR, video)
        images_path = os.path.join(video_path, "images")
        labels_path = os.path.join(video_path, "labels")

        num_images = len(os.listdir(images_path)) if os.path.exists(images_path) else 0
        num_labels = len(os.listdir(labels_path)) if os.path.exists(labels_path) else 0

        split_info[split][video] = {"images": num_images, "labels": num_labels}
        split_images += num_images
        split_labels += num_labels

        # Move images
        if os.path.exists(images_path):
            for img in os.listdir(images_path):
                shutil.copy(os.path.join(images_path, img), os.path.join(DATASET_DIR, split, "images", img))

        # Move labels
        if os.path.exists(labels_path):
            for label in os.listdir(labels_path):
                shutil.copy(os.path.join(labels_path, label), os.path.join(DATASET_DIR, split, "labels", label))

    # Store total per split
    split_info[split]["total"] = {"images": split_images, "labels": split_labels}
    split_info["total"]["images"] += split_images
    split_info["total"]["labels"] += split_labels


# ------------------------------
# Start Hyperparameters
TRAIN_RATIO = 0.80
RAW_DATA_DIR = "../data/labeled_data"  # Parent folder containing video-wise folders
RAW_TEST_SET_DIR = "../data/test_set"
DATASET_DIR = "../data/example_dataset"  # Where YOLO dataset will be stored
LOG_FILE = os.path.join(DATASET_DIR, "train_val_split.json")

# Ensure images and labels exist :)
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(DATASET_DIR, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, split, "labels"), exist_ok=True)

# Get all video folders and perform video-wise train-test split
video_folders = [f for f in os.listdir(RAW_DATA_DIR) if os.path.isdir(os.path.join(RAW_DATA_DIR, f))]
test_videos = [f for f in os.listdir(RAW_TEST_SET_DIR) if os.path.isdir(os.path.join(RAW_TEST_SET_DIR, f))]
random.shuffle(video_folders)
split_index = int(len(video_folders) * TRAIN_RATIO)
train_videos, val_videos = video_folders[:split_index], video_folders[split_index:]

# Dictionary to store dataset statistics
split_info = {"train": {}, "val": {}, "test": {}, "total": {"images": 0, "labels": 0}}

# Move files into YOLO dataset structure and gather stats
move_files(train_videos, "train")
move_files(val_videos, "val")
move_files(test_videos, "test")

# Save the split information
with open(LOG_FILE, "w") as f:
    json.dump(split_info, f, indent=4)

# Print dataset statistics
print("\nDataset Split Summary:")
for split in ["train", "val", "test"]:
    print(f"\n{split.upper()} SET:")
    for video, counts in split_info[split].items():
        if video != "total":
            print(f"  {video}: {counts['images']} images, {counts['labels']} labels")
    print(f"TOTAL: {split_info[split]['total']['images']} images, {split_info[split]['total']['labels']} labels")

print(f"\nOVERALL TOTAL: {split_info['total']['images']} images, {split_info['total']['labels']} labels")
print("\nDataset restructuring complete!")