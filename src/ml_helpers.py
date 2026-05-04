import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import cv2
import os
import tifffile
import numpy as np
import matplotlib.pyplot as plt
# from segment_anything import SamPredictor, sam_model_registry
import numpy as np
import json

'''
A variety of helper functions that are useful for machine learning-based visualization.
This folder also contains the custom tracking methods beyond ByteTrack, relying on the assumption
that cilia bases are anchored in place.
'''

# ------------------

# The following code is adapted from:
# https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax, conf):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))   

    if conf is not None:
        label = f"{conf:.2f}"
        ax.text(x0, y0 - 2, label,
                color='white', fontsize=8, weight='bold',
                bbox=dict(facecolor='green', alpha=0.5, boxstyle='round,pad=0.2'))

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

# --------------

def center(box):
    x1, y1, x2, y2 = box
    return np.array([int((x1 + x2) / 2), int((y1 + y2) / 2)])

def id_to_color(tid):
    """Get a consistent BGR color from a given ID"""
    np.random.seed(int(tid))
    return tuple(int(x) for x in np.random.randint(0, 256, size=3))  # (B, G, R)

def overlay_masks_once(frame, masks, ids, colors=None, alpha=0.7):
    """
    Overlay all masks on frame at once, using consistent colors based on ID.
    This avoids repeated darkening of the image.
    """
    overlay = frame.copy()
    color_layer = np.zeros_like(frame, dtype=np.uint8)

    if colors == None:
        for mask, tid in zip(masks, ids):
            mask = mask.astype(bool)

            color = id_to_color(tid)
            color_layer[mask] = color
    else:
        for mask, tid, color in zip(masks, ids, colors):
            mask = mask.astype(bool)
            color_layer[mask] = color

    # Blend just once
    cv2.addWeighted(color_layer, alpha, overlay, 1 - alpha, 0, overlay)
    return overlay

def match_by_historical_centers(current_boxes, historical_centers, max_dist):
    assigned_ids = [None] * len(current_boxes)
    used_ids = set()
    current_centers = np.array([center(box) for box in current_boxes])

    for i, ctr in enumerate(current_centers):
        best_id = None
        best_dist = float('inf')
        for hid, hctr in historical_centers.items():
            dist = np.linalg.norm(ctr - hctr)
            if dist < max_dist and dist < best_dist and hid not in used_ids:
                best_id = hid
                best_dist = dist
        if best_id is not None:
            assigned_ids[i] = best_id
            used_ids.add(best_id)
    return assigned_ids

def iou(boxA, boxB):
    # Determine intersection over union of two boxes
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def merge_overlapping_boxes(boxes, ids, track_age, iou_thresh=0.4):
    # Merges overlapping boxes based on the intersection over union of different boxes.
    # Takes a greedy approach to matching boxes. Preserves the older ID over the newer one.
    merged_ids = ids.copy()
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if merged_ids[i] != merged_ids[j]:
                if iou(boxes[i], boxes[j]) > iou_thresh:
                    id_i = merged_ids[i]
                    id_j = merged_ids[j]
                    print(f"ID: {id_i} and ID: {id_j} Merged")
                    age_i = track_age.get(id_i, 0)
                    age_j = track_age.get(id_j, 0)
                    # Keep the older ID (higher age)
                    if age_i >= age_j:
                        merged_ids[j] = id_i
                    else:
                        merged_ids[i] = id_j

    # Group boxes by merged ID
    id_to_boxes = defaultdict(list)
    for box, tid in zip(boxes, merged_ids):
        id_to_boxes[tid].append(box)

    # Choose representative box for each merged group
    merged_boxes = []
    final_ids = []
    for tid, group in id_to_boxes.items():
        if len(group) == 1:
            chosen = group[0]
        else:
            # Choose box with largest area (or average, your choice)
            areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in group]
            chosen = group[np.argmax(areas)]
        merged_boxes.append(chosen)
        final_ids.append(tid)

    merged_boxes = np.array(merged_boxes)
    return merged_boxes, final_ids

def tiff_to_mp4(tiff_path, output_path, fps=50):
    # Convert a .tiff file into an .mp4 file.
    # Defaults to 50 fps, which is the fps of our standardized video format.

    output_dir = os.path.dirname(output_path)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load TIFF stack
    tiff_data = tifffile.imread(tiff_path)
    print(f"Loaded {len(tiff_data)} frames from {tiff_path}")
    print(f"Data type: {tiff_data.dtype}")
    print(f"Pixel value range: {tiff_data.min()} to {tiff_data.max()}")
    
    frame_shape = tiff_data[0].shape
    height, width = frame_shape if len(frame_shape) == 2 else frame_shape[:2]

    # Define VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for i, frame in enumerate(tiff_data):
        image_normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
        image_uint8 = image_normalized.astype(np.uint8)
        image_rgb = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2RGB)
        out.write(image_rgb)

    out.release()
    print(f"Video saved successfully to {output_path}")

def tiff_get_metadata(tiff_path):
    # Obtain metadata from the microscope, namely number of pixels per micron and frames per second (FPS)

    with tifffile.TiffFile(tiff_path) as tif:
        first_page = tif.pages[0]
        ij_metadata_tag = first_page.tags.get("IJMetadata")

        if ij_metadata_tag:
            ij_metadata_str = ij_metadata_tag.value
            ij_metadata_str = ij_metadata_str['Info']
            print("IJMetadata found. Parsing as key=value lines")

            # Split lines and parse key=value pairs
            lines = ij_metadata_str.strip().split('\n')
            metadata_dict = {}
            for line in lines:
                if '=' in line:
                    key, val = line.split('=', 1)
                    metadata_dict[key.strip()] = val.strip()

            # Extract pixel -> micron, if present
            pixel_microns = metadata_dict.get('dCalibration')

            # On different microscopes, this can be incorrect, so we instead take the difference in timestamps
            # real_fps = metadata_dict.get('dFps')          

            try:
                timestamp_1 = None
                timestamp_2 = None

                for k in ["timestamp #1", "timestamp #01", "timestamp #001"]:
                    if k in metadata_dict:
                        timestamp_1 = float(metadata_dict[k])
                        break

                for k in ["timestamp #2", "timestamp #02", "timestamp #002"]:
                    if k in metadata_dict:
                        timestamp_2 = float(metadata_dict[k])
                        break

                if timestamp_1 is not None and timestamp_2 is not None:
                    time_diff = timestamp_2 - timestamp_1
                else:
                    time_diff = None
                    print("Warning: could not find valid timestamps.")

            except Exception as e:
                time_diff = None
                print(f"Error while parsing timestamps: {e}")
                exit()

            time_diff = timestamp_2 - timestamp_1
            actual_fps = 1 / time_diff

            return float(pixel_microns), float(actual_fps), metadata_dict
        