
from ml_helpers import *
from quantification_help import *

from scipy.interpolate import splprep, splev
from scipy.ndimage import distance_transform_edt, sobel, label

import skimage
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import os
import time 

from scipy.ndimage import zoom, gaussian_filter, sobel, binary_fill_holes, uniform_filter
from skimage.morphology import skeletonize, thin
from skimage.measure import label, regionprops
from skimage.filters import apply_hysteresis_threshold
from scipy.fft import fft, fftfreq
from skimage import measure, morphology

from ultralytics import YOLO
import torch 
from segment_anything import SamPredictor, sam_model_registry

'''
File that is used to perform benchmarking between ciliaIO and other cilia
quantification segmentation methodologies such as Li Binary Thresholding, Canny Detection, and a frequency-based
segmentation method from Thouvenin et al. 2021
'''

# CONFIGS
VIDEO_FILE_PATH = '../data/video_file.mp4'

CROP_Y = (34, 141)
CROP_X = (78, 323)
SAVE_PATH = 'test'
SAVE_FIG = False 

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# Read metadata to get pixel conversion?
pixels_to_microns, fps, metadata_dict = tiff_get_metadata(VIDEO_FILE_PATH[:-3]+"tif")
print(f"One Pixel = {pixels_to_microns} Microns")
print(f"FPS: {fps}")

# Read in filepath
cap = cv2.VideoCapture(VIDEO_FILE_PATH)

frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame_rgb)
cap.release()

# Get data frames, as well as the middle frame to look at
data = np.stack(frames, axis=0)  # shape: (T, H, W)
image = frames[int(len(frames) / 2)]

SCALE_BAR_LENGTH = 1.0 # micron

# -----------------------------------

def save_axis_high_dpi(ax, filename, dpi=400, transparent=True, pad_inches=0.0):
    # Get the parent figure
    fig = ax.figure
    
    # Save only the specific axis extent
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(
        filename,
        bbox_inches=extent,
        dpi=dpi,
        transparent=transparent,
        pad_inches=pad_inches
    )
    print(f"Saved axis as '{filename}' ({dpi} DPI)")

def thouvenin_frequency_segmentation(imaging_data, fps, local_avg_size=4, num_fft_peaks=5, min_roi_pixels=40):
    '''
    Adapted from Thouvenin et al., 2021 using their original MATLAB scripts, ported to Python.
    '''
    # Filter Data
    filtered_data = uniform_filter(imaging_data, size=(0, local_avg_size, local_avg_size))

    # FFT per pixel 
    T, H, W = filtered_data.shape
    freqs_map = np.zeros((H, W), dtype=np.float32)

    for y in range(H):
        for x in range(W):
            ts = filtered_data[:, y, x]
            ts -= np.mean(ts)  # remove DC
            spectrum = np.abs(fft(ts))
            freqs = fftfreq(T, d=1 / fps)

            # Only positive freqs
            mask = freqs > 0
            spectrum = spectrum[mask]
            freqs = freqs[mask]

            peak_indices = np.argsort(spectrum)[-num_fft_peaks:][::-1]
            main_freq = freqs[peak_indices[0]]
            freqs_map[y, x] = main_freq

    # This matches the MATLAB Thouvenin implementation: FreqBand = AcqFreq/20
    freq_band = fps / 20.0
    
    rois = []
    roi_mask = np.zeros_like(freqs_map, dtype=bool)
    
    # Iterate through 9 bands (ff=1 to 9)
    for ff in range(1, 10):
        lower_bound = ff * freq_band
        upper_bound = (ff + 1) * freq_band
        
        # Create binary map for this bandwidth
        bin_mask = (freqs_map > lower_bound) & (freqs_map < upper_bound)
        
        if not np.any(bin_mask):
            continue
            
        # We fill holes first so the 'Area' calculation is more accurate
        filled_mask = morphology.remove_small_holes(bin_mask.astype(bool), area_threshold=min_roi_pixels)
        
        # Label connected components (8-connectivity)
        labeled_mask = measure.label(filled_mask, connectivity=2)
        
        # Find Regions of Interest
        for region in measure.regionprops(labeled_mask):
            # Only select ROIs containing more than min_roi_pixels
            if region.area >= min_roi_pixels and region.area < 1000:
                rois.append(region)
                # Map these pixels to our master ROI mask
                # This replaces the 'ismember' and 'FrequencyMap' update in MATLAB
                roi_mask[labeled_mask == region.label] = True

    # Create ROI frequency map (background set to 0 for better contrast)
    roi_freq_map = np.where(roi_mask, freqs_map, 0)
    # Plot frequency maps
    # fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # im1 = axes[0].imshow(freqs_map, cmap='plasma')
    # axes[0].set_title("Full Frequency Map")
    # axes[0].axis('off')
    # plt.colorbar(im1, ax=axes[0], label="Frequency (Hz)", shrink=0.7)
    
    # im2 = axes[1].imshow(roi_freq_map, cmap='plasma')
    # axes[1].set_title("ROIs (Frequencies)")
    # axes[1].axis('off')
    # plt.colorbar(im2, ax=axes[1], label="Frequency (Hz)", shrink=0.7)
    
    # plt.tight_layout()

    # Plot frequency maps with black background and white text
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Full frequency map
    im1 = axes[0].imshow(freqs_map, cmap='plasma')
    axes[0].set_title("Full Frequency Map", color='black')
    axes[0].axis('off')
    cbar1 = plt.colorbar(im1, ax=axes[0], label="Frequency (Hz)", shrink=0.7)
    cbar1.ax.yaxis.set_tick_params(color='white', labelcolor='black')  # tick numbers white
    cbar1.outline.set_edgecolor('black')
    cbar1.ax.yaxis.label.set_color('black')

    # ROI frequency map
    im2 = axes[1].imshow(roi_freq_map, cmap='plasma')
    axes[1].set_title("ROIs (Frequencies)", color='black')
    axes[1].axis('off')
    cbar2 = plt.colorbar(im2, ax=axes[1], label="Frequency (Hz)", shrink=0.7)
    cbar2.ax.yaxis.set_tick_params(color='black', labelcolor='black')  # tick numbers white
    cbar2.outline.set_edgecolor('black')
    cbar2.ax.yaxis.label.set_color('black')

    plt.tight_layout()

    if SAVE_FIG:
        plt.savefig(os.path.join(SAVE_PATH,"thouvenin_frequency_graphs.png"), format='png', dpi=400)
        print("Saved axis as 'thouvenin_frequency_graphs.png' (400 dpi)")

    roi_freq_values = []

    for region in rois:
        coords = region.coords  # Nx2 array of (row, col)
        pixel_freqs = freqs_map[coords[:, 0], coords[:, 1]]
        median_freq = np.median(pixel_freqs)
        roi_freq_values.append(median_freq)

    plt.figure(figsize=(8,5))
    plt.hist(roi_freq_values, bins=30, color='green', alpha=0.7)
    plt.xlabel("Median Frequency per ROI (Hz)")
    plt.ylabel("Number of ROIs")
    plt.title("Histogram of ROI Median Frequencies")
    plt.grid(True)

    print(f"Video FPS: {fps:.2f}")
    print(f"Found {len(rois)} ROIs meeting size criteria.")

    return freqs_map, roi_freq_map, rois

def canny_segmentation(image, sigma_gaussian=1.0, low_threshold=0, high_threshold=0.8):
    """
    Applies a 2D image processing pipeline: Gaussian smoothing, Sobel edge detection,
    hysteresis thresholding, and hole filling, similar to CiliaQ paper

    NOTE: Not the same exact algorithm as the paper.
    """

    # Image smoothed with a 2D Gaussian kernel
    smoothed_image = gaussian_filter(image, sigma=sigma_gaussian)

    # Canny Edge Detection
    # Calculate gradients in x and y directions
    edge_magnitude = skimage.feature.canny(smoothed_image, sigma=1)

    # A 2D hysteresis threshold is applied
    binary_edges = apply_hysteresis_threshold(edge_magnitude,
                                               low=low_threshold,
                                               high=high_threshold)

    # Holes encapsulated in two dimensions are filled
    filled_image = binary_fill_holes(binary_edges)
    
    return filled_image


print(f"Is Torch Available: {torch.cuda.is_available()}")

# Constants
# Calculate exact pixel length
bar_length_px = SCALE_BAR_LENGTH / pixels_to_microns  # ~9.35 px

# Create a figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(20, 5))  # 2x2 grid of subplots
axes = axes.ravel()  # Flatten the 2D array of axes for easy iteration
original_image = image.copy()

# Plot the original image
axes[0].imshow(image)
axes[0].set_title("Original Image")
axes[0].axis('off')

# Draw the scale bar
x_start = image.shape[1] - 25 # Position (bottom-left)
y_start = 10

axes[0].plot(
    [x_start, x_start + bar_length_px],
    [y_start, y_start],
    color="white",
    linewidth=2,
)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur
blurred = gaussian_filter(gray, sigma=1)
inverted = cv2.bitwise_not(blurred)
# axes[1].imshow(blurred, cmap='gray')
# axes[1].set_title("Blurred Image")
# axes[1].axis('off')

# Li Binary Thresholidng Methodology
threshold = skimage.filters.threshold_li(blurred)
_, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
axes[2].imshow(binary, cmap='gray')
axes[2].set_title(f"Li Binary Thresholding Segmentation (Threshold = {threshold:.2f})")
axes[2].axis('off')

# Adaptive Threshold
ADAPTIVE_BLOCK_SIZE = 11  # Must be odd and >1
ADAPTIVE_C = 2

adaptive_thresh = cv2.adaptiveThreshold(
    blurred, 255, 
    cv2.ADAPTIVE_THRESH_MEAN_C, 
    cv2.THRESH_BINARY, 
    ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C
)

axes[4].imshow(adaptive_thresh, cmap='gray')
axes[4].set_title(f"Adaptive Thresholding with BLOCK_SIZE: {ADAPTIVE_BLOCK_SIZE}, C: {ADAPTIVE_C}")
axes[4].axis('off')

# canny Segmentation
canny_work = canny_segmentation(gray)

axes[3].imshow(canny_work, cmap='gray')
axes[3].set_title(f"3D Canny Segmentation")
axes[3].axis('off')

# Thouvenin Frequency Segmentation
cap = cv2.VideoCapture(VIDEO_FILE_PATH)

frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frames.append(gray.astype(np.float32))
cap.release()

data = np.stack(frames, axis=0)  # shape: (T, H, W)

frequencies, freq_roi_map, rois = thouvenin_frequency_segmentation(data, fps, min_roi_pixels=40)

im2 = axes[5].imshow(freq_roi_map, cmap='plasma')
axes[5].set_title("Thouvenin Frequency Segmentation")
axes[5].axis('off')

# ---------------------------

# Cilia IO

NUMBER_INTERPOLATION_PTS = 100
MATCHING_CONF = 0.4
MAX_DIST = 30  # max center distance to match historical ID (pixels)
MAX_AGE = 5    # frames to keep unmatched IDs before dropping (Used to match similar IDs together :))
POST_IOU_THRESH = 0.5 # Post ByteTrack IOU threshold for matching different boxes together

device = "cuda"
# Load YOLO model
box_model = YOLO("../yolo/runs_cilia_io_submission/detect/train/weights/best.pt")
box_model.to(device)
print("Loaded in YOLO V11m Model")

# Load SAM model
sam_checkpoint = "../sam/sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)
print("Loaded in SAM Model")

# Persistent tracking structures
historical_centers = {}   # id (str) -> np.array([x,y])
track_memory = {}         # id (str) -> last known box (xyxy)
track_age = {}            # id (str) -> frames since last seen
track_history = defaultdict(list)  # id -> list of center points for visualization

next_id = 0
frame_idx = 0

cilia_lengths = defaultdict(list)
cilia_areas = defaultdict(list)

# Keep center points of the box and of the skeleton
track_center_points_box = defaultdict(list)
track_center_points_skeleton = defaultdict(list)       

# 0) Run YoloV11 Detection
result = box_model.track(image, persist=True, tracker="bytetrack.yaml", conf=MATCHING_CONF)[0]
current_boxes = result.boxes.xyxy.cpu().numpy() if result.boxes else np.zeros((0, 4))   # In the future, keep everything on the GPU to minimize data movement.

# 1) Match boxes to historical centers for difficult matches
# This assumes that the cilia are somewhat FIXED and do not otherwise move their entire positions
# We make the key assumption here that cilia basal body is fixed and cannot really move.
assigned_ids = match_by_historical_centers(current_boxes, historical_centers, MAX_DIST) 

# 2) Assign new IDs to unmatched detections
for i in range(len(assigned_ids)):
    if assigned_ids[i] is None:
        assigned_ids[i] = str(next_id)
        next_id += 1

matched_ids = set(assigned_ids)

# 3) Add in old unmatched track_memory entries and increment their age
augmented_boxes = list(current_boxes)
augmented_ids = list(assigned_ids)

for tid in list(track_memory.keys()):
    if tid not in matched_ids:
        track_age[tid] = track_age.get(tid, 0) + 1
        if track_age[tid] <= MAX_AGE:
            # If the age of tracking is low, keep track of it.
            augmented_boxes.append(track_memory[tid])
            augmented_ids.append(tid)
        else:
            print(f"Removing old track ID {tid} with age {track_age[tid]}")
            track_age.pop(tid, None)
            track_memory.pop(tid, None)
            historical_centers.pop(tid, None)
            track_history.pop(tid, None)

augmented_boxes = np.array(augmented_boxes)

# 4) Merge overlapping boxes (across old + new)
merged_boxes, merged_ids = merge_overlapping_boxes(augmented_boxes, augmented_ids, track_age, iou_thresh=POST_IOU_THRESH) 

# 5) Update track memory, historical centers, and reset age for matched IDs
id_to_boxes = defaultdict(list)
for box, tid in zip(merged_boxes, merged_ids):
    id_to_boxes[tid].append(box)

for tid, boxes in id_to_boxes.items():
    chosen_box = boxes[0]

    track_memory[tid] = chosen_box
    historical_centers[tid] = center(chosen_box)
    # Reset age for IDs matched this frame
    if tid in matched_ids:
        track_age[tid] = 0

# Set SAM image
predictor.set_image(image)

# Prepare all boxes at once for batched inference 
input_boxes_list = []
for tid in track_memory.keys():
    box = track_memory[tid]
    x1, y1, x2, y2 = map(int, box)
    input_boxes_list.append([x1, y1, x2, y2])

input_boxes = torch.tensor(input_boxes_list, device=predictor.device)

# Apply SAM’s box transformation
transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])

# Run SAM once for all boxes
masks, scores, logits = predictor.predict_torch(
    point_coords=None,
    point_labels=None,
    boxes=transformed_boxes,
    multimask_output=False
)
end_time = time.time()

masks_np = np.stack([mask.cpu().numpy().squeeze(0) for mask in masks], axis=0)

# keep track of masks for visualization
new_mask_largest_component = np.zeros((masks_np.shape))
mask_id = 0

# keep track of dorsal vs. ventral cilia based on box center (at runtime)
# 6) Skeletonize
for (cilia_id, mask_np) in zip(track_memory.keys(), masks_np):
    box = track_memory[cilia_id]
    x1, y1, x2, y2 = map(int, box)
    #cv2.putText(frame, str(cilia_id), (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Put centers of the boxes 
    box_center_guy = center(box)
    track_center_points_box[cilia_id].append((frame_idx, (box_center_guy[0], box_center_guy[1])))
    
    # Fix mask (when there are two separate islands :) )
    cur_mask = keep_largest_connected_component(mask_np)
    new_mask_largest_component[mask_id] = cur_mask
    mask_id+=1
    h,w = cur_mask.shape

    # Zero pad mask and try segmentation like that
    padded_mask = pad_mask(cur_mask, 5)
    skeleton, centerline_coords = get_skeleton_from_mask(padded_mask)
    ordered_coords_original = order_path(centerline_coords)

    # Extend skeleton :) (use sobel of distance transform of the mask to get it
    distance_map = distance_transform_edt(cur_mask)

    # Compute sobel gradient magnitude
    grad_x = sobel(distance_map, axis=1)  # Horizontal (∂I/∂x)
    grad_y = sobel(distance_map, axis=0)  # Vertical (∂I/∂y)
    sobel_mask = np.hypot(grad_x, grad_y)
    sobel_mask /= np.linalg.norm(sobel_mask) + 1e-8  # Avoid division by zero

    # Center point of skeleton
    center_index = int(ordered_coords_original.shape[0] / 2)
    skel_center = (ordered_coords_original[center_index, 0], ordered_coords_original[center_index, 1])

    # Extend skeleton in one direction toward edge
    starting_point_0 = (ordered_coords_original[0, 0], ordered_coords_original[0, 1])
    additional_path_0 = extend_skeleton(starting_point_0, sobel_mask, distance_map, cur_mask, skel_center, ordered_coords_original,h,w)
    ordered_coords = np.vstack((additional_path_0[::-1], ordered_coords_original[1:]))

    # Extend skeleton in other direction toward edge
    starting_point_1 = (ordered_coords_original[-1, 0], ordered_coords_original[-1, 1])
    additional_path_1 = extend_skeleton(starting_point_1, sobel_mask, distance_map, cur_mask, skel_center, ordered_coords_original,h,w)
    ordered_coords = np.vstack((ordered_coords[:-1], additional_path_1))

    # Ensure all stays inside the mask
    in_mask = cur_mask[ordered_coords[:,0], ordered_coords[:, 1]] > 0
    ordered_coords = ordered_coords[in_mask]

    # Get rid of sharp turns :)
    ordered_coords = remove_sharp_turns(ordered_coords)

    try:
        tck, u = splprep(ordered_coords.T, s=3)  # s=2 is smoothing factor, tune it

        # Evaluate spline fit
        u_fine = np.linspace(0, 1, NUMBER_INTERPOLATION_PTS)
        x_fit, y_fit = splev(u_fine, tck)

        mid_idx = len(x_fit) // 2
        skeleton_center_guy = (y_fit[mid_idx], x_fit[mid_idx])
        track_center_points_skeleton[cilia_id].append((frame_idx, skeleton_center_guy))

        # Plot on frame to look nice
        points = np.stack((y_fit, x_fit), axis=-1).astype(np.int32)

        # DRAW SKELETON
        cv2.polylines(image, [points], isClosed=False, color=(0,0,0), thickness=1)

        # Quantify length of spline :)
        interpolated_skeleton = np.stack((y_fit, x_fit), axis=1)

    except Exception as e:
        print(e)
        print(f"Problem with Cilia: {cilia_id} Frame: {frame_idx}")
        print("Getting original skeleton coords")

# 8) Visualize
image = overlay_masks_once(image, new_mask_largest_component, track_memory.keys(), alpha=0.45)

for tid in sorted(track_memory.keys(), key=lambda x: int(x)):
    box = track_memory[tid]
    x1, y1, x2, y2 = map(int, box)
    c = center(box)
    skeleton_center = track_center_points_skeleton[tid][-1][1]
    track_history[tid].append((int(skeleton_center[0]), int(skeleton_center[1])))
    if len(track_history[tid]) > 30:
        track_history[tid].pop(0)
    pts = np.array(track_history[tid]).reshape((-1, 1, 2))

    # Visualize tracking the center over time :)
    cv2.rectangle(image, (x1, y1), (x2, y2), (83, 83, 83), 1)
    cv2.putText(image, str(tid), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (83, 83, 83), 1)
    cv2.polylines(image, [pts], isClosed=False, color=(255, 255, 255), thickness=1)

axes[1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[1].set_title(f"ciliaIO")
axes[1].axis('off')

if SAVE_FIG:
    fig.savefig(os.path.join(SAVE_PATH, "raw_image.png"), format="png", dpi=400)

    save_axis_high_dpi(axes[0], os.path.join(SAVE_PATH, "original_image_raw.png"))
    save_axis_high_dpi(axes[1], os.path.join(SAVE_PATH, "cilia_io_raw.png"))
    save_axis_high_dpi(axes[2], os.path.join(SAVE_PATH, "li_binary_raw.png"))
    save_axis_high_dpi(axes[3], os.path.join(SAVE_PATH, "canny_raw.png"))
    save_axis_high_dpi(axes[4], os.path.join(SAVE_PATH, "adaptive_threshold_raw.png"))
    save_axis_high_dpi(axes[5], os.path.join(SAVE_PATH, "thouvenin_frequency_raw.png"))

# Define crop helper
def crop(img):
    return img[CROP_Y[0]:CROP_Y[1], CROP_X[0]:CROP_X[1]]

# === PLOTTING ===
fig, axes = plt.subplots(2, 3, figsize=(20, 5))
axes = axes.ravel()

# Cropped original
cropped_orig = crop(original_image)
axes[0].imshow(cropped_orig)
axes[0].set_title("Original Image")
axes[0].axis('off')

# Draw scale bar (bottom-left)
x_start = cropped_orig.shape[1] - 25
y_start =  10

axes[0].plot(
    [x_start, x_start + bar_length_px],
    [y_start, y_start],
    color="white",
    linewidth=2,
)

# === Other methods ===
axes[1].imshow(cv2.cvtColor(crop(image), cv2.COLOR_BGR2RGB))
axes[1].set_title("ciliaIO")
axes[1].axis('off')

axes[2].imshow(crop(binary), cmap='gray')
axes[2].set_title(f"Li Binary Thresholding (Threshold = {threshold:.2f})")
axes[2].axis('off')

axes[3].imshow(crop(canny_work), cmap='gray')
axes[3].set_title("3D Canny Segmentation")
axes[3].axis('off')

axes[4].imshow(crop(adaptive_thresh), cmap='gray')
axes[4].set_title(f"Adaptive Thresholding (Block={ADAPTIVE_BLOCK_SIZE}, C={ADAPTIVE_C})")
axes[4].axis('off')

im2 = axes[5].imshow(crop(freq_roi_map), cmap='plasma')
axes[5].set_title("Thouvenin Frequency Segmentation")
axes[5].axis('off')

if SAVE_FIG:
    plt.savefig(os.path.join(SAVE_PATH, "cropped.png"), format="png", dpi=400)

    save_axis_high_dpi(axes[0], os.path.join(SAVE_PATH, "original_image_cropped.png"))
    save_axis_high_dpi(axes[1], os.path.join(SAVE_PATH, "cilia_io_cropped.png"))
    save_axis_high_dpi(axes[2], os.path.join(SAVE_PATH, "li_binary_cropped.png"))
    save_axis_high_dpi(axes[3], os.path.join(SAVE_PATH, "canny_cropped.png"))
    save_axis_high_dpi(axes[4], os.path.join(SAVE_PATH, "adaptive_threshold_cropped.png"))
    save_axis_high_dpi(axes[5], os.path.join(SAVE_PATH, "thouvenin_frequency_cropped.png"))

plt.show(block=False)
plt.pause(0.001)
input("hit [enter] to end.")
plt.close("all")