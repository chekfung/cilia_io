import time
import numpy as np
import glob
import pickle
from prettytable import PrettyTable
import os
import math
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt

# ML Imports
from ultralytics import YOLO
import torch
from segment_anything import SamPredictor, sam_model_registry

# Image Imports
import cv2
from scipy.ndimage import distance_transform_edt, sobel
from scipy.interpolate import splprep, splev

# Helper File Imports
from ml_helpers import *
from quantification_help import *

'''
This is the main script for the entire repository. 
See the hyperparameters and the call to __main__ to run the program.

The program is configured to take in raw .tif files in a directory
and then selectively go through and apply Cilia.io on them to perform
cilia morphodynamic quantification.
'''

# Some timing debug
timers = {}
loop_timers = defaultdict(list)

def start_timer(name="default"):
    timers[name] = time.time()

def stop_timer(name="default"):
    if name in timers:
        elapsed_time = time.time() - timers[name]
        print(f"{name}: {elapsed_time:.6f} seconds")
        del timers[name]
    else:
        print(f"No timer found for '{name}'")

def stop_timer_loop(name="default"):
    if name in timers:
        elapsed_time = time.time() - timers[name]
        del timers[name]

        # Put elapsed time into the loop_timers
        loop_timers[name].append(elapsed_time)
    else:
        print(f"No timer found for '{name}'")

def reset_loop_timers():
    loop_timers.clear()

def pretty_print_loop_timers():
    if not loop_timers:
        print("No loop timers recorded.")
        return

    table = PrettyTable()
    table.field_names = ["Timer Name", "Average Time (s)", "Min Time (s)", "Max Time (s)", "Total Time (s)"]

    for name, times in loop_timers.items():
        average_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        total_time = sum(times)

        table.add_row([name, f"{average_time:.6f}", f"{min_time:.6f}", f"{max_time:.6f}", f"{total_time:.6f}"])

    print(table)

# ---- HYPERPARAMETERS ------
# Video Parameters
# NOTE: Edit Here!
dirname = '../data/test_cilia_io'
DORSAL_VENTRAL_THRESHOLD_LIST = [85]

# YOLO Parameters
MATCHING_CONF = 0.4     # Confidence to be identified by YOLO as real point. 

# ByteTrack / Post ByteTrack YOLO Tracker parameters
MAX_DIST = 30  # max center distance to match historical ID (pixels)
MAX_AGE = 5    # frames to keep unmatched IDs before dropping (Used to match similar IDs together :))
POST_IOU_THRESH = 0.5 # Post ByteTrack IOU threshold for matching different boxes together

# SAM Params

# Skeletonization Parameters
NUMBER_INTERPOLATION_PTS = 100
DORSAL_VENTRAL_THRESHOLD_GLOBAL = 80       # Refers to where we make a horizontal cut of what is and is not dorsal / ventral. Note, we group in 0, or 1, as I am not sure which is which :)
STATIC_DORSAL_VENTRAL_THRESHOLD = False
MIN_NUMBER_POINTS_TO_EXAMINE_MOTILITY = 1
OUTPUT_FPS = 5
EARLY_STOP_DEBUG = True
RANDOM_COLORS = True        # Draw random colors for each of the cilia, or use dorsal ventral coloring

DORSAL_COLOR = (179, 110, 59)
VENTRAL_COLOR = (132, 172, 195)

# ML Parameters
device = "cuda"
yolo_model_filepath = "../yolo/runs/detect/train/weights/best.pt"

sam_checkpoint = "../sam/sam_vit_h_4b8939.pth"
model_type = "vit_h"
# ---- END HYPERPARAMETERS ------]

# Load YOLOv11 model
model = YOLO(yolo_model_filepath).to(device)
print("Loaded in Fine-Tuned YoloV11 Model")

# Load SAM model
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)
print("Loaded in SAM Model")

def cilia_io(file, pixels_to_microns, fps, DORSAL_VENTRAL_THRESHOLD):
    # Load in Video File :)
    cap = cv2.VideoCapture(file)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Frame Width: {frame_width}")
    print(f"Frame Height: {frame_height}")

    # Define VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    dirname = os.path.dirname(file)
    fname = os.path.basename(file)
    output_path = os.path.join(dirname, fname.replace(".mp4", f"_{OUTPUT_FPS}fps_ciliaio_output.mp4"))
    out = cv2.VideoWriter(output_path, fourcc, OUTPUT_FPS, (frame_width, frame_height))
    print(f"Saving Cilia.io Output File to: {output_path}")

    # Persistent tracking structures
    historical_centers = {}   # id (str) -> np.array([x,y])
    track_memory = {}         # id (str) -> last known box (xyxy)
    track_age = {}            # id (str) -> frames since last seen
    track_history = defaultdict(list)  # id -> list of center points for visualization

    next_id = 0
    frame_idx = 0

    cilia_lengths = defaultdict(list)
    cilia_straightness_flipped = defaultdict(list)
    cilia_areas = defaultdict(list)

    # Keep center points of the box and of the skeleton
    track_center_points_box = defaultdict(list)
    track_center_points_skeleton = defaultdict(list)           
    start_timer("Full Run")

    while cap.isOpened():
        # Loop through each frame and then do YOLO -> post-process -> SAM -> skeletonize
        success, frame = cap.read()

        if not success:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_idx += 1
        print(f"\nFrame {frame_idx}")

        # 0) Run YoloV11 Detection
        start_timer("Yolo Run")
        result = model.track(rgb_frame, persist=True, tracker="bytetrack.yaml", conf=MATCHING_CONF)[0]
        current_boxes = result.boxes.xyxy.cpu().numpy() if result.boxes else np.zeros((0, 4))   # In the future, keep everything on the GPU to minimize data movement.
        stop_timer_loop("Yolo Run")

        # 1) Match boxes to historical centers for difficult matches
        # This assumes that the cilia are somewhat FIXED and do not otherwise move their entire positions
        # We make the key assumption here that cilia basal body is fixed and cannot really move.
        start_timer("Match Historical Centers")
        assigned_ids = match_by_historical_centers(current_boxes, historical_centers, MAX_DIST) 
        stop_timer_loop("Match Historical Centers")

        # 2) Assign new IDs to unmatched detections
        for i in range(len(assigned_ids)):
            if assigned_ids[i] is None:
                assigned_ids[i] = str(next_id)
                next_id += 1

        matched_ids = set(assigned_ids)

        # 3) Add in old unmatched track_memory entries and increment their age
        augmented_boxes = list(current_boxes)
        augmented_ids = list(assigned_ids)

        start_timer("Assign Tracking IDs")
        for tid in list(track_memory.keys()):
            if tid not in matched_ids:
                track_age[tid] = track_age.get(tid, 0) + 1
                if track_age[tid] <= MAX_AGE:
                    # If the age of tracking is low, keep track of it.
                    augmented_boxes.append(track_memory[tid])
                    augmented_ids.append(tid)
                else:
                    # If an ID is too old, and we have not seen it in a MAX-AGE, we remove it (maybe it died or something?)
                    print(f"Removing old track ID {tid} with age {track_age[tid]}")
                    track_age.pop(tid, None)
                    track_memory.pop(tid, None)
                    historical_centers.pop(tid, None)
                    track_history.pop(tid, None)
        stop_timer_loop("Assign Tracking IDs")

        augmented_boxes = np.array(augmented_boxes)

        # 4) Merge overlapping boxes (across old + new)
        start_timer("Merge Overlapping Boxes")
        merged_boxes, merged_ids = merge_overlapping_boxes(augmented_boxes, augmented_ids, track_age, iou_thresh=POST_IOU_THRESH)  
        stop_timer_loop("Merge Overlapping Boxes")

        # 5) Update track memory, historical centers, and reset age for matched IDs
        start_timer("Update track Memory")
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
        stop_timer_loop("Update track Memory")

        # Set SAM image
        start_timer("SAM")
        start_time = time.time()
        start_timer("SAM: Load Image and Compute Embeddings")
        predictor.set_image(rgb_frame)
        stop_timer_loop("SAM: Load Image and Compute Embeddings")
        
        # Prepare all boxes at once for batched inference 
        input_boxes_list = []
        start_timer("SAM: Get Boxes")
        for tid in track_memory.keys():
            box = track_memory[tid]
            x1, y1, x2, y2 = map(int, box)
            input_boxes_list.append([x1, y1, x2, y2])

        input_boxes = torch.tensor(input_boxes_list, device=predictor.device)
        stop_timer_loop("SAM: Get Boxes")

        # Apply SAM’s box transformation
        start_timer("SAM: Box Transform")
        transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, rgb_frame.shape[:2])
        stop_timer_loop("SAM: Box Transform")

        # Run SAM once for all boxes
        start_timer("SAM: ML-RUN")
        masks, scores, logits = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False
        )
        stop_timer_loop("SAM: ML-RUN")
        stop_timer_loop("SAM")
        end_time = time.time()
        print(f"SAM took: {end_time - start_time}s to run on {len(input_boxes):.2f} detected boxes")

        start_timer("Bring Masks from GPU to CPU")
        masks_np = np.stack([mask.cpu().numpy().squeeze(0) for mask in masks], axis=0)
        stop_timer_loop("Bring Masks from GPU to CPU")

        # keep track of masks for visualization
        new_mask_largest_component = np.zeros((masks_np.shape))
        mask_id = 0

        # keep track of dorsal vs. ventral cilia based on box center (at runtime)
        cilia_id_colors = []

        # 6) Skeletonize
        for (cilia_id, mask_np) in zip(track_memory.keys(), masks_np):
            box = track_memory[cilia_id]
            x1, y1, x2, y2 = map(int, box)
            #cv2.putText(frame, str(cilia_id), (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Put centers of the boxes 
            box_center_guy = center(box)
            track_center_points_box[cilia_id].append((frame_idx, (box_center_guy[0], box_center_guy[1])))

            # Assign runtime coloring for cilia
            if box_center_guy[1] < DORSAL_VENTRAL_THRESHOLD:
                cilia_id_colors.append(DORSAL_COLOR)
            else:
                cilia_id_colors.append(VENTRAL_COLOR)

            start_timer("keep largest component")
            
            # Fix mask (when there are two separate islands :) )
            cur_mask = keep_largest_connected_component(mask_np)
            new_mask_largest_component[mask_id] = cur_mask
            mask_id+=1
            h,w = cur_mask.shape

            stop_timer_loop("keep largest component")

            # Zero pad mask and try segmentation like that
            start_timer("original skeletonize")
            padded_mask = pad_mask(cur_mask, 5)
            skeleton, centerline_coords = get_skeleton_from_mask(padded_mask)
            ordered_coords_original = order_path(centerline_coords)
            stop_timer_loop("original skeletonize")

            # Extend skeleton :) (use sobel of distance transform of the mask to get it
            start_timer("distance transform and sobel")
            distance_map = distance_transform_edt(cur_mask)

            # Compute sobel gradient magnitude
            grad_x = sobel(distance_map, axis=1)  # Horizontal (∂I/∂x)
            grad_y = sobel(distance_map, axis=0)  # Vertical (∂I/∂y)
            sobel_mask = np.hypot(grad_x, grad_y)
            sobel_mask /= np.linalg.norm(sobel_mask) + 1e-8  # Avoid division by zero
            stop_timer_loop("distance transform and sobel")

            # Center point of skeleton
            center_index = int(ordered_coords_original.shape[0] / 2)
            skel_center = (ordered_coords_original[center_index, 0], ordered_coords_original[center_index, 1])

            # Extend skeleton in one direction toward edge
            start_timer("extend skeleton")
            starting_point_0 = (ordered_coords_original[0, 0], ordered_coords_original[0, 1])
            additional_path_0 = extend_skeleton(starting_point_0, sobel_mask, distance_map, cur_mask, skel_center, ordered_coords_original,h,w)
            ordered_coords = np.vstack((additional_path_0[::-1], ordered_coords_original[1:]))

            # Extend skeleton in other direction toward edge
            starting_point_1 = (ordered_coords_original[-1, 0], ordered_coords_original[-1, 1])
            additional_path_1 = extend_skeleton(starting_point_1, sobel_mask, distance_map, cur_mask, skel_center, ordered_coords_original,h,w)
            ordered_coords = np.vstack((ordered_coords[:-1], additional_path_1))
            stop_timer_loop("extend skeleton")

            # Ensure all stays inside the mask
            in_mask = cur_mask[ordered_coords[:,0], ordered_coords[:, 1]] > 0
            ordered_coords = ordered_coords[in_mask]

            # Get rid of sharp turns :)
            start_timer("remove sharp turns")
            ordered_coords = remove_sharp_turns(ordered_coords)
            stop_timer_loop("remove sharp turns")

            # Fit spline curve
            try:
                start_timer("spline fit")
                tck, u = splprep(ordered_coords.T, s=3)  # s=2 is smoothing factor, tune it

                # Evaluate spline fit
                u_fine = np.linspace(0, 1, NUMBER_INTERPOLATION_PTS)
                x_fit, y_fit = splev(u_fine, tck)

                mid_idx = len(x_fit) // 2
                skeleton_center_guy = (y_fit[mid_idx], x_fit[mid_idx])
                track_center_points_skeleton[cilia_id].append((frame_idx, skeleton_center_guy))

                stop_timer_loop("spline fit")
            
                # Plot on frame to look nice
                points = np.stack((y_fit, x_fit), axis=-1).astype(np.int32)

                # Visualize: DRAW SKELETON
                cv2.polylines(frame, [points], isClosed=False, color=(0,0,0), thickness=1)

                # Quantify length of spline :)
                interpolated_skeleton = np.stack((y_fit, x_fit), axis=1)

                cilia_length = arc_length(interpolated_skeleton) * pixels_to_microns

                # Calculate straightness (Note: straightness is with respect to the straight line)
                # 1 = perfectly straght, Otherwise greater. closer to 1, the more straight it is.
                cilia_length_original = arc_length(ordered_coords_original) * pixels_to_microns # Use original skeleton :)
                cilia_shortest_path = math.hypot(ordered_coords_original[-1,0] - ordered_coords_original[0,0], ordered_coords_original[-1,1] - ordered_coords_original[0,1]) * pixels_to_microns
                cilia_straightness_flipped[cilia_id].append(cilia_shortest_path / cilia_length_original)

                # Get Area of cilia
                cilia_area = np.count_nonzero(cur_mask) * (pixels_to_microns * pixels_to_microns)
                cilia_lengths[cilia_id].append(cilia_length)
                cilia_areas[cilia_id].append(cilia_area)

            except Exception as e:
                print(e)
                print(f"Problem with Cilia: {cilia_id} Frame: {frame_idx}")
                print("Getting original skeleton coords")

                mid_idx = len(ordered_coords) // 2

                if len(ordered_coords) != 0:
                    skeleton_center_guy = ordered_coords[mid_idx]
                    track_center_points_skeleton[cilia_id].append((frame_idx, (skeleton_center_guy[1], skeleton_center_guy[0])))
                continue

        # Put the center guys away
        if EARLY_STOP_DEBUG:
            if frame_idx == 5:
                break

        # Draw Masks
        if RANDOM_COLORS:
            frame = overlay_masks_once(frame, new_mask_largest_component, track_memory.keys(), alpha=0.45)
        else:
            frame = overlay_masks_once(frame, new_mask_largest_component, track_memory.keys(), cilia_id_colors, alpha=0.45)

        # 8) Visualize
        for tid in sorted(track_memory.keys(), key=lambda x: int(x)):
            box = track_memory[tid]
            x1, y1, x2, y2 = map(int, box)
            c = center(box)
            try:
                skeleton_center = track_center_points_skeleton[tid][-1][1]
                track_history[tid].append((int(skeleton_center[0]), int(skeleton_center[1])))
                if len(track_history[tid]) > 30:
                    track_history[tid].pop(0)
                pts = np.array(track_history[tid]).reshape((-1, 1, 2))

                # Visualize tracking the center over time :)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (83, 83, 83), 1)
                cv2.putText(frame, str(tid), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (83, 83, 83), 1)
                cv2.polylines(frame, [pts], isClosed=False, color=(255, 255, 255), thickness=1)
            except Exception as e:
                print(f"ERROR: Could not paint cilia: {tid} on frame {frame_idx} due to {e}")

        # Draw horizontal line denoting where we split thef rame
        #cv2.line(frame, (0, DORSAL_VENTRAL_THRESHOLD), (frame_width - 1, DORSAL_VENTRAL_THRESHOLD), color=(255, 0, 0), thickness=1)

        cv2.imshow(f"Cilia.io: {file}", frame)
        out.write(frame)

        # if frame_idx >= 0:
        #     input(f"Paused at frame {frame_idx} — press Enter to continue")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


    # Quantify Cilia Length, Area per cilia
    mean_lengths = {}
    mean_areas = {}

    for cilia_id in cilia_lengths:
        mean_lengths[cilia_id] = sum(cilia_lengths[cilia_id]) / len(cilia_lengths[cilia_id])

    for cilia_id in cilia_areas:
        mean_areas[cilia_id] = sum(cilia_areas[cilia_id]) / len(cilia_areas[cilia_id])

    #print(f"Mean Length / Cilia (microns): {mean_lengths}")
    #print(f"Mean Area / Cilia (microns): {mean_areas}")

    # Quantify Motility Data Outside 
    start_timer("Motility Box Center")
    quantification_data = {}

    for cilia_id, track in track_center_points_box.items():
        # Go through each tracked cilia and go and get the features and plot it
        track = sorted(track, key=lambda z: z[0])

        # Extract times
        times = [p[0] for p in track]

        # Check if times are consecutive integers
        time_diffs = np.diff(times)
        if not all(time_diffs == 1):
            print(f"Cilia ID {cilia_id} has non-consecutive or unordered times: {times}")
            continue

        xy = np.array([p[1] for p in track], dtype=float)

        # Get Features
        points =  xy.reshape(-1,2)
        features = extract_motion_features(cilia_id, points, pixels_to_microns, fps=fps, pixel_group_threshold=DORSAL_VENTRAL_THRESHOLD, MIN_NUMBER_POINTS=MIN_NUMBER_POINTS_TO_EXAMINE_MOTILITY)

        if features != None:
            # While we are here, add in the cilia length and area
            if cilia_id in mean_lengths:
                features["mean_cilia_length"] = mean_lengths[cilia_id]
            else:
                print(f"Error: mean_lengths missing for cilia_id: {cilia_id}")

            if cilia_id in mean_areas:
                features["mean_cilia_area"] = mean_areas[cilia_id]
            else:
                print(f"Error: mean_areas missing for cilia_id: {cilia_id}")
            
            if cilia_id in cilia_straightness_flipped:
                # Note: Straightness flipped still refers to straightness, just a different way of expressing it with the fraction flipped :)
                features["mean_straightness_flipped"] = np.mean(cilia_straightness_flipped[cilia_id])
                features["std_straightness_flipped"] = np.std(cilia_straightness_flipped[cilia_id], ddof=1)
                features["max_straightness_flipped"] = np.max(cilia_straightness_flipped[cilia_id])
                features["min_straightness_flipped"] = np.min(cilia_straightness_flipped[cilia_id])
            else:
                print(f"Error: std_angle missing for cilia_id: {cilia_id}")

            quantification_data[cilia_id] = features

    # Now we have quantification data per guy
    all_features_list = list(quantification_data.values())

    # Create DataFrame and save CSV
    df = pd.DataFrame(all_features_list)
    box_csv = os.path.join(dirname, fname.replace(".mp4", "_quant_box_center.csv"))
    df.to_csv(box_csv, index=False)
    print(f"Saving Box CSV to: {box_csv}")

    stop_timer_loop("Motility Box Center")

    # ------
    # Do for skeleton center
    start_timer("Motility Skel Center")
    quantification_data = {}

    for cilia_id, track in track_center_points_skeleton.items():
        # Go through each tracked cilia and go and get the features and plot it
        track = sorted(track, key=lambda z: z[0])

        # Extract times
        times = [p[0] for p in track]

        # Check if times are consecutive integers
        time_diffs = np.diff(times)
        if not all(time_diffs == 1):
            print(f"Cilia ID {cilia_id} has non-consecutive or unordered times: {times}")
            continue

        xy = np.array([p[1] for p in track], dtype=float)

        # Get Features
        points =  xy.reshape(-1,2)
        features = extract_motion_features(cilia_id, points, pixels_to_microns, fps=fps, pixel_group_threshold=DORSAL_VENTRAL_THRESHOLD, MIN_NUMBER_POINTS=MIN_NUMBER_POINTS_TO_EXAMINE_MOTILITY)

        if features != None:
            # While we are here, add in the cilia length and area
            if cilia_id in mean_lengths:
                features["mean_cilia_length"] = mean_lengths[cilia_id]
            else:
                print(f"Error: mean_lengths missing for cilia_id: {cilia_id}")

            if cilia_id in mean_areas:
                features["mean_cilia_area"] = mean_areas[cilia_id]
            else:
                print(f"Error: mean_areas missing for cilia_id: {cilia_id}")

            if cilia_id in cilia_straightness_flipped:
                # Note: Straightness flipped still refers to straightness, just a different way of expressing it with the fraction flipped :)
                features["mean_straightness_flipped"] = np.mean(cilia_straightness_flipped[cilia_id])
                features["std_straightness_flipped"] = np.std(cilia_straightness_flipped[cilia_id], ddof=1)
                features["max_straightness_flipped"] = np.max(cilia_straightness_flipped[cilia_id])
                features["min_straightness_flipped"] = np.min(cilia_straightness_flipped[cilia_id])

            else:
                print(f"Error: std_angle missing for cilia_id: {cilia_id}")

            quantification_data[cilia_id] = features

    # Now we have quantification data per guy
    all_features_list = list(quantification_data.values())

    # Create DataFrame and save CSV
    df = pd.DataFrame(all_features_list)
    box_csv = os.path.join(dirname, fname.replace(".mp4", "_quant_skel_center.csv"))
    df.to_csv(box_csv, index=False)
    print(f"Saving Box CSV to: {box_csv}")
    stop_timer_loop("Motility Skel Center")

    # -----------
    # Save to pickle 
    pickle_name = os.path.join(dirname, f'{fname}_box_center_points.pkl')
    with open(pickle_name, 'wb') as f:
        pickle.dump(track_center_points_box, f)

    pickle_name = os.path.join(dirname, f'{fname}_skel_center_points.pkl')
    with open(pickle_name, 'wb') as f:
        pickle.dump(track_center_points_skeleton, f)

    cap.release()
    cv2.destroyAllWindows()
    out.release()

    stop_timer("Full Run")
    pretty_print_loop_timers()
    reset_loop_timers()



# -------------
if __name__ == "__main__":
    # ------------------------------------
    # Get the files in a directory
    tif_files = glob.glob(os.path.join(dirname, "*.tif"))
    print(tif_files)
    dorsal_index=0

    for file in tif_files:
        try:
            print(f'Running File: {file}')
            dirname = os.path.dirname(file)
            fname = os.path.basename(file)
            
            # Read metadata to get pixel conversion?
            pixels_to_microns, fps, metadata_dict = tiff_get_metadata(file)
            print(f"One Pixel = {pixels_to_microns} Microns")
            print(f"FPS: {fps}")

            # Convert to mp4
            new_name = os.path.join(dirname, fname.replace(".tif", ".mp4"))
            tiff_to_mp4(file, new_name,fps=fps)

            # run files
            if STATIC_DORSAL_VENTRAL_THRESHOLD:
                print(f"DORSAL THRESHOLD: {DORSAL_VENTRAL_THRESHOLD_GLOBAL}")
                cilia_io(new_name, pixels_to_microns, fps, DORSAL_VENTRAL_THRESHOLD_GLOBAL)
            else:
                print(f"DORSAL THRESHOLD: {DORSAL_VENTRAL_THRESHOLD_LIST[dorsal_index]}")
                cilia_io(new_name, pixels_to_microns, fps, DORSAL_VENTRAL_THRESHOLD_LIST[dorsal_index])
            dorsal_index+=1
        except Exception as e:
            print(f"ERROR: on file: {file} ; Error Str: {e}")
            dorsal_index+=1
            continue
