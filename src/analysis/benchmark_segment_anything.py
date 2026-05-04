import os
import json
import glob
import cv2
import csv
import numpy as np
import torch
from scipy.spatial.distance import cdist

import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')  # Essential for stable GUI windows
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
from matplotlib.lines import Line2D

from segment_anything import sam_model_registry, SamPredictor

DISPLAY_AND_SAVE_PICTURES = False        # Increases time for running by quite a bit

def decode_brush_rle_fast(rle, height, width):
    """
    Optimized Label Studio Brush RLE decoding using NumPy bit manipulation.
    
    Adapted from
    https://stackoverflow.com/questions/74339154/how-to-convert-rle-format-of-label-studio-to-black-and-white-image-masks
    """
    data = np.array(rle, dtype=np.uint8)
    bits = np.unpackbits(data)
    ptr = 0
    def read_bits(n):
        nonlocal ptr
        val = 0
        for _ in range(n):
            val = (val << 1) | bits[ptr]
            ptr += 1
        return val

    num_pixels = read_bits(32)
    word_size = read_bits(5) + 1
    rle_sizes = [read_bits(4) + 1 for _ in range(4)]
    
    out = np.zeros(num_pixels, dtype=np.uint8)
    i = 0
    while i < num_pixels:
        is_run = bits[ptr]; ptr += 1
        length_bits = rle_sizes[read_bits(2)]
        run_length = read_bits(length_bits) + 1
        if is_run:
            val = read_bits(word_size)
            out[i : i + run_length] = val
            i += run_length
        else:
            for _ in range(run_length):
                out[i] = read_bits(word_size); i += 1
    try:
        # LS Brush usually encodes as RGBA; we need the Alpha (3) channel
        return out.reshape((height, width, 4))[:, :, 3]
    except:
        return out.reshape((height, width))

def yolo_to_bbox(yolo_line, img_w, img_h):
    """Converts YOLO [cls cx cy w h] to [x_min, y_min, w, h] pixels."""
    parts = list(map(float, yolo_line.split()))
    if len(parts) < 5: return None
    _, xc, yc, w, h = parts
    bw, bh = w * img_w, h * img_h
    xmin, ymin = (xc * img_w) - (bw/2), (yc * img_h) - (bh/2)
    return [xmin, ymin, bw, bh]

def get_centers(masks, bboxes):
    """Calculates centroids for a list of masks and a list of bboxes."""
    m_centers = []
    for m in masks:
        pos = np.where(m > 0)
        m_centers.append([np.mean(pos[1]), np.mean(pos[0])] if len(pos[0]) > 0 else [0, 0])
    
    b_centers = []
    for b in bboxes:
        # Center of box is x + w/2, y + h/2
        b_centers.append([b[0] + b[2]/2, b[1] + b[3]/2])
    
    return np.array(m_centers), np.array(b_centers)

def calculate_iou(gt_mask, pred_mask):
    """Standard Intersection over Union calculation."""
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    return intersection / union if union > 0 else 0

def calculate_dice(gt_mask, pred_mask):
    """Calculates the Dice Coefficient (F1-score) for two binary masks."""
    gt = gt_mask > 0
    pred = pred_mask > 0
    intersection = np.logical_and(gt, pred).sum()
    total_area = gt.sum() + pred.sum()
    return (2.0 * intersection) / total_area if total_area > 0 else 1.0

def create_segmentation_dataset(json_dir, dist_threshold=100, show_debug=False):
    """
    Creates a matched dataset of YOLO boxes and Label Studio masks.
    Includes a visualization step to audit the synchronization of both sources.
    """

    dataset = {}
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    
    # Global counters for the final report
    total_yolo_boxes = 0
    total_ls_masks = 0
    total_matches = 0

    for json_path in json_files:
        proj = os.path.splitext(os.path.basename(json_path))[0]
        img_dir = f"../data/labeled_data/{proj}/images"
        lbl_dir = f"../data/labeled_data/{proj}/labels"

        with open(json_path, 'r') as f:
            tasks = json.load(f)

        for task in tasks:
            # Extract Image URL/Path
            url = task.get("data", {}).get("img") or task.get("data", {}).get("image")
            if not url: continue
            
            # Match image filename
            fname = os.path.basename(url).split('-', 1)[-1]
            img_paths = glob.glob(os.path.join(img_dir, f"*{fname}"))
            if not img_paths: continue
            
            img = cv2.cvtColor(cv2.imread(img_paths[0]), cv2.COLOR_BGR2RGB)
            h, w, _ = img.shape
            fid = os.path.splitext(os.path.basename(img_paths[0]))[0]
            
            # Load YOLO Bounding Boxes
            boxes = []
            l_path = os.path.join(lbl_dir, f"{fid}.txt")
            if os.path.exists(l_path):
                with open(l_path, 'r') as f:
                    # Uses yolo_to_bbox utility defined earlier in the script
                    boxes = [yolo_to_bbox(line, w, h) for line in f if yolo_to_bbox(line, w, h)]

            # Load Label Studio Masks
            masks = []
            for ann in task.get("annotations", []):
                for res in ann.get("result", []):
                    if res.get('type') == 'brushlabels':
                        # Uses decode_brush_rle_fast utility defined earlier
                        m = (decode_brush_rle_fast(res['value']['rle'], h, w) > 0).astype(np.uint8)

                        pixel_count = np.sum(m)
                        if pixel_count > 10:
                            masks.append(m)
                        else:
                            print("Pixel Count less Than 10!")

            # Proximity Matching (Centroid Distance)
            instances = []
            used_m, used_b = set(), set()
            
            if boxes and masks:
                m_pts, b_pts = get_centers(masks, boxes)
                dists = cdist(m_pts, b_pts) # Matrix of distances
                
                # Sort all possible pairs by proximity
                flat_dists = []
                for m_idx in range(len(masks)):
                    for b_idx in range(len(boxes)):
                        flat_dists.append((dists[m_idx, b_idx], m_idx, b_idx))
                flat_dists.sort() 

                # Matches by closest distance first (so get rid of easy matches, then deal with the hard ones)
                for d, m_idx, b_idx in flat_dists:
                    if m_idx not in used_m and b_idx not in used_b and d < dist_threshold:
                        instances.append({"mask": masks[m_idx], "bbox": boxes[b_idx]})
                        used_m.add(m_idx)
                        used_b.add(b_idx)

            status = "PERFECT" if (len(boxes) == len(masks) == len(instances)) else "MISMATCH"
            print(f"[{status}] {fid:40} | YOLO: {len(boxes):<3} | LS: {len(masks):<3} | Matched: {len(instances):<3}")

            # Debug Visualization
            # Look specifically at one frame
            frame_in_question = "2e903909-wt_3_f001_frame_45"
            EXAMINE_ONE_FRAME = False
            FORCE_FRAME_DEBUG = (fid == frame_in_question) and EXAMINE_ONE_FRAME

            if show_debug and (boxes or masks) or FORCE_FRAME_DEBUG:
                fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                ax.imshow(img)
                colors = cm.get_cmap('hsv', len(instances) + 2)

                # Draw Matched Pairs
                for i, inst in enumerate(instances):
                    color = colors(i)
                    bx, by, bw, bh = inst['bbox']
                    # Box
                    ax.add_patch(Rectangle((bx, by), bw, bh, lw=2, edgecolor=color, facecolor='none'))
                    # Mask
                    m_overlay = np.zeros((*inst['mask'].shape, 4))
                    m_overlay[inst['mask'] > 0] = [*color[:3], 0.5]
                    ax.imshow(m_overlay)
                    ax.text(bx, by-5, f"Match {i}", color=color, fontweight='bold', fontsize=9)

                # Highlight Unmatched Boxes (YOLO prompt with no GT Mask)
                for b_idx, box in enumerate(boxes):
                    if b_idx not in used_b:
                        bx, by, bw, bh = box
                        ax.add_patch(Rectangle((bx, by), bw, bh, lw=1.5, edgecolor='white', ls='--', facecolor='none'))
                        ax.text(bx, by-5, "BOX NO MASK", color='white', fontsize=8, backgroundcolor='red')

                # Highlight Unmatched Masks (GT Mask with no YOLO prompt)
                for m_idx, mask in enumerate(masks):
                    if m_idx not in used_m:
                        m_overlay = np.zeros((*mask.shape, 4))
                        m_overlay[mask > 0] = [0.5, 0.5, 0.5, 0.6] # Gray for unmatched
                        ax.imshow(m_overlay)
                        # Find centroid for text placement
                        pos = np.where(mask > 0)
                        ax.text(np.mean(pos[1]), np.mean(pos[0]), "MASK NO BOX", color='white', fontsize=8, backgroundcolor='gray')

                ax.set_title(f"Sync Audit: {fid}\nMatched: {len(instances)} | Boxes: {len(boxes)} | Masks: {len(masks)}")
                plt.axis('off')
                plt.tight_layout()
                plt.show()

            # Update counters
            total_yolo_boxes += len(boxes)
            total_ls_masks += len(masks)
            total_matches += len(instances)

            dataset[fid] = {
                "image": img, 
                "instances": instances, 
                "project": proj,
                "orig_box_count": len(boxes),
                "orig_mask_count": len(masks)
            }
            
    # Print Final Summary
    print("\n" + "="*50)
    print(" DATASET SYNCHRONIZATION REPORT ")
    print("="*50)
    print(f"Total YOLO Boxes found:    {total_yolo_boxes}")
    print(f"Total LS Masks found:      {total_ls_masks}")
    print(f"Successfully Paired:       {total_matches}")
    print(f"Sync Efficiency:           {(total_matches/max(1, total_yolo_boxes))*100:.1f}%")
    print("="*50 + "\n")

    return dataset

def run_sam_evaluation(dataset, checkpoint_path, visualize=False, save_directory=None):
    """
    Feeds each ground truth box into SAM and compares generated mask against GT.
    Tracks IoU and DICE per frame, per video, and globally.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    print(f"SAM Loaded on {device}")

    # Hierarchical storage: { video_id: { frame_id: [(iou1, dice1), (iou2, dice2)...] } }
    stats = {}

    for frame_id, data in dataset.items():
        # Extract Video ID (Assumes format: name_frame_XXX)
        try:
            after_id = frame_id.split('-', 1)[1]
            video_id = after_id.split('_frame_')[0]
        except IndexError:
            video_id = data.get('project', 'unknown_video')

        if video_id not in stats:
            stats[video_id] = {}
        
        stats[video_id][frame_id] = []
        
        print(f"Evaluating SAM on: {frame_id} (Video: {video_id})...")
        predictor.set_image(data['image'])
        
        all_results = []
        
        for inst in data['instances']:
            # SAM box format: [x1, y1, x2, y2]
            x, y, w, h = inst['bbox']
            input_box = np.array([x, y, x + w, y + h])

            masks, _, _ = predictor.predict(box=input_box[None, :], multimask_output=False)
            pred_mask = masks[0]
            
            # Calculate both metrics
            iou = calculate_iou(inst['mask'], pred_mask)
            dice = calculate_dice(inst['mask'], pred_mask)
            
            stats[video_id][frame_id].append((iou, dice))

            all_results.append({
                'box': [x, y, w, h],
                'gt_mask': inst['mask'],
                'pred_mask': pred_mask,
                'dice': dice
            })
        
        if all_results and DISPLAY_AND_SAVE_PICTURES:
            plt.figure(figsize=(15, 10))
            plt.imshow(data['image'])
            ax = plt.gca()

            for res in all_results:
                # Blue mask for ground truth
                gt_rgba = np.zeros((*res['gt_mask'].shape, 4))
                gt_rgba[res['gt_mask'] > 0] = [0.502, 1.0, 0.502, 0.5] 
                ax.imshow(gt_rgba)

                # 2. Create an RGBA 'Orange' mask for SAM Prediction
                pred_rgba = np.zeros((*res['pred_mask'].shape, 4))
                pred_rgba[res['pred_mask'] > 0] = [0.945, 0.055, 0.835, 0.4]
                ax.imshow(pred_rgba)

                # 3. Plot Bounding Box
                x, y, w, h = res['box']
                rect = patches.Rectangle((x, y), w, h, linewidth=0.6, edgecolor='white', 
                                         facecolor='none', alpha=0.3)
                ax.add_patch(rect)
                
                # Optional: Label IoU near the box
                ax.text(x, y-2, f"{res['dice']:.2f}", color='white', fontsize=6, 
                        bbox=dict(facecolor='black', alpha=0.5, pad=0))
                
            legend_elements = [
                Line2D([0], [0], color=(0.502, 1.0, 0.502), lw=4, label='Ground Truth'),
                Line2D([0], [0], color=(0.945, 0.055, 0.835), lw=4, label='SAM Prediction'),
            ]
            ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)

            # Make the scale bar
            pixels_to_micron = 0.107438780772816        # one pixel = this many microns
            scale_bar_microns = 2.5
            
            bar_width_px = scale_bar_microns / pixels_to_micron 
            
            img_h, img_w = data['image'].shape[:2]
            
            # Define margins and thickness for journal standards
            margin_x = img_w * 0.05  # 5% margin from right
            margin_y = img_h * 0.05  # 5% margin from bottom
            bar_thickness = max(2, img_h * 0.01) # Dynamic thickness based on resolution
            
            # Coordinates for the bar
            bar_x = img_w - margin_x - bar_width_px
            bar_y = img_h - margin_y - bar_thickness
            
            # Draw the bar
            scale_bar = patches.Rectangle(
                (bar_x, bar_y), 
                bar_width_px, bar_thickness, 
                linewidth=0, facecolor='white', edgecolor='none'
            )
            ax.add_patch(scale_bar)

            plt.title(f"Frame: {frame_id}, scale_bar = {scale_bar_microns} microns")
            plt.axis('off')

            plt.savefig(os.path.join(save_directory, f"{frame_id}.png"), format='png', dpi=300)
                
            if visualize:
                plt.show()

            plt.cla()    # Clear the current axes
            plt.clf()    # Clear the current figure
            plt.close('all') # Force close all figure windows

            import gc
            gc.collect() # Force Python to hunt for orphaned memory
            
        
    print("\n" + "="*75)
    print(f"{'SAM EVALUATION SUMMARY':^75}")
    print("="*75)

    global_metrics = [] # Will store all (iou, dice) tuples

    # Threshold for "Good" segmentation
    dice_threshold = 0.7
    
    for vid, frames in stats.items():
        video_metrics = []
        video_dice_over_count = 0  # Counter for DICE > 0.7 in this video
        total_video_instances = 0
        print(f"\nVideo: {vid}")
        print("-" * 30)
        
        for fid, f_metrics in frames.items():
            # Separate IoUs and Dices for the frame
            f_ious = [m[0] for m in f_metrics]
            f_dices = [m[1] for m in f_metrics]

            frame_dice_over = sum(1 for d in f_dices if d > dice_threshold)
            frame_prop = frame_dice_over / len(f_metrics) if f_metrics else 0
            
            # Accumulate for video and global stats
            video_dice_over_count += frame_dice_over
            total_video_instances += len(f_metrics)
            
            frame_iou = np.mean(f_ious) if f_ious else 0
            frame_dice = np.mean(f_dices) if f_dices else 0
            
            video_metrics.extend(f_metrics)
            global_metrics.extend(f_metrics)
            
            print(f"  Frame {fid:25} | IoU: {frame_iou:.4f} | DICE: {frame_dice:.4f} ({len(f_metrics)} inst) | % DICE > 0.7: {frame_prop:.1%}")
        
        v_ious = [m[0] for m in video_metrics]
        v_dices = [m[1] for m in video_metrics]
        v_proportion = video_dice_over_count / total_video_instances if total_video_instances > 0 else 0
        
        print(f"  >>> Video {vid} Average | IoU: {np.mean(v_ious):.4f} | DICE: {np.mean(v_dices):.4f}")
        print(f"  >>> Prop of instances with DICE > {dice_threshold}: {v_proportion:.2%}")

    # Final overall stats
    final_ious = [m[0] for m in global_metrics]
    final_dices = [m[1] for m in global_metrics]

    final_dice_over_count = sum(1 for d in final_dices if d > dice_threshold)
    final_proportion = final_dice_over_count / len(final_dices) if final_dices else 0

    print("\n" + "="*75)
    print(f"OVERALL DATASET MEAN IoU:  {np.mean(final_ious):.4f}")
    print(f"OVERALL DATASET MEAN DICE: {np.mean(final_dices):.4f}")
    print(f"Overall Prop DICE > {dice_threshold}: {final_proportion:.2%}")
    print("="*75)

    return stats

def plot_metric_distribution(stats_dict, save_directory):
    """
    Plots histograms for IoU and DICE.
    Uses list conversion to avoid 'Multi-dimensional indexing' errors 
    common in Seaborn/Pandas version mismatches.
    """
    all_ious = []
    all_dices = []
    
    # Navigate the 3-level hierarchy: { video: { frame: [(iou, dice)] } }
    for video_id, frames in stats_dict.items():
        for frame_id, instances in frames.items():
            for metric_pair in instances:
                if len(metric_pair) == 2:
                    all_ious.append(float(metric_pair[0]))
                    all_dices.append(float(metric_pair[1]))

    if not all_ious:
        print("No data found to plot.")
        return

    # Create the visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # IoU Distribution
    # Note: We pass raw Python lists here. Seaborn handles these 
    # much more reliably than NumPy arrays that might have weird shapes.
    sns.histplot(data=all_ious, bins=20, color='skyblue', ax=ax1)
    ax1.axvline(np.mean(all_ious), color='blue', linestyle='--', 
                label=f'Mean: {np.mean(all_ious):.3f}, STD: {np.std(all_ious):.3f}')
    ax1.set_title(f'IoU Distribution (N={len(all_ious)})')
    ax1.set_xlabel('Intersection over Union')
    ax1.set_xlim(0, 1)
    ax1.legend()

    # DICE Distribution
    sns.histplot(data=all_dices, bins=20, color='salmon', ax=ax2)
    ax2.axvline(np.mean(all_dices), color='red', linestyle='--', 
                label=f'Mean: {np.mean(all_dices):.3f}±{np.std(all_dices):.2f}')
    ax2.axvline(0.7, color='black', linestyle='--', label='DICE=0.7')
    ax2.set_title(f'DICE Distribution (N={len(all_dices)})')
    ax2.set_xlabel('Dice Coefficient')
    ax2.set_xlim(0, 1)
    ax2.legend()

    if DISPLAY_AND_SAVE_PICTURES:
        plt.savefig(os.path.join(save_directory, "histograms.png"), format='png', dpi=300)

    plt.tight_layout()
    plt.show()

# --- Actual main script ---
if __name__ == "__main__":
    MY_DATA_DIR = "../../data/cilia_io_segmentation_dataset/" 
    SAM_CHECKPOINT = "../../sam/sam_vit_h_4b8939.pth"
    SHOW_MATCHING_SEGMENTATION_BOXES = False

    save_directory = '../../writing/segment_anything_benchmark_plots_4_28_26'
    os.makedirs(save_directory, exist_ok=True)

    # Prepare dataset
    cilia_dataset = create_segmentation_dataset(MY_DATA_DIR, show_debug=SHOW_MATCHING_SEGMENTATION_BOXES)
    print("Created Cilia Dataset :)")

    # Run Evaluation 
    if cilia_dataset:
        results = run_sam_evaluation(cilia_dataset, SAM_CHECKPOINT, save_directory=save_directory)
        csv_rows = []
        
        # Threshold for success
        dice_threshold = 0.7

        for vid, frames in results.items():
            for fid, instances in frames.items():
                # instances is a list of tuples: [(iou1, dice1), (iou2, dice2)]
                for inst_tuple in instances:
                    iou_val = inst_tuple[0]
                    dice_val = inst_tuple[1]
                    
                    # Map the tuple to a dictionary for the CSV
                    csv_rows.append({
                        'video_id': vid,
                        'frame_id': fid,
                        'iou': round(iou_val, 4),
                        'dice': round(dice_val, 4),
                        'is_success': 1 if dice_val > dice_threshold else 0
                    })

        # Write to CSV
        if csv_rows:
            keys = csv_rows[0].keys()
            output_path = os.path.join(save_directory, "stats.csv")
            with open(output_path, 'w', newline='') as output_file:
                dict_writer = csv.DictWriter(output_file, fieldnames=keys)
                dict_writer.writeheader()
                dict_writer.writerows(csv_rows)
            print(f"Successfully saved {len(csv_rows)} records to {output_path}")
        else:
            print("Warning: No results found to save.")

        plot_metric_distribution(results, save_directory)
    else:
        print("No valid project data found. Check your JSON and image paths.")


