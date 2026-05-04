import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import cv2

def show_first_frame(path, save_fig=False, dir=""):
    # Open and read ONLY the data
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    success, frame = cap.read()
    
    # IMMEDIATELY release the video to free up the system resources/threads
    cap.release()

    if not success:
        print("Error: Could not read frame.")
        return

    # use Matplotlib to show the image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(frame_rgb)
    plt.title(f"First Frame: {path}")
    plt.axis('off')

    plt.tight_layout()

    if save_fig:
        plt.savefig(os.path.join(dir,'first_frame.png'), dpi=300, format='png')


def generate_temporal_skeleton_plot(pickle_path, target_id, fps, px_to_um, dorsal_ventral_threshold=85,first_frame=0, max_frames=None, sample_mode="first", dark_mode=True, save_fig=False, dir=''):
    # Load data
    try:
        with open(pickle_path, 'rb') as f:
            full_data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{pickle_path}' was not found.")
        return

    if target_id not in full_data:
        available = list(full_data.keys())
        print(f"ID '{target_id}' not found. Available IDs: {available[:5]}...")
        return

    cilia_series = full_data[target_id]
    sorted_frames = sorted(cilia_series.keys())

    # Num frames to display
    if max_frames is not None and max_frames < len(sorted_frames):
        if sample_mode == "even":
            idx = np.linspace(0, len(sorted_frames) - 1, max_frames).astype(int)
            selected_frames = [sorted_frames[i] for i in idx]
        else:
            selected_frames = sorted_frames[first_frame:first_frame+max_frames]
    else:
        selected_frames = sorted_frames

    # We sample the first few frames to see which end (start or end) is more stable
    sample_size = min(1, len(selected_frames))
    all_starts = []
    all_ends = []
    all_centroids_y = []

    for i in range(sample_size):
        pts = np.asarray(cilia_series[selected_frames[i]])
        if pts.ndim == 2 and len(pts) > 0:
            all_starts.append(pts[0])
            all_ends.append(pts[-1])
            all_centroids_y.append(np.mean(pts[:, 1]))

    all_starts = np.array(all_starts)
    all_ends = np.array(all_ends)
    
    # Determine general side (Dorsal vs Ventral)
    mean_centroid_y = np.mean(all_centroids_y)
    is_ventral = mean_centroid_y > dorsal_ventral_threshold

    first_pts = np.asarray(cilia_series[selected_frames[0]])
    if is_ventral:
        root_idx = 0 if first_pts[0, 1] > first_pts[-1, 1] else -1
    else:
        root_idx = 0 if first_pts[0, 1] < first_pts[-1, 1] else -1

    master_base_coords = np.asarray(cilia_series[selected_frames[0]])[root_idx]
    print(f"ID {target_id} | ALIGNMENT: {'Ventral' if is_ventral else 'Dorsal'} | Position: {'Bottom' if is_ventral else 'Top'} | Root: {master_base_coords}")

    # Setup plot
    if dark_mode:
        plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = plt.cm.turbo

    # Processing and Plotting Loop
    valid_time_seconds = []
    all_skeleton_pts = []
    view_limit_um = 3
    limit_px = view_limit_um / px_to_um
    
    for i, frame in enumerate(selected_frames):
        skeleton = np.asarray(cilia_series[frame])
        if skeleton.size == 0 or skeleton.ndim != 2:
            continue

        dist_to_start = np.linalg.norm(skeleton[0] - master_base_coords)
        dist_to_end = np.linalg.norm(skeleton[-1] - master_base_coords)
        
        if dist_to_end < dist_to_start:
            skeleton = np.flip(skeleton, axis=0)

        # Subtract the detected base from every point so the base is at (0,0)
        root = skeleton[0]
        centered = skeleton - root

        x = centered[:, 0]
        y = centered[:, 1]
        all_skeleton_pts.append(centered)

        # Map color by index in the sequence
        color = cmap(i / (len(selected_frames) - 1)) if len(selected_frames) > 1 else cmap(0)
        ax.plot(x, y, color=color, alpha=0.6, linewidth=1.5, zorder=i)
        
        valid_time_seconds.append((frame-first_frame) / fps)

    all_pts_flat = np.concatenate(all_skeleton_pts)
    x_mid = (np.min(all_pts_flat[:, 0]) + np.max(all_pts_flat[:, 0])) / 2
    y_mid = (np.min(all_pts_flat[:, 1]) + np.max(all_pts_flat[:, 1])) / 2

    view_limit_um = 3 
    limit_px = view_limit_um / px_to_um

    ax.set_xlim(x_mid - limit_px, x_mid + limit_px)
    ax.set_ylim(y_mid - limit_px, y_mid + limit_px)
    ax.set_aspect('equal')
    ax.axis('off')

    # Physical Scale Bar (Fixed at 5 Microns)
    scale_um = 1
    bar_size_px = scale_um / px_to_um
    
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()

    bar_x_start = (x_mid + limit_px) - (bar_size_px * 1.3)
    bar_y = (y_mid - limit_px) + (limit_px * 0.15)
    
    if dark_mode:
        text_color='white'
    else:
        text_color='black'

    ax.plot([bar_x_start, bar_x_start + bar_size_px], [bar_y, bar_y], color=text_color, lw=3)
    ax.text(bar_x_start + (bar_size_px / 2), bar_y - (abs(y_lim[1] - y_lim[0]) * 0.02), 
            f"{scale_um} µm", color=text_color, ha='center', va='bottom', fontsize=10)
    ax.invert_yaxis()

    # Temporal Colorbar (Seconds)
    if valid_time_seconds:
        norm = Normalize(vmin=min(valid_time_seconds), vmax=max(valid_time_seconds))
        sm = ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.08, shrink=0.6)
        cbar.set_label('Time (seconds)', fontsize=10, color=text_color)
        cbar.outline.set_edgecolor(text_color)
        plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color=text_color)

    plt.title(f"Cilia: {target_id}", fontsize=14, pad=20)
    plt.tight_layout()

    if save_fig:
        plt.savefig(os.path.join(dir,f'cilia_{target_id}.png'), dpi=300, format='png')

# --- config ---
root_folder = '../../data/test_cilia_io/'
file_name = 'video_name'
dorsal_threshold = 80

dark_mode = True
SAVE_FIGS = True
num_frames = 100
px_to_um = 0.107438780772816
fps = 49.88551663662496

# --------
file = os.path.join(root_folder, f"{file_name}.mp4_full_skeletons.pkl")
video_path = os.path.join(root_folder, f"{file_name}_5fps_ciliaio_output.mp4")

if __name__ == "__main__":
    fig_directory = os.path.join(f'../../img/cilia_skeleton_tracking/{file_name}')

    if SAVE_FIGS:
        # Does figure exist
        os.makedirs(fig_directory, exist_ok=True)

    show_first_frame(video_path, save_fig=SAVE_FIGS, dir=fig_directory)

    for i in range(20):
        target_id = str(i)
        generate_temporal_skeleton_plot(file, target_id, fps, px_to_um, dorsal_ventral_threshold=dorsal_threshold, first_frame=0,max_frames=num_frames, sample_mode="first", dark_mode=dark_mode, save_fig=SAVE_FIGS, dir=fig_directory)

    plt.show(block=False)
    plt.pause(0.001)
    input("hit [enter] to end.")
    plt.close("all")
