import matplotlib.pyplot as plt
import numpy as np 
from skimage.morphology import skeletonize, dilation, footprint_rectangle
from skimage.measure import label, regionprops
from ml_helpers import *
from scipy.spatial import cKDTree
from scipy.stats import zscore
from scipy.ndimage import label
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial import KDTree
from scipy.sparse.linalg import spsolve
from scipy import sparse

'''
A variety of helper functions that define the quantification backend of Cilia.io.
These functions have been divided up into their morphology metrics, as well as the
dynamic motility metrics. These are used in the main scripts.
'''

def arc_length(points):
    # Computes arc length of a set path of points. Technically arc length assumes a circle, but we just
    # calculate the total length of the path from the beginning to the end of the set of points.
    diffs = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    return np.sum(segment_lengths)

def compute_local_widths(skeleton, cur_mask, max_ray_length=10, samples_per_ray=20, search_radius=1, show_figure=False):
    """
    Computes local width of a binary mask at each skeleton point by casting rays
    along normal vectors and finding edge intersections.
    """
    # NOTE: This is not used in the release version of Cilia.io. This is just the 

    # Get edge mask by binary erosion by one pixel
    eroded = binary_erosion(cur_mask)
    edge_mask = cur_mask ^ eroded
    edge_coords = np.argwhere(edge_mask)
    edge_coords = edge_coords[:, [1, 0]]

    edge_tree = KDTree(edge_coords)

    # Get the normal vectors at each skeleton point
    deltas = np.gradient(skeleton.astype(float), axis=0)
    tangents = deltas / (np.linalg.norm(deltas, axis=1, keepdims=True) + 1e-8)
    normals = np.stack([-tangents[:,1], tangents[:,0]], axis=1)  # rotate 90° clockwise

    # For each point, for the two intersection points and then calculate their distances.
    widths = []

    if show_figure:
        fig, ax = plt.subplots(figsize=(10, 10))
        show_mask(cur_mask, ax, random_color=True)  
        plt.plot(skeleton[:,0], skeleton[:,1], color='blue', linewidth=1)
        

    for pt, normal in zip(skeleton, normals):
        intersections = []

        for direction in [-1, 1]:
            # Cast rays in both directions out of the point
            ray = pt + direction * np.linspace(0, max_ray_length, samples_per_ray)[:, None] * normal

            ray_rows, ray_cols = ray[:, 1], ray[:, 0]

            if show_figure:
                plt.plot(ray_cols, ray_rows, color='red', linewidth=1.0)  
                plt.plot(pt[0], pt[1], 'bo')  # pt is (row, col)   


            for probe in ray:
                dist, idx = edge_tree.query(probe, distance_upper_bound=search_radius)
                if dist != np.inf:
                    intersections.append(edge_coords[idx])
                    break

        if len(intersections) == 2:
            # Two intersections when casting rays on both sides of the guy
            width = np.linalg.norm(np.array(intersections[0]) - np.array(intersections[1]))
            widths.append(width)
        else:
            widths.append(np.nan)

    if show_figure:
        plt.show()

    return np.array(widths)

def remove_sharp_turns(coords, angle_threshold_deg=120, min_points=3):
    """
    Iteratively remove points where the local turning angle exceeds the threshold,
    until no sharp turns remain or coordinates become too short.
    """
    # Base case in case we run out of points since everything was bad.
    if len(coords) < min_points:
        return coords.copy()

    threshold_rad = np.deg2rad(angle_threshold_deg)
    coords = coords.copy()
    
    # Iterate until all pairwise have angles less than threshold
    while True:
        if len(coords) < min_points:
            break

        # vec1 is vectors of p_i to p_i+1. vec2 is p_i+1 to p_i+2 for every point
        # this provides pairwise vectors to compare for each point in the skeleton
        vec1 = coords[1:-1] - coords[:-2]
        vec2 = coords[2:] - coords[1:-1]

        # Compute angles of all pairwise vectors
        v1_norm = vec1 / np.linalg.norm(vec1, axis=1, keepdims=True)
        v2_norm = vec2 / np.linalg.norm(vec2, axis=1, keepdims=True)
        dot = np.clip(np.sum(v1_norm * v2_norm, axis=1), -1.0, 1.0)
        angles = np.arccos(dot)

        # Determine where angles exceed allowable threshold and get their indices.
        # If greater, remove p_i+1 from the skeleton
        sharp_points = np.where(angles > threshold_rad)[0] + 1

        if len(sharp_points) == 0:
            break  # no sharp turns left

        coords = np.delete(coords, sharp_points, axis=0)

    return coords


def keep_largest_connected_component(mask):
    labeled_mask, num_features = label(mask)
    if num_features == 0:
        # No components found, return original mask
        return mask
    
    counts = np.bincount(labeled_mask.ravel())
    counts[0] = 0  # ignore background
    
    largest_label = counts.argmax()
    
    return (labeled_mask == largest_label).astype(mask.dtype)

def pad_mask(mask, n):
    """
    Dilates mask by n pixels on each side :)
    """
    selem = footprint_rectangle((2 * n + 1, 2 * n + 1))  # Square structuring element of appropriate size
    return dilation(mask, selem)

def extend_skeleton(starting_point, sobel_mask, dist_mask, segmentation_mask, skel_center, ordered_coords, h, w, max_steps=100):
    # Create path
    path = [[starting_point[0], starting_point[1]]]
    y, x = starting_point[0], starting_point[1]
    y_center, x_center = skel_center[0], skel_center[1]

    normalized_dist_mask = dist_mask / (np.linalg.norm(dist_mask) + 1e-8)

    for _ in range(max_steps):
        if segmentation_mask[y,x] == 0:
            break  # reached edge

        # Get 8 neighbors (within bounds)
        neighbors = [(y + dy, x + dx)
                     for dy in [-1,0,1] for dx in [-1,0,1]
                     if 0 <= y + dy < h and 0 <= x + dx < w and (dy != 0 or dx != 0)]

        neighbors_np = np.array(neighbors)  # Ensure it's an array
        dy_dx = neighbors_np - np.array([y_center, x_center])  # shape (N, 2)
        dists = np.hypot(dy_dx[:, 0], dy_dx[:, 1])  # shape (N,)
        dists = dists / (np.linalg.norm(dists) + 1e-8)

        # Find neighbor with smallest gradient magnitude
        min_grad = np.inf
        next_pos = None
        for i, (ny, nx) in enumerate(neighbors):

            # We need to make sure that this is not already in the skeleton :)
            if [ny,nx] in (np.vstack((ordered_coords, path)).tolist()):
                continue
            
            # Calculate scoring to middle of skeleton
            dist = dists[i]
            scaling_factor = 1
            mag = sobel_mask[ny, nx] - (scaling_factor * dist) #+ (normalized_dist_mask[ny,nx])

            # Find min grad :)
            if mag < min_grad:
                min_grad = mag
                next_pos = (ny, nx)

        # If no better neighbor or stuck, stop
        if next_pos == (y, x) or next_pos is None:
            break

        y, x = next_pos
        
        # Don't include point where equal to 0 since end of mask :)
        if segmentation_mask[y, x] == 0:
            break
        path.append([y, x])

    return np.array(path)

def order_path(points):
    """Greedy nearest-neighbor path ordering."""
    points = points.copy()
    path = [points[0]]
    visited = set([0])
    
    tree = cKDTree(points)
    current_idx = 0

    for _ in range(1, len(points)):
        dists, idxs = tree.query(points[current_idx], k=len(points))
        for idx in idxs:
            if idx not in visited:
                path.append(points[idx])
                visited.add(idx)
                current_idx = idx
                break

    return np.array(path)

def get_skeleton_from_mask(mask):
    # Ensure binary mask
    mask_bin = mask.astype(bool)
    skeleton = skeletonize(mask_bin)
    coords = np.column_stack(np.nonzero(skeleton))  # (row, col) coordinates

    return skeleton, coords


# ----
# Motility Work

def baseline_als(y, lam, p, niter=10):
    # Lambda between 10^2 to 10^9, 0.001 <= p <= 0.1 for signal with positive peaks
    # From paper: Asymmetric Least Squares Smoothing by Paul H C Eilers, Hans Boelens
    # https://www.researchgate.net/publication/228961729_Baseline_Correction_with_Asymmetric_Least_Squares_Smoothing  
  L = len(y)
  D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
  w = np.ones(L)
  for i in range(niter):
    W = sparse.spdiags(w, 0, L, L)
    Z = W + lam * D.dot(D.transpose())
    z = spsolve(Z, w*y)
    w = p * (y > z) + (1-p) * (y < z)
  return z

def extract_motion_features(cilia_id, points, pixels_to_microns, fps=50, pixel_group_threshold=75, MIN_NUMBER_POINTS=10, zero_pad_fft_factor=4):
    points = np.asarray(points)
    T = points.shape[0]

    if T < MIN_NUMBER_POINTS:
        return None

    # Total path length
    diffs = np.diff(points, axis=0)
    step_lengths = np.linalg.norm(diffs, axis=1)
    path_length = np.sum(step_lengths)

    # Get average speed
    dt = 1.0 / fps  # seconds per frame
    speeds = step_lengths / dt  # units per second
    avg_speed = np.mean(speeds)

    # Instantaneous directions
    directions = np.arctan2(diffs[:,1], diffs[:,0])  # angles between steps

    # Angular changes
    angular_changes = np.diff(directions)
    # Normalize angles between -pi and pi
    angular_changes = (angular_changes + np.pi) % (2 * np.pi) - np.pi

    mean_angular_velocity = np.mean(np.abs(angular_changes)) if len(angular_changes) > 0 else 0
    cumulative_angular_displacement = np.abs(np.sum(angular_changes))  

    # Radius of gyration
    centroid = np.mean(points, axis=0)
    rg = np.sqrt(np.mean(np.sum((points - centroid)**2, axis=1)))

    # Determine whether cilium is a dorsal or ventral cilia
    # Note: This is quite naive, and technically only works well if the dorsal / ventral
    #       cilia can be separated by a horizontal line easily. Fixes in the future
    if centroid[1] < pixel_group_threshold:
        group = 0
    else:
        group = 1

    # Eccentricity via PCA
    distances = np.linalg.norm(points - centroid, axis=1)
    z_scores = zscore(distances)
    
    # Filter out points with Z-scores above the threshold
    threshold = 3
    inliers = points[np.abs(z_scores) < threshold]

    centered_points = inliers - np.mean(inliers, axis=0)  # center points
    cov = np.cov(centered_points.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]         # largest → smallest
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]               # reorder columns to match
    #eigvals = np.sort(eigvals)[::-1]  # largest to smallest
    a = np.sqrt(eigvals[0])  # semi-major axis length
    b = np.sqrt(eigvals[1])  # semi-minor axis length
    eccentricity = np.sqrt(1 - (b**2 / a**2)) if a > 0 else 0

    # Mean and max amplitude (projection on major axis)
    major_axis = eigvecs[:, 0]  # eigenvector for major axis
    proj_old = centered_points @ major_axis  # projection on major axis

    # Baseline Correction of Signal
    z = baseline_als(proj_old, 1e3, 0.01)
    proj = proj_old-z
    proj_detrended = proj - np.median(proj)

    mean_amplitude = np.mean(np.abs(proj_detrended))
    max_amplitude = np.max(np.abs(proj_detrended))
    rms_amplitude = np.sqrt(np.mean(proj_detrended**2))

    # Frequency estimation (from projection on major axis)
    window = np.hamming(len(proj_detrended))
    proj_windowed = proj_detrended * window
    
    # Zero pad for extra frequency fidelity with FFT (zero pad by 4x)
    N = len(proj_windowed)
    M = N * zero_pad_fft_factor
    M_pow2 = 2 ** int(np.ceil(np.log2(M)))

    fft_vals = np.fft.rfft(proj_windowed, n=M_pow2)
    fft_freqs = np.fft.rfftfreq(M_pow2, d=1/fps)
    fft_vals[0] = 0
    power = np.abs(fft_vals)**2
    dominant_idx = np.argmax(power)
    frequency = fft_freqs[dominant_idx]

    # While we are here, calculate some other cool things :()
    p_norm = power / power.sum()
    centroid_freq = np.sum(fft_freqs * p_norm)
    entropy = -np.sum(p_norm * np.log2(p_norm + 1e-12))
    entropy = entropy / np.log2(len(p_norm))

    # #1) Plot the old signal, baseline-fixed signal, FFT power spectrum and the dominant frequency (FOR DEBUG)
    # fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
    # time = np.arange(len(proj)) / fps
    # axs[0].plot(time, proj_detrended, color='navy')
    # axs[0].plot(time, proj_old, color='orange')
    # axs[0].set_title(f'Projection on Major Axis, Cilia ID: {cilia_id} \nMean Amp: {mean_amplitude:.3f}, Max Amp: {max_amplitude:.3f}, Entropy: {entropy:.3f}')
    # axs[0].set_xlabel('Time (s)')
    # axs[0].set_ylabel('Projection (units)')
    # axs[0].grid(True, linestyle='--', alpha=0.5)
    # axs[0].legend(['fixed_baseline', 'old'])

    # axs[1].plot(fft_freqs, power, label='Power Spectrum')
    # axs[1].axvline(frequency, color='red', linestyle='--',
    #             label=f'Dominant Frequency: {frequency:.2f} Hz')
    # axs[1].axvline(centroid_freq, color='blue', linestyle='--',
    #         label=f'Centroid Frequency: {centroid_freq:.2f} Hz')
    # axs[1].set_xlabel('Frequency (Hz)')
    # axs[1].set_ylabel('Power')
    # axs[1].set_title('FFT Power Spectrum')
    # axs[1].legend()
    # axs[1].grid(True, linestyle='--', alpha=0.5)

    # plt.tight_layout()
    # plt.show()

    features = {
        "cilia_id": cilia_id,
        "path_length": path_length * pixels_to_microns,
        "average_speed": avg_speed * pixels_to_microns,
        "mean_angular_velocity": mean_angular_velocity,
        "cumulative_angular_displacement": cumulative_angular_displacement,
        "semi_major_axis_length": a * pixels_to_microns,
        "semi_minor_axis_length": b * pixels_to_microns,
        "radius_of_gyration": rg * pixels_to_microns,
        "eccentricity": eccentricity,
        "mean_amplitude": mean_amplitude * pixels_to_microns,
        "max_amplitude": max_amplitude * pixels_to_microns,
        "rms_amplitude": rms_amplitude * pixels_to_microns,
        "frequency": frequency,
        'centroid_frequency': centroid_freq,
        "entropy": entropy,
        "group": group, 
        "num_consecutive_frames": T,
    }

    return features
