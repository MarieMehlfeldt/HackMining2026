import numpy as np

import numpy as np

def find_dirty_clusters(labels, indices, frame_coords, frame_reflectivity,
                        frame_coords_pre, threshold_distance,
                        threshold_deriv, threshold_reflect, n_sectors):

    cluster_perc_threshold = 0.33

    # --- Precompute row/col once ---
    rows, cols = np.unravel_index(indices, (16, 720))

    # --- Precompute sector per point ---
    sectors = np.minimum((cols / 720 * n_sectors).astype(int), n_sectors - 1)

    # --- Gather data ---
    coords_now = frame_coords[rows, cols]          # (N, 3)
    coords_pre = frame_coords_pre[rows, cols]      # (N, 3)
    reflectivity_now = frame_reflectivity[rows, cols]

    # --- Compute distances vectorized ---
    distance_now = np.linalg.norm(coords_now, axis=1)
    distance_pre = np.linalg.norm(coords_pre, axis=1)
    distance_deriv = distance_now - distance_pre

    # --- Dirty mask ---
    is_dirty = (
        (distance_now < threshold_distance) &
        ((distance_deriv < threshold_deriv) |
         (reflectivity_now < threshold_reflect))
    )

    # --- Handle noise points (cluster == -1) directly ---
    noise_mask = labels == -1

    coords_dirty_points = coords_now[noise_mask & is_dirty].tolist()
    coords_clean_points = coords_now[noise_mask & ~is_dirty].tolist()

    dirty_points_in_sectors = np.zeros(n_sectors)
    np.add.at(dirty_points_in_sectors,
              sectors[noise_mask & is_dirty], 1)

    # --- Process real clusters ---
    valid_mask = ~noise_mask
    valid_labels = labels[valid_mask]

    if valid_labels.size == 0:
        return dirty_points_in_sectors, coords_dirty_points, coords_clean_points

    # Remap cluster labels to 0...K-1
    unique_labels, inverse = np.unique(valid_labels, return_inverse=True)

    # Count points per cluster
    counts = np.bincount(inverse)
    dirty_counts = np.bincount(inverse, weights=is_dirty[valid_mask])

    perc_dirty = dirty_counts / counts

    # Determine which clusters are dirty
    cluster_is_dirty = perc_dirty > cluster_perc_threshold

    # Map back to point-level decision
    point_cluster_dirty = cluster_is_dirty[inverse]

    # --- Assign points ---
    cluster_dirty_mask = valid_mask.copy()
    cluster_dirty_mask[valid_mask] = point_cluster_dirty

    cluster_clean_mask = valid_mask & ~cluster_dirty_mask

    # --- Append results ---
    coords_dirty_points.extend(coords_now[cluster_dirty_mask].tolist())
    coords_clean_points.extend(coords_now[cluster_clean_mask].tolist())

    np.add.at(dirty_points_in_sectors,
              sectors[cluster_dirty_mask], 1)

    return dirty_points_in_sectors, coords_dirty_points, coords_clean_points