import numpy as np

def find_dirty_clusters(labels, indices, frame_coords, frame_reflectivity, frame_coords_pre, threshold_distance, threshold_deriv, threshold_reflect, n_sectors, cluster_perc_threshold):
    coords_dirty_points = []
    coords_clean_points = []
    dirty_points_in_sectors = np.zeros(n_sectors, dtype=np.float64)

    height = frame_coords.shape[0]
    width = frame_coords.shape[1]
    n_point_in_sectors = (height * width) / n_sectors

    labels = np.asarray(labels)
    indices = np.asarray(indices)

    # Resolve indexed points in one vectorized pass (C-style flat indexing).
    rows = indices // width
    cols = indices % width
    coords_now_all = frame_coords[rows, cols, :]
    reflectivity_all = frame_reflectivity[rows, cols]
    coords_pre_all = frame_coords_pre[rows, cols, :]

    distance_now_all = np.linalg.norm(coords_now_all, axis=1)
    distance_pre_all = np.linalg.norm(coords_pre_all, axis=1)
    distance_deriv_all = distance_now_all - distance_pre_all
    is_dirty_all = (distance_now_all < threshold_distance) & (
        (distance_deriv_all < threshold_deriv) | (reflectivity_all < threshold_reflect)
    )

    sectors_all = np.minimum((cols * n_sectors) // width, n_sectors - 1)

    cluster_labels_unique = np.unique(labels)
    print(f"n clusters: {len(cluster_labels_unique)}")

    # Iterate by cluster label only; all point-level work stays vectorized.
    for cluster in cluster_labels_unique:
        cluster_mask = labels == cluster
        n_points_in_cluster = int(np.count_nonzero(cluster_mask))
        if n_points_in_cluster == 0:
            continue

        cluster_dirty_mask = is_dirty_all[cluster_mask]
        cluster_coords = coords_now_all[cluster_mask]

        if cluster == -1:
            noise_dirty_coords = cluster_coords[cluster_dirty_mask]
            noise_clean_coords = cluster_coords[~cluster_dirty_mask]

            coords_dirty_points.extend(noise_dirty_coords)
            coords_clean_points.extend(noise_clean_coords)

            dirty_sectors = sectors_all[cluster_mask][cluster_dirty_mask]
            if dirty_sectors.size:
                dirty_points_in_sectors += np.bincount(dirty_sectors, minlength=n_sectors)
            continue

        perc_dirty_in_cluster = float(cluster_dirty_mask.mean() * 100.0)
        print(f"Cluster {cluster} has {perc_dirty_in_cluster}% dirty points.")

        if perc_dirty_in_cluster > cluster_perc_threshold:
            print(f"Cluster {cluster} is deemed dirty with {n_points_in_cluster} points")
            coords_dirty_points.extend(cluster_coords)
            dirty_sectors = sectors_all[cluster_mask]
            dirty_points_in_sectors += np.bincount(dirty_sectors, minlength=n_sectors)
        else:
            coords_clean_points.extend(cluster_coords)

    dirty_perc_in_sectors = (dirty_points_in_sectors / n_point_in_sectors) * 100
    return dirty_perc_in_sectors, coords_dirty_points, coords_clean_points