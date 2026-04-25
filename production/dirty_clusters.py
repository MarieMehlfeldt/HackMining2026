import numpy as np

def find_dirty_clusters(labels, indices, frame_coords, frame_reflectivity, frame_coords_pre, threshold_distance, threshold_deriv, threshold_reflect, n_sectors, cluster_perc_threshold):
    coords_dirty_points = []
    coords_clean_points = []
    dirty_points_in_sectors = np.zeros (n_sectors)
    n_point_in_sectors = (16*720) / n_sectors

    cluster_labels_unique = list(set(labels))
    print (f"n clusters: {len(cluster_labels_unique)}")
    # iterate through the clusters
    for cluster in cluster_labels_unique:
        point_indices = np.where(labels == cluster)[0] # find indices of the points in the cluster
        n_points_in_cluster = len(point_indices)
        n_dirty_in_cluster = 0
        for idx in point_indices:
            row, col = np.unravel_index(indices[idx], (16, 720))
            sector = min(int(col / 720 * n_sectors), n_sectors - 1)
            # determine if point is dirty using classical determinants
            coords_now = frame_coords[row, col, :]
            reflectivity_now = frame_reflectivity[row, col]
            coords_pre = frame_coords_pre[row, col, :]
            distance_now = np.linalg.norm(coords_now)
            distance_pre = np.linalg.norm(coords_pre)
            distance_deriv = distance_now - distance_pre
            is_dirty = (distance_now < threshold_distance) and ((distance_deriv < threshold_deriv) or (reflectivity_now < threshold_reflect))
            if is_dirty:
                n_dirty_in_cluster += 1
                if cluster == -1: # if the point is not part of a cluster
                    coords_dirty_points.append(coords_now)
                    dirty_points_in_sectors[sector] += 1
            else:
                if cluster == -1: # if the point is not part of a cluster
                    coords_clean_points.append(coords_now)
        if n_points_in_cluster == 0 or cluster == -1: # if there are no points in the cluster or the points are not actually clustered
            continue
        perc_dirty_in_cluster = (n_dirty_in_cluster / n_points_in_cluster) * 100
        print (f"Cluster {cluster} has {perc_dirty_in_cluster}% dirty points.")
        if perc_dirty_in_cluster > cluster_perc_threshold: # if more than threshold of the points in the cluster are dirty, consider the whole cluster as dirty
            print(f"Cluster {cluster} is deemed dirty with {len(point_indices)} points")
            for idx in point_indices:
                row, col = np.unravel_index(indices[idx], (16, 720))
                sector = min(int(col / 720 * n_sectors), n_sectors - 1)
                coords_now = frame_coords[row, col, :]
                coords_dirty_points.append(coords_now)
                dirty_points_in_sectors[sector] += 1
        else: # if less than threshold of the points in the cluster are dirty, consider the whole cluster as clean
            for idx in point_indices:
                row, col = np.unravel_index(indices[idx], (16, 720))
                coords_now = frame_coords[row, col, :]
                coords_clean_points.append(coords_now)
    dirty_perc_in_sectors = (dirty_points_in_sectors/n_point_in_sectors) * 100
    return dirty_perc_in_sectors, coords_dirty_points, coords_clean_points