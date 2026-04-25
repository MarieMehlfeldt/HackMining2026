import numpy as np
from sklearn.cluster import DBSCAN, KMeans

def cluster_frame(coords:np.ndarray, eps:float=0.03, min_samples:int=5, max_dist:float=0.3):
    """Cluster a single lidar frame using DBSCAN and return cluster labels.
    
    Args:
        coords (np.ndarray): An (16,720,3) or (16*720,3) array of point coordinates.
        eps (float): The maximum distance between two points for them to be considered
            in the same neighborhood.
        min_samples (int): The minimum number of points required to form a dense region
            (cluster).
        max_dist (float): The maximum distance of a point to be considered for clustering.
        
    Returns:
        out (np.ndarray, np.ndarray): A tuple containing:
            - labels: An array of shape (N,) with cluster labels for each point.
                Noise points are labeled as -1.
            - indices: An array of shape (N,) with the points' indices."""

    if coords.ndim == 3:
        coords = coords.reshape(-1, 3)
    indices = np.arange(coords.shape[0])
    filter_ = np.linalg.norm(coords, axis=1) <= max_dist
    if np.sum(filter_) > 5000:
        return np.ones(indices[filter_].shape[0], dtype=int), indices[filter_]
    coords = coords[filter_]
    indices = indices[filter_]
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(coords)
    # labels = KMeans(n_clusters=10).fit_predict(coords)
    return labels, indices
