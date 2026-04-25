import numpy as np
from .clustering import cluster_frame
from .dirty_clusters import find_dirty_clusters
from dataclasses import dataclass
from . import cloud_state


@dataclass
class AppSettings:
    dirt_clustering_eps:float = 0.03
    dirt_clustering_min_points:int = 5
    dirt_clustering_max_dist:float = 0.3

def process_frame(current_frame:dict[str, np.ndarray],
                  old_frame:dict[str, np.ndarray],
                  app_settings:AppSettings):
    labels, indices = cluster_frame(current_frame["coords"],
                                    eps=app_settings.dirt_clustering_eps,
                                    min_samples=app_settings.dirt_clustering_min_points,
                                    max_dist=app_settings.dirt_clustering_max_dist)
    dirty_points_in_sectors, coords_dirty_points, coords_clean_points = find_dirty_clusters(
        labels, indices, current_frame["coords"], current_frame["reflectivity"],
        old_frame["coords"], threshold_distance=0.3, threshold_deriv=0.1,
        threshold_reflect=0.1, n_sectors=10
    )
    with cloud_state.pointcloud_lock:
        cloud_state.pointcloud_state["clean"] = coords_clean_points.tolist()
        cloud_state.pointcloud_state["dirty"] = coords_dirty_points.tolist()
    print(f"Dirty points in sectors: {dirty_points_in_sectors}")
