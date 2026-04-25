import time
from dataclasses import dataclass

import numpy as np

from . import cloud_state
from .clustering import cluster_frame
from .dirty_clusters import find_dirty_clusters


@dataclass
class AppSettings:
    dirt_clustering_eps: float = 0.03
    dirt_clustering_min_points: int = 5
    dirt_clustering_max_dist: float = 0.3


def process_frame(current_frame: dict[str, np.ndarray],
                  old_frame: dict[str, np.ndarray] | None,
                  app_settings: AppSettings):
    start = time.perf_counter()
    # 🚨 FIX: Prevent crash on the very first payload
    if old_frame is None:
        print("First frame received. Waiting for next frame to compute diff...")
        return

    labels, indices = cluster_frame(current_frame["coords"],
                                    eps=app_settings.dirt_clustering_eps,
                                    min_samples=app_settings.dirt_clustering_min_points,
                                    max_dist=app_settings.dirt_clustering_max_dist)
    print(f"clustering took {start-time.perf_counter()} s.")

    dirty_points_in_sectors, coords_dirty_points, coords_clean_points = find_dirty_clusters(
        labels, indices, current_frame["coords"], current_frame["reflectivity"],
        old_frame["coords"], threshold_distance=0.1, threshold_deriv=0,
        threshold_reflect=100, n_sectors=5, cluster_perc_threshold=10,
    )

    # Save to the unified state
    with cloud_state.data_lock:
        cloud_state.data_state["sectors"] = dirty_points_in_sectors.tolist()
        cloud_state.data_state["dirty"] = [pt.tolist() for pt in coords_dirty_points]
        cloud_state.data_state["clean"] = [pt.tolist() for pt in coords_clean_points]
    print(time.perf_counter() - start)
    print(f"Processed frame. Sectors: {dirty_points_in_sectors}")