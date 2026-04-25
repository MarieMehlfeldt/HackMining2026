import numpy as np
from .clustering import cluster_frame
from dataclasses import dataclass


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
    
    