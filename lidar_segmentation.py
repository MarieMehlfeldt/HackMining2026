"""Plane segmentation utilities for Open3D point clouds."""

from __future__ import annotations

from typing import Any

from matplotlib import pyplot as plt
import numpy as np
import open3d as o3d

class MockSegmenter:
    def __init__(self):
        self._isinit = False
        self.clusters = []

    def segment(self, point_cloud: o3d.geometry.PointCloud) -> list[o3d.geometry.PointCloud]:
        if not self._isinit:
            self._isinit = True
            self.clusters.append(point_cloud)
            return [[], [], [[0, point_cloud]]]
        else:
            return [[[0, point_cloud]], [], []]
        
    def generate_cache_frame(self, updated:list[tuple[int, o3d.geometry.PointCloud]],
                             removed:list[int],
                             new:list[tuple[int, o3d.geometry.PointCloud]]):
        return ([(i, np.asarray(cluster.points, dtype=np.float32))
                            for i, cluster in updated],
                removed,
                [(i, np.asarray(cluster.points, dtype=np.float32))
                for i, cluster in new])
    
    def handle_cache_frame(self, vis, cache_frame):
        updated, removed, new = cache_frame
        for i, points in updated:
            self.clusters[i].points = o3d.utility.Vector3dVector(points)
            vis.update_geometry(self.clusters[i])
        for i in removed:
            vis.remove_geometry(self.clusters[i], reset_bounding_box=False)
        for i, points in new:
            self.clusters[i].points = o3d.utility.Vector3dVector(points)
            vis.add_geometry(self.clusters[i], reset_bounding_box=False)

class DBSegmenter:
    def __init__(self, eps:float=0.2, min_points:int=10, vmax=50):
        """DBSCAN based segmenter with simple cluster tracking based on nearest neighbor distance
        
        Args:
            eps (float): The maximum distance between two points for them to be considered in the same neighborhood.
            min_points (int): The minimum number of points required to form a dense region (cluster).
            vmax (float): The maximum velocity (in km/h) for a cluster to be considered the same as a previous one.
        """
        self.eps = eps
        self.min_points = min_points
        self.vmax = vmax*(1000/3600) # convert to m/s
        self.clusters = []
        self.cmap = plt.get_cmap("tab20")
        self._wasactive = []
        self._chosen = []
    
    def segment(self, point_cloud: o3d.geometry.PointCloud) -> list[o3d.geometry.PointCloud]:
        labels = np.array(point_cloud.cluster_dbscan(eps=self.eps, min_points=self.min_points))
        max_label = labels.max()
        print(f"point cloud has {max_label + 1} clusters")
        cmap = plt.get_cmap("tab20")
        active = [False]*len(self.clusters)
        self._chosen = [False]*len(self.clusters)
        new_clusters = []
        background = point_cloud.select_by_index(np.where(labels == -1)[0])
        if len(self.clusters) == 0:
            self.clusters.append(background)
            active.append(True)
            self._chosen.append(True)
            self._wasactive.append(False)
        else:
            self.clusters[0].points = background.points
            active[0] = True
            self._chosen[0] = True
            self._wasactive[0] = True
        for i in range(0, max_label + 1):
            cluster_cloud = point_cloud.select_by_index(np.where(labels == i)[0])
            idx, _ = self._get_viable_cluster(cluster_cloud)
            if idx is not None:
                self.clusters[idx].points = cluster_cloud.points
                self.clusters[idx].paint_uniform_color(cmap(idx % 20)[:3])
                active[idx] = True
                self._chosen[idx] = True
            else:
                cluster_cloud.paint_uniform_color(cmap(len(self.clusters) % 20)[:3])
                self.clusters.append(cluster_cloud)
                new_clusters.append(cluster_cloud)
                active.append(True)
                self._chosen.append(True)
                self._wasactive.append(False)
        out = ([(i, self.clusters[i]) for i in range(len(self.clusters)) if active[i] and self._wasactive[i]],
                [i for i in range(len(self.clusters)) if not active[i]],
                [(i, self.clusters[i]) for i in range(len(self.clusters)) if active[i] and (i > len(self._wasactive) or not self._wasactive[i])])
        self._wasactive = active
        return out

    def _get_viable_cluster(self, candidate):
        # exclude the background cluster
        index_ = np.arange(len(self.clusters))
        index_ = index_[~np.array(self._chosen)]
        if index_.size == 0:
            return None, None
        centroids = np.array([self.clusters[i].get_center() for i in index_])
        distances = np.linalg.norm(centroids - candidate.get_center(), axis=1)
        closest_idx = np.argmin(distances)
        closest_cluster = index_[np.argmin(distances)]
        # velocity is distance * fps
        if distances[closest_idx]*20 < self.vmax:
            return closest_cluster, self.clusters[closest_cluster]
        else:
            return None, None
        
    def generate_cache_frame(self, updated, removed, new):
        return ([(i, np.asarray(cluster.points, dtype=np.float32))
                            for i, cluster in updated],
                removed,
                [(i, np.asarray(cluster.points, dtype=np.float32))
                for i, cluster in new])
    
    def handle_cache_frame(self, vis, cache_frame):
        updated, removed, new = cache_frame
        for i, points in updated:
            self.clusters[i].points = o3d.utility.Vector3dVector(points)
            if i > 0:
                self.clusters[i].paint_uniform_color(self.cmap(i % 20)[:3])
            vis.update_geometry(self.clusters[i])
        for i in removed:
            vis.remove_geometry(self.clusters[i], reset_bounding_box=False)
        for i, points in new:
            self.clusters[i].points = o3d.utility.Vector3dVector(points)
            self.clusters[i].paint_uniform_color(self.cmap(i % 20)[:3])
            vis.add_geometry(self.clusters[i], reset_bounding_box=False)

def detect_planes(
        point_cloud: o3d.geometry.PointCloud,
) -> list[dict[str, Any]]:
    plane_model, inliers = point_cloud.segment_plane(distance_threshold=0.01,
                                         ransac_n=3,
                                         num_iterations=1000)
    [a, b, c, d] = plane_model
    inlier_cloud = point_cloud.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])
    outlier_cloud = point_cloud.select_by_index(inliers, invert=True)
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    return inlier_cloud, outlier_cloud

def detect_dbscan(
        point_cloud: o3d.geometry.PointCloud,
        eps: float = 0.02,
        min_points: int = 10
) -> list[dict[str, Any]]:
    labels = np.array(point_cloud.cluster_dbscan(eps=eps, min_points=min_points))
    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    clusters = []
    cmap = plt.get_cmap("tab20")
    for i in range(max_label + 1):
        cluster_cloud = point_cloud.select_by_index(np.where(labels == i)[0])
        cluster_cloud.paint_uniform_color(cmap(i / (max_label + 1))[:3])
        clusters.append(cluster_cloud)
    return clusters
