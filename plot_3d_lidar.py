"""This code uses an offline rosbag file and plots it as a 3D point cloud using open3d.

Additionally, it uses a segmenter that tracks clusters in the point cloud.

An important parameter is also the maximum distance that reduces the number of points
based on the distance to the sensor found in max_dist."""

from functools import partial
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Literal
import warnings
import cv2

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import open3d as o3d
from lidar_segmentation import DBSegmenter, MockSegmenter
from rosbag_lidar import get_lidar_data

CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1137
CAMERA_FX = 984.67088410290683
CAMERA_FY = 984.67088410290683
CAMERA_CX = 959.5
CAMERA_CY = 568.0

def _resolve_cmap(
        values: np.ndarray,
        cmap: str | object = "turbo",
        norm:mcolors.Normalize | None = None,
        norm_vmin: float | None = None,
        norm_vmax: float | None = None,
        crit:float|None = None,
        gap:float=0.1,
) -> np.ndarray:
    """Map values to RGB in [0, 1] using matplotlib Normalize + colormap."""
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros((values.size, 3), dtype=np.float32)

    safe_values = np.where(finite, values, np.nan)
    if norm is None:
        vmin = float(np.nanmin(safe_values)) if norm_vmin is None else float(norm_vmin)
        vmax = float(np.nanmax(safe_values)) if norm_vmax is None else float(norm_vmax)
        if vmax <= vmin:
            vmax = vmin + 1.0
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    normalized = norm(np.where(finite, values, vmin))
    if crit is not None and gap is not None:
        normalized = (normalized - gap) / (1.0 - gap)
        normalized[values <= crit] = 0.0
    rgb = np.asarray(cmap_obj(normalized), dtype=np.float32)[:, :3]
    rgb[~finite] = 1.0
    return rgb

class CriticalNormalizer(mcolors.Normalize):
    def __init__(self, vmin:float, vmax:float, crit:float,
                 gap:float=0.1, non_crit_normalizer:Any=None):
        self.crit = crit
        self.gap = gap
        if non_crit_normalizer is None:
            self.non_crit_normalizer = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
        super().__init__(vmin=vmin, vmax=vmax, clip=True)
    
    def __call__(self, value, clip=None):
        value = np.asarray(value, dtype=np.float32)
        value = np.clip(value, self.vmin, self.vmax)
        normalized = np.zeros_like(value, dtype=np.float32)
        crit_mask = value <= self.crit
        non_crit = self.non_crit_normalizer(value[~crit_mask])
        normalized[~crit_mask] = self.gap + non_crit * (1.0 - self.gap)
        return normalized
    
    def inverse(self, value):
        value = np.asarray(value, dtype=np.float32)
        values = np.zeros_like(value, dtype=np.float32)
        crit_mask = value <= self.gap
        values[crit_mask] = self.crit
        non_crit_normalized = (value[~crit_mask] - self.gap) / (1.0 - self.gap)
        values[~crit_mask] = self.non_crit_normalizer.inverse(non_crit_normalized)
        return values

def _build_xy_grid_lineset(o3d, extent: float, step: float):
    """Build a simple XY grid as a LineSet centered at the origin."""
    n = int(np.floor((2.0 * extent) / step))
    ticks = np.linspace(-extent, extent, n + 1, dtype=np.float64)

    points = []
    lines = []
    colors = []
    color = [0.35, 0.35, 0.35]

    for x in ticks:
        i0 = len(points)
        points.append([x, -extent, 0.0])
        points.append([x, extent, 0.0])
        lines.append([i0, i0 + 1])
        colors.append(color)

    for y in ticks:
        i0 = len(points)
        points.append([-extent, y, 0.0])
        points.append([extent, y, 0.0])
        lines.append([i0, i0 + 1])
        colors.append(color)

    grid = o3d.geometry.LineSet()
    grid.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    grid.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    grid.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return grid

def _prepare_cloud_frame(
        frame: tuple[np.ndarray, np.ndarray, np.ndarray],
        color_coding: Literal["intensity", "distance", "reflectivity"] = "intensity",
        min_dist=0,
        max_dist=np.inf
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flatten and downsample one lidar frame for plotting."""
    coords, intensity, reflectivity = frame
    coords = coords.copy()
    coords[:,:,[1,2]] = -coords[:,:,[1,2]]
    points = coords.reshape(-1, 3)
    indices = np.arange(points.shape[0], dtype=np.uint32)
    if color_coding == "intensity":
        color_coder = intensity.reshape(-1)
    elif color_coding == "reflectivity":
        color_coder = reflectivity.reshape(-1)
    else:
        color_coder = np.linalg.norm(points, axis=1)
        color_coder = np.max(color_coder) - color_coder  # invert so closer points are brighter
    color_coder = color_coder.reshape(-1)
    dist = np.linalg.norm(points, axis=1)
    valid = (dist >= min_dist) & (dist <= max_dist)

    valid &= np.isfinite(points).all(axis=1)
    points = points[valid]
    color_coder = color_coder[valid]
    if points.size == 0:
        return (np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32),
                np.empty((0, 3), dtype=np.uint32))

    return (points.astype(np.float32, copy=False), color_coder.astype(np.float32, copy=False),
            indices[valid])


def _add_geometry_backdrop(vis):
    extent = 10
    step = 0.5

    grid_geom = _build_xy_grid_lineset(o3d, extent=extent, step=step)
    # origin_geom = o3d.geometry.TriangleMesh.create_sphere(radius=step * 0.35)
    # origin_geom.paint_uniform_color([1.0, 0.0, 0.0])
    # axis_geom = o3d.geometry.TriangleMesh.create_coordinate_frame(
    #     size=step * 1.8,
    #     origin=[0.0, 0.0, 0.0],
    # )

    vis.add_geometry(grid_geom, reset_bounding_box=False)
    # vis.add_geometry(origin_geom, reset_bounding_box=False)
    # vis.add_geometry(axis_geom, reset_bounding_box=False)

def play_lidar_video_open3d(
        input_: Path | Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]],
        fps: float = 20.0,
        loop: bool = True,
        max_frames: int | None = None,
        output_mp4_path: Path | None = None,
        color_coding:Literal["intensity", "distance", "reflectivity"] = "distance",
        cmap: str | object = "turbo",
        norm: mcolors.Normalize | None = None,
        norm_vmin: float | None = None,
        norm_vmax: float | None = None,
        crit:float|None = None,
        gap:float=0.1,
        max_dist:float=np.inf,
        min_dist:float=0.0,
        cluster_callback:Callable|None = None
) -> None:
    """Play lidar frames as a repeating interactive Open3D animation.

    Close the Open3D window to stop playback.
    """
    if fps <= 0:
        raise ValueError("fps must be > 0")

    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError("open3d is required for video playback") from exc

    o3dv = getattr(o3d, "visualization")
    visualizer_cls = getattr(o3dv, "Visualizer")

    if isinstance(input_, Path):
        bag_path = input_
        win_name = f"Lidar Playback - {bag_path.stem}"
        cached_frames = None
    else:
        # Cache iterable inputs once so looping works even for one-pass generators.
        cached_frames = list(input_)
        if max_frames is not None:
            cached_frames = cached_frames[:max_frames]
        if not cached_frames:
            raise ValueError("Input iterator did not contain any frames")
        bag_path = None
        win_name = "Lidar Playback - Iterable Input"
    if output_mp4_path is None:
        if bag_path is not None:
            output_mp4_path = Path(f"{bag_path.stem}_open3d.mp4")
        else:
            output_mp4_path = Path("open3d_capture.mp4")

    vis = visualizer_cls()

    vis.create_window(window_name=win_name, width=CAMERA_WIDTH, height=CAMERA_HEIGHT)
    render_opt = vis.get_render_option()
    render_opt.point_size = 2.5
    render_opt.background_color = np.array([0.02, 0.02, 0.02], dtype=np.float64)
    point_color_option = getattr(o3dv, "PointColorOption", None)
    if point_color_option is not None:
        render_opt.point_color_option = getattr(point_color_option, "Color")

    # Apply the requested camera intrinsic settings while preserving current extrinsic pose.
    view_control = vis.get_view_control()
    camera_params = view_control.convert_to_pinhole_camera_parameters()
    camera_params.intrinsic = o3d.camera.PinholeCameraIntrinsic(
        CAMERA_WIDTH,
        CAMERA_HEIGHT,
        CAMERA_FX,
        CAMERA_FY,
        CAMERA_CX,
        CAMERA_CY,
    )
    try:
        view_control.convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True)
    except TypeError:
        view_control.convert_from_pinhole_camera_parameters(camera_params)

    cloud = o3d.geometry.PointCloud()
    geometry_added = False
    aux_geometry_added = False
    grid_geom = None
    origin_geom = None
    axis_geom = None
    frame_interval = 1.0 / fps
    segmenter = DBSegmenter(eps=0.1, min_points=10, vmax=50.0)
    # segmenter = MockSegmenter()
    video_writer = None
    recorded_once = False

    def capture_frame_to_mp4() -> None:
        nonlocal video_writer
        if output_mp4_path is None:
            return
        frame = np.asarray(vis.capture_screen_float_buffer(do_render=False), dtype=np.float32)
        if frame.size == 0:
            return
        frame_uint8 = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
        h, w = frame_bgr.shape[:2]
        if video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(str(output_mp4_path), fourcc, fps, (w, h))
            if not video_writer.isOpened():
                raise RuntimeError(f"Could not open video writer for {output_mp4_path}")
        video_writer.write(frame_bgr)

    if isinstance(input_, Path):
        from rosbag_lidar import get_lidar_data
        frame_source = get_lidar_data(bag_path)
    else:
        frame_source = iter(cached_frames)
    # potentially change to preallocated array but needs info from bag file
    frame_cache = []

    def plot_frame(vis, frame, min_dist, max_dist, color_coding, reset_bbox=False, cluster_callback:Callable|None=None):
        nonlocal geometry_added, aux_geometry_added, grid_geom, origin_geom, axis_geom
        
        points, color_coder, indices = _prepare_cloud_frame(
            frame, color_coding=color_coding, min_dist=min_dist, max_dist=max_dist)
        rgb = _resolve_cmap(
            color_coder,
            cmap=cmap,
            norm=norm,
            norm_vmin=norm_vmin,
            norm_vmax=norm_vmax,
            crit=crit,
            gap=gap
        )
        if points.size == 0:
            display_points = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
            display_rgb = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        else:
            display_points = points
            display_rgb = rgb

        display_points = display_points.astype(np.float64, copy=False)
        display_rgb = display_rgb.astype(np.float64, copy=False)

        cloud.points = o3d.utility.Vector3dVector(display_points)
        cloud.colors = o3d.utility.Vector3dVector(display_rgb)
        if cluster_callback is not None:
            try:
                callback = partial(cluster_callback, indices=indices)
            except Exception as exc:
                warnings.warn(f"Failed to create partial callback for cluster_callback: {exc}")
                callback = None
        else:
            callback = None
        updated, removed, new = segmenter.segment(cloud, display_rgb, callback=callback)

        for i, cluster in updated:
            vis.update_geometry(cluster)
        for i in removed:
            cluster = segmenter.clusters[i]
            vis.remove_geometry(cluster, reset_bounding_box=False)
        for i, cluster in new:
            vis.add_geometry(cluster, reset_bounding_box=reset_bbox)

        frame_cache.append(segmenter.generate_cache_frame(updated, removed, new))

        if not vis.poll_events():
            return None
        vis.update_renderer()
        return display_points, display_rgb

    def wait_for_next_frame(deadline, vis):
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return True
            if not vis.poll_events():
                return False
            vis.update_renderer()
            time.sleep(min(0.002, remaining))

    try:
        while True:
            frame_count = 0
            vis.clear_geometries()
            _add_geometry_backdrop(vis)
            if len(frame_cache) == 0:
                for i, frame in enumerate(frame_source):
                    deadline = time.perf_counter() + frame_interval                    
                    display_out = plot_frame(vis, frame, min_dist=min_dist, max_dist=max_dist,
                                             color_coding=color_coding, reset_bbox=(i == 0),
                                             cluster_callback=cluster_callback)
                    if display_out is None:
                        return
                    capture_frame_to_mp4()
                    frame_count += 1
                    if not wait_for_next_frame(deadline, vis):
                        return
                    if max_frames is not None and i + 1 >= max_frames:
                        break
            else:
                for cache_frame in frame_cache:
                    deadline = time.perf_counter() + frame_interval
                    segmenter.handle_cache_frame(vis, cache_frame)
                    vis.update_renderer()
                    if not vis.poll_events():
                        return
                    if not recorded_once:
                        capture_frame_to_mp4()
                    frame_count += 1
                    if not wait_for_next_frame(deadline, vis):
                        return
            if frame_count == 0:
                raise ValueError("No valid frames were rendered")

            if not recorded_once and video_writer is not None:
                video_writer.release()
                video_writer = None
                recorded_once = True
                print(f"Saved Open3D capture to {output_mp4_path} at {fps:.0f} FPS")

            if not loop:
                break
    finally:
        if video_writer is not None:
            video_writer.release()
        vis.destroy_window()


if __name__ == "__main__":
    # this is a minimal example showing the usage of the play_lidar_video_open3d function.
    path = Path("D:\\minehack\\sat_morning\\rosbag2_2026_03_13-18_08_00")
    if not path.is_dir():
        raise ValueError(f"Path {path} is not a directory")
    for file_ in path.iterdir():
        if file_.suffix != ".mcap":
            continue
        try:
            play_lidar_video_open3d(file_,
                                    fps=20,
                                    crit = 0.1,
                                    color_coding="distance",
                                    cmap="turbo",
                                    min_dist=0.0,
                                    max_dist=2)
        except Exception as e:
            print(f"Error processing {file_}: {e}")
