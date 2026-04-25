from pathlib import Path
from rosbag_lidar import get_lidar_data
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import pandas as pd


def find_dirt_perc(frame_coords, frame_reflectivity, frame_coords_pre, threshold_distance, threshold_deriv, threshold_reflect):
    dist = np.linalg.norm(frame_coords, axis=2).flatten()
    reflect_flat = frame_reflectivity.reshape(-1)
    if frame_coords_pre.size == 0:
        dirt_mask = (dist < threshold_distance) & (reflect_flat < threshold_reflect)
    else:
        dist_pre = np.linalg.norm(frame_coords_pre, axis=2).flatten()
        dist_deriv = dist - dist_pre
        dirt_mask = (dist < threshold_distance) & ((dist_deriv < threshold_deriv) | (reflect_flat < threshold_reflect))
    perc_dirt = (np.sum(dirt_mask) / len(dist)) * 100
    return perc_dirt

if __name__ == "__main__":
    csv_path = Path("/Users/mariemehlfeldt/Desktop/HackMining2026/data_key.csv")
    df = pd.read_csv(csv_path)
    n_sectors = 5

    for idx in df.index:
        filename = df.loc[idx, "filename"]
        folder_path = Path(f"/Volumes/T7/minehack/sat_morning/{filename}")

        threshold_distance = 0.10 # threshold distance in meters to consider a point as "dirt"
        threshold_deriv = 0 # threshold for the first derivative of distance to consider a point as "dirt" (e.g., sudden change in distance)
        threshold_reflect = 100 # threshold reflectivity to consider a point as "dirt"
        for file_ in folder_path.iterdir():
            if file_.suffix != ".mcap":
                continue
            try:
                data = list(get_lidar_data(file_))
                coords = [d[0] for d in data]
                intensity = [d[1] for d in data]
                reflectivity = [d[2] for d in data]

                frames_coords = coords
                frames_reflectivity = reflectivity


                
                # precompute dirt percentages for all frames
                for sector in range(n_sectors):
                    dist_all = np.zeros((len(frames_coords), coords[0].shape[0], coords[0].shape[1]//n_sectors), dtype=object)
                    reflect_all = np.zeros((len(frames_coords), coords[0].shape[0], coords[0].shape[1]//n_sectors), dtype=object)
                    first_deriv_all = np.zeros((len(frames_coords), coords[0].shape[0], coords[0].shape[1]//n_sectors), dtype=object)
                    percentage_dirt = np.zeros(len(frames_coords))
                    n_sensors = coords[0].shape[0] * coords[0].shape[1]//n_sectors

                    slice_start = (sector * frames_coords[0].shape[1]) // n_sectors
                    slice_end = ((sector + 1) * frames_coords[0].shape[1]) // n_sectors
                    for frame_idx in range(len(frames_coords)):
                        points = frames_coords[frame_idx][:, slice_start:slice_end, :]
                        dist_all[frame_idx, :] = np.linalg.norm(points, axis=2)
                        reflect = frames_reflectivity[frame_idx][:, slice_start:slice_end]
                        reflect_all[frame_idx, :] = reflect
                        if frame_idx > 0:
                            first_deriv_all[frame_idx, :] = dist_all[frame_idx, :] - dist_all[frame_idx - 1, :]
                        percentage_dirt[frame_idx] = (np.sum((dist_all[frame_idx, :] < threshold_distance) & ((first_deriv_all[frame_idx, :] < threshold_deriv) | (reflect_all[frame_idx, :] < threshold_reflect))) / n_sensors) * 100

                    # write results into csv file
                    if f"dirt_percentage_sector_{sector}" not in df.columns:
                        df[f"dirt_percentage_sector_{sector}"] = np.nan
                    df.loc[idx, f"dirt_percentage_sector_{sector}"] = np.mean(percentage_dirt)
                    print (f"Analyzed sector {sector + 1}/{n_sectors}")
                    print (f"Mean percentage of dirt across all frames: {np.mean(percentage_dirt):.2f}%")
                    print (f"Maximum percentage of dirt in any frame: {np.max(percentage_dirt):.2f}%")
                    print (f"Minimum percentage of dirt in any frame: {np.min(percentage_dirt):.2f}%")
                    print (f"Standard deviation of percentage of dirt across all frames: {np.std(percentage_dirt):.2f}%")

                    if f"mean_reflectivity_sector_{sector}" not in df.columns:
                        df[f"mean_reflectivity_sector_{sector}"] = np.nan
                    df.loc[idx, f"mean_reflectivity_sector_{sector}"] = np.mean(reflect_all)
                    print (f"Mean reflectivity: {np.mean(reflect_all):.2f}")
                    print (f"Max reflectivity: {np.max(reflect_all):.2f}")
                    print (f"Min reflectivity: {np.min(reflect_all):.2f}")
                    print (f"Std reflectivity: {np.std(reflect_all):.2f} \n")
            except Exception as e:
                print (f"Error processing file {filename}: {e}")

    df.to_csv(csv_path, index=False)

    for file_ in folder_path.iterdir():
        if file_.suffix != ".mcap":
            continue
        data = list(get_lidar_data(file_))
        coords = [d[0] for d in data]
        intensity = [d[1] for d in data]
        reflectivity = [d[2] for d in data]

        frames_coords = coords
        frames_reflectivity = reflectivity

        fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3,
        figsize=(18, 4),
        gridspec_kw={"width_ratios": [2, 1, 1]})

        # Scatter plot
        sc = ax1.scatter(np.zeros(1), np.zeros(1), s=1, c="black")
        ax1.set_xlim(0, frames_coords[0].shape[1])
        ax1.set_ylim(0, frames_coords[0].shape[0])
        ax1.invert_yaxis()
        ax1.set_xlabel("Horizontal index")
        ax1.set_ylabel("Vertical index")

        # Histogram distance
        hist_line, = ax2.plot([], [], lw=2, color = 'black')
        ax2.set_xlabel("Distance (m)")
        ax2.set_ylabel("Count")
        ax2.set_title("Distance distribution")
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 5000)

        # Histogram distance first deriv
        hist_line_deriv, = ax3.plot([], [], lw=2, color = 'black')
        ax3.set_xlabel("Distance (m)")
        ax3.set_ylabel("Count")
        ax3.set_title("1st deriv of distance")
        ax3.set_xlim(-10, 10)
        ax3.set_ylim(0, 100)

        def update(frame_idx):
            points = frames_coords[frame_idx]
            reflect = frames_reflectivity[frame_idx]

            h, w, _ = points.shape

            i_idx = np.repeat(np.arange(h), w)
            j_idx = np.tile(np.arange(w), h)

            dist = np.linalg.norm(points, axis=2).flatten()

            # =========================
            # RIGHT: histogram distance first deriv
            # =========================

            if frame_idx > 0:
                prev_points = frames_coords[frame_idx - 1]
                prev_dist = np.linalg.norm(prev_points, axis=2).flatten()

                dist_deriv = dist - prev_dist
                bins_deriv = np.linspace(np.min(dist_deriv), np.max(dist_deriv), 200)
                hist_deriv, edges_deriv = np.histogram(dist_deriv, bins=bins_deriv)
                centers_deriv = (edges_deriv[:-1] + edges_deriv[1:]) / 2
                hist_line_deriv.set_data(centers_deriv, hist_deriv)
            else:
                dist_deriv = np.zeros_like(dist)

            # =========================
            # MIDDLE: histogram distance
            # =========================
            bins = np.linspace(0, np.max(dist), 200)
            hist, edges = np.histogram(dist, bins=bins)

            centers = (edges[:-1] + edges[1:]) / 2

            hist_line.set_data(centers, hist)

            # =========================
            # LEFT: scatter plot
            # ========================
            text = ax1.text(
                0.02, 0.95,
                "",
                transform=ax1.transAxes,
                fontsize=12,
                color="white",
                bbox=dict(facecolor="black", alpha=0.5)
            )

            sizes = 1/(dist) * 50
            reflect_flat = reflect.reshape(-1)
            colors = np.zeros((len(dist), 4))

            r = reflect_flat
            norm_ref = (r - np.min(r)) / (np.max(r) - np.min(r) + 1e-6)

            colors[:] = plt.cm.viridis(norm_ref)
            close_mask = (dist < threshold_distance) & ((dist_deriv < threshold_deriv) | (reflect_flat < threshold_reflect))
            colors[close_mask] = [1, 0, 0, 1]

            sc.set_offsets(np.c_[j_idx, i_idx])
            sc.set_sizes(sizes)
            sc.set_facecolors(colors)

            # calculate number of sensor points with distance less than threshold
            num_dirt_points = np.sum(close_mask)
            perc_dirt = (num_dirt_points / len(dist)) * 100

            text.set_text(f"Dirt (<{threshold_distance}m): {perc_dirt:.2f}%")

            return sc, hist_line, hist_line_deriv

        anim = FuncAnimation(fig, update, frames=len(frames_coords), interval=100)
        plt.show()