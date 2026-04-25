from pathlib import Path
from rosbag_lidar import get_lidar_data
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

if __name__ == "__main__":
    # this is a minimal example showing the usage of the get_lidar_data function.
    folder_path = Path("/Users/mariemehlfeldt/Desktop/Hackmingin2026_Data/rosbag2_2026_03_13-17_21_03")

    threshold_distance = 0.1 # threshold distance in meters to consider a point as "dirt"
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

            fig, ax = plt.subplots(figsize=(14, 4))

            sc = ax.scatter(np.zeros(1), np.zeros(1), s=1, c="black")

            ax.set_xlim(0, 720)
            ax.set_ylim(0, 16)
            ax.invert_yaxis()
            ax.set_xlabel("Horizontal index")
            ax.set_ylabel("Vertical index")

            def update(frame_idx):
                points = frames_coords[frame_idx]
                reflect = frames_reflectivity[frame_idx]

                h, w, _ = points.shape

                i_idx = np.repeat(np.arange(h), w)
                j_idx = np.tile(np.arange(w), h)

                dist = np.linalg.norm(points, axis=2).flatten()
                text = ax.text(
                    0.02, 0.95,
                    "",
                    transform=ax.transAxes,
                    fontsize=12,
                    color="white",
                    bbox=dict(facecolor="black", alpha=0.5)
                )

                sizes = 1/dist * 50

                reflect_flat = reflect.reshape(-1)

                colors = np.zeros((len(dist), 4))

                r = reflect_flat
                norm_ref = (r - np.min(r)) / (np.max(r) - np.min(r) + 1e-6)

                colors[:] = plt.cm.viridis(norm_ref)

                close_mask = dist < threshold_distance
                colors[close_mask] = [1, 0, 0, 1]

                sc.set_offsets(np.c_[j_idx, i_idx])
                sc.set_sizes(sizes)
                sc.set_facecolors(colors)

                # calculate number of sensor points with distance less than threshold
                num_dirt_points = np.sum(close_mask)
                perc_dirt = (num_dirt_points / len(dist)) * 100

                text.set_text(f"Dirt (<{threshold_distance}m): {perc_dirt:.2f}%")

                return sc,

            anim = FuncAnimation(fig, update, frames=len(frames_coords), interval=1)
            plt.show()
        except Exception as e:
            print(f"Error processing {file_}: {e}")