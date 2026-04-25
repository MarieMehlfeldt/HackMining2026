"""
analyze_reflectivity.py
=======================
Offline LiDAR reflectivity analysis for sensor contamination detection.

Goal:
    For each fixed board distance, test whether raw LiDAR reflectivity
    decreases as water spray level (contamination) increases.

Outputs:
    - reflectivity_summary.csv       : per-recording statistics (raw)
    - reflectivity_normalized.csv    : statistics normalized to clean baseline
    - plots/                         : one folder of .png plot files
    - correlations printed to console

Usage:
    1. Set BASE_FOLDER to the folder containing data_key.csv and your recording folders.
    2. Run:  python analyze_reflectivity.py
"""

# ── Imports ─────────────────────────────────────────────────────────────────
from pathlib import Path          # clean cross-platform file paths
import numpy as np                # numerical operations
import pandas as pd               # tables / CSV handling
import matplotlib.pyplot as plt   # plots
import warnings

# Spearman correlation (optional – gracefully skipped if scipy is missing)
try:
    from scipy.stats import spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("NOTE: scipy not found – Spearman correlation will be skipped.")
    print("      Install with:  pip install scipy")

# Your LiDAR data loader
from rosbag_lidar import get_lidar_data


# ── Configuration ────────────────────────────────────────────────────────────
# Set this to the folder that contains data_key.csv and all rosbag2_* folders.
BASE_FOLDER = Path("/Users/olelee/Desktop/Minedata")   # defaults to same folder as this script

CSV_FILE    = BASE_FOLDER / "data_key.csv"
OUTPUT_DIR  = BASE_FOLDER / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)

# Distance threshold (metres) for the "near-sensor artifact" metric
NEAR_SENSOR_THRESHOLD = 0.1


# ── Helper: find the single .mcap file inside a recording folder ─────────────
def find_mcap(recording_folder: Path) -> Path | None:
    """Return the first .mcap file found in recording_folder, or None."""
    mcap_files = list(recording_folder.glob("*.mcap"))
    if len(mcap_files) == 0:
        print(f"  WARNING: no .mcap file found in {recording_folder}")
        return None
    if len(mcap_files) > 1:
        print(f"  WARNING: multiple .mcap files in {recording_folder}, using first")
    return mcap_files[0]


# ── Helper: compute per-frame statistics ─────────────────────────────────────
def compute_frame_stats(points: np.ndarray, reflectivity: np.ndarray) -> dict:
    """
    Given the point array (shape ≈ 16×720×3) and reflectivity array
    (shape ≈ 16×720) for one frame, return a dict of statistics.

    We flatten both arrays and remove NaN / infinite values before computing.
    """
    # Flatten to 1-D arrays
    pts_flat    = points.reshape(-1, 3)          # shape: (N, 3)
    ref_flat    = reflectivity.reshape(-1)        # shape: (N,)

    # Compute distance from the LiDAR origin for every point
    dist_flat   = np.linalg.norm(pts_flat, axis=1)   # shape: (N,)

    # ── Sanity counts (reported but NOT used to filter) ──────────────────
    n_total     = ref_flat.size
    n_nan       = np.sum(np.isnan(ref_flat))
    n_inf       = np.sum(np.isinf(ref_flat))
    n_zero_dist = np.sum(dist_flat == 0.0)

    # ── Build a "valid" mask: remove NaN and infinite reflectivity only ──
    valid_mask  = np.isfinite(ref_flat)
    ref_valid   = ref_flat[valid_mask]
    dist_valid  = dist_flat[valid_mask]

    n_valid     = ref_valid.size

    if n_valid == 0:
        # All points were invalid – return NaN for everything
        return {
            "mean_reflectivity":   np.nan,
            "median_reflectivity": np.nan,
            "std_reflectivity":    np.nan,
            "p10_reflectivity":    np.nan,
            "p25_reflectivity":    np.nan,
            "p75_reflectivity":    np.nan,
            "p90_reflectivity":    np.nan,
            "valid_point_count":   0,
            "near_sensor_pct":     np.nan,
            "n_nan":               n_nan,
            "n_inf":               n_inf,
            "n_zero_dist":         n_zero_dist,
        }

    # ── Reflectivity statistics (raw, not normalized) ────────────────────
    mean_r   = np.mean(ref_valid)
    median_r = np.median(ref_valid)
    std_r    = np.std(ref_valid)
    p10_r    = np.percentile(ref_valid, 10)
    p25_r    = np.percentile(ref_valid, 25)
    p75_r    = np.percentile(ref_valid, 75)
    p90_r    = np.percentile(ref_valid, 90)

    # ── Near-sensor artifact percentage ─────────────────────────────────
    near_sensor_pct = 100.0 * np.sum(dist_valid < NEAR_SENSOR_THRESHOLD) / n_valid

    return {
        "mean_reflectivity":   mean_r,
        "median_reflectivity": median_r,
        "std_reflectivity":    std_r,
        "p10_reflectivity":    p10_r,
        "p25_reflectivity":    p25_r,
        "p75_reflectivity":    p75_r,
        "p90_reflectivity":    p90_r,
        "valid_point_count":   n_valid,
        "near_sensor_pct":     near_sensor_pct,
        "n_nan":               n_nan,
        "n_inf":               n_inf,
        "n_zero_dist":         n_zero_dist,
    }


# ── Main analysis ─────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("LiDAR Reflectivity Analysis")
    print("=" * 60)

    # ── 1. Load the metadata CSV ─────────────────────────────────────────
    print(f"\nReading metadata from: {CSV_FILE}")
    metadata = pd.read_csv(CSV_FILE)

    # Rename columns to standard names if needed
    metadata.columns = metadata.columns.str.strip()
    col_map = {}
    if "filename"    in metadata.columns: col_map["filename"]    = "recording"
    if "distance"    in metadata.columns: col_map["distance"]    = "distance_m"
    if "dirt"        in metadata.columns: col_map["dirt"]        = "spray_level"
    if "spray_level" in metadata.columns: col_map["spray_level"] = "spray_level"
    metadata = metadata.rename(columns=col_map)

    print(f"  Found {len(metadata)} recordings in CSV.")
    print(f"  Columns: {list(metadata.columns)}")

    # ── 2. Loop over every recording and compute statistics ───────────────
    summary_rows = []   # will become our results table

    for idx, row in metadata.iterrows():
        recording_name = str(row["recording"]).strip()
        distance_m     = float(row["distance_m"])
        spray_level    = int(row["spray_level"])

        recording_folder = BASE_FOLDER / recording_name
        print(f"\n[{idx+1}/{len(metadata)}] {recording_name}  "
              f"(dist={distance_m}m, spray={spray_level})")

        # ── Find the .mcap file ──────────────────────────────────────────
        if not recording_folder.exists():
            print(f"  SKIP: folder not found → {recording_folder}")
            continue

        mcap_path = find_mcap(recording_folder)
        if mcap_path is None:
            continue

        print(f"  Loading: {mcap_path.name}")

        # ── Load LiDAR data ──────────────────────────────────────────────
        try:
            data = list(get_lidar_data(mcap_path))
        except Exception as e:
            print(f"  ERROR loading data: {e}")
            continue

        if len(data) == 0:
            print("  SKIP: no frames returned by get_lidar_data")
            continue

        print(f"  Frames loaded: {len(data)}")

        # ── 3. Compute per-frame statistics ──────────────────────────────
        # We use Option B: compute stat for each frame, then average.
        # This gives each frame equal weight regardless of point count.
        frame_stats_list = []

        for frame_idx, d in enumerate(data):
            points       = np.array(d[0])   # shape ≈ (16, 720, 3)
            reflectivity = np.array(d[2])   # shape ≈ (16, 720)

            if points.ndim != 3 or reflectivity.ndim != 2:
                # Unexpected shape – skip this frame
                continue

            stats = compute_frame_stats(points, reflectivity)
            frame_stats_list.append(stats)

        if len(frame_stats_list) == 0:
            print("  SKIP: no valid frames after processing")
            continue

        # ── 4. Aggregate frame-level stats into one recording-level row ──
        # Convert list of dicts → DataFrame for easy column-wise mean
        frame_df = pd.DataFrame(frame_stats_list)

        # Average across frames (Option B)
        agg = frame_df.mean(numeric_only=True)

        # Sanity totals (summed, not averaged)
        total_nan      = frame_df["n_nan"].sum()
        total_inf      = frame_df["n_inf"].sum()
        total_zerodist = frame_df["n_zero_dist"].sum()

        print(f"  mean reflectivity (avg of frame means): "
              f"{agg['mean_reflectivity']:.2f}")
        print(f"  Sanity – NaN points: {total_nan:.0f}, "
              f"Inf: {total_inf:.0f}, zero-dist: {total_zerodist:.0f}")

        summary_rows.append({
            "recording":           recording_name,
            "distance_m":          distance_m,
            "spray_level":         spray_level,
            "num_frames":          len(frame_stats_list),
            "mean_reflectivity":   agg["mean_reflectivity"],
            "median_reflectivity": agg["median_reflectivity"],
            "std_reflectivity":    agg["std_reflectivity"],
            "p10_reflectivity":    agg["p10_reflectivity"],
            "p25_reflectivity":    agg["p25_reflectivity"],
            "p75_reflectivity":    agg["p75_reflectivity"],
            "p90_reflectivity":    agg["p90_reflectivity"],
            "valid_point_count":   agg["valid_point_count"],
            "near_sensor_pct":     agg["near_sensor_pct"],
            "total_nan":           total_nan,
            "total_inf":           total_inf,
            "total_zero_dist":     total_zerodist,
        })

    if len(summary_rows) == 0:
        print("\nERROR: No recordings were successfully processed. Exiting.")
        return

    # ── 5. Build summary DataFrame and save ──────────────────────────────
    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values(["distance_m", "spray_level"]).reset_index(drop=True)

    summary_path = BASE_FOLDER / "reflectivity_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\n[✓] Saved raw summary → {summary_path}")

    # ── 6. Normalize each recording by its clean baseline ────────────────
    # For each distance, the clean recording is the one with spray_level == 0.
    # normalized_mean = mean_reflectivity / mean_reflectivity_of_clean_at_same_distance
    print("\nNormalizing against clean baselines...")

    norm_rows = []

    for dist, group in summary.groupby("distance_m"):
        # Find the clean (spray=0) row for this distance
        clean_rows = group[group["spray_level"] == 0]

        if len(clean_rows) == 0:
            print(f"  WARNING: no clean baseline for distance={dist}m – skipping.")
            continue

        baseline = clean_rows.iloc[0]   # use first if somehow duplicated

        baseline_mean   = baseline["mean_reflectivity"]
        baseline_median = baseline["median_reflectivity"]
        baseline_count  = baseline["valid_point_count"]

        for _, rec in group.iterrows():
            # Avoid division by zero
            norm_mean   = rec["mean_reflectivity"]   / baseline_mean   if baseline_mean   > 0 else np.nan
            norm_median = rec["median_reflectivity"] / baseline_median if baseline_median > 0 else np.nan
            norm_count  = rec["valid_point_count"]   / baseline_count  if baseline_count  > 0 else np.nan

            norm_rows.append({
                **rec.to_dict(),
                "baseline_mean_reflectivity":   baseline_mean,
                "baseline_median_reflectivity": baseline_median,
                "baseline_valid_point_count":   baseline_count,
                "normalized_mean_reflectivity":   norm_mean,
                "normalized_median_reflectivity": norm_median,
                "normalized_point_count":         norm_count,
            })

    normalized = pd.DataFrame(norm_rows)
    normalized = normalized.sort_values(["distance_m", "spray_level"]).reset_index(drop=True)

    norm_path = BASE_FOLDER / "reflectivity_normalized.csv"
    normalized.to_csv(norm_path, index=False)
    print(f"[✓] Saved normalized summary → {norm_path}")

    # ── 7. Plots ──────────────────────────────────────────────────────────
    print("\nGenerating plots...")

    distances  = sorted(normalized["distance_m"].unique())
    cmap       = plt.cm.get_cmap("tab10", len(distances))
    colors     = {d: cmap(i) for i, d in enumerate(distances)}
    spray_vals = sorted(normalized["spray_level"].unique())

    # ─ Plot 1: Raw mean reflectivity vs spray level ───────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for dist in distances:
        grp = normalized[normalized["distance_m"] == dist].sort_values("spray_level")
        ax.plot(grp["spray_level"], grp["mean_reflectivity"],
                marker="o", label=f"{dist} m", color=colors[dist])
    ax.set_title("Raw Mean Reflectivity vs Spray Level")
    ax.set_xlabel("Spray Level (0 = clean)")
    ax.set_ylabel("Mean Reflectivity (raw)")
    ax.set_xticks(spray_vals)
    ax.legend(title="Distance")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_raw_mean_reflectivity.png", dpi=150)
    plt.close(fig)

    # ─ Plot 2: Normalized mean reflectivity vs spray level ───────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for dist in distances:
        grp = normalized[normalized["distance_m"] == dist].sort_values("spray_level")
        ax.plot(grp["spray_level"], grp["normalized_mean_reflectivity"],
                marker="o", label=f"{dist} m", color=colors[dist])
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=1, label="Baseline = 1.0")
    ax.set_title("Normalized Mean Reflectivity vs Spray Level")
    ax.set_xlabel("Spray Level (0 = clean)")
    ax.set_ylabel("Normalized Mean Reflectivity\n(1.0 = clean baseline)")
    ax.set_xticks(spray_vals)
    ax.legend(title="Distance")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_normalized_mean_reflectivity.png", dpi=150)
    plt.close(fig)

    # ─ Plot 3: Normalized median reflectivity vs spray level ─────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for dist in distances:
        grp = normalized[normalized["distance_m"] == dist].sort_values("spray_level")
        ax.plot(grp["spray_level"], grp["normalized_median_reflectivity"],
                marker="s", label=f"{dist} m", color=colors[dist])
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=1, label="Baseline = 1.0")
    ax.set_title("Normalized Median Reflectivity vs Spray Level")
    ax.set_xlabel("Spray Level (0 = clean)")
    ax.set_ylabel("Normalized Median Reflectivity")
    ax.set_xticks(spray_vals)
    ax.legend(title="Distance")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_normalized_median_reflectivity.png", dpi=150)
    plt.close(fig)

    # ─ Plot 4: Normalized valid point count vs spray level ───────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for dist in distances:
        grp = normalized[normalized["distance_m"] == dist].sort_values("spray_level")
        ax.plot(grp["spray_level"], grp["normalized_point_count"],
                marker="^", label=f"{dist} m", color=colors[dist])
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=1, label="Baseline = 1.0")
    ax.set_title("Normalized Valid Point Count vs Spray Level")
    ax.set_xlabel("Spray Level (0 = clean)")
    ax.set_ylabel("Normalized Point Count")
    ax.set_xticks(spray_vals)
    ax.legend(title="Distance")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_normalized_point_count.png", dpi=150)
    plt.close(fig)

    # ─ Plot 5: Near-sensor artifact percentage vs spray level ─────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for dist in distances:
        grp = normalized[normalized["distance_m"] == dist].sort_values("spray_level")
        ax.plot(grp["spray_level"], grp["near_sensor_pct"],
                marker="D", label=f"{dist} m", color=colors[dist])
    ax.set_title(f"Near-Sensor Artifact % vs Spray Level\n"
                 f"(points within {NEAR_SENSOR_THRESHOLD} m of sensor)")
    ax.set_xlabel("Spray Level (0 = clean)")
    ax.set_ylabel("Near-Sensor Points (%)")
    ax.set_xticks(spray_vals)
    ax.legend(title="Distance")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "05_near_sensor_artifact_pct.png", dpi=150)
    plt.close(fig)

    # ─ Plot 6: Boxplot of normalized mean reflectivity by spray level ─────
    # Aggregate across all distances (already normalized, so distances are comparable)
    fig, ax = plt.subplots(figsize=(9, 5))
    box_data = [
        normalized[normalized["spray_level"] == s]["normalized_mean_reflectivity"].dropna().values
        for s in spray_vals
    ]
    ax.boxplot(box_data, labels=[str(s) for s in spray_vals], patch_artist=True,
               boxprops=dict(facecolor="steelblue", alpha=0.6))
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=1, label="Baseline = 1.0")
    ax.set_title("Normalized Mean Reflectivity by Spray Level\n(all distances pooled)")
    ax.set_xlabel("Spray Level")
    ax.set_ylabel("Normalized Mean Reflectivity")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "06_boxplot_normalized_reflectivity.png", dpi=150)
    plt.close(fig)

    print(f"[✓] Saved 6 plots to: {OUTPUT_DIR}")

    # ── 8. Spearman Correlations ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SPEARMAN CORRELATIONS  (spray_level vs normalized_mean_reflectivity)")
    print("=" * 60)
    print("  Interpretation:")
    print("    rho ≈ -1 → strong drop in reflectivity with more spray  ✓")
    print("    rho ≈  0 → no clear trend")
    print("    rho ≈ +1 → reflectivity increases with spray  (unexpected)")
    print()

    if HAS_SCIPY:
        # Per-distance Spearman
        for dist in distances:
            grp = normalized[normalized["distance_m"] == dist].sort_values("spray_level")
            x   = grp["spray_level"].values
            y   = grp["normalized_mean_reflectivity"].values
            mask = np.isfinite(y)
            if mask.sum() < 3:
                print(f"  {dist} m: not enough data points for correlation")
                continue
            rho, p = spearmanr(x[mask], y[mask])
            sig = "  *significant*" if p < 0.05 else ""
            print(f"  {dist:5.1f} m  →  rho = {rho:+.3f},  p = {p:.4f}{sig}")

        # Global Spearman (across all distances, normalized values)
        x_all = normalized["spray_level"].values
        y_all = normalized["normalized_mean_reflectivity"].values
        mask  = np.isfinite(y_all)
        if mask.sum() >= 3:
            rho_all, p_all = spearmanr(x_all[mask], y_all[mask])
            sig = "  *significant*" if p_all < 0.05 else ""
            print(f"\n  GLOBAL →  rho = {rho_all:+.3f},  p = {p_all:.4f}{sig}")
    else:
        print("  (scipy not available – install with: pip install scipy)")

    # ── 9. Quick summary table ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("NORMALIZED MEAN REFLECTIVITY SUMMARY TABLE")
    print("=" * 60)
    pivot = normalized.pivot_table(
        index="distance_m",
        columns="spray_level",
        values="normalized_mean_reflectivity"
    )
    with pd.option_context("display.float_format", "{:.3f}".format,
                           "display.width", 120):
        print(pivot.to_string())

    print("\n" + "=" * 60)
    print("Analysis complete.")
    print(f"  Raw summary:        {summary_path}")
    print(f"  Normalized summary: {norm_path}")
    print(f"  Plots folder:       {OUTPUT_DIR}")
    print("=" * 60)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
