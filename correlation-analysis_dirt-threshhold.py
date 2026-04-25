from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.stats import spearmanr, pearsonr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


CSV_PATH = Path("detection_analysis.csv")
OUTPUT_FOLDER = Path("analysis_outputs")
OUTPUT_FOLDER.mkdir(exist_ok=True)


REQUIRED_COLUMNS = [
    "file_name",
    "distance_m",
    "spray_level",
    "mean_reflectivity",
    "dirt_percent",
    "object_detection_threshold_m",
]


def check_columns(df):
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def clean_data(df):
    """
    Converts relevant columns to numeric values and removes rows
    where dirt_percent or object_detection_threshold_m is missing.
    """

    numeric_columns = [
        "distance_m",
        "spray_level",
        "mean_reflectivity",
        "dirt_percent",
        "object_detection_threshold_m",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)

    df = df.dropna(subset=[
        "dirt_percent",
        "object_detection_threshold_m",
        "mean_reflectivity",
    ])

    after = len(df)

    print(f"Rows before cleaning: {before}")
    print(f"Rows after cleaning:  {after}")

    return df


def calculate_correlation(df, x_col, y_col):
    """
    Calculates Spearman and Pearson correlations between two columns.
    """

    x = df[x_col].values
    y = df[y_col].values

    print("\n" + "=" * 60)
    print(f"Correlation: {x_col} vs {y_col}")
    print("=" * 60)

    if len(x) < 3:
        print("Not enough data points for correlation.")
        return

    if SCIPY_AVAILABLE:
        spearman_corr, spearman_p = spearmanr(x, y)
        pearson_corr, pearson_p = pearsonr(x, y)

        print(f"Spearman correlation: {spearman_corr:.3f}")
        print(f"Spearman p-value:     {spearman_p:.4f}")

        print(f"Pearson correlation:  {pearson_corr:.3f}")
        print(f"Pearson p-value:      {pearson_p:.4f}")

    else:
        print("scipy is not installed.")
        print("Install it with:")
        print("python -m pip install scipy")
        print()
        print("Using pandas correlation as fallback.")

        spearman_corr = df[[x_col, y_col]].corr(method="spearman").iloc[0, 1]
        pearson_corr = df[[x_col, y_col]].corr(method="pearson").iloc[0, 1]

        print(f"Spearman correlation: {spearman_corr:.3f}")
        print(f"Pearson correlation:  {pearson_corr:.3f}")


def plot_scatter_with_trend(df, x_col, y_col, title, output_name):
    """
    Creates a scatter plot with a simple linear trend line.
    """

    x = df[x_col].values
    y = df[y_col].values

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y)

    # Add trend line only if we have enough unique x values
    if len(np.unique(x)) >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(np.min(x), np.max(x), 100)
        y_line = slope * x_line + intercept
        plt.plot(x_line, y_line, linestyle="--")

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()

    output_path = OUTPUT_FOLDER / output_name
    plt.savefig(output_path, dpi=300)
    plt.show()

    print(f"Saved plot: {output_path}")


def plot_by_spray_level(df):
    """
    Optional plot: shows how detection threshold changes with spray level.
    """

    grouped = df.groupby("spray_level")["object_detection_threshold_m"].mean().reset_index()

    plt.figure(figsize=(8, 6))
    plt.plot(
        grouped["spray_level"],
        grouped["object_detection_threshold_m"],
        marker="o"
    )

    plt.xlabel("Spray level")
    plt.ylabel("Mean object detection threshold (m)")
    plt.title("Object detection threshold vs spray level")
    plt.grid(True)
    plt.tight_layout()

    output_path = OUTPUT_FOLDER / "threshold_vs_spray_level.png"
    plt.savefig(output_path, dpi=300)
    plt.show()

    print(f"Saved plot: {output_path}")


def main():
    df = pd.read_csv(CSV_PATH)

    check_columns(df)
    df = clean_data(df)

    print("\nCleaned data preview:")
    print(df.head())

    # Save cleaned version
    cleaned_path = OUTPUT_FOLDER / "cleaned_detection_analysis.csv"
    df.to_csv(cleaned_path, index=False)
    print(f"\nSaved cleaned data to: {cleaned_path}")

    # Main question:
    # Does dirt percentage correlate with detection threshold?
    calculate_correlation(
        df,
        "dirt_percent",
        "object_detection_threshold_m"
    )

    plot_scatter_with_trend(
        df,
        "dirt_percent",
        "object_detection_threshold_m",
        "Dirt percentage vs object detection threshold",
        "dirt_percent_vs_detection_threshold.png"
    )

    # Secondary question:
    # Does reflectivity correlate with detection threshold?
    calculate_correlation(
        df,
        "mean_reflectivity",
        "object_detection_threshold_m"
    )

    plot_scatter_with_trend(
        df,
        "mean_reflectivity",
        "object_detection_threshold_m",
        "Mean reflectivity vs object detection threshold",
        "reflectivity_vs_detection_threshold.png"
    )

    # Previous relationship:
    # Does dirt percentage correlate with reflectivity?
    calculate_correlation(
        df,
        "dirt_percent",
        "mean_reflectivity"
    )

    plot_scatter_with_trend(
        df,
        "dirt_percent",
        "mean_reflectivity",
        "Dirt percentage vs mean reflectivity",
        "dirt_percent_vs_reflectivity.png"
    )

    # Optional spray-level plot
    plot_by_spray_level(df)


if __name__ == "__main__":
    main()