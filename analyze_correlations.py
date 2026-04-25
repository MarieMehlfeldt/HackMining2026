import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Daten laden
csv_path = Path("/Users/mariemehlfeldt/Desktop/HackMining2026/data_key.csv")
df = pd.read_csv(csv_path)

# Sicherstellen, dass die relevante Spalte numerisch ist
df["dirt spray level"] = pd.to_numeric(df["dirt"], errors="coerce")

# omit dirst spray level 5
df = df[df["dirt spray level"] != 5]

# Alle Sektor-Spalten finden
sector_cols = [col for col in df.columns if col.startswith("dirt_percentage_sector_")]

# Nach dirt spray level gruppieren
grouped = df.groupby("dirt spray level")

# Plot
plt.figure()

for col in sector_cols:
    mean = grouped[col].mean()
    std = grouped[col].std()

    x = mean.index.values

    # Linie (Mittelwert)
    plt.plot(x, mean.values, label=col)

    # Standardabweichung als Fläche
    plt.fill_between(
        x,
        mean - std,
        mean + std,
        alpha=0.2
    )

# Achsen & Layout
plt.xlabel("Dirt Spray Level")
plt.ylabel("Dirt Percentage")
plt.legend()
plt.title("Dirt vs Spray Level per Sector")

plt.show()


# Plot of values subtracting the reference (dirt spray level 0)
plt.figure()

for col in sector_cols:
    if col != "dirt_percentage_sector_2":
        continue
    mean = grouped[col].mean() - grouped[col].mean().loc[0]  # Subtract the mean of the reference level (0)
    std = grouped[col].sem()

    x = mean.index.values

    # Linie (Mittelwert)
    plt.plot(x, mean.values, label=col)

    # Standardabweichung als Fläche
    plt.fill_between(
        x,
        mean - std,
        mean + std,
        alpha=0.2
    )

# Achsen & Layout
plt.xlabel("Dirt Spray Level")
plt.ylabel("Dirt Detected (% of sensor points occluded)")
plt.title("Dirt vs Spray Level per Sector")
plt.xticks(np.arange(0, 5))  # Setzt die x-Achse auf die Werte 0 bis 4

plt.show()