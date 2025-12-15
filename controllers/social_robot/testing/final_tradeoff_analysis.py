import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import re

FILE_PATTERNS = {
    "Baseline":     "experiment_data_baseline_*.csv",
    "Conservative": "experiment_data_conservative_*.csv",
    "Neutral":      "experiment_data_neutral_*.csv",
    "Open":         "experiment_data_open_*.csv",
}

OUTPUT_IMAGE = "final_tradeoff_with_errorbars_baseline.png"

def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]

def process_single_run(filename):
    try:
        df = pd.read_csv(filename)

        dx = np.diff(df["X"])
        dy = np.diff(df["Y"])
        total_length = np.sum(np.sqrt(dx**2 + dy**2))

        d_theta = np.diff(df["Theta"])
        d_theta = (d_theta + np.pi) % (2 * np.pi) - np.pi
        smoothness = np.sum(d_theta**2) / total_length if total_length > 0 else 0

        min_dist = df["DistHuman"].min()

        return total_length, smoothness, min_dist
    except Exception as e:
        print(f"[ERR] {filename}: {e}")
        return None

def main():
    stats = {}
    ns = {}
    profiles = ["Baseline", "Conservative", "Neutral", "Open"]

    for profile in profiles:
        files = sorted(glob.glob(FILE_PATTERNS[profile]), key=natural_key)
        print(f"[{profile}] Found {len(files)} files.")

        lengths, smooths, mins = [], [], []
        for f in files:
            res = process_single_run(f)
            if res:
                L, S, M = res
                lengths.append(L)
                smooths.append(S)
                mins.append(M)

        stats[profile] = {
            "Length": (np.mean(lengths), np.std(lengths)),
            "Smoothness": (np.mean(smooths), np.std(smooths)),
            "MinDist": (np.mean(mins), np.std(mins)),
        }
        ns[profile] = len(mins)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    colors = {
        "Baseline": "black",
        "Conservative": "#3366CC",
        "Neutral": "#109618",
        "Open": "#DC3912"
    }

    metrics_map = [
        ("Length", "Efficiency (Path Length)", "Meters (↓ Better)"),
        ("Smoothness", "Comfort (Smoothness)", "Score (↓ Better)"),
        ("MinDist", "Safety (Min Distance)", "Meters (↑ Better)")
    ]

    for i, (key, title, ylabel) in enumerate(metrics_map):
        means = [stats[p][key][0] for p in profiles]
        stds = [stats[p][key][1] for p in profiles]

        bars = axes[i].bar(
            profiles, means, yerr=stds, capsize=10,
            color=[colors[p] for p in profiles],
            alpha=0.85, ecolor="black"
        )

        axes[i].set_title(title, fontweight="bold")
        axes[i].set_ylabel(ylabel)
        axes[i].grid(axis="y", alpha=0.3)

        for bar, p, mean, std in zip(bars, profiles, means, stds):
            axes[i].text(
                bar.get_x() + bar.get_width() / 2,
                mean + std + 0.05 * abs(mean),
                f"n={ns[p]}\n{mean:.2f}±{std:.2f}",
                ha="center", va="bottom", fontsize=9
            )

    axes[2].axhline(0.45, color="red", linestyle="--", label="Intimate (0.45m)")
    axes[2].legend()

    plt.suptitle("Baseline vs Social Navigation Trade-offs (n=12)", fontsize=16)
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"[SUCCESS] Saved {OUTPUT_IMAGE}")

if __name__ == "__main__":
    main()