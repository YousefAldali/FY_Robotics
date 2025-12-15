import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.lines import Line2D

# ---------------- CONFIG ----------------
profiles = ["Baseline", "Conservative", "Neutral", "Open"]
num_runs_per_profile = 12

colors = {
    "Baseline": "black",
    "Conservative": "blue",
    "Neutral": "green",
    "Open": "red"
}
linestyles = {
    "Baseline": "-",
    "Conservative": "-",
    "Neutral": "--",
    "Open": "-."
}

INTIMATE_TH = 0.45
PERSONAL_TH = 1.2

# If your CSV heading column has a different name, add it here:
HEADING_COL_CANDIDATES = ["Theta", "theta", "Yaw", "yaw", "Heading", "heading"]


# ---------------- HELPERS ----------------
def compute_violation_rates(df, personal_th=PERSONAL_TH, intimate_th=INTIMATE_TH):
    total = len(df)
    if total == 0:
        return 0.0, 0.0
    personal = (df["DistHuman"] < personal_th).sum()
    intimate = (df["DistHuman"] < intimate_th).sum()
    return 100.0 * personal / total, 100.0 * intimate / total


def wrap_to_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def get_heading_series(df):
    """Return heading (radians) from a heading column, else estimate from X,Y."""
    for c in HEADING_COL_CANDIDATES:
        if c in df.columns:
            theta = df[c].astype(float).to_numpy()
            return theta

    if {"X", "Y"}.issubset(df.columns):
        dx = df["X"].diff().fillna(0).to_numpy()
        dy = df["Y"].diff().fillna(0).to_numpy()
        theta = np.arctan2(dy, dx)
        return theta

    return None


def compute_smoothness(df):
    """
    S = mean(|wrapToPi(theta[t+1] - theta[t])|)
    Lower = smoother.
    """
    theta = get_heading_series(df)
    if theta is None or len(theta) < 3:
        return np.nan

    dtheta = wrap_to_pi(np.diff(theta))
    return float(np.mean(np.abs(dtheta)))


# ---------------- STORAGE ----------------
metrics = {
    p: {
        "avg_dists": [],
        "min_dists": [],
        "total_dists": [],
        "speeds": [],
        "personal_rates": [],
        "intimate_rates": [],
        "smoothness": [],
    }
    for p in profiles
}

# ---------------- PROCESS ----------------
plt.figure(figsize=(14, 8))
print(f"--- PROCESSING {num_runs_per_profile * len(profiles)} EXPERIMENT FILES ---")

for profile in profiles:
    print(f"\nProcessing Profile: {profile}")
    for run_id in range(1, num_runs_per_profile + 1):
        filename = f"experiment_data_{profile.lower()}_{run_id}.csv"

        if not os.path.exists(filename):
            print(f"  [WARN] Missing: {filename}")
            continue

        try:
            df = pd.read_csv(filename)
            df.columns = [c.strip() for c in df.columns]

            required = {"Time", "DistHuman"}
            if not required.issubset(df.columns):
                print(f"  [ERR] {filename}: Missing required columns {required}. Found: {list(df.columns)}")
                continue

            # core distances
            avg_dist = float(df["DistHuman"].mean())
            min_dist = float(df["DistHuman"].min())

            # violations
            personal_rate, intimate_rate = compute_violation_rates(df)

            # path length + speed if X,Y exist
            if {"X", "Y"}.issubset(df.columns):
                dx = df["X"].diff().fillna(0)
                dy = df["Y"].diff().fillna(0)
                dist_traveled = float(np.sqrt(dx**2 + dy**2).sum())

                dt = df["Time"].diff().fillna(0)
                valid = dt > 0
                speeds = np.sqrt(dx[valid]**2 + dy[valid]**2) / dt[valid]
                mean_speed = float(speeds.mean()) if len(speeds) > 0 else 0.0
            else:
                dist_traveled = 0.0
                mean_speed = 0.0

            # smoothness
            smooth = compute_smoothness(df)

            # store
            metrics[profile]["avg_dists"].append(avg_dist)
            metrics[profile]["min_dists"].append(min_dist)
            metrics[profile]["total_dists"].append(dist_traveled)
            metrics[profile]["speeds"].append(mean_speed)
            metrics[profile]["personal_rates"].append(personal_rate)
            metrics[profile]["intimate_rates"].append(intimate_rate)
            metrics[profile]["smoothness"].append(smooth)

            # plot run
            plt.plot(
                df["Time"], df["DistHuman"],
                color=colors[profile],
                linestyle=linestyles[profile],
                alpha=0.25 if profile != "Baseline" else 0.55,
                linewidth=1.2 if profile != "Baseline" else 1.8
            )

            s_txt = "nan" if np.isnan(smooth) else f"{smooth:.4f}"
            print(
                f"  [OK] Run {run_id}: "
                f"Min={min_dist:.2f}m, Path={dist_traveled:.2f}m, "
                f"<1.2m={personal_rate:.1f}%, <0.45m={intimate_rate:.1f}%, "
                f"S={s_txt}"
            )

        except Exception as e:
            print(f"  [CRITICAL] {filename}: {e}")

# ---------------- PLOT ----------------
plt.axhline(INTIMATE_TH, color="black", linestyle=":", linewidth=2)
plt.axhline(PERSONAL_TH, color="gray", linestyle="--", linewidth=2)

plt.title("Distance to Human Over Time (Baseline vs Social Profiles)", fontsize=16)
plt.xlabel("Time (s)")
plt.ylabel("Distance to Human (m)")
plt.grid(alpha=0.3)

legend_lines = [Line2D([0], [0], color=colors[p], lw=2, linestyle=linestyles[p]) for p in profiles]
legend_lines += [
    Line2D([0], [0], color="black", lw=2, linestyle=":"),
    Line2D([0], [0], color="gray", lw=2, linestyle="--")
]
legend_labels = profiles + [f"Intimate ({INTIMATE_TH}m)", f"Personal ({PERSONAL_TH}m)"]

plt.legend(legend_lines, legend_labels, loc="upper right")
plt.tight_layout()
plt.savefig("48_tests_distance_comparison.png", dpi=300)

# ---------------- SUMMARY TABLE (CONSOLE) ----------------
print("\n" + "=" * 160)
print(
    f"{'Profile':<15} | {'N':<3} | {'MinDist (m)':<16} | "
    f"{'<1.2m (% time)':<16} | {'<0.45m (% time)':<17} | "
    f"{'AvgDist (m)':<14} | {'Path (m)':<10} | {'Smoothness':<16}"
)
print("-" * 160)

summary_rows = []

for p in profiles:
    n = len(metrics[p]["min_dists"])
    if n == 0:
        print(f"{p:<15} | 0   | N/A")
        continue

    mean_min = np.mean(metrics[p]["min_dists"])
    std_min  = np.std(metrics[p]["min_dists"])

    mean_personal = np.mean(metrics[p]["personal_rates"])
    std_personal  = np.std(metrics[p]["personal_rates"])

    mean_intimate = np.mean(metrics[p]["intimate_rates"])
    std_intimate  = np.std(metrics[p]["intimate_rates"])

    mean_avg  = np.mean(metrics[p]["avg_dists"])
    mean_path = np.mean(metrics[p]["total_dists"])

    svals = np.array(metrics[p]["smoothness"], dtype=float)
    svals = svals[~np.isnan(svals)]
    mean_s = np.mean(svals) if len(svals) else np.nan
    std_s  = np.std(svals) if len(svals) else np.nan

    mean_s_txt = "nan" if np.isnan(mean_s) else f"{mean_s:.4f}"
    std_s_txt  = "nan" if np.isnan(std_s) else f"{std_s:.4f}"

    print(
        f"{p:<15} | {n:<3} | "
        f"{mean_min:.2f}±{std_min:.2f}{'':<5} | "
        f"{mean_personal:6.1f}±{std_personal:4.1f}{'':<4} | "
        f"{mean_intimate:6.1f}±{std_intimate:4.1f}{'':<4} | "
        f"{mean_avg:6.2f}{'':<7} | "
        f"{mean_path:6.1f} | "
        f"{mean_s_txt}±{std_s_txt}"
    )

    summary_rows.append((p, n, mean_min, std_min, mean_personal, std_personal,
                         mean_intimate, std_intimate, mean_avg, mean_path, mean_s, std_s))

print("=" * 160)
print("\n[INFO] Saved plot as '48_tests_distance_comparison.png'")

# ---------------- LATEX TABLE ----------------
print("\n\n--- LATEX TABLE (copy/paste into Overleaf) ---\n")

print(r"\begin{table}[!t]")
print(rf"\caption{{Social navigation performance across proxemic profiles ($N=12$ trials per condition, mean $\pm$ std).}}")
print(r"\label{tab:results}")
print(r"\centering")
print(r"\setlength{\tabcolsep}{4pt}")
print(r"\renewcommand{\arraystretch}{1.1}")
print(r"\begin{tabularx}{\columnwidth}{l c c c c c c}")
print(r"\toprule")
print(r"Profile & MinDist [m] & $<1.2$m [\%] & $<0.45$m [\%] & AvgDist [m] & Path [m] & Smoothness $\downarrow$ \\")
print(r"\midrule")

for (p, n, mean_min, std_min, mean_personal, std_personal, mean_intimate, std_intimate, mean_avg, mean_path, mean_s, std_s) in summary_rows:
    s_txt = r"--" if np.isnan(mean_s) else f"{mean_s:.4f} $\\pm$ {std_s:.4f}"
    print(
        rf"{p} & "
        rf"${mean_min:.2f} \pm {std_min:.2f}$ & "
        rf"${mean_personal:.1f} \pm {std_personal:.1f}$ & "
        rf"${mean_intimate:.1f} \pm {std_intimate:.1f}$ & "
        rf"${mean_avg:.2f}$ & "
        rf"${mean_path:.1f}$ & "
        rf"${s_txt}$ \\"
    )

print(r"\bottomrule")
print(r"\end{tabularx}")
print(r"\end{table}")