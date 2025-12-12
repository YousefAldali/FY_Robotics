import pandas as pd
import matplotlib.pyplot as plt
import os

# Files to compare
files = {
    "Conservative": "experiment_data_conservative.csv",
    "Neutral":      "experiment_data_neutral.csv",
    "Open":         "experiment_data_open.csv"
}

plt.figure(figsize=(10, 6))

colors = {"Conservative": "blue", "Neutral": "green", "Open": "red"}
linestyles = {"Conservative": "-", "Neutral": "--", "Open": "-."}

metrics = []

print("--- COMPARATIVE ANALYSIS ---")

for profile, csv_file in files.items():
    if not os.path.exists(csv_file):
        print(f"[WARN] Missing {csv_file}, skipping...")
        continue
        
    # Load data without header assumption first to check, 
    # but based on your previous code, it has headers.
    try:
        df = pd.read_csv(csv_file, header=None, skiprows=1)
        # Assign columns: Time, X, Y, Theta, DistHuman
        df.columns = ["Time", "X", "Y", "Theta", "DistHuman"]
        
        # Calculate Mean Distance
        avg_dist = df["DistHuman"].mean()
        min_dist = df["DistHuman"].min()
        metrics.append([profile, avg_dist, min_dist])
        
        # Plot
        plt.plot(df["Time"], df["DistHuman"], 
                 label=f"{profile} (Avg: {avg_dist:.2f}m)",
                 color=colors[profile],
                 linestyle=linestyles[profile],
                 linewidth=2)
                 
    except Exception as e:
        print(f"[ERROR] processing {profile}: {e}")

# Add Social Zones
plt.axhline(y=0.45, color='black', linestyle=':', linewidth=1.5, label='Intimate Limit (0.45m)')
plt.axhline(y=1.2, color='gray', linestyle=':', linewidth=1.5, label='Personal Limit (1.2m)')

plt.title("Impact of Proxemic Profiles on Human-Robot Distance", fontsize=14)
plt.xlabel("Time (s)", fontsize=12)
plt.ylabel("Distance to Human (m)", fontsize=12)
plt.legend(loc="upper right")
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save
plt.savefig("comparative_proxemics_plot.png", dpi=300)
print("\n[INFO] Graph saved as 'comparative_proxemics_plot.png'")

print("\n--- SUMMARY TABLE ---")
print(f"{'Profile':<15} | {'Avg Dist (m)':<15} | {'Min Dist (m)':<15}")
print("-" * 50)
for m in metrics:
    print(f"{m[0]:<15} | {m[1]:<15.2f} | {m[2]:<15.2f}")