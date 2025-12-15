import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Configuration
FILES = {
    "Conservative": "experiment_data_conservative.csv",
    "Neutral":      "experiment_data_neutral.csv",
    "Open":         "experiment_data_open.csv"
}
OUTPUT_IMAGE = "final_tradeoff_analysis.png"

def calculate_metrics(filename):
    """
    Calculates Path Length, Smoothness, and Safety Violations for a single run.
    """
    if not os.path.exists(filename):
        print(f"[WARN] File not found: {filename}")
        return None

    try:
        df = pd.read_csv(filename)
        
        # 1. Path Length (Efficiency)
        # Sum of Euclidean distances between consecutive points
        dx = np.diff(df['X'])
        dy = np.diff(df['Y'])
        step_dists = np.sqrt(dx**2 + dy**2)
        total_length = np.sum(step_dists)
        
        # 2. Smoothness (Comfort)
        # Metric: Spectral Smoothness (Sum of squared heading changes / Path Length)
        # Lower score = Smoother path
        d_theta = np.diff(df['Theta'])
        # Normalize angles to [-pi, pi] to handle the -3.14 to 3.14 jump
        d_theta = (d_theta + np.pi) % (2 * np.pi) - np.pi 
        
        if total_length > 0:
            smoothness_score = np.sum(d_theta**2) / total_length
        else:
            smoothness_score = 0.0
        
        # 3. Safety (Intimate Zone Violations)
        # Count frames where distance to human < 0.45m
        violations = np.sum(df['DistHuman'] < 0.45)
        
        # 4. Minimum Distance (Proxy for risk)
        min_dist = df['DistHuman'].min()
        
        return {
            "Length": total_length,
            "Smoothness": smoothness_score,
            "Violations": violations,
            "MinDist": min_dist
        }
        
    except Exception as e:
        print(f"[ERROR] Failed to process {filename}: {e}")
        return None

def main():
    results = {}
    
    print("--- Processing Experiment Data ---")
    for profile, filepath in FILES.items():
        metrics = calculate_metrics(filepath)
        if metrics:
            results[profile] = metrics
            print(f"[{profile}] Length: {metrics['Length']:.2f}m | "
                  f"Smoothness: {metrics['Smoothness']:.3f} | "
                  f"Min Dist: {metrics['MinDist']:.2f}m")

    if not results:
        print("[ERROR] No data found. Run the simulation first.")
        return

    # --- Plotting the Comparison ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    profiles = ["Conservative", "Neutral", "Open"]
    # Filter to only profiles we actually found
    profiles = [p for p in profiles if p in results]
    
    colors = ['#3366CC', '#109618', '#DC3912'] # Blue, Green, Red
    active_colors = [colors[i] for i, p in enumerate(["Conservative", "Neutral", "Open"]) if p in results]

    # Data lists
    lengths = [results[p]["Length"] for p in profiles]
    smoothness = [results[p]["Smoothness"] for p in profiles]
    min_dists = [results[p]["MinDist"] for p in profiles]

    # Plot 1: Efficiency (Path Length)
    bars1 = axes[0].bar(profiles, lengths, color=active_colors, alpha=0.8)
    axes[0].set_title("Efficiency (Path Length)", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Total Distance (m) - Lower is Better")
    axes[0].grid(axis='y', alpha=0.3)
    # Add values on top
    for bar in bars1:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.1f}m', ha='center', va='bottom')

    # Plot 2: Comfort (Smoothness)
    bars2 = axes[1].bar(profiles, smoothness, color=active_colors, alpha=0.8)
    axes[1].set_title("Comfort (Path Jerkiness)", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("Smoothness Score - Lower is Better")
    axes[1].grid(axis='y', alpha=0.3)
    for bar in bars2:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.3f}', ha='center', va='bottom')

    # Plot 3: Safety (Minimum Distance)
    # We use Min Distance because Violations might be 0 for all if code works well
    bars3 = axes[2].bar(profiles, min_dists, color=active_colors, alpha=0.8)
    axes[2].set_title("Safety Margin (Min Distance)", fontsize=12, fontweight='bold')
    axes[2].set_ylabel("Distance (m) - Higher is Safer")
    axes[2].axhline(y=0.45, color='red', linestyle='--', label='Intimate Limit')
    axes[2].legend(loc='lower right')
    axes[2].grid(axis='y', alpha=0.3)
    for bar in bars3:
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.2f}m', ha='center', va='bottom')

    plt.suptitle("Trade-off Analysis: Efficiency vs. Comfort vs. Safety", fontsize=16)
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"\n[SUCCESS] Comparison chart saved to {OUTPUT_IMAGE}")

if __name__ == "__main__":
    main()