import numpy as np
import csv
import matplotlib.pyplot as plt
import math
import os

# Configuration
INPUT_FILE = '/Users/faridkhan/Desktop/Robotics/editting_version/V3/FY_Robotics copy 3/controllers/social_robot/experiment_data.csv'
OUTPUT_REPORT = 'social_navigation_report.txt'

def calculate_smoothness(theta_arr, dist_arr):

    # Calculate change in heading (d_theta)
    d_theta = np.diff(theta_arr)
    
    # Normalize d_theta to range [-pi, pi] to handle 359->1 degree transitions
    d_theta = (d_theta + np.pi) % (2 * np.pi) - np.pi
    
    # Calculate change in distance (d_s)
    d_s = dist_arr + 1e-6
    
    # Calculate Curvature (Kappa) = d_theta / d_s
    curvature = np.abs(d_theta) / d_s
    
    # Metric 1: Average Curvature (The primary smoothness metric)
    avg_curvature = np.mean(curvature)
    
    # Metric 2: Spectral Smoothness (Sum of squared jerk/changes) represents "wobble"
    # This is often used in human-robot interaction studies
    spectral_metric = np.sum(d_theta**2) / np.sum(d_s)
    
    return avg_curvature, spectral_metric

def analyze_experiment(filename):
    if not os.path.exists(filename):
        print(f"[ERROR] File {filename} not found. Run the simulation first!")
        return

    print(f"Loading data from {filename}...")
    
    # Load Data: [Time, X, Y, Theta, DistHuman]
    try:
        data = np.loadtxt(filename, delimiter=',', skiprows=1)
    except Exception as e:
        print(f"[ERROR] Could not read CSV: {e}")
        return

    # Extract columns
    time = data[:, 0]
    x = data[:, 1]
    y = data[:, 2]
    theta = data[:, 3]
    dist_human = data[:, 4]

    # 1. Calculate Path Length 
    dx = np.diff(x)
    dy = np.diff(y)
    step_distances = np.sqrt(dx**2 + dy**2)
    total_path_length = np.sum(step_distances)

    #  2. Calculate Smoothness 
    # We pass the theta array and the step_distances array
    avg_curv, smooth_score = calculate_smoothness(theta, step_distances)

    # --- 3. Safety & Comfort Metrics ---
    min_human_dist = np.min(dist_human)
    avg_human_dist = np.mean(dist_human)
    
    # Count strict proxemic violations (e.g., entering "Intimate" zone < 0.45m)
    intimate_zone_violations = np.sum(dist_human < 0.45)

    # Generate Report 
    report = (
        f"SOCIAL NAVIGATION EXPERIMENT REPORT\n"
        f"Data Source:       {filename}\n"
        f"Duration:          {time[-1] - time[0]:.2f} seconds\n"
        f"Total Samples:     {len(time)}\n\n"
    
        f"NAVIGATION METRICS \n"
        f"Total Path Length: {total_path_length:.2f} meters\n"
        f"Avg. Velocity:     {total_path_length / (time[-1] - time[0]):.2f} m/s\n\n"
    
        f"SMOOTHNESS METRICS (LOWER IS BETTER)\n"
        f"Avg. Curvature:    {avg_curv:.4f} rad/m\n"
        f"Smoothness Index:  {smooth_score:.4f}\n\n"
    
        f" PROXEMICS & SAFETY \n"
        f"Min Dist to Human: {min_human_dist:.2f} meters\n"
        f"Avg Dist to Human: {avg_human_dist:.2f} meters\n"
        f"Intimate Violations (<0.45m): {intimate_zone_violations}\n"
)

    print(report)
    
    # Save Report to File
    with open(OUTPUT_REPORT, 'w') as f:
        f.write(report)
    print(f"[INFO] Report saved to {OUTPUT_REPORT}")

    # Optional: Plotting for Visual Confirmation 
    plot_results(x, y, dist_human, time)

def plot_results(x, y, dist_human, time):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Trajectory
    ax1.plot(x, y, label='Robot Path', color='blue')
    ax1.set_title("Robot Trajectory")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.grid(True)
    ax1.axis('equal')
    
    # Plot 2: Proxemics over Time
    ax2.plot(time, dist_human, color='green')
    ax2.axhline(y=0.45, color='red', linestyle='--', label='Intimate Zone (0.45m)')
    ax2.axhline(y=1.2, color='orange', linestyle='--', label='Personal Zone (1.2m)')
    ax2.set_title("Distance to Human Over Time")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Distance (m)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("experiment_analysis_plot.png")
    print("[INFO] Analysis plot saved to experiment_analysis_plot.png")

if __name__ == "__main__":
    analyze_experiment(INPUT_FILE)