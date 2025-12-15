import matplotlib
matplotlib.use("MacOSX")
import math
import yaml
from PIL import Image
import matplotlib.pyplot as plt

# ---------- USER INPUTS ----------
MAP_IMAGE_PATH = "/Users/faridkhan/Downloads/FY_Robotics copy 5/controllers/social_robot/map_snapshot_0840.png"
MAP_YAML_PATH  = None   
RESOLUTION_M_PER_PX = 0.05   

GROUND_TRUTH_M = {
    "Feature A (corridor length)": 16.04,   # Full corridor from bottom to top
    "Feature B (kitchen width)": 10.6,      # Kitchen room width
    "Feature C (doorway width)": 4.3,       # Bedroom doorway gap
}


# --------------------------------

def load_resolution(yaml_path):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return float(data["resolution"])

def pick_two_points(ax, title):
    ax.set_title(title + "\nClick START then END point (2 clicks)")
    plt.draw()
    pts = []
    print(f"\nWaiting for 2 clicks for: {title} ... (click inside the image window)")
    while len(pts) < 2:
        new_pts = plt.ginput(1, timeout=-1)
        if not new_pts:
            continue
        x, y = new_pts[0]
        pts.append((x, y))
        ax.plot([x], [y], marker="x")
        plt.draw()
        print(f"  Got click #{len(pts)} at (x={x:.1f}, y={y:.1f})")
    return pts[0], pts[1]


def dist_px(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

# Load image
img = Image.open(MAP_IMAGE_PATH)
# Many occupancy maps are grayscale; show as-is
fig, ax = plt.subplots()
ax.imshow(img, cmap="gray")
ax.axis("off")

plt.show(block=False)
plt.pause(0.5)

# Resolution
if MAP_YAML_PATH:
    res = load_resolution(MAP_YAML_PATH)
else:
    if RESOLUTION_M_PER_PX is None:
        raise ValueError("Set RESOLUTION_M_PER_PX (e.g., 0.05) or provide MAP_YAML_PATH.")
    res = float(RESOLUTION_M_PER_PX)

results = []

for feature_name, real_m in GROUND_TRUTH_M.items():
    if real_m is None:
        raise ValueError(f"Fill in ground truth meters for: {feature_name}")

    p1, p2 = pick_two_points(ax, feature_name)
    px = dist_px(p1, p2)
    measured_m = px * res
    error_m = abs(real_m - measured_m)
    pct_error = (error_m / real_m) * 100 if real_m != 0 else float("nan")

    # Draw the measured line
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]])

    results.append((feature_name, real_m, px, res, measured_m, error_m, pct_error))

plt.show()

print("\n--- Measurement Results ---")
print(f"Resolution used: {res:.6f} m/px\n")
for (name, real_m, px, res, measured_m, error_m, pct_error) in results:
    print(f"{name}")
    print(f"  Pixels:         {px:.2f} px")
    print(f"  Measured:       {measured_m:.3f} m")
    print(f"  Ground truth:   {real_m:.3f} m")
    print(f"  Abs error:      {error_m:.3f} m")
    print(f"  Percent error:  {pct_error:.2f}%\n")
