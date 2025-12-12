import math
import numpy as np
from breezyslam.algorithms import RMHC_SLAM
from breezyslam.sensors import Laser


class WebotsLidar(Laser):
    def __init__(self, lidar_device, time_step_ms):
        n_beams = lidar_device.getHorizontalResolution()
        fov_rad = lidar_device.getFov()
        fov_deg = math.degrees(fov_rad)

        max_range_m = lidar_device.getMaxRange()
        max_range_mm = int(max_range_m * 1000)

        scan_rate_hz = 1000.0 / float(time_step_ms)

        super().__init__(n_beams, scan_rate_hz, fov_deg, max_range_mm)

        self.max_range_m = max_range_m


class SlamBackend:
    def __init__(self, lidar_device, time_step_ms):
        self.lidar_model = WebotsLidar(lidar_device, time_step_ms)
        self.max_range_m = self.lidar_model.max_range_m

        self.MAP_SIZE_PIXELS = 800
        self.MAP_SIZE_METERS = 20

        self.slam = RMHC_SLAM(self.lidar_model,
                              self.MAP_SIZE_PIXELS,
                              self.MAP_SIZE_METERS,
                              sigma_theta_degrees=5,
                              sigma_xy_mm=50,
                              hole_width_mm=200,
                              map_quality=1)

        self.mapbytes = bytearray(self.MAP_SIZE_PIXELS * self.MAP_SIZE_PIXELS)

        self.x_mm = 0
        self.y_mm = 0
        self.theta_deg = 0

        self.scan_period_sec = time_step_ms / 1000.0

    def update(self, ranges_m, d_center_m=None, dtheta_rad=None):
        max_range_m = self.max_range_m
        scan_mm = []
        for r in ranges_m:
            if math.isinf(r) or r <= 0.0:
                r = max_range_m
            scan_mm.append(int(r * 1000))

        pose_change = None
        if d_center_m is not None and dtheta_rad is not None:
            d_center_mm = int(round(d_center_m * 1000))
            dtheta_deg = -math.degrees(dtheta_rad)
            pose_change = (d_center_mm, dtheta_deg, self.scan_period_sec)
        if pose_change is None:
            self.slam.update(scan_mm)
        else:
            self.slam.update(scan_mm, pose_change)

    def get_pose(self):
        x_mm, y_mm, theta_deg = self.slam.getpos()
        self.x_mm = x_mm
        self.y_mm = y_mm
        self.theta_deg = theta_deg

        x_m = x_mm / 1000.0
        y_m = y_mm / 1000.0
        theta_rad = math.radians(theta_deg)
        return x_m, y_m, theta_rad

    def get_map_grid(self):
        self.slam.getmap(self.mapbytes)
        size = self.MAP_SIZE_PIXELS
        grid = np.frombuffer(self.mapbytes, dtype=np.uint8).reshape((size, size))
        return grid


def process_lidar_ranges(raw_ranges, n_beams, n_layers):
    if n_layers > 1:
        middle_layer = n_layers // 2
        ranges = raw_ranges[middle_layer * n_beams : (middle_layer + 1) * n_beams]
    else:
        ranges = raw_ranges[:n_beams]
    return ranges


def get_lidar_safety_info(ranges, obstacle_stop_dist=0.25):
    too_close = False
    min_range = None
    center_range = None
    front_min_range = None

    if ranges and len(ranges) > 0:
        min_range = min(ranges)
        center_index = len(ranges) // 2
        center_range = ranges[center_index]

        # front window: ±20 beams around center
        half_window = int(len(ranges) * 0.20)
        start_idx = max(0, center_index - half_window)
        end_idx = min(len(ranges), center_index + half_window + 1)
        front_window = ranges[start_idx:end_idx]

        if front_window:
            front_min_range = min(front_window)

        if front_min_range is not None and front_min_range < obstacle_stop_dist:
            too_close = True

    return too_close, min_range, center_range, front_min_range


def find_frontiers(grid, free_threshold=230, unknown_min=100, unknown_max=200):
    try:
        import cv2
    except ImportError:
        print("[ERROR] OpenCV (cv2) not found. Install with: pip install opencv-python")
        return np.zeros_like(grid, dtype=np.uint8), []

    # Step A: Isolate free space (high values = free in BreezySLAM)
    free_mask = (grid > free_threshold).astype(np.uint8)

    # Step B: Dilate free space to find boundaries
    kernel = np.ones((3, 3), np.uint8)
    dilated_free = cv2.dilate(free_mask, kernel, iterations=1)

    # Step C: Isolate unknown space (mid-range values)
    unknown_mask = ((grid >= unknown_min) & (grid <= unknown_max)).astype(np.uint8)

    # Step D: Find intersection (frontier)
    frontier_mask = (dilated_free == 1) & (unknown_mask == 1)

    # Extract frontier points as list of coordinates
    frontier_coords = np.argwhere(frontier_mask)  
    frontier_points = [(int(j), int(i)) for i, j in frontier_coords]

    return frontier_mask.astype(np.uint8), frontier_points