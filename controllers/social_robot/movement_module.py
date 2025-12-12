
import math
import numpy as np
import heapq
import cv2


# Configuration constants
OCC_THRESHOLD = 100  # Occupancy threshold for obstacle detection
OBSTACLE_STOP_DIST = 0.3  # meters
GOAL_TOLERANCE = 0.8  # meters
FINAL_STUCK_LIMIT = 120  # steps
STUCK_STEP_LIMIT = 300  # steps


class PoseEstimator:

    def __init__(self, wheel_radius, axle_length):
        self.wheel_radius = wheel_radius
        self.axle_length = axle_length

        self.prev_left = None
        self.prev_right = None

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0  # in radians

    def update_from_encoders(self, left_val, right_val):
        if self.prev_left is None:
            self.prev_left = left_val
            self.prev_right = right_val
            return 0.0, 0.0, 0.0

        d_left = left_val - self.prev_left
        d_right = right_val - self.prev_right

        self.prev_left = left_val
        self.prev_right = right_val

        d_left_m = d_left * self.wheel_radius
        d_right_m = d_right * self.wheel_radius

        d_center = (d_left_m + d_right_m) / 2.0
        d_theta = (d_right_m - d_left_m) / self.axle_length

        theta_mid = self.theta + d_theta / 2.0
        self.x += d_center * math.cos(theta_mid)
        self.y += d_center * math.sin(theta_mid)
        self.theta += d_theta

        self.theta = (self.theta + math.pi) % (2 * math.pi) - math.pi

        return d_center, 0.0, d_theta

    def get_pose(self):
        return self.x, self.y, self.theta

class OccupancyAStarPlanner:

    def __init__(self, occ_grid):
      
        self.grid = occ_grid
        self.h, self.w = occ_grid.shape

    def in_bounds(self, i, j):
        return 0 <= i < self.w and 0 <= j < self.h

    def passable(self, i, j):
      
        if not self.in_bounds(i, j):
            return False
        cell = self.grid[j, i]
        return cell != 1  # Only obstacle (1) is impassable

    def neighbors(self, i, j):
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if self.passable(ni, nj):
                yield (ni, nj)

    def heuristic(self, a, b):
        (i1, j1), (i2, j2) = a, b
        return math.hypot(i2 - i1, j2 - j1)

    def plan(self, start, goal):
        start = tuple(start)
        goal = tuple(goal)

        open_set = []
        heapq.heappush(open_set, (0.0, start))

        came_from = {}
        g_cost = {start: 0.0}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            for nb in self.neighbors(*current):
                cell_value = self.grid[nb[1], nb[0]]
                movement_cost = 1.0

                if cell_value == 2:
                    movement_cost = 8.0  # High cost discourages hugging walls

                tentative_g = g_cost[current] + movement_cost
                if nb not in g_cost or tentative_g < g_cost[nb]:
                    g_cost[nb] = tentative_g
                    f = tentative_g + self.heuristic(nb, goal)
                    heapq.heappush(open_set, (f, nb))
                    came_from[nb] = current

        return []

class SocialAStarPlanner(OccupancyAStarPlanner):
    def __init__(self, occ_grid, human_pos, profile="Neutral"):
        # Initialize the standard grid from the parent class
        super().__init__(occ_grid)
        self.human_pos = human_pos  # (grid_x, grid_y)

        # Phase 4 Preview: Profiles allow you to change behavior easily
        if profile == "Conservative":
            self.sigma = 15.0  # Big bubble (pixels/cells)
            self.amplitude = 20.0 # High cost
        elif profile == "Open":
            self.sigma = 8.0   # Small bubble
            self.amplitude = 10.0
        else: # Neutral
            self.sigma = 10.0
            self.amplitude = 15.0

    def get_social_cost(self, i, j):
        # Calculate distance from this node (i,j) to the human
        hx, hy = self.human_pos
        dist_sq = (i - hx)**2 + (j - hy)**2

        # Gaussian function: Cost is highest at human position, drops off with distance
        # Cost = A * e^(-dist^2 / (2*sigma^2))
        cost = self.amplitude * math.exp(-dist_sq / (2 * self.sigma**2))
        return cost

    def plan(self, start, goal):
        # This is mostly identical to your standard A*, but adds 'social_cost'
        start = tuple(start)
        goal = tuple(goal)

        open_set = []
        heapq.heappush(open_set, (0.0, start))

        came_from = {}
        g_cost = {start: 0.0}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            for nb in self.neighbors(*current):
                # --- THE CHANGE IS HERE ---
                # Standard cost is 1.0. We add the social cost to it.
                step_cost = 1.0 + self.get_social_cost(nb[0], nb[1])

                tentative_g = g_cost[current] + step_cost
                if nb not in g_cost or tentative_g < g_cost[nb]:
                    g_cost[nb] = tentative_g
                    f = tentative_g + self.heuristic(nb, goal)
                    heapq.heappush(open_set, (f, nb))
                    came_from[nb] = current
        return []

class AStarPlanner:

    def __init__(self, grid, occ_thresh=OCC_THRESHOLD):
        self.grid = grid
        self.occ_thresh = occ_thresh
        self.h, self.w = grid.shape

    def in_bounds(self, i, j):
        return 0 <= i < self.w and 0 <= j < self.h

    def passable(self, i, j):
        return not is_occupied(self.grid, i, j, self.occ_thresh)

    def neighbors(self, i, j):
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if self.in_bounds(ni, nj) and self.passable(ni, nj):
                yield (ni, nj)

    def heuristic(self, a, b):
        (i1, j1), (i2, j2) = a, b
        return math.hypot(i2 - i1, j2 - j1)

    def plan(self, start, goal):
        start = tuple(start)
        goal = tuple(goal)

        open_set = []
        heapq.heappush(open_set, (0.0, start))

        came_from = {}
        g_cost = {start: 0.0}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            for nb in self.neighbors(*current):
                cell_value = self.grid[nb[1], nb[0]]
                movement_cost = 1.0

                if cell_value == 2:
                    movement_cost = 8.0  # High cost discourages hugging walls

                tentative_g = g_cost[current] + movement_cost
                if nb not in g_cost or tentative_g < g_cost[nb]:
                    g_cost[nb] = tentative_g
                    f = tentative_g + self.heuristic(nb, goal)
                    heapq.heappush(open_set, (f, nb))
                    came_from[nb] = current

        return []




# Room coordinates (SLAM world coordinates in meters)
# Updated with real SLAM coordinates
ROOM_GOALS = {
    "start_point": (10.01, 10.00),  # Start point in hallway
    "bed1":        (14.52, 4.94),
    "bed2":        (9.12,  4.58),
    "living":      (4.80,  6.75),
    "bathroom":    (15.13, 15.25),
    "kitchen":     (4.35,  15.09)
}


def slam_pose_to_grid(x_m, y_m, slam_backend):
    size = slam_backend.MAP_SIZE_PIXELS
    meters = slam_backend.MAP_SIZE_METERS
    scale = size / float(meters)

    i = int(x_m * scale)
    j = int(y_m * scale)
    return i, j


def grid_to_world(i, j, slam_backend):
    size = slam_backend.MAP_SIZE_PIXELS
    meters = slam_backend.MAP_SIZE_METERS
    scale = size / float(meters)

    x_m = i / scale
    y_m = j / scale
    return x_m, y_m


def is_occupied(grid, i, j, occ_threshold=OCC_THRESHOLD):

    h, w = grid.shape
    if i < 0 or i >= w or j < 0 or j >= h:
        return True

    padding = 12

    # Vectorized bounds check
    i_min = max(0, i - padding)
    i_max = min(w, i + padding + 1)
    j_min = max(0, j - padding)
    j_max = min(h, j + padding + 1)

    # Check entire region at once
    region = grid[j_min:j_max, i_min:i_max]
    return np.any(region < occ_threshold)


def find_nearest_free_cell(grid, i_start, j_start, max_radius=20):
   
    if not is_occupied(grid, i_start, j_start):
        return (i_start, j_start)

    h, w = grid.shape
    for r in range(1, max_radius + 1):
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                i = i_start + di
                j = j_start + dj
                if 0 <= i < w and 0 <= j < h and not is_occupied(grid, i, j):
                    return (i, j)

    return None


def find_forward_free_goal(slam_x, slam_y, slam_theta, slam_backend, grid):
  
    # 1) Try forward ray first
    step = 0.1
    distance = 1.0  
    max_distance = 3.0  

    while distance < max_distance:
        x_goal = slam_x + distance * math.cos(slam_theta)
        y_goal = slam_y + distance * math.sin(slam_theta)

        i, j = slam_pose_to_grid(x_goal, y_goal, slam_backend)

        if not is_occupied(grid, i, j):
            print(f"[PLAN] Forward free cell at dist={distance:.2f}m -> grid=({i},{j})")
            return (i, j)

        distance += step

    print("[PLAN] Forward ray blocked; searching around robot...")

    # 2) Fallback: search a small circle around the robot
    radii = np.arange(0.5, 2.2, 0.2)
    num_angles = 16

    for r in radii:
        for k in range(num_angles):
            angle = slam_theta + 2.0 * math.pi * (k / num_angles)
            x_goal = slam_x + r * math.cos(angle)
            y_goal = slam_y + r * math.sin(angle)

            i, j = slam_pose_to_grid(x_goal, y_goal, slam_backend)

            if not is_occupied(grid, i, j):
                print(f"[PLAN] Fallback free cell at r={r:.2f}m, angle_offset={2*math.pi*(k/num_angles):.2f} -> grid=({i},{j})")
                return (i, j)

    return None


def compute_wheel_velocities(v, w, wheel_radius, axle_length, max_wheel_speed=10.0):

    v_left = (2 * v - w * axle_length) / (2 * wheel_radius)
    v_right = (2 * v + w * axle_length) / (2 * wheel_radius)

    # Clamp to motor limits
    v_left = max(min(v_left, max_wheel_speed), -max_wheel_speed)
    v_right = max(min(v_right, max_wheel_speed), -max_wheel_speed)

    return v_left, v_right


def apply_complementary_filter(odo_theta, imu_yaw, alpha=0.98):
    
    # Normalize IMU yaw to [-pi, pi]
    imu_yaw = (imu_yaw + math.pi) % (2 * math.pi) - math.pi

    # Fuse odometry and IMU
    fused_theta = alpha * odo_theta + (1 - alpha) * imu_yaw

    return fused_theta

def mark_frontier_region_visited(visited_set, center, radius=3):
  
    ci, cj = center
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            visited_set.add((ci + di, cj + dj))

#
# def find_nearest_frontier(grid, start_i, start_j, visited, prefer_distant=False):
#
#     if grid is None:
#         return None
#
#     h, w = grid.shape
#
#     # 0) Clamp robot index to valid range
#     start_i = int(np.clip(start_i, 0, w - 1))
#     start_j = int(np.clip(start_j, 0, h - 1))
#
#     # Ensure uint8 for OpenCV
#     g = grid.astype(np.uint8)
#
#     # 1) Free and non-free masks
#     free_mask = (g == 1)         # True where cell is free
#     nonfree_mask = (g == 0)      # True where cell is obstacle or unknown
#
#     free_u8    = free_mask.astype(np.uint8)    # 0/1
#     nonfree_u8 = nonfree_mask.astype(np.uint8) # 0/1
#
#     # 2) Dilate non-free, then frontiers are free cells adjacent to non-free
#     kernel = np.ones((3, 3), np.uint8)
#     dilated_nonfree = cv2.dilate(nonfree_u8, kernel, iterations=1)
#
#     frontier_mask = free_mask & (dilated_nonfree > 0)
#
#     #  3) Remove visited frontiers directly from the mask
#     for (vi, vj) in visited:
#         if 0 <= vj < h and 0 <= vi < w:
#             frontier_mask[vj, vi] = False
#
#     # 4) Extract frontier pixels
#     frontier_pixels = np.argwhere(frontier_mask)
#     if frontier_pixels.size == 0:
#         return None
#
#     max_frontiers = 5000
#     if len(frontier_pixels) > max_frontiers:
#         idxs = np.linspace(0, len(frontier_pixels) - 1, max_frontiers).astype(int)
#         frontier_pixels = frontier_pixels[idxs]
#
#         # Compute distances
#     dy = frontier_pixels[:, 0] - start_j
#     dx = frontier_pixels[:, 1] - start_i
#     d2 = dx * dx + dy * dy
#
#     if prefer_distant:
#         distances = np.sqrt(d2)
#
#         # Weight: prefer distance 50-200 cells, penalize very close or very far
#         weights = np.ones_like(distances)
#         weights[distances < 20] = 0.1  # Penalize very close
#         weights[distances > 200] = 0.5  # Penalize very far
#
#         # Combine distance with weight
#         scores = distances * weights
#         best_idx = int(np.argmax(scores))  # Maximize weighted distance
#     else:
#         # Original behavior
#         best_idx = int(np.argmin(d2))
#
#     fj, fi = frontier_pixels[best_idx]
#     return (int(fi), int(fj))
def find_nearest_frontier(grid, start_i, start_j, visited, prefer_distant=False):
    if grid is None:
        return None

    h, w = grid.shape

    # 0) Clamp robot index to valid range
    start_i = int(np.clip(start_i, 0, w - 1))
    start_j = int(np.clip(start_j, 0, h - 1))

    # Ensure uint8 for OpenCV
    g = grid.astype(np.uint8)

    # 1) Free and non-free masks
    free_mask = (g == 1)  # True where cell is free
    nonfree_mask = (g == 0)  # True where cell is obstacle or unknown

    free_u8 = free_mask.astype(np.uint8)  # 0/1
    nonfree_u8 = nonfree_mask.astype(np.uint8)  # 0/1

    # 2) Dilate non-free
    kernel = np.ones((3, 3), np.uint8)
    dilated_nonfree = cv2.dilate(nonfree_u8, kernel, iterations=1)

    frontier_mask = free_mask & (dilated_nonfree > 0)

    # 3) Remove visited frontiers
    for (vi, vj) in visited:
        if 0 <= vj < h and 0 <= vi < w:
            frontier_mask[vj, vi] = False

    # 4) Group frontiers into connected components (Clusters)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(frontier_mask.astype(np.uint8),
                                                                            connectivity=8)

    best_target = None
    max_score = -1.0

    # Thresholds
    MIN_DIST_PIXELS = 80  # ~2 meter. Don't target anything closer than this.
    MIN_AREA = 15  # Ignore tiny jagged corners

    # Iterate over clusters (skip label 0, which is background)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        center_x, center_y = centroids[i]

        # A. IGNORE NOISE: Filter out small corners
        if area < MIN_AREA:
            continue

        # B. CALCULATE DISTANCE
        dx = center_x - start_i
        dy = center_y - start_j
        dist = math.hypot(dx, dy)

        # C. SCORING FUNCTION
        # Goal: Aggressively prefer FURTHER targets.

        if dist < MIN_DIST_PIXELS:
            # Penalize nearby targets heavily to prevent "Ping-Pong"
            # We effectively divide the score by 10
            score = area * 0.1
        else:
            # Reward distant targets exponentially
            # Score = Area * (Distance ^ 2)
            # A target 2x further is 4x more valuable.
            score = area * (dist ** 2)

        if score > max_score:
            max_score = score
            best_target = (int(center_x), int(center_y))

    # Fallback: If no good clusters found (map mostly explored)
    if best_target is None:
        frontier_pixels = np.argwhere(frontier_mask)
        if frontier_pixels.size == 0:
            return None
        # Pick random valid pixel
        idx = len(frontier_pixels) // 2
        fj, fi = frontier_pixels[idx]
        return (int(fi), int(fj))

    return best_target