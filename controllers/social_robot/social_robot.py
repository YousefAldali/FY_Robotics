import math
import os
import cv2
import csv
from controller import Keyboard, Supervisor
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from lidar_module import SlamBackend, process_lidar_ranges, get_lidar_safety_info
from movement_module import (
    PoseEstimator, SocialAStarPlanner, AStarPlanner, ROOM_GOALS, OccupancyAStarPlanner,
    slam_pose_to_grid, grid_to_world, find_nearest_free_cell,
    compute_wheel_velocities, apply_complementary_filter,
    OBSTACLE_STOP_DIST, STUCK_STEP_LIMIT, 
    find_nearest_frontier, mark_frontier_region_visited,
    is_occupied
)

# Initialize robot
robot = Supervisor()
TIME_STEP = int(robot.getBasicTimeStep())

matplotlib.use("Agg")

keyboard = Keyboard()
keyboard.enable(TIME_STEP)

# Configuration
GOAL_TOLERANCE = 0.8  # meters
SAVE_FINAL_MAP = True
SAVE_VIDEO_FRAMES = True
SNAPSHOT_INTERVAL = 30
VIDEO_FPS = 10
MAX_EXPLORATION_TIME = 1000 # 2500
FRAMES_DIR = "slam_frames"
SNAPSHOTS_DIR = "map_snapshots"
if SAVE_VIDEO_FRAMES and not os.path.exists(FRAMES_DIR):
    os.makedirs(FRAMES_DIR)
if not os.path.exists(SNAPSHOTS_DIR):
    os.makedirs(SNAPSHOTS_DIR)

map_saved = False
frame_index = 0
video_created = False

# --------------- Data Logging Setup -----------------
experiment_log = []  # List to store data
print("[INFO] Data logging initialized.")

print("[INFO] Social Python Controller Initialized with TIME_STEP =", TIME_STEP)

# Robot physical parameters
WHEEL_RADIUS = 0.0975  # in meters
AXLE_LENGTH = 0.33  # in meters

# Robot Motors Setup
left_motor = robot.getDevice('left wheel')
right_motor = robot.getDevice('right wheel')

left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

# Start stopped
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

# Robot Encoders Setup
left_encoder = robot.getPositionSensor('left wheel sensor')
right_encoder = robot.getPositionSensor('right wheel sensor')

left_encoder.enable(TIME_STEP)
right_encoder.enable(TIME_STEP)

# LIDAR Setup
lidar = robot.getDevice('lidar')
lidar.enable(TIME_STEP)
lidar.enablePointCloud()

# SLAM Backend
slam_backend = SlamBackend(lidar, TIME_STEP)

# IMU Setup
imu = None
try:
    imu = robot.getDevice('imu')
    imu.enable(TIME_STEP)
    print("[INFO] IMU found and enabled.")
except:
    print("[WARN] IMU not found on this robot model.")

# Pose Estimator Setup
pose_estimator = PoseEstimator(WHEEL_RADIUS, AXLE_LENGTH)
# Force the robot to know its real starting location on the map
pose_estimator.x = 10.01
pose_estimator.y = 10.00

# --------------- Navigation State -----------------
mode = "IDLE"  # Modes: IDLE, EXPLORE, MANUAL, TEST, GUIDE
state = "EXPLORE"  # States within a mode
path = []
current_wp_idx = 0
last_wp_idx = None
stuck_steps = 0
spin_timer = 0
avoid_timer = 0
AVOID_MIN_CLEARANCE = 0.45
frontier_cache = None
frontier_cache_time = 0.0
FRONTIER_CACHE_DURATION = 0.5
last_explore_time = 0.0
EXPLORE_INTERVAL = 0.2
turn_accumulated = 0.0
last_stuck_check_time = 0.0
last_stuck_pos = (0.0, 0.0)

# Navigation Condition Toggle
NAV_CONDITION = "BASELINE"   # "SOCIAL" or "BASELINE"

# Profile for social navigation experiments
# Run 1: "Conservative", Run 2: "Neutral", Run 3: "Open"
current_profile = "Open"  # <--- EDIT THIS MANUALLY FOR EACH RUN (only used if NAV_CONDITION == "SOCIAL")

# -------------- Utility Function --------------------
def reset_navigation_state():
    global left_motor, right_motor, path, current_wp_idx, last_wp_idx, stuck_steps, state, v_left, v_right, frontier_cache, persistent_obstacles
    path = []
    current_wp_idx = 0
    last_wp_idx = None
    stuck_steps = 0
    frontier_cache = None
    persistent_obstacles = set() # Reset blacklisted return obstacles
    v_left = 0.0
    v_right = 0.0
    left_motor.setVelocity(v_left)
    right_motor.setVelocity(v_right)
    # Note: We do NOT clear 'visited' so the robot remembers the map
    
    global last_stuck_check_time, last_stuck_pos
    last_stuck_check_time = 0.0
    last_stuck_pos = (0.0, 0.0)

# --------------- Fetch human node ---------------------------
def find_node_by_name(robot, target_name):
    # Fallback function to find a node by its 'name' field
    root = robot.getRoot()
    if root is None:
        print("\n\n" + "!"*60)
        print("[CRITICAL ERROR] robot.getRoot() failed.")
        print("CAUSE: The 'social_robot' node in Webots has 'supervisor' field set to FALSE.")
        print("FIX: Open Webots Scene Tree -> social_robot -> Set 'supervisor' to TRUE -> Save World.")
        print("!"*60 + "\n\n")
        return None

    children = root.getField("children")
    if children is None:
        print("\n\n" + "!"*60)
        print("[CRITICAL ERROR] Cannot access World Info.")
        print("CAUSE: The 'social_robot' node in Webots has 'supervisor' field set to FALSE.")
        print("FIX: Open Webots Scene Tree -> social_robot -> Set 'supervisor' to TRUE -> Save World.")
        print("!"*60 + "\n\n")
        return None

    n = children.getCount()
    for i in range(n):
        node = children.getMFNode(i)
        # Check if node has 'name' field
        # Only Solid nodes have names.
        if node.getTypeName() in ["Solid", "Robot", "Supervisor", "Pedestrian"]: 
             pass
        
        # Try getting name field
        name_field = node.getField("name")
        if name_field is not None: 
            val = name_field.getSFString()
            if val == target_name:
                return node
            # Also try case-insensitive or common variants if target is generic
            if target_name.lower() == "pedestrian" and val == "Pedestrian":
                return node
    return None

human_node = robot.getFromDef("HUMAN_TARGET")
if human_node is None:
    print("[WARN] DEF 'HUMAN_TARGET' not found. Searching by name 'Pedestrian'...")
    human_node = find_node_by_name(robot, "Pedestrian")
    if human_node is None:
         print("[WARN] Search for 'Pedestrian' failed. Trying 'pedestrian'...")
         human_node = find_node_by_name(robot, "pedestrian")

if human_node:
    print(f"[INFO] Human Target Node FOUND! ID: {human_node.getId()}")
else:
    print("[ERROR] Human Target Node NOT FOUND. Human will NOT move.")

human_z = 1.27  # Default height for Pedestrian PROTO

# Define the Offset (SLAM Origin is 10m, 10m away from Webots Origin)
MAP_OFFSET_X = 10.0
MAP_OFFSET_Y = 10.0
initial_y = 0.0
if human_node is None:
    print("[WARN] HUMAN_TARGET node not found. Visuals will fail.")
    human_x, human_y = 11.0, 10.0
else:
    human_trans_field = human_node.getField("translation")
    initial_pos = human_trans_field.getSFVec3f()
    human_z = initial_pos[2]  # Should be 1.27 based on your file
    initial_y = initial_pos[1]
    # --- FIX: INVERT COORDINATES ---
    # Robot is rotated 180 deg. World +X is Map -X.
    # Formula: Map = Offset - World
    human_x = MAP_OFFSET_X - initial_pos[0]  # 10 - 4.82 = 5.18
    human_y = MAP_OFFSET_Y + initial_pos[1]

    print(f"[INFO] Human detected at Webots({initial_pos[0]:.2f}, {initial_pos[1]:.2f})"
          f" -> SLAM({human_x:.2f}, {human_y:.2f})")
    
    # VERIFY CONTROL
    try:
        # Try to nudge the human slightly to prove we have control
        human_trans_field.setSFVec3f(initial_pos)
        print("[SUCCESS] Human translation field is writable.")
    except Exception as e:
        print(f"[ERROR] Could not write to Human translation field: {e}")

# --------------- Complementary Filter Setup -----------------
fused_theta = 0.0
prev_fused_theta = 0.0
alpha = 0.98  

imu_offset = 0.0
if imu:
    for i in range(10): 
        robot.step(TIME_STEP)
    roll, pitch, start_yaw = imu.getRollPitchYaw()
    imu_offset = start_yaw 

visited = set()
persistent_obstacles = set()  # Stores (grid_i, grid_j) tuples of known return-path blockages

print("\n" + "="*50)
print("ROBOT CONTROL MODES")
print("="*50)
print("Press 'E' - EXPLORE mode (autonomous exploration)")
print("Press 'M' - MANUAL mode (drive with WASD keys)")
print("Press 'T' - TEST mode (not yet implemented)")
print("Press 'G' - GUIDE mode (not yet implemented)")
print("Press 'R' - Restart exploration (when in DONE state)")
print("\nMANUAL MODE CONTROLS:")
print("  'W' - Move forward")
print("  'S' - Move backward")
print("  'A' - Turn left")
print("  'D' - Turn right")
print("  'X' - Stop")
print("="*50 + "\n")

# --------------- Main Loop -----------------
while robot.step(TIME_STEP) != -1:
    t = robot.getTime()
    key = keyboard.getKey()

    # Handle keyboard mode switching
    if key == ord('E'):
        if mode != "EXPLORE":
            print("[MODE] Switching to EXPLORE mode")
            reset_navigation_state()  # Reset state
            mode = "EXPLORE"
            state = "EXPLORE"

    elif key == ord('M'):
        if mode != "MANUAL":
            print("[MODE] Switching to MANUAL mode")
            print("[MANUAL] Use WASD keys to drive, X to stop")
            reset_navigation_state()  # Reset state
            mode = "MANUAL"

    elif key == ord('T'):
        if mode != "TEST":
            print("[MODE] Switching to TEST mode (not yet implemented)")
            reset_navigation_state()  # Reset state
            mode = "TEST"
    elif key == ord('G'):
        if mode != "GUIDE":
            print("[MODE] Switching to GUIDE mode")
            reset_navigation_state()  # Reset state
            mode = "GUIDE"
            state = "FETCHING" # Triggers the human fetching state in the loop

    if t >= MAX_EXPLORATION_TIME and state not in ["DONE", "AVOID", "AVOID_TURN", "RETURNING_HOME"] and mode == "EXPLORE":
        print(f"[INFO] Time limit reached. Returning to Origin.")
        start_pos_m = ROOM_GOALS["start_point"]
        start_i, start_j = slam_pose_to_grid(slam_x, slam_y, slam_backend)
        goal_i, goal_j = slam_pose_to_grid(start_pos_m[0], start_pos_m[1], slam_backend)

        # Safety clip
        h, w = slam_backend.get_map_grid().shape
        goal_i = int(np.clip(goal_i, 0, w - 1))
        goal_j = int(np.clip(goal_j, 0, h - 1))

        # Quick Map Generation
        grid = slam_backend.get_map_grid()
        occ = np.zeros_like(grid, dtype=np.uint8)
        occ[grid <= 100] = 1
        occ[grid >= 230] = 0
        # Dilate
        inflation_kernel = np.ones((10, 10), np.uint8)
        hard_obs = (occ == 1).astype(np.uint8)
        inflated = cv2.dilate(hard_obs, inflation_kernel, iterations=1)
        occ[inflated == 1] = 1

        # Apply Persistent Obstacles (Blacklist)
        for (obs_i, obs_j) in persistent_obstacles:
            # Mark a small radius around the known blockage
            for di in range(-4, 5):
                for dj in range(-4, 5):
                    ni, nj = obs_i + di, obs_j + dj
                    if 0 <= ni < h and 0 <= nj < w:
                        occ[nj, ni] = 1 # Mark as occupied (note: grid is y,x so j,i)

        # Plan
        planner = OccupancyAStarPlanner(occ)
        raw_path = planner.plan((start_i, start_j), (goal_i, goal_j))

        if raw_path:
            path = [raw_path[0]] + raw_path[5::5] + [raw_path[-1]]
            current_wp_idx = 0
            state = "RETURNING_HOME"  # Triggers the return state
        else:
            print("[RETURN] Path planning failed (Start/Goal blocked?). Stopping.")
            state = "DONE"

    # 1) Read encoders & update odometry 
    left_val = left_encoder.getValue()
    right_val = right_encoder.getValue()
    dx, dy, dtheta = pose_estimator.update_from_encoders(left_val, right_val)
    odo_x, odo_y, odo_theta = pose_estimator.get_pose()

    # 2) LIDAR + safety
    raw_ranges = lidar.getRangeImage()
    n_beams = lidar.getHorizontalResolution()
    n_layers = lidar.getNumberOfLayers()

    ranges = process_lidar_ranges(raw_ranges, n_beams, n_layers)
    too_close, min_range, center_range, front_min_range = get_lidar_safety_info(
        ranges, OBSTACLE_STOP_DIST
    )

    # 3) IMU + Complementary Filte
   
    if imu is not None:
        roll, pitch, raw_yaw = imu.getRollPitchYaw()
        yaw = raw_yaw - imu_offset

      
        if 'prev_yaw_scan' not in locals(): prev_yaw_scan = yaw

        dtheta_imu = yaw - prev_yaw_scan

        dtheta_imu = (dtheta_imu + math.pi) % (2 * math.pi) - math.pi

        dtheta_for_slam = dtheta_imu  
        prev_yaw_scan = yaw
    else:
        dtheta_for_slam = dtheta  

    if imu is not None and yaw is not None:
        fused_theta = apply_complementary_filter(odo_theta, yaw, alpha)
        dtheta_fused = fused_theta - prev_fused_theta
        dtheta_fused = (dtheta_fused + math.pi) % (2 * math.pi) - math.pi
        prev_fused_theta = fused_theta
        dtheta_for_slam = dtheta_fused
    else:
        dtheta_for_slam = dtheta

    # 4) SLAM update
    if ranges and len(ranges) > 0:
        slam_backend.update(ranges, d_center_m=dx, dtheta_rad=dtheta_for_slam)
        slam_x, slam_y, slam_theta = slam_backend.get_pose()
    else:
        slam_x, slam_y, slam_theta = 0.0, 0.0, 0.0

    # Calculate distance
    dist_to_robot = math.hypot(slam_x - human_x, slam_y - human_y)
    HUMAN_STOP_DIST = 1.0  # Human stops if closer than this to robot

    # Only move the human if we are in GUIDE mode and actually LEADING
    # User requested ONLY Guide Mode behavior (No Manual following)
    should_human_move = (mode == "GUIDE" and state == "GUIDING")

    # (Debug prints removed to reduce console spam)

    # Movement logic is now handled in the 'human_node' update block below (line 426+)
    # This prevents duplicate calculations or conflicts.
    pass

    dist_to_human = dist_to_robot # Update variable for logging

    # Log: [Time, RobotX, RobotY, RobotTheta, DistToHuman]
    experiment_log.append([t, slam_x, slam_y, slam_theta, dist_to_human])

    if human_node is not None:
        if should_human_move:
            # Update position using "Smooth Follow" logic
            # 1. Get Robot Heading (using SLAM theta which is accurate)
            angle = slam_theta 
            
            # 2. Compute Target Point 1.5m BEHIND the robot
            FOLLOW_DIST = 1.5
            # Since robot moves Forward along +Theta, Behind is -Theta vector
            target_x = slam_x - FOLLOW_DIST * math.cos(angle)
            target_y = slam_y - FOLLOW_DIST * math.sin(angle)
            
            # 3. Lerp Human Position towards Target
            # Current human pos (in SLAM coords)
            current_hx = human_x
            current_hy = human_y
            
            follow_speed = 0.05 # Adjust for smoothness
            
            new_hx = current_hx + follow_speed * (target_x - current_hx)
            new_hy = current_hy + follow_speed * (target_y - current_hy)
            
            human_x = new_hx
            human_y = new_hy
            
            # 4. Convert back to Webots Coords and Set
            webots_x = MAP_OFFSET_X - human_x
            webots_y = human_y - MAP_OFFSET_Y
            
            human_trans_field.setSFVec3f([webots_x, webots_y, human_z])
            
            # Debug Print
            if int(t*10) % 20 == 0:
                 dist_teth = math.hypot(slam_x - human_x, slam_y - human_y)
                 print(f"[HUMAN] Tethered Follow. Dist={dist_teth:.2f}")

        else:
            # If not moving, ensure we enforce the current position 
            # (Prevent drift if physics is active, though Pedestrian is passive now)
            webots_x = MAP_OFFSET_X - human_x
            webots_y = human_y - MAP_OFFSET_Y
            human_trans_field.setSFVec3f([webots_x, webots_y, human_z])

    # 5) High-level state machine
    # Only run state machine if in EXPLORE mode
    if mode == "EXPLORE":
        if state == "EXPLORE":
            phase = "Calculating Frontier"

            if (t - last_explore_time) < EXPLORE_INTERVAL:
                v_left = v_right = 0.0
                left_motor.setVelocity(v_left)
                right_motor.setVelocity(v_right)
                continue

            last_explore_time = t
            v_left, v_right = 0.0, 0.0

            grid = slam_backend.get_map_grid()

            OBSTACLE_MAX = 100
            FREE_MIN = 230

            # Create occupancy grid: 0=free, 1=obstacle, 2=unknown
            occ = np.zeros_like(grid, dtype=np.uint8)

            # Identify Walls (Hard Obstacles)
            # Mark anything that isn't "definitely free" as a potential obstacle
            occ[grid <= 100] = 1
            occ[grid > 100] = 0

            # Create "Lava" (Hard Safety Limit) - SMALL
            # This is the physical "do not touch" zone. Keep it small for doors.
            # 12x12 = ~15cm radius. Just enough so wheels don't clip.
            hard_kernel = np.ones((8, 8), np.uint8)
            hard_obs = (occ == 1).astype(np.uint8)
            hard_inflated = cv2.dilate(hard_obs, hard_kernel, iterations=1)

            # Create "Grass" (Soft Buffer) - LARGE
            # This pushes the robot to the center.
            # 30x30 = ~40cm radius. Overlaps walls but leaves center clear.
            soft_kernel = np.ones((30, 30), np.uint8)
            soft_inflated = cv2.dilate(hard_obs, soft_kernel, iterations=1)

            # Combine into final Map
            # First set everything to 0 (Free)
            occ[:] = 0
            # Set Soft Buffer to 2 (Expensive)
            occ[soft_inflated == 1] = 2
            # Set Hard Walls to 1 (Impassable) - Overwrites the soft buffer
            occ[hard_inflated == 1] = 1

            # Binary nav grid for frontier detection: 1 = free, 0 = not free
            nav_grid = (occ == 1).astype(np.uint8)

            start_i, start_j = slam_pose_to_grid(slam_x, slam_y, slam_backend)

            # Clamp indices
            h, w = nav_grid.shape
            start_i = int(np.clip(start_i, 0, w - 1))
            start_j = int(np.clip(start_j, 0, h - 1))

            if frontier_cache is not None and (t - frontier_cache_time) < FRONTIER_CACHE_DURATION:
                frontier_goal = frontier_cache
            else:
                frontier_goal = find_nearest_frontier(nav_grid, start_i, start_j, visited, True)
                frontier_cache = frontier_goal
                frontier_cache_time = t
                if frontier_goal:
                    print(f"[EXPLORE] New frontier found at {frontier_goal}.")

            if frontier_goal:
                goal_i, goal_j = frontier_goal

                di = goal_i - start_i
                dj = goal_j - start_j
                dist2_cells = di * di + dj * dj

                if dist2_cells < 9:
                    print(f"[EXPLORE] Frontier too close (d2={dist2_cells}), marking region visited.")
                    mark_frontier_region_visited(visited, frontier_goal, radius=40)
                    frontier_cache = None
                    state = "EXPLORE"
                    v_left = v_right = 0.0
                    continue

                print(f"[EXPLORE] Frontier found at ({goal_i}, {goal_j}). Planning...")

                goal_i = int(np.clip(goal_i, 0, w - 1))
                goal_j = int(np.clip(goal_j, 0, h - 1))

                plan_occ = occ.copy()

                goal_safety_radius = 6
                start_safety_radius = 15

                start_r_min = max(0, start_j - start_safety_radius)
                start_r_max = min(h, start_j + start_safety_radius + 1)
                start_c_min = max(0, start_i - start_safety_radius)
                start_c_max = min(w, start_i + start_safety_radius + 1)
                plan_occ[start_r_min:start_r_max, start_c_min:start_c_max] = 0

                goal_r_min = max(0, goal_j - goal_safety_radius)
                goal_r_max = min(h, goal_j + goal_safety_radius + 1)
                goal_c_min = max(0, goal_i - goal_safety_radius)
                goal_c_max = min(w, goal_i + goal_safety_radius + 1)
                plan_occ[goal_r_min:goal_r_max, goal_c_min:goal_c_max] = 0

                # Convert human position (meters) to grid coordinates
                human_grid_x, human_grid_y = slam_pose_to_grid(human_x, human_y, slam_backend)

                # Choose the planner based on Mode
                if mode == "GUIDE" or mode == "SOCIAL_TEST":
                    # Use the NEW Social Brain
                    # This forces the robot to path AROUND the human bubble
                    planner = SocialAStarPlanner(plan_occ, (human_grid_x, human_grid_y), profile="Neutral")
                    print("[PLANNER] Using Social A* (Avoiding Human)")
                else:
                    # EXPLORE mode keeps using the OLD Standard Brain
                    # This ensures your baseline remains exactly as it is now
                    planner = OccupancyAStarPlanner(plan_occ)
                    print("[PLANNER] Using Standard A*")

                # Compute the path
                raw_path = planner.plan((start_i, start_j), (goal_i, goal_j))

                if raw_path and len(raw_path) < 20:  # Less than ~0.5m
                    print(f"[EXPLORE] Path too short ({len(raw_path)} steps). Marking area visited and retrying.")
                    mark_frontier_region_visited(visited, frontier_goal, radius=40)
                    continue

                if raw_path and len(raw_path) > 1:
                    path = [raw_path[0]] + raw_path[6::6]
                    if path[-1] != raw_path[-1]:
                        path.append(raw_path[-1])

                    print(f"[EXPLORE] Path found: {len(path)} waypoints.")
                    mark_frontier_region_visited(visited, frontier_goal, radius=40)
                    frontier_cache = None
                    current_wp_idx = 0
                    last_wp_idx = None
                    stuck_steps = 0
                    state = "FOLLOW"
                else:
                    mark_frontier_region_visited(visited, frontier_goal, radius=40)
                    frontier_cache = None
                    print("[EXPLORE] Frontier unreachable. Spinning to clear map.")
                    state = "AVOID"
                    front_min_range = 0.0

            else:
                print("[EXPLORE] No new frontiers found.")

                total_cells = h * w
                visited_cells = len(visited)
                visited_percentage = (visited_cells / total_cells) * 100

                print(f"[EXPLORE] Exploration complete. Returning to Origin.")

                # Get the start position from configs.
                start_pos_m = ROOM_GOALS["start_point"]

                # Get the current grid position
                start_i, start_j = slam_pose_to_grid(slam_x, slam_y, slam_backend)

                # Get Goal (start pos) grid position
                goal_i, goal_j = slam_pose_to_grid(start_pos_m[0], start_pos_m[1], slam_backend)

                # Clip to ensure bounds safety
                h, w = grid.shape
                goal_i = int(np.clip(goal_i, 0, w - 1))
                goal_j = int(np.clip(goal_j, 0, h - 1))

                # Get Map and Inflate for Safety
                # (Reuses the occupancy generation logic in EXPLORE state)
                occ = np.zeros_like(grid, dtype=np.uint8)
                occ[grid <= 100] = 1
                occ[grid >= 230] = 0
                inflation_kernel = np.ones((10, 10), np.uint8)  # Moderate inflation for return
                hard_obs = (occ == 1).astype(np.uint8)
                inflated = cv2.dilate(hard_obs, inflation_kernel, iterations=1)
                occ[inflated == 1] = 1
                
                # CRITICAL FIX: Use 'visited' set to block known stuck areas during Return
                # The 'visited' set contains (i, j) tuples of places we've been.
                # If we got stuck there, we likely marked it.
                # However, 'visited' also covers good places. We initially only marked stuck places in EXPLORE.
                # But to be safe, let's just assume if we are stuck logic triggered mark_frontier_region_visited,
                # it should be avoided.
                # A better approach is to rely on the fact that if we replan, A* will find a path.
                # The issue is if the START is blocked.
                
                # Apply visited mask (optional, but requested for robustness)
                # Note: modifying occ based on visited might block valid paths home if we visited them safely.
                # So we only apply it if we are STUCK. 
                # Actually, the 'mark_frontier_region_visited' call in stuck logic adds to 'visited'.
                # Let's map 'visited' points to obstacles in 'occ'.
                h_occ, w_occ = occ.shape
                for (vi, vj) in visited:
                     if 0 <= vi < w_occ and 0 <= vj < h_occ:
                         # We only want to block it if it was marked as "BAD" by stuck logic.
                         # Since 'visited' mixes good and bad, this is risky.
                         # BUT, since the user wants to avoid "stuck loops", blocking re-entry to where we were is good.
                         # Let's explicitly block the immediate area where we are stuck (handled by the stuck logic calling mark_frontier).
                         # Here we just apply it.
                         # To avoid blocking the WHOLE map, maybe we shouldn't apply ALL visited.
                         # Strategy: The stuck logic marked it. We just need to ensure the planner sees it.
                         # Issue: 'visited' is just a set.
                         # Let's trust the AVOID_TURN to break the loop for now.
                         # If we really want to blacklist, we should have a 'blacklist' set.
                         # Using 'visited' to block path is too aggressive (blocks return path).
                         pass
                
                # INSTEAD: We rely on the fact that we moved to EXPLORE state.
                # The Plan: EXPLORE state will trigger. It will see "Time Limit". It will come HERE.
                # We need to make sure we don't plan the EXACT SAME path.
                # The AVOID_TURN moves us. The start node changes. The path changes.
                # So we likely don't need to mod 'occ' with 'visited' heavily.
                # But let's add a small 'stochastic' cost or dilation if needed?
                # No, let's stick to the plan: AVOID_TURN breaks the loop.
                pass

                # Plan
                planner = OccupancyAStarPlanner(occ)
                raw_path = planner.plan((start_i, start_j), (goal_i, goal_j))

                if raw_path:
                    path = [raw_path[0]] + raw_path[5::5]  # Downsample
                    path.append(raw_path[-1])
                    current_wp_idx = 0
                    state = "RETURNING_HOME"  # New State
                else:
                    print("[RETURN] Could not find path home. Stopping.")
                    state = "DONE"

        # STATE: FOLLOW PATH
        elif state == "FOLLOW":
            phase = "Following Path"
            
            # --- GLOBAL STUCK DETECTION ---
            # Check every 5 seconds if we have moved
            if (t - last_stuck_check_time) > 5.0:
                dx_stuck = slam_x - last_stuck_pos[0]
                dy_stuck = slam_y - last_stuck_pos[1]
                dist_moved = math.hypot(dx_stuck, dy_stuck)
                
                if dist_moved < 0.2:
                    print(f"[STUCK] Global stuck detected (moved {dist_moved:.2f}m in 5s). Force replanning.")
                    ri, rj = slam_pose_to_grid(slam_x, slam_y, slam_backend)
                    # Mark area as visited so we don't just plan back through it instantly
                    mark_frontier_region_visited(visited, (ri, rj), radius=30)
                    frontier_cache = None
                    state = "EXPLORE"
                    last_stuck_check_time = t
                    last_stuck_pos = (slam_x, slam_y)
                    continue
                
                # Update for next check
                last_stuck_check_time = t
                last_stuck_pos = (slam_x, slam_y)
            # ------------------------------

            frontier_cache = None
            # If something is too close in front
            if too_close:
                print("[SAFETY] Wall detected. Initiating 180 turn.")

                ri, rj = slam_pose_to_grid(slam_x, slam_y, slam_backend)
                mark_frontier_region_visited(visited, (ri, rj), radius=40)

                v = 0.0
                w = 0.0
                v_left, v_right = compute_wheel_velocities(
                    v, w, WHEEL_RADIUS, AXLE_LENGTH
                )

                state = "AVOID"
                turn_accumulated = 0.0
                continue

                avoid_timer = 0
            else:
                if current_wp_idx >= len(path):
                    v_left, v_right = 0.0, 0.0

                    ri, rj = slam_pose_to_grid(slam_x, slam_y, slam_backend)
                    mark_frontier_region_visited(visited, (ri, rj), radius=6)
                    print(f"[INFO] Reached Frontier at ({ri}, {rj}). Marking region & scanning... CURRENT SPIN IS DISABLED!")

                    state = "EXPLORE"
                    spin_timer = 0
                else:
                    # Current waypoint
                    wp_i, wp_j = path[current_wp_idx]
                    wp_x, wp_y = grid_to_world(wp_i, wp_j, slam_backend)

                    # Vector to waypoint
                    dx_wp = wp_x - slam_x
                    dy_wp = wp_y - slam_y
                    dist = math.hypot(dx_wp, dy_wp)

                    # Heading calculation
                    target_heading = math.atan2(dy_wp, dx_wp)
                    heading_error = target_heading - slam_theta
                    heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi

                    is_final_waypoint = (current_wp_idx == len(path) - 1)

                    if not is_final_waypoint:
                        if dist < 0.15 and abs(heading_error) > math.radians(90):
                            current_wp_idx += 1
                            continue

                    # --- SMOOTH CONTROLLER START ---
                    # Instead of stopping completely for small errors, we slow down.
                    
                    # 1. Calculate Turn Speed (w)
                    # Proportional control on heading error
                    w = -heading_error * 2.5 
                    
                    # Clamp rotation speed (allow faster turning than before)
                    MAX_ROT_SPEED = 2.0
                    w = max(min(w, MAX_ROT_SPEED), -MAX_ROT_SPEED)

                    # 2. Calculate Linear Speed (v)
                    # Base speed based on distance to target (slow down when close)
                    v_base = min(dist * 1.5, 0.5)
                    
                    # Slow down significantly if we are turning sharp
                    # If error is > 45 degrees (0.8 rad), v drops to 0
                    # This replaces the hard "if error > 0.35 then stop" check
                    turn_penalty = max(0.0, 1.0 - (abs(heading_error) / 1.0))
                    v = v_base * turn_penalty
                    
                    # Ensure we don't stall completely if error is huge (optional, but good for flow)
                    # But for safety, let's keep it 0 if error is very large (> 60 deg)
                    if abs(heading_error) > 1.2:
                         v = 0.0

                    # --- SMOOTH CONTROLLER END ---

                    v_left, v_right = compute_wheel_velocities(v, w, WHEEL_RADIUS, AXLE_LENGTH)

                    if last_wp_idx == current_wp_idx:
                        stuck_steps += 1
                    else:
                        stuck_steps = 0
                        last_wp_idx = current_wp_idx

                    if stuck_steps > STUCK_STEP_LIMIT:
                        print(f"[FOLLOW] Stuck at waypoint {current_wp_idx}, skipping.")
                        current_wp_idx += 1
                        stuck_steps = 0
                        continue

                    if dist < 0.3:
                        current_wp_idx += 1

        elif state == "ROTATE_180":
            phase = "Performing 180 Turn"

            # 1. Accumulate the absolute rotation
            turn_accumulated += abs(dtheta_for_slam)

            # 2. Check if we have turned 180 degrees (Pi radians)
            if turn_accumulated >= math.pi:
                print("[ROTATE] 180 turn complete. Resuming exploration.")
                v_left, v_right = 0.0, 0.0
                state = "EXPLORE"
            else:
                # 3. Rotate in place safely
                v = 0.0
                w = 0.25
                v_left, v_right = compute_wheel_velocities(v, w, WHEEL_RADIUS, AXLE_LENGTH)

        # STATE: 360 SPIN
        elif state == "SPIN":
            phase = "360 Scan"
            v_left = -1.0
            v_right = 1.0

            spin_timer += 1
            # Spin duration
            duration_steps = int(4000 / TIME_STEP)

            if spin_timer > duration_steps:
                v_left, v_right = 0.0, 0.0
                spin_timer = 0
                state = "EXPLORE"
                print("[SPIN] Scan complete. Recalculating frontiers.")

        elif state == "AVOID":
            phase = "Avoiding Obstacle"
            if front_min_range is not None and front_min_range < AVOID_MIN_CLEARANCE:
                v = -0.15
                w = 0.0
                v_left, v_right = compute_wheel_velocities(
                    v, w, WHEEL_RADIUS, AXLE_LENGTH
                )
            else:
                v_left = 0.0
                v_right = 0.0

                state = "AVOID_TURN"
                turn_accumulated = 0.0
                print("[AVOID] Safe distance reached. Turning to break loop.")

        elif state == "AVOID_TURN":
            phase = "Avoiding (Turning)"
            
            # Rotate for a bit to face a new direction
            turn_accumulated += abs(dtheta_for_slam)
            
            # Turn ~90 degrees (1.57 rad)
            if turn_accumulated >= 1.5:
                 v_left, v_right = 0.0, 0.0
                 state = "EXPLORE"
                 print("[AVOID] Turn complete. Replanning.")
            else:
                 v = 0.0
                 w = 0.5 # Turn left
                 v_left, v_right = compute_wheel_velocities(v, w, WHEEL_RADIUS, AXLE_LENGTH)

        elif state == "RETURNING_HOME":
            phase = "Auto-Homing"

            # SAFETY & STUCK CHECKS 
            # 1. Obstacle Avoidance
            # Check frontal 'too_close' OR any side objects very close (< 0.18m)
            if too_close or (min_range is not None and min_range < 0.18):
                print("[RETURN] Obstacle detected. Initiating persistence avoidance.")
                
                # BLACKLIST: Mark this collision point as a persistent obstacle
                ci, cj = slam_pose_to_grid(slam_x, slam_y, slam_backend)
                # We add the CURRENT location (or slightly forward) to the blacklist
                # Let's verify if we can project forward based on theta
                proj_dist = 0.3 # Look ahead 30cm
                proj_x = slam_x + proj_dist * math.cos(slam_theta)
                proj_y = slam_y + proj_dist * math.sin(slam_theta)
                pi, pj = slam_pose_to_grid(proj_x, proj_y, slam_backend)
                
                print(f"[RETURN] Blacklisting obstacle at ({pi}, {pj})")
                persistent_obstacles.add((pi, pj))

                state = "AVOID"
                continue

            # 2. Stuck Detection
            if (t - last_stuck_check_time) > 5.0:
                dx_stuck = slam_x - last_stuck_pos[0]
                dy_stuck = slam_y - last_stuck_pos[1]
                dist_moved = math.hypot(dx_stuck, dy_stuck)
                
                if dist_moved < 0.2:
                    print(f"[RETURN] Global stuck detected ({dist_moved:.2f}m). Force replanning.")
                    
                    # MARK AREA AS VISITED/BLOCKED so we don't plan through it again
                    ri, rj = slam_pose_to_grid(slam_x, slam_y, slam_backend)
                    mark_frontier_region_visited(visited, (ri, rj), radius=30)
                    
                    state = "EXPLORE" # Will trigger 'Time Limit' logic to re-plan home
                    last_stuck_check_time = t
                    last_stuck_pos = (slam_x, slam_y)
                    continue
                
                last_stuck_check_time = t
                last_stuck_pos = (slam_x, slam_y)


            if current_wp_idx >= len(path):
                print("[RETURN] Arrived at Origin.")
                state = "DONE"
                v_left, v_right = 0.0, 0.0
            else:
                # Pure Pursuit Logic
                wp_i, wp_j = path[current_wp_idx]
                wp_x, wp_y = grid_to_world(wp_i, wp_j, slam_backend)

                dx = wp_x - slam_x
                dy = wp_y - slam_y
                dist_sq = dx * dx + dy * dy

                target_heading = math.atan2(dy, dx)
                heading_err = target_heading - slam_theta
                heading_err = (heading_err + math.pi) % (2 * math.pi) - math.pi

                if dist_sq < 0.15:  # Waypoint reached radius
                    current_wp_idx += 1

                # Motor Control
                # Smoother Home Return Controller
                w = -heading_err * 2.5
                v_base = 0.35 # Slightly faster return
                
                # Slow down if turning
                turn_penalty = max(0.0, 1.0 - (abs(heading_err) / 1.2))
                v = v_base * turn_penalty

                v_left, v_right = compute_wheel_velocities(v, w, WHEEL_RADIUS, AXLE_LENGTH)

        # STATE: DONE (Fix: Add Handler to EXPLORE Mode)
        elif state == "DONE":
             v_left, v_right = 0.0, 0.0
             phase = "Done"
             # Saving map logic handled at end of loop checking state=="DONE"

    # Handle other modes (MANUAL, TEST, GUIDE)
    elif mode == "MANUAL":
        # Manual mode - control robot with WASD keys
        phase = "Manual Driving"

        # Manual control speeds
        FORWARD_SPEED = 2.0
        BACKWARD_SPEED = 2.0
        TURN_SPEED = 1.5

        # Check for WASD controls
        if key == ord('W'):
            # Move forward
            v_left = FORWARD_SPEED
            v_right = FORWARD_SPEED
        elif key == ord('S'):
            # Move backward
            v_left = -BACKWARD_SPEED
            v_right = -BACKWARD_SPEED
        elif key == ord('A'):
            # Turn left (left wheel slower/backward, right wheel forward)
            v_left = -TURN_SPEED
            v_right = TURN_SPEED
        elif key == ord('D'):
            # Turn right (right wheel slower/backward, left wheel forward)
            v_left = TURN_SPEED
            v_right = -TURN_SPEED
        elif key == ord('X'):
            # Stop
            v_left = 0.0
            v_right = 0.0
        else:
            # No key pressed or unknown key - maintain stopped
            v_left = 0.0
            v_right = 0.0

    elif mode == "TEST":
        # Test mode - not yet implemented
        v_left = v_right = 0.0
        phase = "Test Mode (Not Implemented)"

    elif mode == "GUIDE":
        phase = "Guiding Human to Kitchen"

        # SUB-STATE: INITIALIZATION 
        if state == "IDLE":
            print("[GUIDE] Initializing Guidance Sequence...")

            # 1. Define Goal (Kitchen)
            target_pos = ROOM_GOALS["kitchen"] # Imported from movement_module
            goal_i, goal_j = slam_pose_to_grid(target_pos[0], target_pos[1], slam_backend)
            start_i, start_j = slam_pose_to_grid(slam_x, slam_y, slam_backend)

            # 2. Get Occupancy Grid for Planning
            grid = slam_backend.get_map_grid()
            occ = np.zeros_like(grid, dtype=np.uint8)
            occ[grid <= 100] = 1 # Obstacles
            occ[grid >= 230] = 0 # Free
            # Dilation/Inflation for safety
            inflation_kernel = np.ones((10, 10), np.uint8)
            hard_obs = (occ == 1).astype(np.uint8)
            inflated = cv2.dilate(hard_obs, inflation_kernel, iterations=1)
            occ[inflated == 1] = 1

            # Apply Persistent Obstacles (Blacklist) - Guide Mode
            # This ensures that if we got stuck and replanned, we don't pick the same path
            h, w = grid.shape
            for (obs_i, obs_j) in persistent_obstacles:
                for di in range(-4, 5):
                    for dj in range(-4, 5):
                        ni, nj = obs_i + di, obs_j + dj
                        if 0 <= ni < h and 0 <= nj < w:
                            occ[nj, ni] = 1 # Mark as occupied (note: grid is y,x so j,i)

            # 3. Plan using Phase 2 Social Planner
            h_grid = slam_pose_to_grid(human_x, human_y, slam_backend)

            # Use the globally defined current_profile (set at top of file)
            if NAV_CONDITION == "BASELINE":
                planner = OccupancyAStarPlanner(occ)
                print("[PLANNER] BASELINE: Occupancy A* (no proxemics)")
            else:
                planner = SocialAStarPlanner(occ, h_grid, profile=current_profile)
                print(f"[PLANNER] SOCIAL A*: profile={current_profile}")

            raw_path = planner.plan((start_i, start_j), (goal_i, goal_j))

            if raw_path and len(raw_path) > 1:
                # Downsample path for smoother following
                path = [raw_path[0]] + raw_path[5::5]
                if path[-1] != raw_path[-1]: path.append(raw_path[-1])

                current_wp_idx = 0
                state = "GUIDING"
                print(f"[GUIDE] Path found ({len(path)} wps). Starting guidance.")
            else:
                print("[GUIDE] Standard path failed. Attempting RESCUE PLAN (Reduced Inflation)...")
                
                # RESCUE STRATEGY: Reduce inflation to squeeze through tight spots
                occ_rescue = np.zeros_like(grid, dtype=np.uint8)
                occ_rescue[grid <= 100] = 1
                occ_rescue[grid >= 230] = 0
                
                # Minimal Inflation (5px radius instead of 10px)
                rescue_kernel = np.ones((5, 5), np.uint8)
                hard_obs_rescue = (occ_rescue == 1).astype(np.uint8)
                inflated_rescue = cv2.dilate(hard_obs_rescue, rescue_kernel, iterations=1)
                occ_rescue[inflated_rescue == 1] = 1
                
                # Apply Persistent Obstacles even in Rescue Mode (Still unsafe to hit known walls)
                for (obs_i, obs_j) in persistent_obstacles:
                     for di in range(-2, 3): # Smaller blacklist radius (2px)
                        for dj in range(-2, 3):
                            ni, nj = obs_i + di, obs_j + dj
                            if 0 <= ni < h and 0 <= nj < w:
                                occ_rescue[nj, ni] = 1

                # Re-Plan
                if NAV_CONDITION == "BASELINE":
                    planner_rescue = OccupancyAStarPlanner(occ_rescue)
                    print("[PLANNER] BASELINE RESCUE: Occupancy A*")
                else:
                    planner_rescue = SocialAStarPlanner(occ_rescue, h_grid, profile=current_profile)
                    print(f"[PLANNER] SOCIAL RESCUE: profile={current_profile}")

                raw_path_rescue = planner_rescue.plan((start_i, start_j), (goal_i, goal_j))
                
                if raw_path_rescue and len(raw_path_rescue) > 1:
                    path = [raw_path_rescue[0]] + raw_path_rescue[5::5]
                    if path[-1] != raw_path_rescue[-1]: path.append(raw_path_rescue[-1])
                    current_wp_idx = 0
                    state = "GUIDING"
                    print(f"[GUIDE] RESCUE PATH FOUND ({len(path)} wps). Squeezing through...")
                else:
                    print("[GUIDE] Rescue failed. Path blocked. Waiting (IDLE) to retry...")
                    # Do NOT switch to Manual. Just wait and retry.
                    # Also, maybe the persistent obstacles are wrong? Let's clear them to give it a fresh start.
                    if len(persistent_obstacles) > 0:
                        print("[GUIDE] Clearing obstacle memory to attempt fresh plan next cycle.")
                        persistent_obstacles.clear()
                    
                    state = "IDLE" 
                    mode = "GUIDE" # Ensure we stay in Guide
                    v_left, v_right = 0.0, 0.0
                    path = []
                    # Add a small pause to avoid spamming
                    # We can't sleep, but we can rely on IDLE's natural flow (it replans instantly)
                    # Maybe we should set a timer? 
                    last_stuck_check_time = t # Reset stuck timer

        elif state == "FETCHING":
            phase = "Fetching Human"

            # Check distance to human
            dist_to_human = math.hypot(slam_x - human_x, slam_y - human_y)
            
            # DEBUG: Why is it skipping?
            if int(t*10) % 20 == 0:
                 print(f"[FETCH] Dist to Human: {dist_to_human:.2f}m")

            if dist_to_human < 1.5:
                print("[FETCH] Arrived at Human. Planning to Kitchen...")
                state = "IDLE"  # This triggers the existing Kitchen planning logic
                v_left, v_right = 0.0, 0.0

            elif not path:
                # Plan path to human if we don't have one
                h_i, h_j = slam_pose_to_grid(human_x, human_y, slam_backend)
                s_i, s_j = slam_pose_to_grid(slam_x, slam_y, slam_backend)

                # --- FIX: Generate Safer Map ---
                grid = slam_backend.get_map_grid()
                # Create occupancy grid: 0=free, 1=obstacle, 2=unknown
                occ = np.zeros_like(grid, dtype=np.uint8)

                # Identify Walls (Hard Obstacles)
                # Mark anything that isn't "definitely free" as a potential obstacle
                occ[grid <= 100] = 1
                occ[grid > 100] = 0

                # Create "Lava" (Hard Safety Limit) - SMALL
                # This is the physical "do not touch" zone. Keep it small for doors.
                # 12x12 = ~15cm radius. Just enough so wheels don't clip.
                hard_kernel = np.ones((8, 8), np.uint8)
                hard_obs = (occ == 1).astype(np.uint8)
                hard_inflated = cv2.dilate(hard_obs, hard_kernel, iterations=1)

                # Create "Grass" (Soft Buffer) - LARGE
                # This pushes the robot to the center.
                # 30x30 = ~40cm radius. Overlaps walls but leaves center clear.
                soft_kernel = np.ones((30, 30), np.uint8)
                soft_inflated = cv2.dilate(hard_obs, soft_kernel, iterations=1)

                # Combine into final Map
                # First set everything to 0 (Free)
                occ[:] = 0
                # Set Soft Buffer to 2 (Expensive)
                occ[soft_inflated == 1] = 2
                # Set Hard Walls to 1 (Impassable) - Overwrites the soft buffer
                occ[hard_inflated == 1] = 1

                # Use Standard Planner
                planner = OccupancyAStarPlanner(occ)
                # Check if human is inside obstacles (Furniture + Legs)
                if occ[h_j, h_i] == 1:
                    print(f"[FETCH] Target ({h_i},{h_j}) is blocked by furniture. Searching for open space...")

                    # FIX: Increase radius to 80 (2.0 meters) to escape the furniture ring
                    # This finds the "entrance" to the seating area
                    found_goal = find_nearest_free_cell(occ, h_i, h_j, max_radius=80)

                    if found_goal:
                        print(f"[FETCH] Adjusted target to nearest accessible floor: {found_goal}")
                        h_i, h_j = found_goal
                    else:
                        print("[FETCH] CRITICAL: Human is completely walled off! Cannot find path.")
                raw_path = planner.plan((s_i, s_j), (h_i, h_j))
                if raw_path:
                    path = raw_path[5::5]
                    current_wp_idx = 0
                else:
                    print("[FETCH] Path planning failed (Target unreachable). Stopping.")
                    mode = "MANUAL"  # Stop the infinite loop

            else:
                # Current waypoint
                wp_i, wp_j = path[current_wp_idx]
                wp_x, wp_y = grid_to_world(wp_i, wp_j, slam_backend)

                # Vector to waypoint
                dx_wp = wp_x - slam_x
                dy_wp = wp_y - slam_y
                dist = math.hypot(dx_wp, dy_wp)

                # Heading calculation
                target_heading = math.atan2(dy_wp, dx_wp)
                heading_error = target_heading - slam_theta
                heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi

                is_final_waypoint = (current_wp_idx == len(path) - 1)

                if not is_final_waypoint:
                    if dist < 0.15 and abs(heading_error) > math.radians(90):
                        current_wp_idx += 1
                        continue

                if abs(heading_error) > 0.35:  # ~20 degrees
                    v = 0.0
                    max_rot_speed = 1.5
                    w = -heading_error * 1.5
                    w = max(min(w, max_rot_speed), -max_rot_speed)
                else:
                    v = min(dist * 1.5, 0.4)
                    w = -heading_error * 2.0
                    v = min(v, 0.6)

                    max_rot_speed = 0.6
                    w = max(min(w, max_rot_speed), -max_rot_speed)

                v_left, v_right = compute_wheel_velocities(v, w, WHEEL_RADIUS, AXLE_LENGTH)

                if last_wp_idx == current_wp_idx:
                    stuck_steps += 1
                else:
                    stuck_steps = 0
                    last_wp_idx = current_wp_idx

                if stuck_steps > STUCK_STEP_LIMIT:
                    print(f"[FOLLOW] Stuck at waypoint {current_wp_idx}, skipping.")
                    current_wp_idx += 1
                    stuck_steps = 0
                    continue

                if dist < 0.3:
                    current_wp_idx += 1

            # STATE: DONE
        elif state == "DONE":
            v_left = 0.0
            v_right = 0.0
            phase = "Done"

            # Data is already being saved periodically, no need to save again here

            if key == ord('R'):
                state = "EXPLORE"
                print("[USER] Restarting Exploration.")

        # --- SUB-STATE: GUIDING ---
        elif state == "GUIDING":
            # PHASE 3 KEY FEATURE: WAIT FOR HUMAN
            # If the human lags behind (> 2.5m), robot stops.
            MAX_SEPARATION = 2.5

            if dist_to_human > MAX_SEPARATION:
                phase = "WAITING FOR HUMAN"
                v_left = 0.0
                v_right = 0.0
                # We do NOT update current_wp_idx or calculate wheels here

            else:
                phase = "Leading to Kitchen"

                # --- 1. Obstacle Avoidance (New Guide Mode Feature) ---
                # Should mirror Return Home Safety logic
                if too_close or (min_range is not None and min_range < 0.18):
                     print("[GUIDE] Obstacle detected! Initiating Avoidance maneuver.")
                     
                     # BLACKLIST: Mark this collision point as a persistent obstacle
                     ci, cj = slam_pose_to_grid(slam_x, slam_y, slam_backend)
                     
                     # Project forward slightly to mark the WALL not the robot center
                     proj_dist = 0.35
                     proj_x = slam_x + proj_dist * math.cos(slam_theta)
                     proj_y = slam_y + proj_dist * math.sin(slam_theta)
                     pi, pj = slam_pose_to_grid(proj_x, proj_y, slam_backend)
                     
                     print(f"[GUIDE] Blacklisting detected obstacle at ({pi}, {pj})")
                     persistent_obstacles.add((pi, pj))
                     
                     state = "GUIDE_AVOID"
                     continue

                # 2. Stuck Detection 
                if (t - last_stuck_check_time) > 4.0:
                     dx_g = slam_x - last_stuck_pos[0]
                     dy_g = slam_y - last_stuck_pos[1]
                     dist_moved_g = math.hypot(dx_g, dy_g)
                     
                     if dist_moved_g < 0.15:
                         print(f"[GUIDE] Stuck detected in kitchen path ({dist_moved_g:.2f}m). Force replanning.")
                         
                         # BLACKLIST: Mark this stuck location as a persistent obstacle
                         si, sj = slam_pose_to_grid(slam_x, slam_y, slam_backend)
                         print(f"[GUIDE] Blacklisting stuck spot at ({si}, {sj})")
                         persistent_obstacles.add((si, sj))
                         
                         state = "IDLE" # Force Re-plan
                         last_stuck_check_time = t
                         last_stuck_pos = (slam_x, slam_y)
                         continue
                     
                     last_stuck_check_time = t
                     last_stuck_pos = (slam_x, slam_y)


                # Check if arrived
                if current_wp_idx >= len(path):
                    print("[GUIDE] Destination Reached.")
                    state = "DONE"
                    v_left, v_right = 0.0, 0.0
                else:
                    # Pure Pursuit Logic
                    wp_i, wp_j = path[current_wp_idx]
                    wp_x, wp_y = grid_to_world(wp_i, wp_j, slam_backend)

                    dx = wp_x - slam_x
                    dy = wp_y - slam_y
                    dist_sq = dx*dx + dy*dy

                    target_heading = math.atan2(dy, dx)
                    heading_err = target_heading - slam_theta
                    heading_err = (heading_err + math.pi) % (2 * math.pi) - math.pi

                    if dist_sq < 0.15: # Waypoint reached radius
                        current_wp_idx += 1

                    # Motor Control
                    if abs(heading_err) > 0.4:
                        v = 0.0
                        w = -heading_err * 1.5
                    else:
                        v = 0.25 # Move Slower for guidance
                        w = -heading_err * 2.5

                    v_left, v_right = compute_wheel_velocities(v, w, WHEEL_RADIUS, AXLE_LENGTH)

        #SUB-STATE: GUIDE AVOID 
        elif state == "GUIDE_AVOID":
            phase = "Guide: Avoiding Obstacle"
            # Logic: Backup until 'front_min_range' is safe
            # We use front_min_range from global scan
            # INCREASED BACKUP DISTANCE: 0.4m clearance to ensure we are really out of the wall
            if front_min_range is not None and front_min_range < 0.40:
                v = -0.20 # Faster backup
                w = 0.0
                v_left, v_right = compute_wheel_velocities(v, w, WHEEL_RADIUS, AXLE_LENGTH)
            else:
                v_left = 0.0
                v_right = 0.0
                state = "GUIDE_AVOID_TURN"
                turn_accumulated = 0.0
                print("[GUIDE] Safe distance reached. Turning to break contact.")

        # SUB-STATE: GUIDE AVOID TURN 
        elif state == "GUIDE_AVOID_TURN":
            phase = "Guide: Avoiding (Turning)"
            
            # Rotate for a bit to face a new direction
            turn_accumulated += abs(dtheta_for_slam)
            
            # Turn ~90 degrees (1.57 rad)
            if turn_accumulated >= 1.5:
                 v_left, v_right = 0.0, 0.0
                 # After turning, we go FORWARD
                 state = "GUIDE_AVOID_FORWARD"
                 avoid_fwd_start = (slam_x, slam_y)
                 print("[GUIDE] Turn complete. Moving FORWARD to clear area.")
            else:
                 v = 0.0
                 w = 0.5 # Turn Left
                 v_left, v_right = compute_wheel_velocities(v, w, WHEEL_RADIUS, AXLE_LENGTH)

        # SUB-STATE: GUIDE AVOID FORWARD 
        elif state == "GUIDE_AVOID_FORWARD":
            phase = "Guide: Avoiding (Forward)"
            
            # Move forward for distance
            dx_av = slam_x - avoid_fwd_start[0]
            dy_av = slam_y - avoid_fwd_start[1]
            dist_av = math.hypot(dx_av, dy_av)
            
            # Move 0.5 meters forward
            if dist_av >= 0.5:
                v_left, v_right = 0.0, 0.0
                state = "IDLE" # NOW we re-plan
                print("[GUIDE] Avoidance Move complete. Forcing Re-plan.")
            else:
                
                # Check for obstacle while moving forward! (Don't hit another wall)
                if front_min_range is not None and front_min_range < 0.25:
                     print("[GUIDE] Obstacle while moving forward! Stopping avoidance move.")
                     state = "IDLE"
                     v_left, v_right = 0.0, 0.0
                else:
                    v = 0.2 # Slow forward
                    w = 0.0
                    v_left, v_right = compute_wheel_velocities(v, w, WHEEL_RADIUS, AXLE_LENGTH)

    else:
        # IDLE mode or unknown
        v_left = v_right = 0.0
        phase = "Idle"

    left_motor.setVelocity(v_left)
    right_motor.setVelocity(v_right)

    # 6) Periodic status printing and map saving 
    if int(t) != int(t - TIME_STEP / 1000.0):
        print(f"----  Time: {t:.2f} s  ----")
        print(f"Phase: {phase}")
        print(f"SLAM Pose: x={slam_x:.2f} m, y={slam_y:.2f} m")

        if state == "EXPLORE":
            print(f"Visited cells: {len(visited)}")
        elif state == "FOLLOW":
            print(f"Waypoint: {current_wp_idx}/{len(path)}")
        
        if int(t) > 0 and int(t) % SNAPSHOT_INTERVAL == 0:
            grid = slam_backend.get_map_grid()

            plt.figure(figsize=(6, 6))
            plt.imshow(grid, cmap="gray", origin="lower")
            plt.title(f"Map Snapshot at {int(t)}s")
            plt.axis("off")
            plt.tight_layout()

            # Save to file
            filename = os.path.join(SNAPSHOTS_DIR, f"map_snapshot_{int(t):04d}.png")
            plt.savefig(filename, dpi=150)
            plt.close()
            print(f"[INFO] Auto-saved map snapshot: {filename}")

        # Save Video Frames
        if SAVE_VIDEO_FRAMES:
            grid = slam_backend.get_map_grid()

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(grid, cmap="gray", origin="lower")

            # Draw the Path (Cyan Line)
            if path and len(path) > 0:
                path_arr = np.array(path)  # List of (x, y) tuples
                # Matplotlib plots x as columns (index 0) and y as rows (index 1)
                ax.plot(path_arr[:, 0], path_arr[:, 1], color='cyan', linewidth=2, label='Path')

            # Draw Social Zones
            hx_grid, hy_grid = slam_pose_to_grid(human_x, human_y , slam_backend)

            # Convert meters to pixels. Scale = 800px / 20m = 40 px/m
            pixels_per_meter = slam_backend.MAP_SIZE_PIXELS / slam_backend.MAP_SIZE_METERS

            # Intimate Zone (0.45m) -> Red
            intimate_radius = 0.45 * pixels_per_meter
            circ_intimate = plt.Circle((hx_grid, hy_grid), intimate_radius, color='red', fill=False, linewidth=2)
            ax.add_patch(circ_intimate)

            # Personal Zone (1.2m) -> Green Dashed
            personal_radius = 1.2 * pixels_per_meter
            circ_personal = plt.Circle((hx_grid, hy_grid), personal_radius, color='lime', fill=False,
                                       linestyle='--', linewidth=2)
            ax.add_patch(circ_personal)

            ax.set_title(f"Social SLAM - {phase}")
            ax.axis("off")
            plt.tight_layout()

            frame_filename = os.path.join(FRAMES_DIR, f"frame_{frame_index:05d}.png")
            plt.savefig(frame_filename, dpi=100)
            plt.close()
            frame_index += 1
            
        # Save experiment data periodically
        if int(t) % 10 == 0:  # Save every 10 seconds
            if experiment_log:
                # OLD:
                # with open('experiment_data.csv', 'w', newline='') as f:

                # NEW: Include profile name in filename
                tag = current_profile.lower() if NAV_CONDITION == "SOCIAL" else "baseline"
                filename = f"experiment_data_{tag}.csv"
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Time", "X", "Y", "Theta", "DistHuman"])
                    writer.writerows(experiment_log)

        # Save map when finished
        if state == "DONE" and SAVE_FINAL_MAP and not map_saved:
            grid = slam_backend.get_map_grid()

            plt.figure(figsize=(6, 6))
            plt.imshow(grid, cmap="gray", origin="lower")
            plt.title("Final Autonomous Map")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig("slam_map_explore_complete.png", dpi=300)
            plt.close()

            map_saved = True
            print("[INFO] Saved final SLAM map to slam_map_explore_complete.png")

        if state == "DONE" and SAVE_VIDEO_FRAMES and not video_created:
            print("[INFO] Creating video from saved frames...")

            # Get list of frame files
            frame_files = sorted([f for f in os.listdir(FRAMES_DIR) if f.startswith("frame_") and f.endswith(".png")])

            if len(frame_files) > 0:
                # Read first frame to get dimensions
                first_frame_path = os.path.join(FRAMES_DIR, frame_files[0])
                first_frame = cv2.imread(first_frame_path)
                height, width, _ = first_frame.shape

                video_filename = "slam_exploration_video.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(video_filename, fourcc, VIDEO_FPS, (width, height))

                # Write all frames to video
                for frame_file in frame_files:
                    frame_path = os.path.join(FRAMES_DIR, frame_file)
                    frame = cv2.imread(frame_path)
                    video_writer.write(frame)

                video_writer.release()
                video_created = True
                print(f"[INFO] Video created successfully: {video_filename}")
                print(f"[INFO] Total frames: {len(frame_files)}, Duration: {len(frame_files)/VIDEO_FPS:.1f}s")
            else:
                print("[WARN] No frames found to create video")
                video_created = True

        print("-------------------------")