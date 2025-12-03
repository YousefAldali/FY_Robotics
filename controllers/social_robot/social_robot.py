import math
import os
import cv2
from controller import Robot, Keyboard
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from lidar_module import SlamBackend, process_lidar_ranges, get_lidar_safety_info
from movement_module import (
    PoseEstimator, AStarPlanner, ROOM_GOALS, OccupancyAStarPlanner,
    slam_pose_to_grid, grid_to_world, find_nearest_free_cell,
    compute_wheel_velocities, apply_complementary_filter,
    OBSTACLE_STOP_DIST, STUCK_STEP_LIMIT, 
    find_nearest_frontier, mark_frontier_region_visited
)

# Initialize robot
robot = Robot()
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
MAX_EXPLORATION_TIME = 2500
FRAMES_DIR = "slam_frames"
if SAVE_VIDEO_FRAMES and not os.path.exists(FRAMES_DIR):
    os.makedirs(FRAMES_DIR)

map_saved = False
frame_index = 0
video_created = False

print("[INFO] Social Python Controller Initialized with TIME_STEP =", TIME_STEP)

# Robot physical parameters
WHEEL_RADIUS = 0.0975  # in meters
AXLE_LENGTH = 0.33  # in meters

# --------------- Robot Motors Setup -----------------
left_motor = robot.getDevice('left wheel')
right_motor = robot.getDevice('right wheel')

left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

# Start stopped
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

# --------------- Robot Encoders Setup -----------------
left_encoder = robot.getPositionSensor('left wheel sensor')
right_encoder = robot.getPositionSensor('right wheel sensor')

left_encoder.enable(TIME_STEP)
right_encoder.enable(TIME_STEP)

# --------------- LIDAR Setup -----------------
lidar = robot.getDevice('lidar')
lidar.enable(TIME_STEP)
lidar.enablePointCloud()

# --------------- SLAM Backend -----------------
slam_backend = SlamBackend(lidar, TIME_STEP)

# --------------- IMU Setup -----------------
imu = None
try:
    imu = robot.getDevice('imu')
    imu.enable(TIME_STEP)
    print("[INFO] IMU found and enabled.")
except:
    print("[WARN] IMU not found on this robot model.")

# --------------- Pose Estimator Setup -----------------
pose_estimator = PoseEstimator(WHEEL_RADIUS, AXLE_LENGTH)

# --------------- Navigation State -----------------
state = "EXPLORE"  # Starts in Autonomous Explore Mode
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

# --------------- Complementary Filter Setup -----------------
fused_theta = 0.0
prev_fused_theta = 0.0
alpha = 0.98  

# Test
imu_offset = 0.0
if imu:
    for i in range(10): 
        robot.step(TIME_STEP)
    roll, pitch, start_yaw = imu.getRollPitchYaw()
    imu_offset = start_yaw 

visited = set()

# --------------- Main Loop -----------------
while robot.step(TIME_STEP) != -1:
    t = robot.getTime()
    key = keyboard.getKey()

    if t >= MAX_EXPLORATION_TIME and state != "DONE":
        print(f"[INFO] Maximum exploration time ({MAX_EXPLORATION_TIME}s) reached. Stopping exploration.")
        state = "DONE"
        v_left = v_right = 0.0
        left_motor.setVelocity(v_left)
        right_motor.setVelocity(v_right)

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

    # 5) High-level state machine 

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
        occ[grid <= OBSTACLE_MAX] = 1  # Hard obstacles
        occ[(grid > OBSTACLE_MAX) & (grid < FREE_MIN)] = 2  # Unknown
        occ[grid >= FREE_MIN] = 0  # Free space
        inflation_kernel = np.ones((32, 32), np.uint8)
        hard_obs = (occ == 1).astype(np.uint8)
        inflated = cv2.dilate(hard_obs, inflation_kernel, iterations=1)
        occ[inflated == 1] = 1

        # Binary nav grid for frontier detection: 1 = free, 0 = not free
        nav_grid = (occ == 0).astype(np.uint8)

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
                mark_frontier_region_visited(visited, frontier_goal, radius=5)
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

            planner = OccupancyAStarPlanner(plan_occ)
            raw_path = planner.plan((start_i, start_j), (goal_i, goal_j))

            if raw_path and len(raw_path) > 1:
                path = [raw_path[0]] + raw_path[6::6]
                if path[-1] != raw_path[-1]:
                    path.append(raw_path[-1])

                print(f"[EXPLORE] Path found: {len(path)} waypoints.")
                mark_frontier_region_visited(visited, frontier_goal, radius=3)
                frontier_cache = None  
                current_wp_idx = 0
                last_wp_idx = None
                stuck_steps = 0
                state = "FOLLOW"
            else:
                mark_frontier_region_visited(visited, frontier_goal, radius=6)
                frontier_cache = None  
                print("[EXPLORE] Frontier unreachable. Spinning to clear map.")
                state = "AVOID"
                front_min_range = 0.0  

        else:
            print("[EXPLORE] No new frontiers found.")

            total_cells = h * w
            visited_cells = len(visited)
            visited_percentage = (visited_cells / total_cells) * 100

            if visited_percentage > 50:
                print(f"[EXPLORE] {visited_percentage:.1f}% of map marked visited. Exploration complete.")
                state = "DONE"
            else:
                print(f"[EXPLORE] Only {visited_percentage:.1f}% visited but no frontiers. Clearing visited set.")
                visited.clear()
                frontier_cache = None
                state = "SPIN"
                spin_timer = 0


    # STATE: FOLLOW PATH
    elif state == "FOLLOW":
        phase = "Following Path"
        frontier_cache = None
        # If something is too close in front
        if too_close:
            print("[SAFETY] Wall detected. Initiating 180 turn.")

            ri, rj = slam_pose_to_grid(slam_x, slam_y, slam_backend)
            mark_frontier_region_visited(visited, (ri, rj), radius=6)

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
                print(f"[INFO] Reached Frontier at ({ri}, {rj}). Marking region & scanning...")

                state = "SPIN"
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

            state = "ROTATE_180"
            turn_accumulated = 0.0
            print("[AVOID] Safe distance reached. Starting 180 turn.")
    

    # STATE: DONE 
    elif state == "DONE":
        v_left = 0.0
        v_right = 0.0
        phase = "Done"
        
        if key == ord('R'):
            state = "EXPLORE"
            print("[USER] Restarting Exploration.")

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
            filename = f"map_snapshot_{int(t):04d}.png"
            plt.savefig(filename, dpi=150)
            plt.close()
            print(f"[INFO] Auto-saved map snapshot: {filename}")

        # Save Video Frames
        if SAVE_VIDEO_FRAMES:
            grid = slam_backend.get_map_grid()

            plt.figure(figsize=(8, 8))
            plt.imshow(grid, cmap="gray", origin="lower")
            plt.title(f"SLAM Map - Time: {t:.1f}s")
            plt.axis("off")
            plt.tight_layout()

            # Save frame
            frame_filename = os.path.join(FRAMES_DIR, f"frame_{frame_index:05d}.png")
            plt.savefig(frame_filename, dpi=100)
            plt.close()
            frame_index += 1
            
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