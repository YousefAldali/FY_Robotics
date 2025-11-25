import math
import os
from controller import Robot, Keyboard
import matplotlib
import matplotlib.pyplot as plt

from lidar_module import SlamBackend, process_lidar_ranges, get_lidar_safety_info
from movement_module import (
    PoseEstimator, AStarPlanner, ROOM_GOALS,
    slam_pose_to_grid, grid_to_world, find_nearest_free_cell,
    compute_wheel_velocities, apply_complementary_filter,
    OBSTACLE_STOP_DIST, STUCK_STEP_LIMIT
)

# Initialize robot
robot = Robot()
TIME_STEP = int(robot.getBasicTimeStep())

matplotlib.use("Agg")

keyboard = Keyboard()
keyboard.enable(TIME_STEP)

GOAL_TOLERANCE = 0.8  # meters
SAVE_FINAL_MAP = True
SAVE_VIDEO_FRAMES = False

FRAMES_DIR = "slam_frames"
if SAVE_VIDEO_FRAMES and not os.path.exists(FRAMES_DIR):
    os.makedirs(FRAMES_DIR)

map_saved = False
frame_index = 0

print("[INFO] Social Python Controller Initialized with TIME_STEP =", TIME_STEP)

WHEEL_RADIUS = 0.0975  # in meters
AXLE_LENGTH = 0.33  # in meters

# --------------- Robot Motors Setup -----------------
left_motor = robot.getDevice('left wheel')
right_motor = robot.getDevice('right wheel')

left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

forward_speed = 1.0  # Forward speed in rad/s
left_motor.setVelocity(forward_speed)
right_motor.setVelocity(forward_speed)

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
state = "EXPLORE"  # EXPLORE -> PLAN -> FOLLOW -> DONE
path = []  
current_wp_idx = 0
last_wp_idx = None
stuck_steps = 0

# --------------- Complementary Filter Setup -----------------
fused_theta = 0.0
prev_fused_theta = 0.0
alpha = 0.98  

# --------------- Target Room for Navigation -----------------
TARGET_ROOM = "kitchen"
kitchen_goal = None

# Test
imu_offset = 0.0
if imu:
    for i in range(10): 
        robot.step(TIME_STEP)
    roll, pitch, start_yaw = imu.getRollPitchYaw()
    imu_offset = start_yaw 

# --------------- Main Loop -----------------
while robot.step(TIME_STEP) != -1:
    t = robot.getTime()
    key = keyboard.getKey()

    left_val = left_encoder.getValue()
    right_val = right_encoder.getValue()
    dx, dy, dtheta = pose_estimator.update_from_encoders(left_val, right_val)
    odo_x, odo_y, odo_theta = pose_estimator.get_pose()

    raw_ranges = lidar.getRangeImage()
    n_beams = lidar.getHorizontalResolution()
    n_layers = lidar.getNumberOfLayers()

    ranges = process_lidar_ranges(raw_ranges, n_beams, n_layers)
    too_close, min_range, center_range, front_min_range = get_lidar_safety_info(
        ranges, OBSTACLE_STOP_DIST
    )

    yaw_deg = None
    if imu is not None:
        roll, pitch, raw_yaw = imu.getRollPitchYaw()  # yaw in radians
        yaw = raw_yaw - imu_offset
        yaw_deg = yaw * (180.0 / math.pi)
    else:
        yaw = None

    if imu is not None and yaw is not None:
        fused_theta = apply_complementary_filter(odo_theta, yaw, alpha)
        dtheta_fused = fused_theta - prev_fused_theta
        dtheta_fused = (dtheta_fused + math.pi) % (2 * math.pi) - math.pi
        prev_fused_theta = fused_theta
        dtheta_for_slam = dtheta_fused
    else:
        dtheta_for_slam = dtheta

    # if abs(dx) < 1e-6 and abs(dtheta) < 1e-6:
    #     dtheta_for_slam = 0.0

    if ranges and len(ranges) > 0:
        slam_backend.update(ranges, d_center_m=dx, dtheta_rad=dtheta_for_slam)
        slam_x, slam_y, slam_theta = slam_backend.get_pose()
    else:
        slam_x, slam_y, slam_theta = 0.0, 0.0, 0.0

    if state == "EXPLORE":
        v_left = 0.0
        v_right = 0.0
        phase = "Teleop"

        if key == ord('K'):
            kitchen_goal = (slam_x, slam_y)
            print(f"[MARK] Kitchen goal set at SLAM coords: x={slam_x:.2f}, y={slam_y:.2f}")

        if key == ord('W'):      # forward
            v_left = 3.0
            v_right = 3.0
        elif key == ord('S'):    # backward
            v_left = -3.0
            v_right = -3.0
        elif key == ord('A'):    # turn left
            v_left = -2.0
            v_right = 2.0
        elif key == ord('D'):    # turn right
            v_left = 2.0
            v_right = -2.0

        if key == ord('M'):
            v_left = 0.0
            v_right = 0.0
            phase = "Stopped"
            state = "PLAN"
            print("[EXPLORE] Mapping finished, switching to PLAN.")

            grid = slam_backend.get_map_grid()
            plt.figure(figsize=(10, 10))
            plt.imshow(grid, cmap="gray", origin="lower")
            plt.title("SLAM Occupancy Grid - Exploration Complete", fontsize=14)
            plt.colorbar(label="Occupancy Value (0=obstacle, 255=free)")
            plt.xlabel("Grid X (pixels)")
            plt.ylabel("Grid Y (pixels)")

            robot_i, robot_j = slam_pose_to_grid(slam_x, slam_y, slam_backend)
            plt.plot(robot_i, robot_j, 'ro', markersize=10, label=f'Robot Position ({slam_x:.2f}, {slam_y:.2f})')

            if kitchen_goal is not None:
                kitchen_i, kitchen_j = slam_pose_to_grid(kitchen_goal[0], kitchen_goal[1], slam_backend)
                plt.plot(kitchen_i, kitchen_j, 'g*', markersize=15, label=f'Kitchen Goal ({kitchen_goal[0]:.2f}, {kitchen_goal[1]:.2f})')

            plt.legend()
            plt.tight_layout()
            plt.savefig("slam_map_explore_complete.png", dpi=300)
            plt.close()
            print("[INFO] Saved exploration map to slam_map_explore_complete.png")

    elif state == "PLAN":
        v_left = 0.0
        v_right = 0.0
        phase = "Planning"

        grid = slam_backend.get_map_grid()
        start_i, start_j = slam_pose_to_grid(slam_x, slam_y, slam_backend)

        start_free = find_nearest_free_cell(grid, start_i, start_j)
        if start_free is None:
            print(f"[PLAN] Robot is stuck in a wall (no free cell).")
            state = "DONE"
        else:
            start_i, start_j = start_free

            if TARGET_ROOM == "kitchen" and kitchen_goal is not None:
                ROOM_GOALS["kitchen"] = kitchen_goal

            if TARGET_ROOM not in ROOM_GOALS:
                print(f"[PLAN] Unknown room: {TARGET_ROOM}")
                state = "DONE"
            else:
                goal_x, goal_y = ROOM_GOALS[TARGET_ROOM]
                goal_i, goal_j = slam_pose_to_grid(goal_x, goal_y, slam_backend)

                goal_free = find_nearest_free_cell(grid, goal_i, goal_j, max_radius=20)
                if goal_free is None:
                    print("[PLAN] Goal is inside an obstacle.")
                    state = "DONE"
                else:
                    goal_i, goal_j = goal_free
                    print(f"[PLAN] Planning from ({start_i},{start_j}) to ({goal_i},{goal_j})...")

                    planner = AStarPlanner(grid)
                    raw_path = planner.plan((start_i, start_j), (goal_i, goal_j))

                    if raw_path:
                        print(f"[PLAN] A* found path with {len(raw_path)} steps.")

                        stride = 4
                        path = [raw_path[0]] + raw_path[stride::stride]

                        if path[-1] != raw_path[-1]:
                            path.append(raw_path[-1])

                        print(f"[PLAN] Smoothed path to {len(path)} waypoints.")
                        current_wp_idx = 0
                        state = "FOLLOW"
                    else:
                        print("[PLAN] No path found! (Goal might be unreachable)")
                        state = "DONE"

    elif state == "FOLLOW":
        phase = "Following"

        if too_close:
            if front_min_range is not None:
                print(f"[SAFETY] Obstacle too close (front_min_range={front_min_range:.2f} m). Stopping.")
            else:
                print("[SAFETY] Obstacle too close (front_min_range unknown). Stopping.")
            v_left = 0.0
            v_right = 0.0
            state = "DONE"
        else:
            if current_wp_idx >= len(path):
                v_left = 0.0
                v_right = 0.0
                state = "DONE"
                print("[INFO] Reached end of path list.")
            else:
                wp_i, wp_j = path[current_wp_idx]
                wp_x, wp_y = grid_to_world(wp_i, wp_j, slam_backend)

                dx_wp = wp_x - slam_x
                dy_wp = wp_y - slam_y
                dist = math.hypot(dx_wp, dy_wp)

                target_heading = math.atan2(dy_wp, dx_wp)
                heading_error = target_heading - slam_theta
                heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi

                is_final_waypoint = (current_wp_idx == len(path) - 1)

                if not is_final_waypoint:
                    if dist < 0.15 and abs(heading_error) > math.radians(90):
                        print(f"[SMART] Skipping waypoint {current_wp_idx} (Behind robot)")
                        current_wp_idx += 1
                        continue

                if abs(heading_error) > 0.35:  # ~20 degrees
                    v = 0.0
                    max_rot_speed = 1.5
                    w = -heading_error * 1.5
                    w = max(min(w, max_rot_speed), -max_rot_speed)
                    print(f"[FOLLOW] Point turning. Error: {math.degrees(heading_error):.1f} deg")
                else:
                    v = min(dist * 1.5, 1.0)
                    w = -heading_error * 2.0

                    v = min(v, 1.0)
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
                    print(f"[FOLLOW] Reached waypoint {current_wp_idx}/{len(path)}")

    else:  # state == "DONE"
        v_left = 0.0
        v_right = 0.0
        phase = "Done"

    left_motor.setVelocity(v_left)
    right_motor.setVelocity(v_right)

    if int(t) != int(t - TIME_STEP / 1000.0):
        print(f"----  Time: {t:.2f} s  ----")
        print(f"Phase: {phase}")
        print(f"Odometry: x={odo_x:.2f} m, y={odo_y:.2f} m, theta={odo_theta*180/math.pi:.1f} deg")
        print(f"SLAM Pose: x={slam_x:.2f} m, y={slam_y:.2f} m, theta={slam_theta*180/math.pi:.1f} deg")
        print(f"SLAM raw theta_deg: {slam_backend.theta_deg:.1f}")
        print(f"Encoders: left={left_val:.2f} rad, right={right_val:.2f} rad")
        if front_min_range is not None:
            print(f"Lidar: front_min={front_min_range:.2f} m, center={center_range:.2f} m")
        if yaw_deg is not None:
            print(f"IMU: yaw={yaw_deg:.2f} degrees")

        if int(t) % 5 == 0:
            grid = slam_backend.get_map_grid()
            print(f"[MAP] mean={grid.mean():.1f}, occupied={(grid > 200).sum()}")

        if SAVE_FINAL_MAP and (phase == "Stopped") and not map_saved:
            grid = slam_backend.get_map_grid()

            plt.figure(figsize=(6, 6))
            plt.imshow(grid, cmap="gray", origin="lower")
            plt.title("SLAM Occupancy Grid")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig("slam_map_final.png", dpi=300)
            plt.close()

            map_saved = True
            print("[INFO] Saved final SLAM map to slam_map_final.png")

        if SAVE_VIDEO_FRAMES:
            grid = slam_backend.get_map_grid()
            fname = os.path.join(FRAMES_DIR, f"map_{frame_index:04d}.png")
            plt.figure(figsize=(6, 6))
            plt.imshow(grid, cmap="gray", origin="lower")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(fname, dpi=150)
            plt.close()
            frame_index += 1

        print("-------------------------")
