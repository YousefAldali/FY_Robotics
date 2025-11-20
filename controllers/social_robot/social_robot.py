import math
from controller import Robot
import numpy as np
from breezyslam.algorithms import RMHC_SLAM
from breezyslam.sensors import Laser


robot = Robot()
TIME_STEP = int(robot.getBasicTimeStep())

print("[INFO] Social Python Controller Initialized with TIME_STEP =", TIME_STEP)

WHEEL_RADIUS = 0.0975 # in meters
AXLE_LENGTH = 0.33  # in meters

# --------------- Robot Motors Setup -----------------
# Initialise motors
left_motor = robot.getDevice('left wheel')
right_motor = robot.getDevice('right wheel')

left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))   

forward_speed = 2.0 # Forward speed in rad/s
left_motor.setVelocity(forward_speed)
right_motor.setVelocity(forward_speed)

# --------------- Robot Encoders Setup -----------------
# Initialise encoders
left_encoder = robot.getPositionSensor('left wheel sensor')
right_encoder = robot.getPositionSensor('right wheel sensor')

left_encoder.enable(TIME_STEP)
right_encoder.enable(TIME_STEP)

# --------------- Pose Estimator Class -----------------
class PoseEstimator:
    def __init__(self, wheel_radius, axle_length):
        self.wheel_radius = wheel_radius
        self.axle_length = axle_length

        self.prev_left = 0.0
        self.prev_right = 0.0

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0 # in radians

    def update_from_encoders(self, left_val, right_val):
        if self.prev_left is None:
            self.prev_left = left_val
            self.prev_right = right_val
            return
        
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
    
class WebotsLidar(Laser):
    def __init__(self, lidar_device, time_step_ms):
        n_beams = lidar_device.getHorizontalResolution()
        fov_rad = lidar_device.getFov()
        fov_deg = math.degrees(fov_rad)

        max_range_m = lidar_device.getMaxRange()
        max_range_mm = int(max_range_m * 1000)

        scan_rate_hz = 1000.0 / float(time_step_ms)

        super().__init__ (n_beams, scan_rate_hz, fov_deg, max_range_mm)

        self.max_range_m = max_range_m



class SlamBackend:
    def __init__ (self, lidar_device, time_step_ms):
        self.lidar_model = WebotsLidar(lidar_device, time_step_ms)
        self.max_range_m = self.lidar_model.max_range_m
        
        self.MAP_SIZE_PIXELS = 800
        self.MAP_SIZE_METERS = 20

        self.slam = RMHC_SLAM(self.lidar_model,
                         self.MAP_SIZE_PIXELS,
                         self.MAP_SIZE_METERS)
        
        self.mapbytes = bytearray(self.MAP_SIZE_PIXELS * self.MAP_SIZE_PIXELS)

        self.x_mm = 0
        self.y_mm = 0
        self.theta_deg = 0

    def update(self, ranges_m):
        max_range_m = self.max_range_m
        scan_mm = []
        for r in ranges_m:
            if math.isinf(r) or r <= 0.0:
                r = max_range_m
            scan_mm.append(int(r * 1000))

        self.slam.update(scan_mm)

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

# --------------- LIDAR Setup -----------------
# Initialise LIDAR
lidar = robot.getDevice('lidar')
lidar.enable(TIME_STEP)

lidar.enablePointCloud()

# --------------- SLAM Backend -----------------
slam_backend = SlamBackend(lidar, TIME_STEP)

# --------------- IMU Setup -----------------
# Initialise IMU
imu = None

try: 
    imu = robot.getDevice('imu')
    imu.enable(TIME_STEP)
    print("[INFO] IMU found and enabled.")
except:
    print("[WARN] IMU not found on this robot model.")

# --------------- Pose Estimator Setup -----------------
pose_estimator = PoseEstimator(WHEEL_RADIUS, AXLE_LENGTH)

# --------------- Main Loop -----------------
while robot.step(TIME_STEP) != -1:
    t = robot.getTime()
    
    if t < 5.0:
        v_left = 3.0
        v_right = 3.0
        phase = "Forward"
    elif t < 10.0:
        v_left = 2.0
        v_right = -2.0
        phase = "Turning"
    else:
        v_left = 0.0
        v_right = 0.0
        phase = "Stopped"

    left_motor.setVelocity(v_left)
    right_motor.setVelocity(v_right)

    left_val = left_encoder.getValue()
    right_val = right_encoder.getValue()

    dx, dy, dtheta = pose_estimator.update_from_encoders(left_val, right_val)
    odo_x, odo_y, odo_theta = pose_estimator.get_pose()

    raw_ranges = lidar.getRangeImage()

    n_beams = lidar.getHorizontalResolution()
    n_layers = lidar.getNumberOfLayers()

    if n_layers > 1:
        middle_layer = n_layers // 2
        ranges = raw_ranges[middle_layer * n_beams : (middle_layer + 1) * n_beams]
    else:
        ranges = raw_ranges[:n_beams]

    min_range = None
    center_range = None
    if ranges and len(ranges) > 0:
        min_range = min(ranges)
        center_index = len(ranges) // 2
        center_range = ranges[center_index]
    
    yaw_deg = None
    if imu is not None:
        roll, pitch, yaw = imu.getRollPitchYaw()
        yaw_deg = yaw * (180.0 / math.pi)

    if ranges and len(ranges) > 0:
        slam_backend.update(ranges)
        slam_x, slam_y, slam_theta = slam_backend.get_pose()
    else:
        slam_x, slam_y, slam_theta = 0.0, 0.0, 0.0
        
    if int(t) != int(t - TIME_STEP / 1000.0):
        print(f"----  Time: {t:.2f} s  ----")
        print(f"Phase: {phase}")
        print(f"Odometry: x={x:.2f} m, y={y:.2f} m, theta={theta*180/math.pi:.1f} deg")
        print(f"Encoders: left={left_val:.2f} rad, right={right_val:.2f} rad")
        if min_range is not None:
            print(f"Lidar: min={min_range:.2f} m, center={center_range:.2f} m")
        if yaw_deg is not None:
            print(f"IMU: yaw={yaw_deg:.2f} degrees")
        print("-------------------------")
        