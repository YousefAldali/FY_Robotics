import math
from controller import Robot

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

# --------------- LIDAR Setup -----------------
# Initialise LIDAR
lidar = robot.getDevice('lidar')
lidar.enable(TIME_STEP)

lidar.enablePointCloud()

# --------------- IMU Setup -----------------
# Initialise IMU
imu = None

try: 
    imu = robot.getDevice('imu')
    imu.enable(TIME_STEP)
    print("[INFO] IMU found and enabled.")
except:
    print("[WARN] IMU not found on this robot model.")


# --------------- Odometry Variables -----------------
prev_left = 0.0
prev_right = 0.0
x = 0.0
y = 0.0
theta = 0.0
first_step = True

def update_odometry():
    global prev_left, prev_right, x, y, theta, first_step

    left_val = left_encoder.getValue()
    right_val = right_encoder.getValue()
    if first_step:
        prev_left = left_val
        prev_right = right_val
        first_step = False
        return
    
    d_left = left_val - prev_left
    d_right = right_val - prev_right
    prev_left = left_val
    prev_right = right_val

    dl_distance = d_left * WHEEL_RADIUS
    dr_distance = d_right * WHEEL_RADIUS

    d_center = (dl_distance + dr_distance) / 2.0
    d_theta = (dr_distance - dl_distance) / AXLE_LENGTH

    theta_mid = theta + d_theta / 2.0
    x += d_center * math.cos(theta_mid)
    y += d_center * math.sin(theta_mid)
    theta += d_theta

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

    update_odometry()
    
    left_val = left_encoder.getValue()
    right_val = right_encoder.getValue()

    ranges = lidar.getRangeImage()
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
        