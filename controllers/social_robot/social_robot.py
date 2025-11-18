from controller import Robot

robot = Robot()
TIME_STEP = int(robot.getBasicTimeStep())

print("[INFO] Social Python Controller Initialized with TIME_STEP =", TIME_STEP)

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



while robot.step(TIME_STEP) != -1:
    left_val = left_encoder.getValue()
    right_val = right_encoder.getValue()

    t = robot.getTime()
    if int(t*2) % 2 == 0:
        print(f"[TIME {t:.2f} s] Left Encoder: {left_val:.4f} rad, Right Encoder: {right_val:.4f} rad")