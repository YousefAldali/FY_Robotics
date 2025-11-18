from controller import Robot

robot = Robot()
TIME_STEP = int(robot.getBasicTimeStep())

print("[INFO] Social Python Controller Initialized with TIME_STEP =", TIME_STEP)

# Initialise motors
left_motor = robot.getDevice('left wheel')
right_motor = robot.getDevice('right wheel')

left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))   

forward_speed = 2.0 # Forward speed in rad/s
left_motor.setVelocity(forward_speed)
right_motor.setVelocity(forward_speed)

while robot.step(TIME_STEP) != -1:
    pass