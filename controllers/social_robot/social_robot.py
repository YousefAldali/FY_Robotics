from controller import Robot

robot = Robot()
TIME_STEP = int(robot.getBasicTimeStep())

print("[INFO] Social Python Controller Initialized with TIME_STEP =", TIME_STEP)

while robot.step(TIME_STEP) != -1:
    pass