from controller import Supervisor
import math

class Pedestrian (Supervisor):


    def __init__(self):

        self.BODY_PARTS_NUMBER = 13
        self.WALK_SEQUENCES_NUMBER = 8
        self.ROOT_HEIGHT = 1.27
        self.CYCLE_TO_DISTANCE_RATIO = 0.22
        self.speed = 1.15
        self.current_height_offset = 0
        self.joints_position_field = []
        self.joint_names = [
            "leftArmAngle", "leftLowerArmAngle", "leftHandAngle",
            "rightArmAngle", "rightLowerArmAngle", "rightHandAngle",
            "leftLegAngle", "leftLowerLegAngle", "leftFootAngle",
            "rightLegAngle", "rightLowerLegAngle", "rightFootAngle",
            "headAngle"
        ]
        self.height_offsets = [
            -0.02, 0.04, 0.08, -0.03, -0.02, 0.04, 0.08, -0.03
        ]
        self.angles = [
            [-0.52, -0.15, 0.58, 0.7, 0.52, 0.17, -0.36, -0.74],
            [0.0, -0.16, -0.7, -0.38, -0.47, -0.3, -0.58, -0.21],
            [0.12, 0.0, 0.12, 0.2, 0.0, -0.17, -0.25, 0.0],
            [0.52, 0.17, -0.36, -0.74, -0.52, -0.15, 0.58, 0.7],
            [-0.47, -0.3, -0.58, -0.21, 0.0, -0.16, -0.7, -0.38],
            [0.0, -0.17, -0.25, 0.0, 0.12, 0.0, 0.12, 0.2],
            [-0.55, -0.85, -1.14, -0.7, -0.56, 0.12, 0.24, 0.4],
            [1.4, 1.58, 1.71, 0.49, 0.84, 0.0, 0.14, 0.26],
            [0.07, 0.07, -0.07, -0.36, 0.0, 0.0, 0.32, -0.07],
            [-0.56, 0.12, 0.24, 0.4, -0.55, -0.85, -1.14, -0.7],
            [0.84, 0.0, 0.14, 0.26, 1.4, 1.58, 1.71, 0.49],
            [0.0, 0.0, 0.42, -0.07, 0.07, 0.07, -0.07, -0.36],
            [0.18, 0.09, 0.0, 0.09, 0.18, 0.09, 0.0, 0.09]
        ]
        Supervisor.__init__(self)

    def run(self):
        """Set the Pedestrian pose and position."""
        self.time_step = int(self.getBasicTimeStep())
        self.root_node_ref = self.getSelf()
        self.root_translation_field = self.root_node_ref.getField("translation")
        self.root_rotation_field = self.root_node_ref.getField("rotation")
        
        for i in range(0, self.BODY_PARTS_NUMBER):
            self.joints_position_field.append(self.root_node_ref.getField(self.joint_names[i]))

        # Track previous position to calculate speed
        current_pos = self.root_translation_field.getSFVec3f()
        last_pos = current_pos
        distance_covered = 0.0

        while not self.step(self.time_step) == -1:
            current_pos = self.root_translation_field.getSFVec3f()
            
            # Calculate distance moved since last step
            dx = current_pos[0] - last_pos[0]
            dy = current_pos[1] - last_pos[1]
            move_dist = math.sqrt(dx*dx + dy*dy)
            
            # Threshold to prevent jitter animation when standing still
            if move_dist > 0.001:
                distance_covered += move_dist
                
                # Update Animation based on Distance Covered
                current_sequence = int((distance_covered / self.CYCLE_TO_DISTANCE_RATIO) % self.WALK_SEQUENCES_NUMBER)
                ratio = (distance_covered / self.CYCLE_TO_DISTANCE_RATIO) - int(distance_covered / self.CYCLE_TO_DISTANCE_RATIO)

                for i in range(0, self.BODY_PARTS_NUMBER):
                    current_angle = self.angles[i][current_sequence] * (1 - ratio) + \
                        self.angles[i][(current_sequence + 1) % self.WALK_SEQUENCES_NUMBER] * ratio
                    self.joints_position_field[i].setSFFloat(current_angle)

                # adjust height
                self.current_height_offset = self.height_offsets[current_sequence] * (1 - ratio) + \
                    self.height_offsets[(current_sequence + 1) % self.WALK_SEQUENCES_NUMBER] * ratio
                    
              
              
                if move_dist > 0.005:
                    angle = math.atan2(dy, dx)
                    rotation = [0, 0, 1, angle]
                    self.root_rotation_field.setSFRotation(rotation)
                
            last_pos = current_pos

controller = Pedestrian()
controller.run()
