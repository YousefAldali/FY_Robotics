# System Architecture

This document provides a detailed technical overview of the social navigation robot system.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      Webots Simulator                        │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                    Physics Engine                       │ │
│  │  • Robot (TiagoIron)    • Human (Pedestrian)           │ │
│  │  • Environment (Care Home)                             │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────┐
│               social_robot.py (Main Controller)              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Sensors    │  │     IMU      │  │   Encoders   │     │
│  │   (LiDAR)    │  │  (Optional)  │  │ (Odometry)   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│           ↓                ↓                ↓               │
│  ┌────────────────────────────────────────────────────┐    │
│  │          Complementary Filter Fusion                │    │
│  └────────────────────────────────────────────────────┘    │
│           ↓                                                 │
│  ┌────────────────────────────────────────────────────┐    │
│  │            lidar_module.py (SLAM Backend)           │    │
│  │  • BreezySLAM RMHC_SLAM                            │    │
│  │  • Map Generation (800x800 grid)                   │    │
│  │  • Pose Estimation (x, y, θ)                       │    │
│  └────────────────────────────────────────────────────┘    │
│           ↓                                                 │
│  ┌────────────────────────────────────────────────────┐    │
│  │        movement_module.py (Planning & Control)      │    │
│  │  • Frontier Detection                              │    │
│  │  • A* Path Planning (Standard + Social)            │    │
│  │  • Motion Control (Differential Drive)             │    │
│  └────────────────────────────────────────────────────┘    │
│           ↓                                                 │
│  ┌────────────────────────────────────────────────────┐    │
│  │              State Machine Controller               │    │
│  │  • EXPLORE  • GUIDE  • MANUAL  • TEST              │    │
│  └────────────────────────────────────────────────────┘    │
│           ↓                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Left Motor  │  │ Right Motor  │  │ Human Node   │     │
│  │   Commands   │  │   Commands   │  │  Position    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

---

## Module Breakdown

### 1. Main Controller (`social_robot.py`)

**Responsibility**: Central orchestration of all subsystems

**Key Components**:

#### State Machine
```python
Modes:
  - EXPLORE: Autonomous frontier-based exploration
  - GUIDE: Human guidance with social navigation
  - MANUAL: Teleoperation via keyboard
  - TEST: Automated test sequences (not implemented)

States (within EXPLORE mode):
  - EXPLORE: Frontier selection and path planning
  - FOLLOW: Path following with waypoint tracking
  - SPIN: 360° scan for map refinement
  - AVOID: Obstacle avoidance (backup)
  - AVOID_TURN: Turn to escape obstacle
  - RETURNING_HOME: Return to start after time limit
  - DONE: Mission complete

States (within GUIDE mode):
  - IDLE: Initial planning
  - FETCHING: Navigate to human position
  - GUIDING: Lead human to destination
  - GUIDE_AVOID: Obstacle avoidance during guidance
  - GUIDE_AVOID_TURN: Turn during avoidance
  - GUIDE_AVOID_FORWARD: Move forward after turn
  - DONE: Mission complete
```

#### Sensor Integration
- **LiDAR**: 360° range data (270° FoV, 180 beams)
- **Encoders**: Left/right wheel position sensors
- **IMU**: Roll, pitch, yaw (optional, for drift correction)
- **Keyboard**: Mode switching and manual control

#### Data Logging
- Periodic CSV export (Time, X, Y, Theta, DistHuman)
- Map snapshots every 30 seconds
- Video frame generation with path visualization

---

### 2. SLAM Module (`lidar_module.py`)

**Responsibility**: Localization and mapping

**Architecture**:

```python
class WebotsLidar(breezyslam.Laser):
    Purpose: Adapter for Webots LiDAR → BreezySLAM format

    Methods:
      __init__(lidar_device, time_step_ms):
        - Extracts: num_beams, FoV, max_range
        - Converts units: radians→degrees, meters→mm
        - Calculates scan rate from timestep

class SlamBackend:
    Purpose: High-level SLAM interface with odometry fusion

    Attributes:
      - lidar_model: WebotsLidar instance
      - slam: BreezySLAM RMHC_SLAM instance
      - mapbytes: 800x800 occupancy grid buffer
      - MAP_SIZE_PIXELS: 800
      - MAP_SIZE_METERS: 20

    Methods:
      update(ranges_m, d_center_m, dtheta_rad):
        - Convert ranges to mm
        - Pass odometry delta to SLAM
        - Updates internal map

      get_pose() → (x_m, y_m, theta_rad):
        - Returns current robot pose in meters/radians

      get_map_grid() → numpy array:
        - Returns 800x800 occupancy grid
        - Values: 0 (obstacle) → 127 (unknown) → 255 (free)
```

**SLAM Algorithm**: BreezySLAM's RMHC_SLAM
- **Type**: Particle filter-based scan matching
- **Input**: LiDAR ranges + odometry
- **Output**: Pose estimate + occupancy grid
- **Parameters**:
  - `sigma_theta_degrees=5`: Angular uncertainty
  - `sigma_xy_mm=50`: Position uncertainty
  - `hole_width_mm=200`: Gap-filling threshold
  - `map_quality=1`: Resolution setting

---

### 3. Movement Module (`movement_module.py`)

**Responsibility**: Path planning, motion control, and social navigation

#### A. Pose Estimation

```python
class PoseEstimator:
    Purpose: Dead reckoning from wheel encoders

    Algorithm: Differential drive odometry
      d_left = (encoder_left - prev_left) * wheel_radius
      d_right = (encoder_right - prev_right) * wheel_radius
      d_center = (d_left + d_right) / 2
      d_theta = (d_right - d_left) / axle_length

      x += d_center * cos(theta + d_theta/2)
      y += d_center * sin(theta + d_theta/2)
      theta += d_theta

    Note: Used for SLAM odometry input, not primary localization
```

#### B. Standard Path Planning

```python
class OccupancyAStarPlanner:
    Purpose: Cost-based A* with terrain modulation

    Cost Model:
      cell_value = 0 → Free space (cost: 1.0)
      cell_value = 1 → Obstacle (impassable)
      cell_value = 2 → Buffer zone (cost: 8.0)

    Algorithm:
      1. Expand nodes in priority queue (f = g + h)
      2. g_cost = path cost from start
      3. h_cost = Euclidean distance to goal
      4. step_cost = base_cost (1.0 or 8.0)

    Result: Minimum-cost path avoiding obstacles
```

#### C. Social Path Planning (Novel Contribution)

```python
class SocialAStarPlanner(OccupancyAStarPlanner):
    Purpose: Proxemics-aware path planning

    Innovation: Gaussian cost field around human

    get_social_cost(i, j):
      dist_sq = (i - human_x)² + (j - human_y)²
      cost = amplitude * exp(-dist_sq / (2 * sigma²))

      Returns: Additional cost based on proximity to human

    Profiles (tunable parameters):
      Conservative:
        - sigma = 15 cells (~37.5cm radius @ 40px/m)
        - amplitude = 20 (strong avoidance)

      Neutral:
        - sigma = 10 cells (~25cm radius)
        - amplitude = 15 (moderate avoidance)

      Open:
        - sigma = 8 cells (~20cm radius)
        - amplitude = 10 (weak avoidance)

    Total Cost:
      step_cost = base_cost + social_cost

      Example: Cell 1m from human (Conservative profile)
        social_cost ≈ 20 * exp(-40² / (2*15²)) ≈ 2.7
        total_cost ≈ 1.0 + 2.7 = 3.7

        → Robot prefers 3.7x longer path through free space
          over cutting close to human
```

#### D. Frontier Detection

```python
find_nearest_frontier(nav_grid, start_i, start_j, visited):
    Purpose: Identify unexplored boundaries for exploration

    Algorithm:
      1. Binary grid: 1 = free, 0 = not free
      2. Dilate free space by 1 cell
      3. Find intersection with unknown space → frontier
      4. BFS from start to nearest frontier (not in visited set)

    Output: (goal_i, goal_j) coordinates of nearest frontier
```

#### E. Motion Control

```python
compute_wheel_velocities(v, w, wheel_radius, axle_length):
    Purpose: Convert linear/angular velocity to wheel speeds

    Differential Drive Kinematics:
      v_left = (v - w * axle_length/2) / wheel_radius
      v_right = (v + w * axle_length/2) / wheel_radius

    Inputs:
      v: Linear velocity (m/s)
      w: Angular velocity (rad/s)

    Outputs:
      v_left, v_right: Wheel angular velocities (rad/s)
```

#### F. Complementary Filter (IMU Fusion)

```python
apply_complementary_filter(odo_theta, imu_theta, alpha=0.98):
    Purpose: Reduce odometry drift using IMU

    Algorithm:
      fused_theta = alpha * odo_theta + (1-alpha) * imu_theta

    Rationale:
      - Odometry drifts over time (low-frequency error)
      - IMU is noisy but unbiased (high-frequency error)
      - alpha=0.98: Trust odometry for short-term, IMU for long-term
```

---

### 4. Pedestrian Controller (`pedestrian.py`)

**Responsibility**: Human animation synchronized with movement

**Mode**: Passive (position controlled by `social_robot.py`)

**Animation Algorithm**:
```python
1. Track distance moved since last frame
2. Update animation phase = distance / cycle_distance_ratio
3. Interpolate joint angles from 8-frame walk cycle
4. Update height offset for realistic gait
5. Rotate body to face movement direction
```

**Walk Cycle**: 8 keyframes for:
- Left/right arm swing
- Left/right leg motion
- Head bob
- Torso height oscillation

**Integration**:
- Position set by supervisor from `social_robot.py`
- Animation triggers only when position changes >0.001m

---

## Data Flow

### Sensor → SLAM → Planning Loop

```
Every timestep (64ms):

  1. Read Sensors
     └─> LiDAR: raw_ranges[540] (3 layers × 180 beams)
     └─> Encoders: left_val, right_val
     └─> IMU: roll, pitch, yaw

  2. Preprocess
     └─> Extract middle layer: ranges[180]
     └─> Calculate odometry: dx, dy, dtheta
     └─> Fuse IMU: dtheta_fused = complementary_filter()

  3. Update SLAM
     └─> slam_backend.update(ranges, dx, dtheta_fused)
     └─> Get pose: slam_x, slam_y, slam_theta
     └─> Get map: grid[800, 800]

  4. State Machine Decision
     └─> If EXPLORE:
         ├─> Find frontier: (goal_i, goal_j)
         ├─> Plan path: A* on occupancy grid
         └─> Execute: Follow waypoints

     └─> If GUIDE:
         ├─> Get human position: (human_x, human_y)
         ├─> Plan social path: SocialAStarPlanner
         ├─> Check separation: dist_to_human
         ├─> If human far: Wait
         └─> Else: Follow waypoints

  5. Motion Control
     └─> Calculate heading error: target_heading - slam_theta
     └─> Set velocities: v, w
     └─> Convert to wheels: v_left, v_right
     └─> Send commands: motor.setVelocity()

  6. Human Control (if GUIDE mode)
     └─> Calculate target: robot_pos - 1.5m behind
     └─> Lerp human position: smooth following
     └─> Update Webots node: translation field

  7. Data Logging
     └─> Log: [time, x, y, theta, dist_human]
     └─> Save frame: SLAM map + path + social zones
```

---

## Key Design Decisions

### 1. Why BreezySLAM?

**Pros**:
- Lightweight (no ROS dependency)
- Easy Webots integration
- Reasonable accuracy for indoor environments
- Good performance on modest hardware

**Cons**:
- No loop closure (drift on large maps)
- Limited tuning options
- No 3D mapping

**Alternative considered**: GMapping (rejected due to ROS dependency)

### 2. Why Gaussian Social Costs?

**Rationale**:
- Smooth, differentiable cost function
- Interpretable parameters (sigma = "comfort distance")
- Computationally cheap (closed-form)
- Resembles Hall's proxemic zones naturally

**Alternative considered**: Hard constraint circles (rejected: too rigid, poor paths)

### 3. Why Frontier-Based Exploration?

**Rationale**:
- No prior map required
- Naturally maximizes coverage
- Simple to implement
- Proven in literature (Yamauchi 1997)

**Alternative considered**: Random walk (rejected: inefficient, redundant coverage)

### 4. Why Supervisor Control for Human?

**Rationale**:
- Precise control (no physics errors)
- Simple implementation
- Predictable behavior for testing
- Avoids collision resolution issues

**Limitation**: Not realistic for dynamic human behavior

**Future work**: Replace with physics-based human agent

---

## Performance Characteristics

### Computational Complexity

| Component | Time Complexity | Notes |
|-----------|----------------|-------|
| SLAM Update | O(N × M) | N=beams (180), M=particles (~100) |
| Occupancy Grid | O(1) | Fixed 800×800 |
| A* Planning | O(E log V) | E=edges, V=cells (~640k max) |
| Frontier Detection | O(W²) | W=map width (800) |
| Motion Control | O(1) | Closed-form kinematics |

**Typical Performance** (on M1 MacBook):
- SLAM: ~5ms/update
- A*: ~50ms/plan (typical path: 200 nodes)
- Full loop: ~64ms (real-time capable)

### Map Accuracy

**Error Sources**:
1. LiDAR noise: ±2cm (Webots default)
2. Odometry drift: ~5% of distance traveled
3. SLAM error: ~10cm after 20m travel (no loop closure)

**Mitigation**:
- Complementary filter reduces drift by ~30%
- Frequent SLAM updates (every 64ms)
- Static environment assumption

---

## Testing & Validation

See [README.md Test Plan](README.md#test-plan) for full details.

**Key Metrics**:
- **Map Quality**: Compared to ground truth (Webots coordinates)
- **Path Efficiency**: Ratio of actual to optimal path length
- **Social Distance**: Min distance maintained during guidance
- **Smoothness**: Angular acceleration variance

---

## Known Limitations

1. **No Dynamic Obstacles**: Assumes static environment
2. **No Multi-Robot**: Single robot only
3. **No Loop Closure**: SLAM drifts on long missions
4. **Fixed Social Model**: Gaussian only (no learning)
5. **Tethered Human**: Not autonomous pedestrian
6. **2D Only**: No stairs, ramps, or 3D obstacles

---

## Future Architecture Enhancements

### Short-Term
- [ ] ROS bridge for hardware deployment
- [ ] Dynamic human tracking (sensor-based)
- [ ] Multi-goal mission planner

### Long-Term
- [ ] Deep RL for social navigation (replace hand-tuned Gaussian)
- [ ] 3D mapping (add camera SLAM)
- [ ] Multi-robot coordination
- [ ] Adaptive comfort profiles (learn from user feedback)

---

**Last Updated**: December 2024
