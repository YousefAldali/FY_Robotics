# Social Navigation Robot for Dementia Care

A Webots-based social robot simulation that implements SLAM, autonomous exploration, and socially-aware navigation to guide dementia patients in care home environments.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Attribution: What Was Implemented vs Pre-Built](#attribution-what-was-implemented-vs-pre-built)
3. [Repository Structure](#repository-structure)
4. [Dependencies](#dependencies)
5. [Installation Instructions](#installation-instructions)
6. [Running the Simulation](#running-the-simulation)
7. [Robot Operation Modes](#robot-operation-modes)
8. [Test Plan](#test-plan)
9. [Technical Implementation](#technical-implementation)
10. [Results & Data Collection](#results--data-collection)

---

## Project Overview

This project develops a socially-aware robot designed to assist dementia patients in care home environments. The robot:
- **Maps** the environment autonomously using LiDAR-based SLAM
- **Navigates** safely while avoiding obstacles
- **Respects personal space** using proxemic zones (intimate, personal, social)
- **Guides** patients to destinations (e.g., kitchen, bathroom, bedroom)
- **Adapts behavior** based on configurable social profiles (Conservative, Neutral, Open)

The simulation runs in Webots and implements a complete autonomous navigation stack from scratch.

---

## Attribution: What Was Implemented vs Pre-Built

This section clearly outlines what code was written by me versus what came from external libraries/packages.

### Custom Implementation (Written by Me)

All code in the following files was **entirely implemented from scratch** for this project:

#### 1. **Main Robot Controller** ([`social_robot.py`](controllers/social_robot/social_robot.py))
   - **What it does**: Main control loop integrating all subsystems
   - **Custom implementations**:
     - Multi-mode state machine (EXPLORE, GUIDE, MANUAL, TEST)
     - Keyboard-based mode switching
     - Human tracking and tethered following logic
     - Supervisor control for pedestrian positioning
     - Data logging system for experiments
     - Video frame generation for visualization
     - Exploration algorithm with frontier detection
     - Return-to-home navigation
     - Stuck detection and recovery mechanisms
     - Persistent obstacle blacklisting
     - Profile-based social navigation switching

#### 2. **SLAM Module** ([`lidar_module.py`](controllers/social_robot/lidar_module.py))
   - **What it does**: Wraps BreezySLAM library for Webots integration
   - **Custom implementations**:
     - `WebotsLidar` class: Adapts Webots LiDAR parameters to BreezySLAM format
     - `SlamBackend` class: Complete odometry integration with pose estimation
     - `process_lidar_ranges()`: Multi-layer LiDAR data extraction
     - `get_lidar_safety_info()`: Collision avoidance zone detection
     - Complementary filter integration for IMU + odometry fusion
   - **Pre-built**: Uses BreezySLAM's RMHC_SLAM algorithm (credited below)

#### 3. **Movement & Planning Module** ([`movement_module.py`](controllers/social_robot/movement_module.py))
   - **What it does**: All path planning, social navigation, and motion control
   - **Custom implementations**:
     - `PoseEstimator` class: Differential drive odometry from encoder readings
     - `OccupancyAStarPlanner` class: A* path planning with multi-cost terrain (hard obstacles + soft buffer zones)
     - **`SocialAStarPlanner` class**: Novel social A* algorithm with Gaussian cost fields for proxemics
       - Implements 3 social profiles (Conservative, Neutral, Open)
       - Dynamic human position integration
       - Phase-based cost modulation
     - `AStarPlanner` class: Standard grid-based A* (baseline comparison)
     - Frontier detection algorithm for autonomous exploration
     - `find_nearest_frontier()`: BFS-based frontier search with visited tracking
     - `find_nearest_free_cell()`: Obstacle-aware goal adjustment
     - `compute_wheel_velocities()`: Differential drive kinematics
     - `apply_complementary_filter()`: IMU drift correction
     - All helper functions (grid conversions, collision checks, etc.)

#### 4. **Pedestrian Controller** ([`pedestrian.py`](controllers/pedestrian/pedestrian.py))
   - **What it does**: Animates walking motion based on robot movement
   - **Custom modifications**:
     - Converted from active (physics-based) to **passive animation mode**
     - Distance-based animation triggering (not time-based)
     - Automatic rotation to face movement direction
     - Integration with supervisor control from `social_robot.py`
   - **Pre-built base**: Cyberbotics Pedestrian PROTO template (heavily modified)

#### 5. **Analysis Scripts** ([`testing/`](controllers/social_robot/testing/))
   - `compare_profiles.py`: Profile comparison visualization
   - `final_tradeoff_analysis.py`: Performance metrics analysis
   - **All analysis code is custom-written**

#### 6. **World Design** ([`worlds/dementia_care_world.wbt`](worlds/dementia_care_world.wbt))
   - **Custom-designed** care home layout:
     - Bedrooms, kitchen, living room, bathroom, hallways
     - Furniture placement (chairs, tables, beds, TV)
     - Lighting configuration
     - Robot spawn point and human target positioning
   - **Pre-built**: Webots PROTO objects (TiagoIron robot, Pedestrian, furniture models)

---

### Pre-Built Packages & Libraries Used

#### 1. **BreezySLAM** (SLAM Library)
   - **Author**: Simon D. Levy
   - **Repository**: https://github.com/simondlevy/BreezySLAM
   - **License**: LGPL v3.0
   - **What it provides**: Core SLAM algorithm (RMHC_SLAM)
   - **What I implemented on top**: Webots integration, odometry fusion, map processing

#### 2. **Webots** (Simulation Platform)
   - **Developer**: Cyberbotics Ltd.
   - **License**: Apache 2.0
   - **What it provides**: Physics engine, robot models, sensors
   - **What I implemented on top**: All robot control logic, world design, social behaviors

#### 3. **Standard Python Libraries**
   - `numpy`: Matrix operations, array handling
   - `matplotlib`: Visualization, map snapshots, video frames
   - `opencv-python` (cv2): Image dilation for obstacle inflation
   - `math`, `heapq`, `csv`: Standard utilities

---

## Repository Structure

```
FY_Robotics/
├── controllers/                  # Robot control code
│   ├── social_robot/            # Main robot controller
│   │   ├── social_robot.py      # [CUSTOM] Main control loop
│   │   ├── lidar_module.py      # [CUSTOM] SLAM integration
│   │   ├── movement_module.py   # [CUSTOM] Path planning & social navigation
│   │   ├── measure_map.py       # [CUSTOM] Map measurement utility
│   │   ├── testing/             # Analysis scripts
│   │   │   ├── compare_profiles.py
│   │   │   └── final_tradeoff_analysis.py
│   │   ├── map_snapshots/       # Generated map images
│   │   └── slam_frames/         # Video frame storage
│   ├── pedestrian/              # Human animation controller
│   │   └── pedestrian.py        # [MODIFIED] Passive animation mode
│   ├── mirror/                  # (Unused - can be removed)
│   └── television_switch_on/    # (Unused - can be removed)
├── worlds/                      # Webots simulation worlds
│   └── dementia_care_world.wbt  # [CUSTOM] Care home environment
├── README.md                    # This file
├── requirements.txt             # Python dependencies
└── .gitignore                   # Git exclusions

Generated Files (not in repo):
├── experiment_data_*.csv        # Logged trajectory data
├── slam_map_explore_complete.png
└── slam_exploration_video.mp4
```

---

## Dependencies

### Required Software

1. **Webots R2023b or later**
   - Download: https://cyberbotics.com/
   - Used for: Robot simulation, physics, sensors

2. **Python 3.11+**
   - Check version: `python3 --version`

### Python Packages

Install via pip:
```bash
pip install numpy matplotlib opencv-python
```

### BreezySLAM (Manual Installation)

BreezySLAM requires compilation from source:

```bash
git clone https://github.com/simondlevy/BreezySLAM
cd BreezySLAM/python
sudo python3 setup.py install
```

**Troubleshooting**:
- On macOS: May need Xcode Command Line Tools (`xcode-select --install`)
- On Linux: May need `build-essential` package
- On Windows: Requires Visual Studio Build Tools

---

## Installation Instructions

### Step 1: Clone the Repository

```bash
git clone <your-repo-url>
cd FY_Robotics
```

### Step 2: Install Dependencies

```bash
# Install Python packages
pip install -r requirements.txt

# Install BreezySLAM (see manual installation above)
git clone https://github.com/simondlevy/BreezySLAM
cd BreezySLAM/python
sudo python3 setup.py install
cd ../..
```

### Step 3: Install Webots

1. Download Webots from https://cyberbotics.com/
2. Install following platform-specific instructions
3. Add Webots to your PATH (optional but recommended)

### Step 4: Verify Installation

```bash
# Test BreezySLAM
python3 -c "from breezyslam.algorithms import RMHC_SLAM; print('BreezySLAM OK')"

# Test OpenCV
python3 -c "import cv2; print('OpenCV OK')"
```

---

## Running the Simulation

### Method 1: Launch from Webots GUI

1. Open Webots
2. **File → Open World**
3. Navigate to `worlds/dementia_care_world.wbt`
4. Click **Play** (▶️) button

### Method 2: Command Line (macOS/Linux)

```bash
webots worlds/dementia_care_world.wbt
```

### Method 3: Command Line (Windows)

```cmd
"C:\Program Files\Webots\msys64\mingw64\bin\webots.exe" worlds\dementia_care_world.wbt
```

### Important: Enable Supervisor Mode

**Critical Step** - The robot **MUST** have supervisor privileges to control the human:

1. In Webots Scene Tree, expand the robot node (likely named `TiagoIron` or `social_robot`)
2. Find the `supervisor` field
3. Change value from `FALSE` to `TRUE`
4. **Save the world** (Ctrl+S / Cmd+S)
5. Restart the simulation

**If you see errors about "getRoot() failed"**, this is the issue.

---

## Robot Operation Modes

The robot has 4 operational modes controlled via keyboard:

### **EXPLORE Mode** (Press `E`)
- **Purpose**: Autonomous mapping and exploration
- **Behavior**:
  - Robot navigates to unexplored frontiers
  - Builds SLAM map from LiDAR data
  - Marks visited regions
  - Returns to start point after time limit (default: 1000 steps)
- **Use case**: Initial environment mapping

### **GUIDE Mode** (Press `G`)
- **Purpose**: Lead a person to the kitchen
- **Behavior**:
  - Drives to human position
  - Plans path to kitchen using social A* (respects proxemics)
  - Waits if human falls behind (>2.5m)
  - Human animates walking motion via tether
- **Configuration**:
  - **Line 118**: Change `NAV_CONDITION` to `"SOCIAL"` or `"BASELINE"`
  - **Line 122**: Change `current_profile` to `"Conservative"`, `"Neutral"`, or `"Open"`

### **MANUAL Mode** (Press `M`)
- **Purpose**: Direct teleoperation
- **Controls**:
  - `W` - Move forward
  - `S` - Move backward
  - `A` - Turn left
  - `D` - Turn right
  - `X` - Stop
- **Use case**: Manual positioning, debugging

### **TEST Mode** (Press `T`)
- **Status**: Not yet implemented
- **Intended use**: Automated test sequences

### Additional Commands
- **`R`**: Restart exploration (when in DONE state)

---

## Test Plan

The project implements the following test matrix (details in original README):

| Test ID | Category | Purpose |
|---------|----------|---------|
| T1 | Mapping (SLAM) | Validate LiDAR map accuracy |
| T2 | Localisation | Check position estimate stability |
| T3 | Navigation | Point-to-point path following |
| T4 | Social Distance | Verify proxemic zone avoidance |
| T5 | Guidance Behaviour | Person-following distance maintenance |
| T6 | Comfort Profiles | Profile switching validation |
| T7 | Trajectory Smoothness | Motion quality analysis |
| T8 | Obstacle Avoidance | Dynamic obstacle handling |
| T9 | Ethical/Safe Behaviour | Dementia-safe motion patterns |
| T10 | Full System Test | End-to-end integration |

---

## Technical Implementation

### Key Algorithms

#### 1. **SLAM** (Simultaneous Localisation and Mapping)
- **Library**: BreezySLAM RMHC_SLAM
- **Sensors**: LiDAR (primary) + Wheel encoders + IMU (optional)
- **Integration**: Complementary filter fuses odometry with IMU to reduce drift
- **Output**: 800x800 occupancy grid (20m x 20m environment)

#### 2. **Social A* Path Planning**
Novel algorithm extending standard A* with social cost fields:

```python
# Standard cost: terrain difficulty
base_cost = 1.0 (free space) or 8.0 (near walls)

# Social cost: Gaussian field around human
social_cost = amplitude * exp(-distance² / (2 * sigma²))

# Total cost
step_cost = base_cost + social_cost
```

**Profiles**:
- **Conservative**: sigma=15, amplitude=20 (large, expensive bubble)
- **Neutral**: sigma=10, amplitude=15 (balanced)
- **Open**: sigma=8, amplitude=10 (small, cheap bubble)

#### 3. **Frontier-Based Exploration**
- Detects boundaries between known free space and unknown space
- Uses BFS to find nearest unvisited frontier
- Marks 40-cell radius as visited when reached (prevents oscillation)

#### 4. **Obstacle Avoidance**
- **Safety zones**: 0.18m emergency stop, 0.45m avoidance trigger
- **Recovery**: Backup → Turn 90° → Move forward → Replan
- **Persistent blacklisting**: Remembers collision points across replans

---

## Results & Data Collection

### Logged Data

The robot automatically logs to CSV files:

**Filename**: `experiment_data_<profile>.csv`
- `baseline.csv` - Standard navigation (no social costs)
- `conservative.csv` - Conservative profile run
- `neutral.csv` - Neutral profile run
- `open.csv` - Open profile run

**Columns**:
```
Time, X, Y, Theta, DistHuman
```

### Generated Outputs

1. **Map Snapshots** (`map_snapshots/map_snapshot_XXXX.png`)
   - Saved every 30 seconds during exploration

2. **Video Frames** (`slam_frames/frame_XXXXX.png`)
   - Includes path visualization
   - Social zones overlaid (red = intimate, green = personal)

3. **Final Map** (`slam_map_explore_complete.png`)
   - High-resolution map saved at completion

4. **Video** (`slam_exploration_video.mp4`)
   - Compiled from frames at 10 FPS

---

## Configuration

Key parameters in [`social_robot.py`](controllers/social_robot/social_robot.py):

```python
# Line 30: Goal tolerance
GOAL_TOLERANCE = 0.8  # meters

# Line 31-32: Output settings
SAVE_FINAL_MAP = True
SAVE_VIDEO_FRAMES = True

# Line 35: Exploration time limit
MAX_EXPLORATION_TIME = 1000  # timesteps (~16 minutes at 64ms/step)

# Line 118: Navigation mode
NAV_CONDITION = "BASELINE"  # or "SOCIAL"

# Line 122: Social profile
current_profile = "Open"  # or "Conservative", "Neutral"
```

---

## Known Issues & Limitations

1. **Human node requires supervisor mode**: Must manually enable in Webots GUI
2. **BreezySLAM installation**: Requires compilation, can be tricky on Windows
3. **Video generation**: Requires sufficient disk space (~1GB for long runs)
4. **Stuck detection**: May trigger false positives in tight spaces
5. **Social costs**: Tuned for 20m x 20m environment (may need adjustment for other scales)

---

## Future Improvements

- [ ] Implement TEST mode with automated test sequences
- [ ] Add ROS bridge for real robot deployment
- [ ] Implement dynamic human tracking (vs. tethered following)
- [ ] Add voice interaction for guidance confirmation
- [ ] Implement multi-goal planning (tour multiple rooms)
- [ ] Add fatigue detection (slow down if patient struggles)

---

## Credits & License

**Author**: [Your Name]
**Institution**: [Your University]
**Course**: Final Year Robotics Project
**Year**: 2024-2025

**External Libraries**:
- BreezySLAM: Simon D. Levy (LGPL v3.0)
- Webots: Cyberbotics Ltd. (Apache 2.0)

**License**: MIT (or specify your chosen license)

---

## Contact

For questions or issues:
- **Email**: [your.email@university.edu]
- **GitHub Issues**: [link-to-issues-page]

---

## Acknowledgments

Special thanks to:
- Dr. [Supervisor Name] for project guidance
- Cyberbotics for Webots platform
- Simon D. Levy for BreezySLAM library
