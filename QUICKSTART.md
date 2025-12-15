# Quick Start Guide

Get your social navigation robot running in 5 minutes!

---

## Prerequisites Check

Before starting, verify you have:
- ✅ Webots R2023b or later installed
- ✅ Python 3.11+ installed
- ✅ 5GB free disk space (for generated data)

---

## Installation (5 Steps)

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd FY_Robotics
```

### 2. Install Python Dependencies
```bash
pip install numpy matplotlib opencv-python
```

### 3. Install BreezySLAM
```bash
git clone https://github.com/simondlevy/BreezySLAM
cd BreezySLAM/python
sudo python3 setup.py install
cd ../..
```

### 4. Open in Webots
```bash
# macOS/Linux
webots worlds/dementia_care_world.wbt

# Windows
"C:\Program Files\Webots\msys64\mingw64\bin\webots.exe" worlds\dementia_care_world.wbt
```

### 5. Enable Supervisor Mode
In Webots:
1. Open Scene Tree (left panel)
2. Find the robot node (e.g., `TiagoIron`)
3. Expand → Find `supervisor` field
4. Change from `FALSE` to `TRUE`
5. Save (Ctrl+S / Cmd+S)
6. Restart simulation

---

## Running Your First Simulation

### Scenario 1: Autonomous Mapping

**Goal**: Robot explores and builds a map

**Steps**:
1. Press ▶️ (Play) in Webots
2. Wait for controller to initialize (~5 seconds)
3. Press `E` key (EXPLORE mode)
4. Watch robot explore for ~5 minutes
5. Map saves automatically to `slam_map_explore_complete.png`

**What to expect**:
- Robot drives to unexplored areas
- Map updates in real-time (check `map_snapshots/`)
- Returns to start after 1000 timesteps

---

### Scenario 2: Guide Human to Kitchen

**Goal**: Robot leads person to kitchen using social navigation

**Steps**:
1. Open [`social_robot.py`](controllers/social_robot/social_robot.py)
2. Set **Line 118**: `NAV_CONDITION = "SOCIAL"`
3. Set **Line 122**: `current_profile = "Neutral"`
4. Save file
5. In Webots, press ▶️ (Play)
6. Press `G` key (GUIDE mode)
7. Watch robot:
   - Drive to human
   - Plan path avoiding human's personal space
   - Lead human to kitchen (human follows via tether)

**What to expect**:
- Robot maintains 1.2m+ distance from human
- Path curves around human's position
- Human animates walking motion
- Data logged to `experiment_data_neutral.csv`

---

### Scenario 3: Manual Teleoperation

**Goal**: Drive robot manually to position it

**Steps**:
1. Press ▶️ (Play) in Webots
2. Press `M` key (MANUAL mode)
3. Use keyboard:
   - `W` = Forward
   - `S` = Backward
   - `A` = Turn left
   - `D` = Turn right
   - `X` = Stop

**Use case**: Position robot before experiment or debug stuck situations

---

## Experiment Workflow

### Running Social Profile Comparison

**Goal**: Compare Conservative vs Neutral vs Open profiles

**Steps**:

#### Run 1: Conservative Profile
1. Edit [`social_robot.py:122`](controllers/social_robot/social_robot.py#L122)
   ```python
   current_profile = "Conservative"
   ```
2. Run simulation (`G` for GUIDE mode)
3. Wait until "Destination Reached" appears
4. Data saved to `experiment_data_conservative.csv`

#### Run 2: Neutral Profile
1. Edit line 122:
   ```python
   current_profile = "Neutral"
   ```
2. Restart simulation (Ctrl+Shift+R)
3. Press `G` again
4. Data saved to `experiment_data_neutral.csv`

#### Run 3: Open Profile
1. Edit line 122:
   ```python
   current_profile = "Open"
   ```
2. Restart simulation
3. Press `G`
4. Data saved to `experiment_data_open.csv`

#### Baseline (No Social Navigation)
1. Edit line 118:
   ```python
   NAV_CONDITION = "BASELINE"
   ```
2. Restart simulation
3. Press `G`
4. Data saved to `experiment_data_baseline.csv`

### Analyze Results
```bash
cd controllers/social_robot/testing
python3 compare_profiles.py
```

**Output**: Comparison plots showing distance profiles

---

## Viewing Results

### Generated Files

| File | Location | Description |
|------|----------|-------------|
| Final map | `slam_map_explore_complete.png` | High-res occupancy grid |
| Map snapshots | `map_snapshots/map_snapshot_XXXX.png` | Periodic saves |
| Video frames | `slam_frames/frame_XXXXX.png` | With path overlay |
| Experiment data | `experiment_data_<profile>.csv` | Trajectory logs |
| Video | `slam_exploration_video.mp4` | Compiled animation |

### Quick Data Check
```bash
# View logged trajectory
head -20 experiment_data_neutral.csv

# Expected format:
# Time, X, Y, Theta, DistHuman
# 0.064, 10.01, 10.00, 0.000, 5.18
# 0.128, 10.01, 10.00, 0.002, 5.18
# ...
```

---

## Troubleshooting

### Problem: "getRoot() failed" error

**Cause**: Supervisor mode not enabled

**Fix**:
1. Scene Tree → Robot node
2. `supervisor` field → `TRUE`
3. Save world
4. Restart simulation

---

### Problem: "BreezySLAM not found"

**Cause**: Installation incomplete

**Fix**:
```bash
cd BreezySLAM/python
sudo python3 setup.py install

# Verify
python3 -c "from breezyslam.algorithms import RMHC_SLAM; print('OK')"
```

---

### Problem: Robot doesn't move after pressing `E`

**Possible causes**:
1. Wrong keyboard focus → Click Webots 3D viewport
2. Simulation paused → Press ▶️
3. Controller crashed → Check Console output (red errors)

**Fix**:
- Check Console tab in Webots (bottom panel)
- Look for Python errors
- Restart simulation (Ctrl+Shift+R)

---

### Problem: Human doesn't move in GUIDE mode

**Cause**: Human node not found

**Fix**:
1. Check world file has human named `HUMAN_TARGET` or `Pedestrian`
2. Verify supervisor mode enabled
3. Check Console for "Human Target Node FOUND" message

---

### Problem: Map is all gray/blank

**Cause**: SLAM not receiving data

**Check**:
1. LiDAR enabled? (Console should show ranges)
2. Robot actually moving? (Try MANUAL mode first)
3. Encoders working? (Check odometry output)

---

## Tips & Best Practices

### Performance Optimization
- **Disable video frames** if running slow:
  ```python
  # social_robot.py line 32
  SAVE_VIDEO_FRAMES = False
  ```
- Close other applications
- Use "Fast" mode in Webots (top toolbar)

### Clean Slate Between Runs
```bash
# Delete generated data
rm experiment_data_*.csv
rm slam_map_*.png
rm -rf slam_frames/ map_snapshots/
```

### Experiment Log Organization
```bash
# Rename data after each run
mv experiment_data_neutral.csv results/run1_neutral.csv
mv slam_map_explore_complete.png results/run1_map.png
```

---

## Next Steps

Once comfortable with basics:

1. **Tune Parameters** ([`social_robot.py`](controllers/social_robot/social_robot.py))
   - Line 30: `GOAL_TOLERANCE` (how close to waypoint)
   - Line 35: `MAX_EXPLORATION_TIME` (how long to explore)
   - Line 389: `HUMAN_STOP_DIST` (personal space threshold)

2. **Modify Social Profiles** ([`movement_module.py:131-139`](controllers/social_robot/movement_module.py#L131-L139))
   - Adjust `sigma` (bubble size)
   - Adjust `amplitude` (avoidance strength)

3. **Design New World**
   - Duplicate `dementia_care_world.wbt`
   - Add/remove rooms
   - Update `ROOM_GOALS` in [`movement_module.py:258`](controllers/social_robot/movement_module.py#L258)

4. **Run Full Test Suite**
   - See [README Test Plan](README.md#test-plan)
   - Implement missing tests (T1-T10)

---

## Cheat Sheet

### Keyboard Commands
| Key | Mode | Action |
|-----|------|--------|
| `E` | Any | Switch to EXPLORE mode |
| `G` | Any | Switch to GUIDE mode |
| `M` | Any | Switch to MANUAL mode |
| `T` | Any | Switch to TEST mode (not implemented) |
| `R` | DONE | Restart exploration |
| `W` | MANUAL | Move forward |
| `S` | MANUAL | Move backward |
| `A` | MANUAL | Turn left |
| `D` | MANUAL | Turn right |
| `X` | MANUAL | Stop |

### File Locations
```
Key Config Files:
  controllers/social_robot/social_robot.py      # Main controller
  controllers/social_robot/movement_module.py   # Planning algorithms
  controllers/social_robot/lidar_module.py      # SLAM integration
  worlds/dementia_care_world.wbt                # Simulation world

Key Config Lines:
  social_robot.py:118   NAV_CONDITION (SOCIAL/BASELINE)
  social_robot.py:122   current_profile (Conservative/Neutral/Open)
  social_robot.py:35    MAX_EXPLORATION_TIME
  movement_module.py:258 ROOM_GOALS (destinations)
```

---

**Need Help?**
- Check [README.md](README.md) for full documentation
- Check [ARCHITECTURE.md](ARCHITECTURE.md) for technical details
- Open GitHub issue for bugs

**Ready to dive deeper?**
- Read full [Technical Implementation](README.md#technical-implementation)
- Review [System Architecture](ARCHITECTURE.md)
- Explore testing scripts in `controllers/social_robot/testing/`
