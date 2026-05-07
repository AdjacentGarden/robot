# Frontier Exploration Package

Autonomous 2D frontier-based exploration for ROS2 Humble. Uses a custom
lightweight navigator (A* + pure-pursuit) instead of Nav2.

---

## Quick Start

```bash
# Build
cd ~/order_exploration
colcon build --packages-select exploration
source install/setup.bash

# Desktop test (simulated environment)
ros2 launch exploration exploration_test_rviz.launch.py seed:=42

# Real robot deployment
ros2 launch exploration exploration_real_robot.launch.py
```

The robot will:
1. Spin 360° to build an initial map
2. Detect frontiers (boundaries between known and unknown space)
3. Navigate to the best frontier using A* + pure-pursuit
4. Re-evaluate and switch targets if a better frontier appears
5. Repeat until no frontiers remain
6. Return to the start position and save the map

---

## Architecture

```
                    ┌──────────────────┐
                    │  sim_environment │ (test only)
                    │  or real drivers │
                    └────────┬─────────┘
                             │ /map, /scan, TF
                    ┌────────▼─────────┐
                    │ custom_navigator │
                    │  A* + Pure-Pursuit│
                    └────────┬─────────┘
               NavigateToPose│action     /cmd_vel
                    ┌────────▼─────────┐      │
                    │frontier_explorer │      │
                    │  State Machine   │      │
                    └──────────────────┘      ▼ robot
```

### Key Components

| Component | File | Description |
|-----------|------|-------------|
| **Custom Navigator** | `custom_navigator.py` | A* global planner with distance-weighted cost + pure-pursuit local planner. Replaces all Nav2 nodes. |
| **Frontier Explorer** | `frontier_explorer.py` | Frontier detection via connected components, score-based goal selection, smart replanning. |
| **Sim Environment** | `sim_environment.py` | 2D random map generator with raycasting LiDAR and SLAM-like map reveal. Test only. |

---

## Launch Files

| File | Use Case | Components |
|------|----------|------------|
| `exploration_test_rviz.launch.py` | Desktop testing | sim + navigator + explorer + RViz |
| `exploration_real_robot.launch.py` | Real robot | drivers + SLAM + navigator + explorer + RViz(optional) |

See `docs/LAUNCH_GUIDE.md` for detailed comparison.

---

## RViz Visualization

| Display | Topic | Color |
|---------|-------|-------|
| SLAM Map | `/map` | Grey/white/black |
| LiDAR | `/scan` | Red points |
| Frontiers | `/exploration/frontiers` | Blue spheres |
| Current Goal | `/exploration/frontiers` (id=9999) | Yellow sphere |
| Global Path | `/navigation/path` | Green line |
| Local Goal | `/navigation/local_goal` | Orange sphere |
| Local Trajectory | `/navigation/local_traj` | Cyan line |

---

## Configuration

### Navigator (`config/custom_nav_params.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_vel_x` | 0.22 | Max forward speed (m/s) |
| `max_vel_theta` | 0.7 | Max angular speed (rad/s) |
| `inflation_mult` | 1.5 | A* inflation = mult × robot_radius |
| `proximity_weight` | 3.0 | Cost penalty for paths near walls |
| `goal_tolerance` | 0.25 | Goal reach distance (m) |
| `replan_interval` | 5.0 | A* recomputation interval (s) |

### Explorer (`config/explore_params.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gain_weight` | 1.5 | Prefer larger frontiers |
| `cost_weight` | 1.0 | Penalize distant frontiers |
| `replan_interval` | 3.0 | Frontier re-evaluation interval (s) |
| `goal_timeout` | 90.0 | Give up on goal after (s) |

---

## File Structure

```
exploration/
├── exploration/              # Python nodes
│   ├── custom_navigator.py   # A* + pure-pursuit navigator
│   ├── frontier_explorer.py  # Frontier detection + state machine
│   └── sim_environment.py    # 2D test simulator
├── config/
│   ├── custom_nav_params.yaml
│   └── explore_params.yaml
├── launch/
│   ├── exploration_test_rviz.launch.py
│   └── exploration_real_robot.launch.py
├── rviz/
│   └── exploration.rviz
├── docs/
│   ├── DEVELOPMENT_LOG.md    # Problem history and solutions
│   └── LAUNCH_GUIDE.md       # Launch file comparison
├── video/
│   └── record_exploration.sh # Screen recording script
└── README.md
```

---

## Performance (seed=42, 10m×10m map)

| Metric | Result |
|--------|--------|
| Exploration time | ~170s |
| Goals reached | 9-11 |
| Stuck events | 0 |
| A* failures | 0 (fallback works) |
| Return home | Success |
| Map saved | Yes |
