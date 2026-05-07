# Start-Pose Localization Test

这个目录专门验证“已知地图导航启动时，系统能否把理论起点修正到真实起点附近”。

这个测试不走真实机器人 launch，而是走纯仿真：

- 静态地图由 `map_server` 提供
- 机器人真实位置由 `sim_environment` 放在 `actual_start_*`
- 系统理论起点由 `initial_pose_*` 提供
- `auto_initial_pose_estimator` 在理论起点附近搜索并修正 AMCL 初始位姿

## Build

```bash
cd /home/yang/vibe_coding/progress/order/exploration
colcon build --packages-select exploration
source install/setup.bash
```

## Launch

下面这个例子里，理论起点和真实起点故意保留一个小偏差：

```bash
ros2 launch exploration localization_start_pose_test.launch.py \
  map:=/home/yang/vibe_coding/progress/order/exploration/src/exploration/map/exploration_map.yaml \
  initial_pose_x:=0.0 \
  initial_pose_y:=0.0 \
  initial_pose_yaw:=0.0 \
  actual_start_x:=0.08 \
  actual_start_y:=-0.04 \
  actual_start_yaw:=0.18
```

## Parameter Meaning

- `initial_pose_*`: 你告诉系统的理论起点
- `actual_start_*`: 仿真里机器人真实落点

## RViz Check

1. 启动后先不要发导航目标，先看终端日志。
2. 重点看两行日志：
   - `Local Match Result`
   - `Estimated start offset from configured pose`
3. 在 RViz 里确认激光点云和静态地图墙体基本贴合。
4. 用 `2D Goal Pose` 在近处点一个目标，再在远处点一个目标，看路径是否从真实当前位置出发。

## Suggested Cases

- `actual_start_x/y` 偏离 `initial_pose_x/y` 2 到 5 cm
- `actual_start_x/y` 偏离 `initial_pose_x/y` 10 到 15 cm
- `actual_start_yaw` 偏离 `initial_pose_yaw` 10 到 20 deg

如果近距离场景稳定、远一点场景不稳定，后续重点调这些参数：

- `search_x_range`
- `search_y_range`
- `search_yaw_range_deg`
- `min_match_ratio`
