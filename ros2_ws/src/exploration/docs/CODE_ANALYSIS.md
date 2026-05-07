# Exploration Package Code Analysis

## 1. Package Purpose and Overall Architecture

这个 ROS2 Humble 软件包实现的是一套“未知二维环境自主探索 + 建图 + 后续导航”的轻量方案，核心思路不是直接使用完整 Nav2 栈，而是把系统拆成三个节点：

1. `sim_environment`
   - 仅用于仿真。
   - 负责生成随机室内地图、模拟激光雷达、积分机器人位姿、发布 TF 和增量地图。
2. `custom_navigator`
   - 作为自定义导航器，替代完整 Nav2。
   - 内部包含 A* 全局规划、基于 pure-pursuit 的局部跟踪、碰撞前瞻检查、卡住恢复。
   - 提供 `navigate_to_pose` action server。
3. `frontier_explorer`
   - 负责 frontier 检测、目标点评分、探索状态机调度。
   - 作为 `navigate_to_pose` 的 action client 调用 `custom_navigator`。
   - 探索结束后支持回原点、保存地图、进入空闲导航模式。

整体数据流如下：

```text
仿真模式:
sim_environment
  -> /map
  -> /scan
  -> TF(map -> odom -> base_footprint -> lidar_frame)
  -> /odom

实机模式:
robot drivers + SLAM Toolbox
  -> /scan
  -> /map
  -> TF(map -> odom -> base_footprint)

frontier_explorer
  -> 调用 navigate_to_pose action
  -> 发布 /exploration/frontiers, /exploration/status
  -> 初始扫描阶段直接发布 /cmd_vel

custom_navigator
  <- /map, /scan, TF
  <- navigate_to_pose goal
  -> /cmd_vel
  -> /navigation/path
  -> /navigation/local_goal
  -> /navigation/local_traj
  -> /navigation/status
```

---

## 2. Core Code Logic

## 2.1 `sim_environment.py`

这个节点是一个不依赖 Gazebo 的最小 2D 探索仿真器，目标是快速验证探索算法，而不是高保真动力学仿真。

### 2.1.1 地图生成逻辑

启动后会根据参数生成一个离散栅格地图：

- 地图尺寸由 `map_width`、`map_height`、`resolution` 决定。
- 先生成四周外墙。
- 再随机插入若干横向或纵向隔断墙，墙上随机留门洞。
- 最后随机加入若干箱体障碍物，模拟家具。
- 起始位置附近会强制清空，保证机器人初始可用。

这样得到的是一张 ground-truth 地图，保存在 `self.gt_map`。同时节点维护一张 `self.revealed` 掩码，记录“哪些格子已经被激光扫描发现”。

### 2.1.2 运动学更新

节点订阅 `/cmd_vel` 和 `/controller/cmd_vel`，把线速度和角速度存到内部状态。随后定时器以 50Hz 做以下工作：

1. 使用简单差分积分更新 `x/y/yaw`。
2. 用新位置做碰撞检测。
3. 如果新位置落在障碍物内，则拒绝平移，仅保留朝向变化。

这意味着仿真运动模型比较简化，但足够验证：

- 路径是否合理；
- 跟踪器是否稳定；
- 探索策略是否能覆盖未知区域。

### 2.1.3 TF / Odom 发布

仿真节点会持续发布：

- `map -> odom`
  - 恒等变换，表示仿真里没有里程计漂移。
- `odom -> base_footprint`
  - 机器人当前位姿。
- `base_footprint -> lidar_frame`
  - 恒等变换。

同时发布 `/odom`，便于调试和可视化。

### 2.1.4 激光扫描模拟

激光扫描通过 raycasting 完成：

- 以 `lidar_frame` 为坐标系；
- 对每个角度沿射线逐步前进；
- 遇到障碍物则记录距离；
- 沿途经过的网格被标记为 `revealed=True`。

这一步不仅生成 `/scan`，还会直接推进“地图已知区域”的扩大，因此系统虽然没有真正运行 SLAM，也具备“扫描越多，地图越完整”的效果。

### 2.1.5 增量地图发布

节点不会直接发布 ground-truth 给探索器，而是构造一张“SLAM 风格”的地图：

- 未扫描区域发布为 `-1`；
- 已扫描区域填入真实占据值；
- 以 `/map` 形式发布。

这样 `frontier_explorer` 在仿真和实机里看到的输入语义是一致的：都是“部分已知、部分未知”的 OccupancyGrid。

### 2.1.6 仿真辅助接口

额外还支持两个接口：

- `/sim/reset`
  - 重新生成随机地图并把机器人放回起点。
  - 用于多轮测试。
- `/sim/teleport`
  - 立即把机器人传送到指定 Pose，并马上更新扫描和地图。
  - 主要用于调试。

---

## 2.2 `custom_navigator.py`

这个节点是整个包里最关键的导航执行器。它把“规划”和“控制”集中在一个轻量 action server 里。

### 2.2.1 输入输出角色

输入：

- `/map`
- `/scan`
- TF: `map -> base_footprint`
- action goal: `navigate_to_pose`

输出：

- `/cmd_vel`
- `/navigation/path`
- `/navigation/local_goal`
- `/navigation/local_traj`
- `/navigation/status`

### 2.2.2 核心执行流程

当收到一个 `NavigateToPose` goal 后，执行流程是：

1. 读取当前 TF 位姿。
2. 基于 `/map` 运行 A* 全局规划。
3. 发布全局路径到 `/navigation/path`。
4. 进入控制循环：
   - 定期检查是否到达目标；
   - 定期重规划；
   - 使用 pure-pursuit 选择局部跟踪点；
   - 对速度指令做前向碰撞预测；
   - 发送 `/cmd_vel`；
   - 持续回传 action feedback。
5. 若机器人长时间无位移进展，则触发恢复动作：
   - 后退；
   - 转向脱困；
   - 重新规划。

### 2.2.3 距离场代价图

节点在收到 `/map` 后，会把占据栅格转成 distance field：

- 障碍物格子为 0；
- 自由区每个点记录到最近障碍物的距离；
- 使用 `cv2.distanceTransform` 加速计算。

这个距离场后续有两个作用：

1. 给 A* 增加“靠墙惩罚”；
2. 给局部控制器做速度安全检查。

因此它不是传统的 local costmap，但实现了类似的“离障碍越近，代价越高”的效果。

### 2.2.4 A* 全局规划

这个 A* 不是简单最短路径，而是“硬膨胀 + 靠墙惩罚”的版本。

#### 1. 硬膨胀

先根据 `robot_radius * inflation_mult` 对障碍物做膨胀：

- 被膨胀后的格子直接不可通行；
- 作用是预留安全边界，避免路径贴墙。

#### 2. 距离惩罚

在搜索邻接节点时，如果该点距离障碍物太近，会附加额外代价：

- 距离越近，代价越高；
- 路径会更偏向通道中心。

#### 3. 失败回退

如果在主膨胀半径下找不到路，节点会自动尝试更小的膨胀半径再规划一次。

这个设计解决了两个常见问题：

- 窄通道因为安全边界太大导致完全不可达；
- 但默认情况下路径仍然尽量远离墙体。

### 2.2.5 Pure-Pursuit 局部跟踪

局部控制逻辑可以概括为：

1. 在全局路径上找到离机器人最近的路径点。
2. 沿路径累计距离，找到 `lookahead_dist` 对应的局部目标点。
3. 计算机器人朝向和局部目标点之间的角度误差。
4. 根据误差输出角速度，根据对齐程度输出线速度。

这个 pure-pursuit 还有几个增强：

- 离目标很近时进入“近目标模式”，直接朝最终目标点逼近，避免绕圈。
- 如果默认 lookahead 被判定不安全，会尝试更短 lookahead。
- 如果前进速度不安全，会降速，再降速，最后尝试纯旋转。
- 如果仍然不安全，会做一组角速度搜索，找一个能脱困的转向动作。

另外，针对实机底盘的硬件限制，又增加了几层约束：

- 当航向误差较大时，控制器会主动压低线速度；
- 当航向误差接近或超过 90° 时，只允许很小的爬行速度，避免急转打滑；
- 控制输出会参考上一拍速度做斜率限制，而不是瞬间跳变；
- 当角速度较大时，还会按侧向加速度上限进一步压低线速度。

### 2.2.6 前向碰撞检查

每次即将发送 `(v, w)` 之前，不是直接信任 pure-pursuit，而是先在未来 `safety_horizon` 时间内做轨迹离散仿真：

- 若仿真轨迹最小障碍距离小于阈值，则当前速度组合不允许使用；
- 控制器会换更慢的速度，或者只转不走；
- 保证输出的 `/cmd_vel` 不会明显穿墙。

### 2.2.7 路径失败补救

实机中有时会出现“目标点看起来合理，但规划器突然找不到路径”的问题。当前版本在 A* 层面增加了两级补救：

1. 目标吸附
   - 若目标点落在未知边缘、障碍边缘或不可达栅格附近，会在周围搜索最近的安全自由点。
   - 若找到，就把规划目标吸附到这个附近可达点，再重新规划。

2. 多档 inflation 重试
   - 不再只尝试一档安全膨胀。
   - 现在会从默认膨胀逐步降到更保守的回退档位，尽量减少“窄路刚好被堵死”的误判。

### 2.2.8 卡住恢复

如果一段时间内：

- 平移位移没有超过 `stuck_radius`；
- 且角度变化也不足；

则判定为 stuck。恢复动作固定为：

1. 倒车；
2. 低速曲线转向；
3. 重新规划。

恢复次数超过 `max_recoveries` 后，当前 goal 会被终止。

---

## 2.3 `frontier_explorer.py`

这个节点负责“探索策略”，也就是决定机器人下一步应该去哪里。

### 2.3.1 状态机

状态机包含以下阶段：

- `WAITING`
  - 等待 `/map`、TF 和 `navigate_to_pose` action server 就绪。
- `INITIAL_SPIN`
  - 原地旋转一段时间，先建立初始局部地图。
- `EXPLORING`
  - 从当前地图中提取 frontier 并挑选最佳目标。
- `NAVIGATING`
  - 等待 `custom_navigator` 执行 action。
- `RETURNING`
  - 探索完成后返回起始点。
- `SAVING_MAP`
  - 调用地图保存命令。
- `RANDOM_NAV`
  - 在已知可行区域内随机导航若干点。
- `RESETTING`
  - 仿真多轮测试时重置地图。
- `IDLE`
  - 探索彻底结束，等待外部下发 `/exploration/goal`。

### 2.3.2 初始自旋

探索不是一启动就找 frontier，而是先原地转一圈附近：

- 直接向 `/cmd_vel` 发布角速度；
- 让 SLAM 或仿真地图先得到一圈局部观测；
- 这样后续 frontier 提取才不会因为地图过小而误判。

### 2.3.3 Frontier 提取逻辑

frontier 的定义是：

- 当前栅格是 free；
- 并且它 8 邻域内至少有一个 unknown。

节点先找出所有满足条件的 frontier 栅格，然后：

1. 对 frontier 栅格做聚类；
2. 过滤掉小于 `min_frontier_size` 的簇；
3. 计算每个簇的中心；
4. 过滤距离机器人过近或者靠近黑名单位置的候选点。

### 2.3.4 目标点评分

每个 frontier 会被打一个分，大体形式是：

```text
score = gain_weight * 信息增益 - cost_weight * 距离代价
```

其中：

- frontier cluster 越大，说明潜在未知区域越大，增益越高；
- 距离机器人越远，代价越高；
- 最后选择得分最高的 frontier。

这样比“只选最近 frontier”更合理，因为它兼顾了探索效率和路程开销。

### 2.3.5 导航期间重评估

机器人在赶往某个 frontier 的过程中，地图仍然在变化，所以 explorer 不会一条路走到黑，而是持续检查：

1. 当前目标附近是否已经不再是 frontier；
2. 是否出现了明显更优的新 frontier；
3. 当前 goal 是否超时或被拒绝。

如果满足切换条件，就取消当前 goal，回到 `EXPLORING` 重新选点。

### 2.3.6 黑名单机制

对失败目标会进入黑名单，避免反复尝试同一个不可达点。

黑名单的意义：

- 防止 A* 对不可达 frontier 无限重试；
- 给探索过程增加“失败记忆”；
- 但也不是永久屏蔽，到时间后会自动清空，避免地图变化后仍然错失可达区域。

### 2.3.7 失败后的补救探索

对于实机里偶发出现的“这次没规划出来，不代表永远不可达”，当前版本在 explorer 侧增加了 rescue goals：

1. 普通 frontier goal 失败后，会在机器人当前位姿附近生成几组短距离 detour 点；
2. 这些点要求落在已知自由区内，目的是让机器人先挪一步、换一个观测角度；
3. rescue goal 会优先执行，执行完后再回到正常 frontier 选择流程；
4. rescue goal 自身失败时不会污染普通 frontier 的 blacklist。

这个设计适合处理地图局部断裂、观测还不完整、或者底盘当前姿态不利于继续前进的情况。

### 2.3.8 完成判定

当连续多次都检测不到 frontier 时，不会立刻宣布结束，而是再检查：

- 当前地图自由区是否已经足够大；
- 是否只是由于起始地图太小造成的误判。

如果自由区太小，会重新进入 `INITIAL_SPIN`；
如果自由区足够大，则认为探索完成。

### 2.3.9 结束后的行为

探索完成后节点可以：

1. 回到起点；
2. 调用地图保存；
3. 生成若干随机目标点做额外导航验证；
4. 所有 cycle 完成后进入 `IDLE`；
5. 在 `IDLE` 状态下支持外部通过 `/exploration/goal` 再发导航目标。

---

## 3. Simulation Launch Analysis

## 3.1 `exploration_test_rviz.launch.py`

这套 launch 主要用于电脑本地仿真验证，启动顺序如下：

1. 立即启动 `sim_environment`
2. 立即启动 `rviz2`
3. 延迟 3 秒启动 `custom_navigator`
4. 延迟 8 秒启动 `frontier_explorer`

这样做的原因是：

- 先让仿真节点把 `/map`、`/scan`、TF 稳定起来；
- 再让 navigator 接管 action server；
- 最后 explorer 再开始状态机，避免一启动就因为话题未就绪而报错。

### 3.1.1 Launch 参数

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `seed` | `-1` | 随机种子，固定后可复现实验地图 |
| `map_width` | `10.0` | 地图宽度 |
| `map_height` | `10.0` | 地图高度 |
| `num_rooms` | `6` | 隔断墙数量 |
| `num_obstacles` | `8` | 随机障碍物数量 |
| `lidar_range` | `8.0` | 雷达最大量程 |

### 3.1.2 仿真中可以看到的效果

根据节点逻辑和当前 README / 开发记录，这套仿真会体现出以下结果：

- 地图从起点周围开始逐步展开，而不是一次性完整给出；
- 机器人启动后先原地旋转建立首帧局部地图；
- RViz 中会持续看到 frontier 点簇、当前目标点、全局路径、局部目标点、局部预测轨迹；
- 机器人会在未知边界之间来回切换，逐步覆盖所有可达空间；
- 探索结束后会回到起点并保存地图；
- 在 `seed=42` 的典型 10m x 10m 随机场景中，文档记录的完成时间约为 170 到 180 秒。

如果用于录屏或论文演示，这套 launch 的优点是：

- 不依赖 Gazebo；
- 启动快；
- 结果稳定；
- RViz 信息完整。

---

## 4. Real Robot Launch Analysis

## 4.1 `exploration_real_robot.launch.py`

这套 launch 用于真实机器人部署。其设计思路是：

- 底层驱动和 SLAM 仍然沿用外部 `slam` 包提供的 launch；
- 本包只接管“导航执行”和“frontier 探索决策”；
- 即，用 `custom_navigator + frontier_explorer` 替代传统 Nav2 多节点组合。

启动顺序如下：

1. 启动机器人底盘、雷达、IMU 等驱动
2. 5 秒后启动 SLAM
3. 15 秒后启动 `custom_navigator`
4. 20 秒后启动 `frontier_explorer`
5. 可选启动 RViz

### 4.1.1 Launch 参数

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `sim` | `false` | 传给底层 `robot.launch.py` 的仿真标志 |
| `master_name` | 环境变量 `MASTER` 或 `master` | 多机器人 / 主控名称 |
| `robot_name` | 环境变量 `HOST` 或 `/` | 机器人命名空间 |
| `rviz` | `true` | 是否启动 RViz |

### 4.1.2 真实机器人必须提供什么

本包本身不直接负责底层驱动，因此实机要能正常跑起来，外部系统至少要提供以下输入：

| 必需接口 | 类型 | 说明 |
|---|---|---|
| `/scan` | `sensor_msgs/LaserScan` | 180 度雷达数据源 |
| `/map` | `nav_msgs/OccupancyGrid` | SLAM Toolbox 输出的建图结果 |
| `map -> odom -> base_footprint` TF | TF | 导航和 frontier 计算所需的实时位姿 |
| 底盘接收速度控制接口 | 通常是 `geometry_msgs/Twist` | 默认接收 `/cmd_vel` |

进一步说明：

- `custom_navigator` 不直接订阅 `/odom`，但它依赖 TF 里的机器人位姿；
- `frontier_explorer` 也依赖 `map -> base_footprint` 的 TF；
- 如果你的底盘控制话题不是 `/cmd_vel`，需要修改 `cmd_vel_topic` 参数，或者在系统中做桥接/remap；
- 真实机器人使用的是 180 度雷达，本包的导航器只依赖 `/scan`，不要求必须 360 度，只要 TF、量程、噪声水平能支撑避障即可。

### 4.1.3 本包会给真实机器人输出什么

本包面向实机的主要输出如下：

| 输出接口 | 类型 | 作用 |
|---|---|---|
| `/cmd_vel` | `geometry_msgs/Twist` | 发送给底盘的线速度 / 角速度 |
| `navigate_to_pose` | `nav2_msgs/action/NavigateToPose` | 本包内部导航 action server，也可供外部复用 |
| `/navigation/path` | `nav_msgs/Path` | A* 全局路径可视化 |
| `/navigation/local_goal` | `visualization_msgs/Marker` | pure-pursuit 当前局部跟踪点 |
| `/navigation/local_traj` | `visualization_msgs/Marker` | 当前速度下的局部预测轨迹 |
| `/navigation/status` | `std_msgs/String` | 导航状态文本 |
| `/exploration/frontiers` | `visualization_msgs/MarkerArray` | frontier 可视化 |
| `/exploration/status` | `std_msgs/String` | 探索状态机文本 |

附加说明：

- `frontier_explorer` 在初始扫描阶段也会直接向 `/cmd_vel` 发旋转命令；
- 实机 launch 中不会实际使用 `/sim/reset`；
- `frontier_explorer` 在 `IDLE` 状态接受 `/exploration/goal`，可把系统当作“探索结束后的简易导航器”继续使用。

### 4.1.4 与真实机器人底层接口对接时要特别确认的点

1. 雷达坐标系和 `base_footprint` 的 TF 是否正确。
2. SLAM 输出的 `map_frame`、`odom_frame`、`base_frame` 是否与 launch 中一致。
3. 底盘是否确实订阅 `/cmd_vel`，以及速度单位是否标准化。
4. 雷达是否会稳定发布 `/scan`，且时间戳不乱跳。
5. 机器人起步时 SLAM 是否已经建出足够的初始局部地图。

### 4.1.5 当前版本针对实机问题做了什么

针对实际部署中暴露出来的几个问题，这一版代码已经做了对应处理：

1. 大于 90° 的转向更容易打滑
   - 导航器增加了分段降速和侧向加速度限制。
   - 急转时不再维持原先的前向速度。

2. 不能原地旋转
   - 默认继续保持 `allow_in_place_rotation=false`。
   - 恢复动作也改成了低速曲线脱困，而不是原地打转。

3. 加速度慢
   - 新增线速度和角速度加速度限制参数，让命令输出更贴近底盘真实响应。

4. 偶发找不到路径
   - 规划器增加了目标吸附和多档 inflation 重试；
   - 探索器增加了 rescue goals，在失败后先尝试短距离补救位姿。

这些修改的目标不是让控制器更激进，而是让它在实机上更保守、更稳定、更可恢复。

---

## 5. Differences Between Simulation and Real-Robot Launch

| 对比项 | 仿真 `exploration_test_rviz.launch.py` | 实机 `exploration_real_robot.launch.py` |
|---|---|---|
| 地图来源 | `sim_environment` 增量揭示地图 | `SLAM Toolbox` 实时建图 |
| 雷达来源 | 仿真 raycasting | 真实 180 度雷达 |
| 运动执行 | 仿真器内部积分 `/cmd_vel` | 真实底盘执行 `/cmd_vel` |
| TF 来源 | 仿真节点直接发布 | 底层驱动 + SLAM 发布 |
| 是否依赖外部包 | 否 | 依赖 `slam` 包内的 `robot.launch.py` 和 `slam_base.launch.py` |
| RViz | 默认总是启动 | 可通过 `rviz:=false` 关闭 |
| 启动延迟 | 3 秒启动导航，8 秒启动探索 | 5 秒启动 SLAM，15 秒导航，20 秒探索 |
| 多轮测试 | 支持 `num_cycles` + `/sim/reset` | 通常只跑一轮，不使用 reset |
| 调试便利性 | 高，地图可复现、可快速试参 | 更接近真实效果，但受传感器噪声和底盘误差影响 |

### 5.1 本质差异

两者最大的区别不是 launch 写法，而是输入质量不同：

- 仿真中的 `/map` 和 `/scan` 非常干净；
- 实机中的 `/scan` 有噪声、遮挡、反光误差；
- 实机中的 TF 和 SLAM 也会有抖动、漂移、延迟；
- 因此很多参数在仿真可用，在实机上需要更保守。

### 5.2 实机比仿真更容易暴露的问题

- 贴墙时局部控制抖动；
- 雷达量程较短导致 frontier 识别更碎；
- 起步阶段地图过小，容易误判“无 frontier”；
- 底盘执行速度与指令不一致，造成近目标转圈或过冲；
- TF 延迟导致控制器判断的当前位姿滞后。

---

## 6. Standalone Map Navigation Mode

除了“探索 + 建图”主流程外，当前包还新增了一条单独模式：基于已有地图做导航，不启动 `frontier_explorer`。

新增链路包含：

1. `navigation_with_map.launch.py`
   - 启动底盘驱动、`nav2_map_server`、`nav2_amcl`、`custom_navigator`、`map_goal_bridge` 和可选 RViz。

2. `map_goal_bridge.py`
   - 订阅 `/goal_pose` 和 `/move_base_simple/goal`；
   - 把外部 `PoseStamped` 目标转成 `navigate_to_pose` action；
   - 这样就可以直接复用自定义导航器，而不必启动探索状态机。

3. `config/amcl_params.yaml`
   - 提供已有地图定位所需的 AMCL 参数；
   - `map_server` 则直接加载指定的 `.yaml` 地图文件。

这个模式适合：

- 已经完成建图，只想做导航测试；
- 先调底盘控制稳定性，不想把 frontier 探索一起带上；
- 用 RViz 直接点目标点，验证地图定位和自定义导航器的联调。

注意：

- 启动后需要先在 RViz 里做一次 `2D Pose Estimate`；
- 然后再通过 `2D Nav Goal` 或外部发布 `PoseStamped` 目标。

---

## 7. Parameters Worth Tuning

## 7.1 导航器参数 `config/custom_nav_params.yaml`

这些参数主要影响“能否走得稳、走得安全”。

| 参数 | 默认值 | 调整建议 |
|---|---:|---|
| `robot_radius` | `0.15` | 实机必须匹配真实底盘外扩半径，偏小会擦碰，偏大会堵住窄路 |
| `max_vel_x` | `0.22` | 实机如果底盘惯性大，可适当降低 |
| `max_vel_theta` | `0.7` | 实机转向抖动时可降低 |
| `max_accel_x` | `0.18` | 实机加速慢时建议保守一些，避免指令跳变 |
| `max_decel_x` | `0.28` | 刹车距离偏长时可进一步减小 |
| `max_accel_theta` | `1.0` | 转向电机响应慢时可降低 |
| `max_lateral_accel` | `0.10` | 急转容易打滑时优先降低这个值 |
| `goal_tolerance` | `0.25` | 实机定位噪声大时可适当调大 |
| `lookahead_dist` | `0.45` | 大一些更平滑，小一些更灵活 |
| `safety_horizon` | `0.5` | 增大更保守，减小更激进 |
| `sharp_turn_angle` | `1.05` | 提前进入急转降速的阈值 |
| `very_sharp_turn_angle` | `1.57` | 接近 90° 时进入极低速爬行 |
| `sharp_turn_speed` | `0.06` | 急转时的最高线速度 |
| `very_sharp_turn_speed` | `0.03` | 超大角度转向时的最高线速度 |
| `inflation_mult` | `1.5` | 实机贴墙风险高时可以增大 |
| `proximity_weight` | `3.0` | 越大越偏向通道中心，但可能让窄门更难通过 |
| `goal_search_radius` | `0.8` | A* 失败时搜索附近可达目标的半径 |
| `goal_search_step` | `0.1` | 附近目标搜索步长 |
| `stuck_timeout` | `3.5` | 实机动作慢时可略增大 |
| `max_recoveries` | `5` | 复杂环境可增大，但会拖长失败目标处理时间 |
| `backup_vel` | `-0.10` | 实机若倒车危险或轮胎打滑，可减小 |
| `goal_timeout` | `90.0` | 实机大场景可适当增大 |
| `replan_interval` | `5.0` | 动态环境或 SLAM 变化快时可减小 |
| `cmd_vel_topic` | `"/cmd_vel"` | 若底盘接口不同，需要优先改这个参数 |

## 7.2 探索器参数 `config/explore_params.yaml`

这些参数主要影响“去哪里探索、什么时候结束”。

| 参数 | 默认值 | 调整建议 |
|---|---:|---|
| `min_frontier_size` | `5` | 雷达噪声大时可增大，过滤碎 frontier |
| `min_frontier_dist` | `0.3` | 避免选到离当前位姿太近的无效目标 |
| `cost_weight` | `1.0` | 越大越偏向近处 frontier |
| `gain_weight` | `1.5` | 越大越偏向大 frontier |
| `blacklist_radius` | `0.6` | 过小会重复撞同一片死区，过大会误伤邻近可达点 |
| `blacklist_clear_interval` | `120.0` | 地图变化慢时可更长 |
| `goal_timeout` | `90.0` | frontier 长路径场景可适当增加 |
| `replan_interval` | `1.5` | 实机地图变化快时保留较小值通常更好 |
| `post_goal_pause` | `0.3` | SLAM 稳定较慢时可略加大 |
| `enable_rescue_goals` | `true` | 实机建议开启，用于失败后的补救位姿 |
| `rescue_goal_radius` | `0.6` | 补救位姿距离，太小作用不明显，太大又可能引入新风险 |
| `max_rescue_goals` | `4` | 一次失败后最多尝试多少个补救点 |
| `no_frontier_threshold` | `5` | 实机雷达视野不稳定时可适当增大 |
| `min_explored_cells` | `300` | 大地图可以适当提高，减少过早结束 |
| `enable_return_home` | `true` | 根据任务需要决定是否回原点 |
| `initial_spin_duration` | `8.0` | 180 度雷达实机上通常比仿真更需要足够长的初始旋转 |
| `spin_angular_vel` | `0.8` | 实机雷达采样较慢时应避免过快旋转 |
| `num_cycles` | `1` | 实机一般保持 1，仿真压测才会用多轮 |
| `num_random_goals` | `5` | 实机若只做探索可设为 0 |
| `nav_action_name` | `"navigate_to_pose"` | 若外部 action 名称修改，需要同步调整 |
| `cmd_vel_topic` | `"/cmd_vel"` | explorer 初始自旋命令的输出话题 |

## 7.3 仿真参数

仿真独有参数主要影响“测试环境难度”：

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `seed` | `-1` | 固定地图复现 |
| `map_width` / `map_height` | `10.0` | 调整场景尺度 |
| `num_rooms` | `6` | 调整拓扑复杂度 |
| `num_obstacles` | `8` | 调整局部避障难度 |
| `lidar_range` | `8.0` | 调整可见范围 |

---

## 8. Recommended Use and Documentation Summary

如果要把这个包用于真实机器人，推荐按下面的顺序做接入：

1. 先用 `exploration_test_rviz.launch.py` 固定 `seed` 做参数调试。
2. 确认 RViz 中 `/map`、`/scan`、frontier、path、local goal 显示完整。
3. 上实机前先确认底盘真实接收的速度话题。
4. 确认 SLAM 输出的 TF 树为 `map -> odom -> base_footprint`。
5. 实机首轮建议降低线速度、降低角速度、增大 `goal_tolerance` 和 `inflation_mult`。

这套代码的工程特点是：

- 架构简单，便于理解和二次开发；
- 仿真链路轻量，适合快速做 frontier 策略验证；
- 实机依赖项较少，只要求稳定的 `/scan`、`/map`、TF 和 `/cmd_vel` 接口；
- 与完整 Nav2 相比更容易掌控每一步行为，但也需要自己负责参数整定和异常场景处理。
