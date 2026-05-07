import os
import cv2
import time
import math
import signal
import threading
import multiprocessing
import numpy as np
from enum import Enum, auto

# ==========================================
# 硬件抽象与控制模块
# ==========================================
try:
    from tracking_person.wheel_control import Board
except ImportError:
    class Board:
        def set_motor_speed(self, speeds): 
            # 虚拟输出，防止报错
            pass

def set_motor(board, speed_right, speed_left, max_speed=60):
    """
    双轮差速控制底层映射
    speed_right/left 范围: [-1.0, 1.0]
    """
    if board is None: return
    try:
        # 针对具体小车电机的正负极性调整
        # 1号电机通常为右轮，2号为左轮（反向补偿）
        board.set_motor_speed([
            [1, int(max_speed * speed_right)],
            [2, int(max_speed * speed_left * -1)],
        ])
    except Exception as e:
        print(f"Motor Control Error: {e}")

def _safe_remove_pid(pid_file: str) -> None:
    if not pid_file or not os.path.exists(pid_file):
        return
    try:
        os.remove(pid_file)
    except OSError:
        try:
            with open(pid_file, "w") as f: f.write("")
        except OSError: pass

# ==========================================
# 视频流读取模块 (多线程保证实时性)
# ==========================================
class CameraReader:
    def __init__(self, src=0, width=640, height=480):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened(): 
            raise RuntimeError(f"can not open camera: {src}")
        
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.ret, self.frame = self.cap.read()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while not self._stop.is_set():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self._lock:
                    self.ret = ret
                    # 针对 OpenBot 等安装方式，通常需要镜像或旋转翻转
                    self.frame = cv2.flip(frame, -1) 
                    #self.frame = cv2.flip(self.frame, 1)
            else: 
                time.sleep(0.005)
        if self.cap.isOpened():
            self.cap.release()

    def read(self):
        with self._lock:
            if self.frame is not None: 
                return self.ret, self.frame.copy()
        return False, None
        
    def isOpened(self): 
        return not self._stop.is_set() and self.cap.isOpened()
    
    def release(self):
        self._stop.set()
        if self.thread.is_alive(): 
            self.thread.join(timeout=2.0)
        if self.cap.isOpened(): 
            self.cap.release()

# ==========================================
# ArUco 检测配置
# ==========================================
try:
    # 兼容 OpenCV 不同版本的 ArUco API
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    def detect_aruco(img): return aruco_detector.detectMarkers(img)
except AttributeError:
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters_create()
    def detect_aruco(img): return cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)

# 标记点物理尺寸 (单位: 米)
MARKER_LENGTH = 0.187 
# 定义 ArUco 码四个角的 3D 物理坐标
OBJ_POINTS = np.array([
    [-MARKER_LENGTH/2,  MARKER_LENGTH/2, 0],
    [ MARKER_LENGTH/2,  MARKER_LENGTH/2, 0],
    [ MARKER_LENGTH/2, -MARKER_LENGTH/2, 0],
    [-MARKER_LENGTH/2, -MARKER_LENGTH/2, 0]
], dtype=np.float32)

# 加载相机内参
_NAV_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_FILE_PATH = os.path.join(_NAV_DIR, "camera_params.npz")

if os.path.exists(CALIB_FILE_PATH):
    with np.load(CALIB_FILE_PATH) as X:
        CAMERA_MATRIX = X['camera_matrix']
        DIST_COEFFS = X['dist_coeffs']
else:
    # 虚拟参数保底
    CAMERA_MATRIX = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    DIST_COEFFS = np.zeros((5, 1), dtype=np.float32)

class TrackerState(Enum):
    SEARCHING = auto()
    ALIGNING = auto()
    ARRIVED = auto()

# ==========================================
# 核心导航追踪器
# ==========================================
class ArucoTracker:
    def __init__(self, target_z=0.40):
        """
        target_z: 回仓停止的理想距离 (单位: 米)
        """
        self.TARGET_Z = target_z     
        # 停止的死区阈值（必须全部满足才能判定为对齐）
        self.DEAD_ZONE_Z = 0.02      # 距离误差 2cm
        self.DEAD_ZONE_X = 0.02      # 横向偏移 2cm (要求极高，保证正对)
        self.DEAD_ZONE_YAW = 3.0     # 偏航角误差 3度 (保证平行于墙面)
        
        # PID 比例系数 (可根据场地摩擦力微调)
        self.KP_Z = 0.7              # 前进控制
        self.KP_X = 1.2              # 横向平移补偿 (通过差速模拟)
        self.KP_YAW = 0.015          # 旋转控制

        self.currentState = TrackerState.SEARCHING
        self.currentL = 0.0
        self.currentR = 0.0
        self.lastMoveDir = 1.0       # 1.0 代表顺时针
        self.arrived_start_time = None

    def updateTarget(self, x_offset, z_dist, yaw_angle):
        """
        x_offset: 标记点中心相对于相机光轴的横向距离 (tvec[0])
        z_dist: 标记点中心距离相机的深度 (tvec[2])
        yaw_angle: 标记点平面与相机平面的水平偏角
        """
        tL = tR = 0.0

        if self.currentState == TrackerState.ALIGNING and z_dist is not None:
            # 1. 计算误差
            error_z = z_dist - self.TARGET_Z
            error_x = x_offset
            error_yaw = yaw_angle # 标记点绕Y轴的旋转

            # 2. 判断是否“完美对齐”
            if (abs(error_z) <= self.DEAD_ZONE_Z and 
                abs(error_x) <= self.DEAD_ZONE_X and 
                abs(error_yaw) <= self.DEAD_ZONE_YAW):
                
                if self.arrived_start_time is None:
                    self.arrived_start_time = time.time()
                elif time.time() - self.arrived_start_time > 1.5: 
                    # 持续稳定 1.5 秒则判定到达
                    self.currentState = TrackerState.ARRIVED
            else:
                self.arrived_start_time = None
                
                # 3. 前进速度逻辑：当角度偏差过大时，大幅减慢前进，优先原地修正角度
                fwd = self.KP_Z * error_z
                fwd = max(-0.30, min(0.30, fwd))
                
                # 如果偏航角大于 12 度或横向偏移严重，限制推进，强制先对正
                if abs(error_yaw) > 12.0 or abs(error_x) > 0.12:
                    fwd *= 0.2 
                elif abs(error_yaw) > 6.0:
                    fwd *= 0.5

                # 4. 转向速度逻辑 (复合控制)
                # steer = 横向位移补偿 + 偏航角纠正
                # 这部分是保证“正对”的关键：如果小车在左边，要右转；同时如果小车没平行于墙，也要偏航纠正
                steer = (self.KP_X * error_x) + (self.KP_YAW * error_yaw)
                steer = max(-0.40, min(0.40, steer)) 
                

                # 注意：这里 steer 的方向需根据电机接线实测调整
                # 差速模型：L = fwd - steer, R = fwd + steer
                tL, tR = fwd + steer, fwd - steer

                
                # 记忆最后看见目标的方向，方便丢失后搜索
                self.lastMoveDir = 1.0 if steer < 0 else -1.0

        elif self.currentState == TrackerState.SEARCHING:
            # 原地旋转搜索逻辑
            spd = 0.22 
            tL = spd if self.lastMoveDir > 0 else -spd
            tR = -spd if self.lastMoveDir > 0 else spd

        # 5. 速度平滑滤波 (LPF)
        alpha = 0.4
        self.currentL = (1 - alpha) * self.currentL + alpha * tL
        self.currentR = (1 - alpha) * self.currentR + alpha * tR

        # 6. 低速截断 (克服死区电压)
        if abs(self.currentL) < 0.06: self.currentL = 0.0
        if abs(self.currentR) < 0.06: self.currentR = 0.0

        return self.currentL, self.currentR


def background_aruco_nav_task(video_source, target_id, pid_file, tts_mp_q=None):
    """
    ArUco 导航后台进程主循环
    """
    import speaker
    speaker.init_mp_queue(tts_mp_q)

    is_running = True
    def handle_sigterm(signum, frame_obj):
        nonlocal is_running
        is_running = False

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)

    board = Board()
    cap = None
    # target_z=0.40 表示停在 40 厘米处
    tracker = ArucoTracker(target_z=0.40)
    target_id = int(target_id)
    
    last_seen_time = 0.0 
    speaker.speak(f"开始精准回仓系统")

    try:
        cap = CameraReader(video_source)
        while cap.isOpened() and is_running:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # 检测标记点
            corners, ids, rejected = detect_aruco(frame)
            
            x_offset = z_dist = yaw_angle = None
            vis = frame.copy() # 用于调试显示的图像

            if ids is not None:
                # 遍历所有发现的 ID
                for i in range(len(ids)):
                    if ids[i][0] == target_id:
                        tracker.currentState = TrackerState.ALIGNING
                        corner = corners[i][0]
                        
                        # 绘制外框
                        cv2.polylines(vis, [corner.astype(np.int32)], True, (0, 255, 0), 2)
                        
                        # 核心：解算 6 自由度位姿 (使用 IPPE_SQUARE 针对矩形码优化)
                        success, rvec, tvec = cv2.solvePnP(
                            OBJ_POINTS, corner, CAMERA_MATRIX, DIST_COEFFS, flags=cv2.SOLVEPNP_IPPE_SQUARE
                        )
                        
                        if success:
                            # tvec[0]: X-平移, tvec[1]: Y-平移(上下), tvec[2]: Z-距离
                            x_offset = tvec[0][0]  
                            z_dist = tvec[2][0]    
                            
                            # 将旋转向量转换为欧拉角，提取偏航角(Yaw)
                            # 注意：即使摄像头俯仰，Yaw 依然是相对于标记平面的左右夹角
                            rmat, _ = cv2.Rodrigues(rvec)
                            euler_angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                            yaw_angle = euler_angles[1] 
                            
                            # 绘制坐标轴辅助线
                            cv2.drawFrameAxes(vis, CAMERA_MATRIX, DIST_COEFFS, rvec, tvec, 0.1)
                            last_seen_time = time.time()
                        break

            # 丢失目标逻辑
            if z_dist is None and tracker.currentState != TrackerState.ARRIVED:
                if time.time() - last_seen_time > 1.5:
                    tracker.currentState = TrackerState.SEARCHING

            # 获取轮速控制指令
            L, R = tracker.updateTarget(x_offset, z_dist, yaw_angle)
            
            # 注意：此处 R/L 赋值需与 Board.set_motor 内部通道一一对应
            # 这里的交换是因为控制逻辑与电机物理通道的映射
            vL, vR = R, L 
            set_motor(board, speed_right=vR, speed_left=vL)

            # 打印调试信息 (在线实时诊断)
            if z_dist:
                print(f"[NAV] Dist:{z_dist:.2f}m | X:{x_offset:.2f}m | Yaw:{yaw_angle:.1f}°")

            # 可视化增强
            state_text = tracker.currentState.name
            cv2.rectangle(vis, (5, 5), (350, 150), (0, 0, 0), -1)
            cv2.putText(vis, f"STATE: {state_text}", (15, 30), 1, 1.5, (0, 255, 255), 2)
            if z_dist:
                cv2.putText(vis, f"D: {z_dist:.3f}m", (15, 65), 1, 1.2, (255, 255, 255), 1)
                cv2.putText(vis, f"X: {x_offset:.3f}m", (15, 95), 1, 1.2, (255, 255, 255), 1)
                cv2.putText(vis, f"Y: {yaw_angle:.1f}deg", (15, 125), 1, 1.2, (255, 255, 255), 1)

            cv2.imshow("Docking System Monitor", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # 判定到达并停止
            if tracker.currentState == TrackerState.ARRIVED:
                set_motor(board, 0.0, 0.0)
                print("==> 精准回仓完成，位置已锁死。")
                speaker.speak(f"目标{target_id}回仓成功")
                time.sleep(1.0)
                break

    except Exception as e:
        print(f"Critical Navigation Error: {e}")
    finally:
        # 安全退出
        set_motor(board, 0.0, 0.0)
        if cap: cap.release()
        cv2.destroyAllWindows()
        _safe_remove_pid(pid_file)
        print("[进程结果] ArUco 导航任务结束")


class ArucoNavigationSystem:
    """
    ArUco 导航系统接口类，负责进程管理
    """
    def __init__(self):
        self.pid_file = "/tmp/aruco_nav_pid.txt"
        self._process: multiprocessing.Process = None

    def start_aruco_navigation(self, video_source, target_id):
        """
        外部调用入口
        """
        import speaker
        if self._process is not None and self._process.is_alive():
            print("警告: 导航任务正在运行...")
            return
            
        _safe_remove_pid(self.pid_file)
        ctx = multiprocessing.get_context('spawn')
        
        # 确保语音队列存在
        if speaker._mp_q is None:
            speaker.init_mp_queue(ctx.Queue())

        p = ctx.Process(
            target=background_aruco_nav_task,
            args=(video_source, target_id, self.pid_file, speaker._mp_q),
            daemon=False,
        )
        p.start()
        self._process = p
        
        try:
            with open(self.pid_file, "w") as f: 
                f.write(str(p.pid))
        except OSError: pass

    def stop_aruco_navigation(self):
        """
        强制终止
        """
        import speaker
        print("正在请求终止 ArUco 导航...")
        try:
            self._terminate_process()
        finally:
            _safe_remove_pid(self.pid_file)
        # 补发一次停止指令
        set_motor(None, 0.0, 0.0)
        speaker.speak("导航已手动关闭")

    def _terminate_process(self):
        if self._process is not None:
            try:
                if self._process.is_alive():
                    self._process.terminate()
                    self._process.join(timeout=2.0)
                    if self._process.is_alive():
                        self._process.kill()
                        self._process.join(timeout=1.0)
            except Exception as e:
                print(f"Terminate Error: {e}")
            finally: 
                self._process = None
            return

        # PID 文件兜底逻辑
        if not os.path.exists(self.pid_file): return
        try:
            with open(self.pid_file, "r") as f:
                pid_str = f.read().strip()
                if pid_str.isdigit():
                    pid = int(pid_str)
                    os.kill(pid, signal.SIGTERM)
                    time.sleep(1.0)
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError: pass
        except Exception: pass

# ==========================================
# 测试入口
# ==========================================
if __name__ == "__main__":
    # 测试代码，实际使用时由主控制程序实例化
    test_nav = ArucoNavigationSystem()
    # 假设 ArUco 码 ID 为 1
    test_nav.start_aruco_navigation("/dev/video0", 1)
    
    # 模拟运行一段时间
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        test_nav.stop_aruco_navigation()
