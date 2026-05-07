import os
import cv2
import time
import math
import signal
import threading
import multiprocessing
import numpy as np
from enum import Enum, auto

try:
    from tracking_person.wheel_control import Board
except ImportError:
    class Board:
        def set_motor_speed(self, speeds): pass

def set_motor(board, speed_right, speed_left, max_speed=60):
    if board is None: return
    try:
        board.set_motor_speed([
            [1, int(max_speed * speed_right)],
            [2, int(max_speed * speed_left * -1)],
        ])
    except Exception:
        pass

def _safe_remove_pid(pid_file: str) -> None:
    if not pid_file or not os.path.exists(pid_file):
        return
    try:
        os.remove(pid_file)
    except OSError:
        try:
            with open(pid_file, "w") as f: f.write("")
        except OSError: pass

class CameraReader:
    def __init__(self, src=0, width=640, height=480):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened(): raise RuntimeError(f"can not open camera: {src}")
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
                    self.frame = cv2.flip(frame, -1) 
            else: time.sleep(0.005)
        if self.cap.isOpened():
            self.cap.release()

    def read(self):
        with self._lock:
            if self.frame is not None: return self.ret, self.frame.copy()
        return False, None
        
    def isOpened(self): return not self._stop.is_set() and self.cap.isOpened()
    
    def release(self):
        self._stop.set()
        if self.thread.is_alive(): self.thread.join(timeout=2.0)
        if self.cap.isOpened(): self.cap.release()

try:
    # 适配 OpenCV 4.7+
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    def detect_aruco(img): return aruco_detector.detectMarkers(img)
except AttributeError:
    # 适配 OpenCV 4.6 及以下
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters_create()
    def detect_aruco(img): return cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)

MARKER_LENGTH = 0.187  # 单位：米
OBJ_POINTS = np.array([
    [-MARKER_LENGTH/2,  MARKER_LENGTH/2, 0],
    [ MARKER_LENGTH/2,  MARKER_LENGTH/2, 0],
    [ MARKER_LENGTH/2, -MARKER_LENGTH/2, 0],
    [-MARKER_LENGTH/2, -MARKER_LENGTH/2, 0]
], dtype=np.float32)

# ================= 真实摄像头内参 (从标定文件读取) =================
_NAV_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_FILE_PATH = os.path.join(_NAV_DIR, "camera_params.npz")

if os.path.exists(CALIB_FILE_PATH):
    print(f"成功加载真实的相机标定文件: {CALIB_FILE_PATH}")
    with np.load(CALIB_FILE_PATH) as X:
        CAMERA_MATRIX = X['camera_matrix']
        DIST_COEFFS = X['dist_coeffs']
else:
    print("未找到 camera_params.npz，正在使用默认虚拟参数！")
    CAMERA_MATRIX = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    DIST_COEFFS = np.zeros((5, 1), dtype=np.float32)

# ================= 状态机与追踪器 =================
class TrackerState(Enum):
    SEARCHING = auto()
    ALIGNING = auto()
    ARRIVED = auto()

class ArucoTracker:
    def __init__(self, target_z=0.40): # 设定为距离 40 厘米
        self.TARGET_Z = target_z     
        self.DEAD_ZONE_Z = 0.03      # 距离允许误差 3 厘米
        self.DEAD_ZONE_X = 0.04      # 横向允许误差 4 厘米
        self.DEAD_ZONE_YAW = 5.0     # 角度允许误差 6 度 (确保严格正对)
        
        self.KP_Z = 0.8              # 距离比例系数
        self.KP_X = 1.0              # 横向偏移比例系数
        self.KP_YAW = 0.010          # 偏航角比例系数 (角度纠正力度)

        self.currentState = TrackerState.SEARCHING
        self.currentL = 0.0
        self.currentR = 0.0
        self.lastMoveDir = 1.0       # 寻找时的旋转方向
        self.arrived_start_time = None

    def updateTarget(self, x_offset, z_dist, yaw_angle):
        tL = tR = 0.0

        if self.currentState == TrackerState.ALIGNING and z_dist is not None:
            error_z = z_dist - self.TARGET_Z
            error_x = x_offset
            error_yaw = yaw_angle

            # 1. 检查是否完全对齐并到达位置
            if abs(error_z) <= self.DEAD_ZONE_Z and abs(error_x) <= self.DEAD_ZONE_X and abs(error_yaw) <= self.DEAD_ZONE_YAW:
                if self.arrived_start_time is None:
                    self.arrived_start_time = time.time()
                elif time.time() - self.arrived_start_time > 1.0: # 稳定对齐 1 秒确认到达
                    self.currentState = TrackerState.ARRIVED
            else:
                self.arrived_start_time = None
                
                # 2. 智能限速逻辑：如果角度或横向偏差太大，先刹车减速，优先把车头调正！
                fwd = self.KP_Z * error_z
                fwd = max(-0.35, min(0.35, fwd))
                if abs(error_yaw) > 15.0 or abs(error_x) > 0.15:
                    fwd *= 0.3  # 大幅降低前进速度，等待车头转正

                # 3. 融合 X 轴偏移和 Yaw 角偏移产生转向力
                steer = (self.KP_X * error_x) + (self.KP_YAW * error_yaw)
                steer = max(-0.35, min(0.35, steer)) 

                steer = -steer # 若发现方向反了，去掉或加上这个负号
                tL, tR = fwd - steer, fwd + steer
                self.lastMoveDir = 1.0 if x_offset < 0 else -1.0

        elif self.currentState == TrackerState.SEARCHING:
            # 慢速原地旋转寻找
            spd = 0.25 
            tL = spd if self.lastMoveDir > 0 else -spd
            tR = -spd if self.lastMoveDir > 0 else spd

        # 速度平滑滤波
        self.currentL = 0.5 * self.currentL + 0.5 * tL
        self.currentR = 0.5 * self.currentR + 0.5 * tR

        # 限制死区，避免电机微小啸叫
        if abs(self.currentL) < 0.05: self.currentL = 0.0
        if abs(self.currentR) < 0.05: self.currentR = 0.0

        return self.currentL, self.currentR


def background_aruco_nav_task(video_source, target_id, pid_file, tts_mp_q=None):
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
    tracker = ArucoTracker(target_z=0.40) # 距墙面 40cm
    target_id = int(target_id)
    
    speaker.speak(f"启动自动回仓，目标标记点 {target_id}")

    try:
        cap = CameraReader(video_source)
        while cap.isOpened() and is_running:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            corners, ids, rejected = detect_aruco(frame)
            
            x_offset = z_dist = yaw_angle = None
            vis = frame.copy()

            if ids is not None:
                for i in range(len(ids)):
                    if ids[i][0] == target_id:
                        tracker.currentState = TrackerState.ALIGNING
                        corner = corners[i][0]
                        cv2.polylines(vis, [corner.astype(np.int32)], True, (0, 255, 0), 2)
                        
                        success, rvec, tvec = cv2.solvePnP(
                            OBJ_POINTS, corner, CAMERA_MATRIX, DIST_COEFFS, flags=cv2.SOLVEPNP_IPPE_SQUARE
                        )
                        
                        if success:
                            # 核心降维提取：完全抛弃 tvec[1][0] (高度)
                            x_offset = tvec[0][0]  # X: 横向偏移 (米)
                            z_dist = tvec[2][0]    # Z: 纵向距离 (米)
                            
                            # 提取纯平面偏航角 Yaw，抛弃上下俯仰角 Pitch
                            rmat, _ = cv2.Rodrigues(rvec)
                            euler_angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                            yaw_angle = euler_angles[1] # Y 轴旋转 (偏航角)
                            
                            cv2.drawFrameAxes(vis, CAMERA_MATRIX, DIST_COEFFS, rvec, tvec, 0.05)
                        break

            # 如果中途丢了视野，切回寻找模式
            if z_dist is None and tracker.currentState != TrackerState.ARRIVED:
                tracker.currentState = TrackerState.SEARCHING

            # 获取并下发电机控制指令
            L, R = tracker.updateTarget(x_offset, z_dist, yaw_angle)
            set_motor(board, speed_right=R, speed_left=L)

            # UI 绘制
            state_text = tracker.currentState.name
            color = (0, 255, 0) if z_dist is not None else (0, 165, 255)
            cv2.putText(vis, f"Target ID: {target_id} | State: {state_text}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            if z_dist is not None:
                cv2.putText(vis, f"Dist(Z): {z_dist:.2f}m", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(vis, f"Offset(X): {x_offset:.2f}m", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(vis, f"Yaw Angle: {yaw_angle:.1f} deg", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            cv2.imshow("ArUco Docking", vis)
            key = cv2.waitKey(1) & 0xFF
            if key in [ord("e"), ord("q"), 27]:
                print("\n收到按键退出指令")
                break

            if tracker.currentState == TrackerState.ARRIVED:
                print(f"成功对齐！小车已正对目标停靠在 {tracker.TARGET_Z*100}cm 处。")
                speaker.speak(f"已完美到达并对齐标记 {target_id}")
                break

    except Exception as e:
        print(f"ArUco Nav 异常: {e}")
    finally:
        set_motor(board, 0.0, 0.0)
        if cap: cap.release()
        cv2.destroyAllWindows()
        for _ in range(10): cv2.waitKey(1)
        _safe_remove_pid(pid_file)
        print("ArUco 回仓进程已退出并释放资源")


class ArucoNavigationSystem:
    def __init__(self):
        self.pid_file = "/tmp/aruco_nav_pid.txt"
        self._process: multiprocessing.Process = None

    def start_aruco_navigation(self, video_source, target_id):
        import speaker
        print(f"启动自动回仓任务: ArUco ID {target_id}")
        if self._process is not None and self._process.is_alive():
            print("ArUco 导航已经在后台运行中，请先停止")
            return
            
        _safe_remove_pid(self.pid_file)
        ctx = multiprocessing.get_context('spawn')
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
        import speaker
        print("Kill ArUco 导航进程")
        try:
            self._terminate_process()
        finally:
            _safe_remove_pid(self.pid_file)
        set_motor(None, 0.0, 0.0)
        speaker.speak("已停止自动回仓")

    def _terminate_process(self):
        if self._process is not None:
            try:
                if self._process.is_alive():
                    self._process.terminate()
                    self._process.join(timeout=3.0)
                    if self._process.is_alive():
                        self._process.kill()
                        self._process.join(timeout=2.0)
            except Exception: pass
            finally: self._process = None
            return

        if not os.path.exists(self.pid_file):
            return
        try:
            pid_str = open(self.pid_file).read().strip()
            if not pid_str.isdigit():
                return
            pid = int(pid_str)
            try:
                os.kill(pid, signal.SIGTERM)
                time.sleep(3.0)
                try:
                    os.kill(pid, 0)
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError: pass
            except ProcessLookupError: pass
            except PermissionError: pass
        except Exception: pass
