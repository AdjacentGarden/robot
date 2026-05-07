import os
import cv2
import time
import math
import signal
import threading
import multiprocessing
import numpy as np
from enum import Enum, auto

# ================= 硬件控制基础 =================
try:
    from tracking_person.wheel_control import Board
except ImportError:
    class Board:
        def set_motor_speed(self, speeds): pass

def set_motor(board, speed_right, speed_left, max_speed=150):
    if board is None: return
    try:
        board.set_motor_speed([
            [1, int(max_speed * speed_right)],
            [2, int(max_speed * speed_left * -1)],
        ])
    except Exception:
        pass

def _safe_remove_pid(pid_file: str):
    if os.path.exists(pid_file):
        try: os.remove(pid_file)
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

    def read(self):
        with self._lock:
            if self.frame is not None: return self.ret, self.frame.copy()
        return False, None
        
    def isOpened(self): return not self._stop.is_set() and self.cap.isOpened()
    
    def release(self):
        self._stop.set()
        if self.thread.is_alive(): self.thread.join(timeout=2.0)
        if self.cap.isOpened(): self.cap.release()

# ================= ArUco 兼容性与 3D 解算配置 =================
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
# 自动获取当前 aruco_camera.py 所在的文件夹路径
_NAV_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_FILE_PATH = os.path.join(_NAV_DIR, "camera_params.npz")

if os.path.exists(CALIB_FILE_PATH):
    print(f"成功加载真实的相机标定文件: {CALIB_FILE_PATH}")
    with np.load(CALIB_FILE_PATH) as X:
        CAMERA_MATRIX = X['camera_matrix']
        DIST_COEFFS = X['dist_coeffs']
else:
    print("未找到 camera_params.npz，正在使用默认虚拟参数（测距和角度可能不准）！")
    CAMERA_MATRIX = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    DIST_COEFFS = np.zeros((5, 1), dtype=np.float32)
# ===================================================================

# ================= 状态机与追踪器 =================
class TrackerState(Enum):
    SEARCHING = auto()
    ALIGNING = auto()
    ARRIVED = auto()

class ArucoTracker:
    def __init__(self, target_z=0.3):
        # PID 控制参数
        self.TARGET_Z = target_z     # 目标停车距离（米）
        self.DEAD_ZONE_Z = 0.05      # 距离允许误差 5 厘米
        self.DEAD_ZONE_X = 0.05      # 横向允许误差 5 厘米
        self.DEAD_ZONE_YAW = 10.0    # 角度允许误差 10 度
        
        self.KP_Z = 0.8              # 距离比例系数
        self.KP_X = 1.2              # 横向偏移比例系数
        self.KP_YAW = 0.005          # 偏航角比例系数

        self.currentState = TrackerState.SEARCHING
        self.currentL = 0.0
        self.currentR = 0.0
        self.lastMoveDir = 1.0       # 寻找时的旋转方向
        self.arrived_start_time = None

    def updateTarget(self, x_offset, z_dist, yaw_angle):
        tL = tR = 0.0

        if self.currentState == TrackerState.ALIGNING and z_dist is not None:
            # 计算各项误差
            error_z = z_dist - self.TARGET_Z
            error_x = x_offset
            error_yaw = yaw_angle

            # 1. 检查是否完全对齐并到达位置
            if abs(error_z) <= self.DEAD_ZONE_Z and abs(error_x) <= self.DEAD_ZONE_X and abs(error_yaw) <= self.DEAD_ZONE_YAW:
                if self.arrived_start_time is None:
                    self.arrived_start_time = time.time()
                elif time.time() - self.arrived_start_time > 1.0:
                    self.currentState = TrackerState.ARRIVED
            else:
                self.arrived_start_time = None
                
                # 2. 计算基础前进速度
                fwd = self.KP_Z * error_z
                fwd = max(-0.35, min(0.35, fwd)) # 限速

                # 3. 融合 X 轴偏移和 Yaw 角偏移，计算转向速度！
                # 这里的加减号可能需要根据你小车的电机实际接线方向调整
                steer = (self.KP_X * error_x) + (self.KP_YAW * error_yaw)
                steer = max(-0.35, min(0.35, steer)) # 限速

                # 混合速度
                steer = -steer # 如果发现小车反向逃离，请把这里的负号去掉
                tL, tR = fwd - steer, fwd + steer

                self.lastMoveDir = 1.0 if x_offset < 0 else -1.0

        elif self.currentState == TrackerState.SEARCHING:
            # 原地旋转寻找
            spd = 0.35
            tL = spd if self.lastMoveDir > 0 else -spd
            tR = -spd if self.lastMoveDir > 0 else spd

        # 速度平滑
        self.currentL = 0.5 * self.currentL + 0.5 * tL
        self.currentR = 0.5 * self.currentR + 0.5 * tR

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
    tracker = ArucoTracker(target_z=0.30) # 设定停在距离二维码正前方 30 厘米处
    target_id = int(target_id) # ArUco 的 ID 必须是整数，比如 0, 1, 2
    
    speaker.speak(f"开始精准回仓至标记点 {target_id}")

    try:
        cap = CameraReader(video_source)
        while cap.isOpened() and is_running:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # 检测 ArUco 码
            corners, ids, rejected = detect_aruco(frame)
            
            x_offset = z_dist = yaw_angle = None
            vis = frame.copy()

            if ids is not None:
                for i in range(len(ids)):
                    if ids[i][0] == target_id:
                        tracker.currentState = TrackerState.ALIGNING
                        corner = corners[i][0]
                        
                        # 画出识别框
                        cv2.polylines(vis, [corner.astype(np.int32)], True, (0, 255, 0), 2)
                        
                        # 3D 姿态解算魔法！
                        success, rvec, tvec = cv2.solvePnP(
                            OBJ_POINTS, corner, CAMERA_MATRIX, DIST_COEFFS, flags=cv2.SOLVEPNP_IPPE_SQUARE
                        )
                        
                        if success:
                            x_offset = tvec[0][0]  # 横向偏移 (米)
                            z_dist = tvec[2][0]    # 纵向距离 (米)
                            
                            # 提取偏航角 Yaw
                            rmat, _ = cv2.Rodrigues(rvec)
                            euler_angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                            yaw_angle = euler_angles[1] # 提取 Y 轴旋转角度
                            
                            # 在画面上画出 3D 坐标轴，极度极客！
                            cv2.drawFrameAxes(vis, CAMERA_MATRIX, DIST_COEFFS, rvec, tvec, 0.05)
                        break

            if z_dist is None and tracker.currentState != TrackerState.ARRIVED:
                tracker.currentState = TrackerState.SEARCHING

            # 获取电机控制量
            L, R = tracker.updateTarget(x_offset, z_dist, yaw_angle)
            set_motor(board, speed_right=R, speed_left=L)

            # 在屏幕上打印 3D 数据
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
                break

            if tracker.currentState == TrackerState.ARRIVED:
                print("已经成功对齐并停靠在回仓点正前方！")
                speaker.speak(f"已完美到达位置 {target_id}，对齐完毕")
                break

    except Exception as e:
        print(f"ArUco Nav 异常: {e}")
    finally:
        set_motor(board, 0.0, 0.0)
        if cap: cap.release()
        cv2.destroyAllWindows()
        for _ in range(10): cv2.waitKey(1)
        _safe_remove_pid(pid_file)


class ArucoNavigationSystem:
    def __init__(self):
        self.pid_file = "/tmp/aruco_nav_pid.txt"
        self._process = None

    def start_aruco_navigation(self, video_source, target_id):
        import speaker
        if self._process is not None and self._process.is_alive():
            print("ArUco 导航正在运行")
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
            with open(self.pid_file, "w") as f: f.write(str(p.pid))
        except OSError: pass

    def stop_aruco_navigation(self):
        import speaker
        if self._process is not None:
            try:
                if self._process.is_alive():
                    self._process.terminate()
                    self._process.join(timeout=2.0)
            except Exception: pass
            finally: self._process = None
        _safe_remove_pid(self.pid_file)
        set_motor(None, 0.0, 0.0)
        speaker.speak("已停止 ArUco 导航")
