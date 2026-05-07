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

def set_motor(board, speed_right, speed_left, max_speed=150):
    if board is None:
        return
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
        pass

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
                    self.frame = cv2.flip(frame, -1) # 保持与你主程序一致的翻转
            else:
                time.sleep(0.005)

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

class RectF:
    def __init__(self, l, t, r, b):
        self.left, self.top, self.right, self.bottom = float(l), float(t), float(r), float(b)
    def width(self): return max(0.0, self.right - self.left)
    def height(self): return max(0.0, self.bottom - self.top)
    def centerX(self): return (self.left + self.right) * 0.5
    def centerY(self): return (self.top + self.bottom) * 0.5

class TrackerState(Enum):
    IDLE = auto()
    SEARCHING = auto()
    TRACKING = auto()
    ARRIVED = auto()

class QRTracker:
    def __init__(self, target_width=180.0):
        # 控制参数与行人追踪保持一致
        self.BASE_SPEED = 0.35
        self.STEERING_GAIN = 0.57
        self.DEAD_ZONE = 0.10
        self.MIN_STEER = 0.15
        
        self.TARGET_WIDTH = target_width  # 目标宽度(像素)，越大离二维码越近
        self.WIDTH_DEAD_ZONE = 15.0       # 距离误差死区

        self.currentState = TrackerState.SEARCHING
        self.currentL = 0.0
        self.currentR = 0.0
        self.frameW, self.frameH = 640, 480
        self.lastMoveDir = 1.0 # 1为右转，-1为左转
        self.arrived_start_time = None

    def updateTarget(self, qr_rect: RectF):
        tL = tR = 0.0

        if self.currentState == TrackerState.TRACKING and qr_rect:
            # 计算水平偏差 (与人脸追踪完全一致)
            error = 1.0 - 2.0 * qr_rect.centerX() / float(self.frameW)
            w = qr_rect.width()

            width_error = self.TARGET_WIDTH - w
            fwd = 0.0

            # 1. 检查是否到达
            if abs(error) <= self.DEAD_ZONE and abs(width_error) <= self.WIDTH_DEAD_ZONE:
                if self.arrived_start_time is None:
                    self.arrived_start_time = time.time()
                elif time.time() - self.arrived_start_time > 1.0:
                    self.currentState = TrackerState.ARRIVED
            else:
                self.arrived_start_time = None
                # 2. 计算前进/后退速度
                if width_error > self.WIDTH_DEAD_ZONE:
                    # 距离太远，前进，按比例减速
                    raw_fwd = self.BASE_SPEED * min(1.0, width_error / 80.0)
                    fwd = max(0.20, raw_fwd)
                elif width_error < -self.WIDTH_DEAD_ZONE:
                    # 距离太近，后退
                    fwd = -0.30

                # 3. 计算转向差速
                steer = 0.0
                if abs(error) > self.DEAD_ZONE:
                    steer = error * self.STEERING_GAIN
                    if 0 < steer < self.MIN_STEER: steer = self.MIN_STEER
                    if 0 > steer > -self.MIN_STEER: steer = -self.MIN_STEER
                
                steer = -steer # 适配硬件连线方向
                tL, tR = fwd - steer, fwd + steer

                # 更新最后一次的移动方向，用于丢失时搜索
                self.lastMoveDir = 1.0 if error < 0 else -1.0

        elif self.currentState == TrackerState.SEARCHING:
            # 原地旋转寻找
            spd = 0.35
            tL = spd if self.lastMoveDir > 0 else -spd
            tR = -spd if self.lastMoveDir > 0 else spd

        # 速度滤波平滑 (防止电机电流突变)
        self.currentL = 0.5 * self.currentL + 0.5 * tL
        self.currentR = 0.5 * self.currentR + 0.5 * tR

        self.currentL = max(-1.0, min(1.0, self.currentL))
        self.currentR = max(-1.0, min(1.0, self.currentR))

        if abs(self.currentL) < 0.05: self.currentL = 0.0
        if abs(self.currentR) < 0.05: self.currentR = 0.0

        return self.currentL, self.currentR

def background_qr_nav_task(video_source, target_qr_text, pid_file, tts_mp_q=None):
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
    qr_detector = cv2.QRCodeDetector()
    tracker = QRTracker(target_width=150.0) # <--- 调参：150为停止距离的像素宽度
    
    speaker.speak(f"开始寻找二维码基地：{target_qr_text}")

    try:
        cap = CameraReader(video_source)
        while cap.isOpened() and is_running:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            h, w = frame.shape[:2]
            tracker.frameW, tracker.frameH = w, h

            # OpenCV 解码二维码
            retval, decoded_info, points, _ = qr_detector.detectAndDecodeMulti(frame)
            
            qr_rect = None
            if retval and len(decoded_info) > 0:
                for i, info in enumerate(decoded_info):
                    if info == target_qr_text:
                        pts = points[i]
                        xs = pts[:, 0]
                        ys = pts[:, 1]
                        qr_rect = RectF(np.min(xs), np.min(ys), np.max(xs), np.max(ys))
                        tracker.currentState = TrackerState.TRACKING
                        break
            
            if qr_rect is None and tracker.currentState != TrackerState.ARRIVED:
                tracker.currentState = TrackerState.SEARCHING

            # 计算控制量
            L, R = tracker.updateTarget(qr_rect)
            set_motor(board, speed_right=R, speed_left=L)

            # UI 绘制
            vis = frame.copy()
            state_text = tracker.currentState.name
            color = (0, 255, 0) if qr_rect else (0, 165, 255)
            cv2.putText(vis, f"Target: [{target_qr_text}] State: {state_text}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            if qr_rect:
                cv2.rectangle(vis, (int(qr_rect.left), int(qr_rect.top)), 
                              (int(qr_rect.right), int(qr_rect.bottom)), (0, 255, 0), 3)

            cv2.imshow("QR Navigation", vis)
            key = cv2.waitKey(1) & 0xFF
            if key in [ord("e"), ord("q"), 27]:
                break

            if tracker.currentState == TrackerState.ARRIVED:
                print("已经成功对齐并停靠在二维码正前方！")
                speaker.speak(f"已到达{target_qr_text}，停车完毕")
                break

    except Exception as e:
        print(f"QR Nav 异常: {e}")
    finally:
        set_motor(board, 0.0, 0.0)
        if cap: cap.release()
        cv2.destroyAllWindows()
        for _ in range(10): cv2.waitKey(1)
        _safe_remove_pid(pid_file)


class QRNavigationSystem:
    def __init__(self):
        self.pid_file = "/tmp/qr_nav_pid.txt"
        self._process: multiprocessing.Process = None

    def start_qr_navigation(self, video_source, target_qr_text):
        import speaker
        if self._process is not None and self._process.is_alive():
            print("二维码导航正在运行")
            return
            
        _safe_remove_pid(self.pid_file)
        ctx = multiprocessing.get_context('spawn')
        if speaker._mp_q is None:
            speaker.init_mp_queue(ctx.Queue())

        p = ctx.Process(
            target=background_qr_nav_task,
            args=(video_source, target_qr_text, self.pid_file, speaker._mp_q),
            daemon=False,
        )
        p.start()
        self._process = p
        try:
            with open(self.pid_file, "w") as f: f.write(str(p.pid))
        except OSError: pass

    def stop_qr_navigation(self):
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
        speaker.speak("已停止二维码导航")