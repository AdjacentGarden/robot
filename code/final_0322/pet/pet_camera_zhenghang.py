import os
import sys
import cv2
import numpy as np
import onnxruntime as ort
import time
import math
import signal
import threading
import multiprocessing
from enum import Enum, auto
from typing import Dict, List
import warnings
from speaker import speak

warnings.filterwarnings("ignore")

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

DETECTOR_MODEL = "/home/test/openbot_test_zhenghang/model_0320/openbot_validify_code/openbot_model_tflite_and_onnx_0315_official/detector.onnx"
DET_CONF = 0.45
COCO_LABELS: Dict[int, str] = {17: "cat", 18: "dog"}

try:
    from wheel_control import Board
except ImportError:
    print("\n底盘驱动导入失败，使用虚拟车轮")
    class Board:
        def set_motor_speed(self, speeds): pass

def set_motor(board, speed_right, speed_left, max_speed=150):
    if board is None: return
    try:
        board.set_motor_speed([
            [1, int(max_speed * speed_right)],
            [2, int(max_speed * speed_left * -1)],
        ])
    except Exception: pass

def _safe_remove_pid(pid_file: str) -> None:
    if not pid_file or not os.path.exists(pid_file): return
    try: os.remove(pid_file)
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
        if not self.cap.isOpened(): raise RuntimeError(f"无法打开摄像头流: {src}")
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
                    self.frame = cv2.flip(frame, 1)
            else:
                time.sleep(0.005)
        if self.cap.isOpened(): self.cap.release()

    def read(self):
        with self._lock:
            if self.frame is not None: return self.ret, self.frame.copy()
        return False, None

    def isOpened(self): return not self._stop.is_set() and self.cap.isOpened()
    def release(self):
        self._stop.set()
        if self.thread.is_alive(): self.thread.join(timeout=2.0)
        if self.cap.isOpened(): self.cap.release()

class RectF:
    def __init__(self, left, top, right, bottom):
        self.left, self.top     = float(left),  float(top)
        self.right, self.bottom = float(right), float(bottom)
    def width(self):   return max(0.0, self.right  - self.left)
    def height(self):  return max(0.0, self.bottom - self.top)
    def centerX(self): return (self.left  + self.right)  * 0.5
    def centerY(self): return (self.top   + self.bottom) * 0.5

class Detection:
    def __init__(self, title, confidence, rect: RectF):
        self.title, self.confidence, self.rect = title, float(confidence), rect

class OnnxDetector:
    INPUT_W = 300; INPUT_H = 300
    def __init__(self, path, conf=0.45):
        self.conf = conf
        so = ort.SessionOptions(); so.intra_op_num_threads = 2; so.log_severity_level = 3
        self.sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
        self.inp_name = self.sess.get_inputs()[0].name
        self.inp_type = self.sess.get_inputs()[0].type
        all_out = [o.name for o in self.sess.get_outputs()]
        self._idx = {n: i for i, n in enumerate(all_out)}
        if "StatefulPartitionedCall:3" in self._idx:
            self._nm = {
                "num":     "StatefulPartitionedCall:0",
                "scores":  "StatefulPartitionedCall:1",
                "classes": "StatefulPartitionedCall:2",
                "boxes":   "StatefulPartitionedCall:3",
            }
        else: self._nm = None

    def _pre(self, bgr):
        img = cv2.resize(bgr, (self.INPUT_W, self.INPUT_H), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) if "float" in self.inp_type else img.astype(np.uint8)
        return img[np.newaxis]

    def detect(self, bgr, W, H, target_classes: List[str]) -> List[Detection]:
        inp  = self._pre(bgr)
        outs = self.sess.run(None, {self.inp_name: inp})
        if self._nm:
            def g(k): return outs[self._idx[self._nm[k]]]
            boxes, clss, scs, nd = g("boxes").squeeze(), g("classes").squeeze(), g("scores").squeeze(), int(g("num").squeeze())
        else:
            boxes, clss, scs = outs[0].squeeze(), outs[1].squeeze(), outs[2].squeeze()
            nd = int(outs[3].squeeze()) if len(outs) >= 4 else boxes.shape[0]

        res = []
        for i in range(min(nd, int(boxes.shape[0]))):
            sc = float(scs[i])
            if sc < self.conf: continue
            lbl = COCO_LABELS.get(int(round(float(clss[i]))) + 1, "?")
            if lbl not in target_classes: continue
            ymin, xmin, ymax, xmax = boxes[i]
            x1 = max(0.0, min(float(xmin) * W, float(W)))
            y1 = max(0.0, min(float(ymin) * H, float(H)))
            x2 = max(0.0, min(float(xmax) * W, float(W)))
            y2 = max(0.0, min(float(ymax) * H, float(H)))
            if x2 > x1 and y2 > y1: res.append(Detection(lbl, sc, RectF(x1, y1, x2, y2)))
        return res
    
class TrackerState(Enum):
    IDLE = auto(); TRACKING = auto(); BUFFER_WAIT = auto(); SEARCHING = auto()

class DrawBox:
    def __init__(self, rect: RectF, title: str, score: float, is_target: bool):
        self.rect, self.title, self.score, self.is_target = rect, title, score, is_target

class MultiBoxTracker:
    MIN_SIZE      = 24.0; BASE_SPEED    = 0.40; STEERING_GAIN = 0.57
    DEAD_ZONE     = 0.03; MIN_STEER     = 0.15; RAMP_STEP     = 0.27

    def __init__(self):
        self.currentState        = TrackerState.IDLE
        self.lastKnownLocation: RectF = None
        self.lastMoveDirection   = 0.0
        self.currentLeftSpeed    = 0.0
        self.currentRightSpeed   = 0.0
        self.frameWidth          = 640
        self.frameHeight         = 480
        self.drawBoxes: List[DrawBox] = []
        
        self.lastSeenTime    = time.time()
        self.searchStartTime = 0.0

        self.stationary_start_time = None 
        self.is_backing_up = False

    def trackResults(self, results: List[Detection], currFrame=None):
        if currFrame is not None:
            self.frameWidth, self.frameHeight = currFrame.shape[1], currFrame.shape[0]
        self.drawBoxes.clear()
        valid = [r for r in results if r.rect.width() >= self.MIN_SIZE and r.rect.height() >= self.MIN_SIZE]

        if self.currentState == TrackerState.IDLE:
            best, best_sc = None, -1.0
            for det in valid:
                dx = det.rect.centerX() - self.frameWidth  / 2.0
                dy = det.rect.centerY() - self.frameHeight / 2.0
                sc = (det.rect.width() * det.rect.height()) * (1.0 - math.sqrt(dx * dx + dy * dy) / self.frameWidth)
                if sc > best_sc: best_sc, best = sc, det
                self.drawBoxes.append(DrawBox(det.rect, det.title, det.confidence, False))
            if best is not None:
                self.lastKnownLocation = best.rect
                self.currentState      = TrackerState.TRACKING
                self.lastSeenTime      = time.time()  
            return

        best, min_d = None, float("inf")
        for det in valid:
            if self.lastKnownLocation is not None:
                dx   = det.rect.centerX() - self.lastKnownLocation.centerX()
                dy   = det.rect.centerY() - self.lastKnownLocation.centerY()
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < min_d and dist < self.frameWidth * 0.4:
                    min_d, best = dist, det
            else:
                best = det

        current_time = time.time()

        if best is not None:
            if self.lastKnownLocation is not None:
                dx = best.rect.centerX() - self.lastKnownLocation.centerX()
                self.lastMoveDirection = self.lastMoveDirection * 0.8 + dx * 0.2
            self.lastKnownLocation = best.rect
            self.currentState      = TrackerState.TRACKING
            self.lastSeenTime      = current_time  
            for det in valid:
                self.drawBoxes.append(DrawBox(det.rect, det.title, det.confidence, det is best))
        else:
            if self.currentState == TrackerState.TRACKING:
                self.currentState = TrackerState.BUFFER_WAIT
            elif self.currentState == TrackerState.BUFFER_WAIT:
                if current_time - self.lastSeenTime > 2.0:
                    self.currentState = TrackerState.SEARCHING
                    self.searchStartTime = current_time
            elif self.currentState == TrackerState.SEARCHING:
                if current_time - self.searchStartTime > 6.0:
                    self.currentState = TrackerState.IDLE
                    self.lastKnownLocation = None
            for det in valid:
                self.drawBoxes.append(DrawBox(det.rect, det.title, det.confidence, False))

    def updateTarget(self):
        tL = tR = 0.0
        if self.currentState == TrackerState.TRACKING and self.lastKnownLocation is not None:
            error  = 1.0 - 2.0 * self.lastKnownLocation.centerX() / float(self.frameWidth)
            abs_e  = abs(error)
            
            area   = (self.lastKnownLocation.width() * self.lastKnownLocation.height()) / (self.frameWidth * self.frameHeight)
            hr     = self.lastKnownLocation.height() / float(self.frameHeight)
            fwd    = 0.0

            is_stationary = abs(self.currentLeftSpeed) < 0.05 and abs(self.currentRightSpeed) < 0.05
            if is_stationary:
                if self.stationary_start_time is None:
                    self.stationary_start_time = time.time()
            else:
                self.stationary_start_time = None

            if self.is_backing_up:
                if hr < 0.90:
                    self.is_backing_up = False  
                else:
                    fwd = -self.BASE_SPEED 
            else:
                if hr < 0.80 and area < 0.40:
                    if hr <= 0.50:
                        raw_fwd = self.BASE_SPEED * 2.5
                    else:
                        raw_fwd = self.BASE_SPEED * ((0.80 - hr) / 0.30) * 2.5
                    fwd = max(0.0, raw_fwd)
                
                if self.stationary_start_time is not None and (time.time() - self.stationary_start_time) > 1.0:
                    if hr > 0.90 or area > 0.55:
                        self.is_backing_up = True

            dz = (self.DEAD_ZONE * 3.0 + area * 0.05) if fwd == 0.0 else self.DEAD_ZONE
            steer = 0.0
            
            if abs_e > dz:
                steer = error * self.STEERING_GAIN * 0.40
                if fwd < 0.0: 
                    steer = -steer 
                ms = self.MIN_STEER * 2.0 if fwd == 0.0 else self.MIN_STEER
                if 0   < steer < ms:  steer =  ms
                if 0   > steer > -ms: steer = -ms
            tL, tR = fwd - steer, fwd + steer

        elif self.currentState == TrackerState.BUFFER_WAIT:
            tL = tR = 0.0
            self.is_backing_up = False
            self.stationary_start_time = None

        elif self.currentState == TrackerState.SEARCHING:
            spd = 0.4
            tL  =  spd if self.lastMoveDirection > 0 else -spd
            tR  = -spd if self.lastMoveDirection > 0 else  spd
            self.is_backing_up = False
            self.stationary_start_time = None
            
        elif self.currentState == TrackerState.IDLE:
            tL = tR = 0.0
            self.is_backing_up = False
            self.stationary_start_time = None

        self.currentLeftSpeed  = 0.5 * self.currentLeftSpeed  + 0.5 * tL
        self.currentRightSpeed = 0.5 * self.currentRightSpeed + 0.5 * tR

        self.currentLeftSpeed  = max(-1.0, min(1.0, self.currentLeftSpeed))
        self.currentRightSpeed = max(-1.0, min(1.0, self.currentRightSpeed))

        if abs(self.currentLeftSpeed)  < 0.05: self.currentLeftSpeed  = 0.0
        if abs(self.currentRightSpeed) < 0.05: self.currentRightSpeed = 0.0

        return self.currentLeftSpeed, self.currentRightSpeed

def _spd_bar(canvas, cx, cy, speed, label):
    bh, bw = 100, 26; x0, y0 = cx - bw // 2, cy - bh // 2
    cv2.rectangle(canvas, (x0, y0), (x0 + bw, y0 + bh), (50, 50, 50), -1)
    fill = int(abs(speed) * bh / 2)
    col  = (50, 220, 50) if speed >= 0 else (50, 50, 220)
    if speed >= 0: cv2.rectangle(canvas, (x0 + 2, y0 + bh // 2 - fill), (x0 + bw - 2, y0 + bh // 2), col, -1)
    else:          cv2.rectangle(canvas, (x0 + 2, y0 + bh // 2), (x0 + bw - 2, y0 + bh // 2 + fill), col, -1)
    cv2.rectangle(canvas, (x0, y0), (x0 + bw, y0 + bh), (180, 180, 180), 1)
    cv2.line(canvas, (x0, cy), (x0 + bw, cy), (255, 255, 255), 1)
    cv2.putText(canvas, label, (cx - 8, y0 + bh + 16), cv2.FONT_HERSHEY_SIMPLEX, .44, (220, 220, 220), 1)
    cv2.putText(canvas, f"{speed:+.2f}", (cx - 22, y0 + bh + 30), cv2.FONT_HERSHEY_SIMPLEX, .44, (220, 220, 220), 1)


def draw_tracking_ui(frame, tracker: MultiBoxTracker, target_pet: str):
    vis = frame.copy(); H, W = vis.shape[:2]
    for db in tracker.drawBoxes:
        color  = (0, 255, 0) if db.is_target else (0, 255, 255)
        status = "Target" if db.is_target else "Other"
        x1, y1 = int(db.rect.left), int(db.rect.top)
        x2, y2 = int(db.rect.right), int(db.rect.bottom)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
        cv2.putText(vis, f"{db.title}|{status}({db.score:.2f})",
                    (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    _spd_bar(vis, 36, H - 75, tracker.currentLeftSpeed,  "L")
    _spd_bar(vis, 76, H - 75, tracker.currentRightSpeed, "R")
    for i, t in enumerate([f"Mode: Follow [{target_pet}]", f"State: {tracker.currentState.name}"]):
        cv2.putText(vis, t, (W - 240, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2, cv2.LINE_AA)
    return vis

def background_pet_search_task(video_source, model_path, target_pet):
    board = Board()
    detector = OnnxDetector(model_path, conf=DET_CONF)
    cap = None
    found = False
    
    pet_dict = {"cat": "小猫", "dog": "小狗"}
    pet_name = pet_dict.get(target_pet, target_pet)

    try:
        cap = CameraReader(video_source)
        # print(f"开启 6 秒原地旋转搜索模式寻找 [{pet_name}]...")
        start_time = time.time()
        while cap.isOpened() and (time.time() - start_time) < 6.0:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue
            h, w  = frame.shape[:2]
            dets  = detector.detect(frame, w, h, target_classes=[target_pet])
            vis   = frame.copy()
            cv2.putText(vis, f"Searching for [{pet_name}]...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
            
            if dets:
                found = True
                set_motor(board, 0.0, 0.0)
                for det in dets:
                    x1, y1 = int(det.rect.left), int(det.rect.top)
                    x2, y2 = int(det.rect.right), int(det.rect.bottom)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(vis, "FOUND", (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Pet Search", vis)
                cv2.waitKey(800)
                break
                    
            cv2.imshow("Pet Search", vis)
            cv2.waitKey(1)
            set_motor(board, speed_right=0.4, speed_left=-0.4)

    finally:
        set_motor(board, 0.0, 0.0)
        if found:
            speak(f"这里有一只{pet_name}")
        else:
            speak(f"抱歉，我没有发现{pet_name}")
        if cap is not None:
            try: cap.release()
            except Exception: pass
        try:
            cv2.destroyAllWindows()
            for _ in range(10): 
                cv2.waitKey(1) 
        except Exception: pass

def background_tracking_task(video_source, model_path, target_pet, pid_file_path):
    is_running = True
    def handle_sigterm(signum, frame_obj):
        nonlocal is_running
        is_running = False     

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT,  handle_sigterm)
    pet_dict = {"cat": "小猫", "dog": "小狗"}
    pet_name = pet_dict.get(target_pet, "宠物")
    cap   = None
    board = None

    try:
        board    = Board()
        detector = OnnxDetector(model_path, conf=DET_CONF)
        tracker  = MultiBoxTracker()
        cap      = CameraReader(video_source)
        
        has_tracked = False  
        start_time = time.time()

        while cap.isOpened() and is_running:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            h, w = frame.shape[:2]
            dets = detector.detect(frame, w, h, target_classes=[target_pet])
            tracker.trackResults(dets, frame)
            L, R = tracker.updateTarget()
            set_motor(board, speed_right=R, speed_left=L)

            vis = draw_tracking_ui(frame, tracker, target_pet)
            cv2.imshow("Pet Follower", vis)
            key = cv2.waitKey(1) & 0xFF
            if key in [ord("x"), ord("e"), ord("q"), 27]:
                print("\n收到按键退出指令")
                break

            if tracker.currentState == TrackerState.TRACKING:
                has_tracked = True
            elif tracker.currentState == TrackerState.IDLE and has_tracked:
                speak(f"{pet_name}不见了")
                # with open('/tmp/pet_tracking_result.txt', 'w', encoding='utf-8') as f: 
                #     f.write(msg) 
                break
            elif not has_tracked and (time.time() - start_time) > 5.0:
                speak(f"寻找超时，附近没有发现{pet_name}，已自动停止")
                break

    except Exception as e:
        print(f"异常: {e}")

    finally:
        set_motor(board, 0.0, 0.0)
        if cap is not None:
            try: cap.release()
            except Exception:
                pass
        try:
            cv2.destroyAllWindows()
            for _ in range(10): cv2.waitKey(1) 
        except Exception: pass
    
        _safe_remove_pid(pid_file_path)
        # msg = f'已退出，摄像头资源已释放'
        # with open('/tmp/pet_tracking_result.txt', 'w', encoding='utf-8') as f: 
        #             f.write(msg)

class PetTrackingSystem:
    def __init__(self, model_path=DETECTOR_MODEL):
        self.model_path = model_path
        self.pid_file   = "/tmp/pet_tracking_pid.txt"   
        self._process: multiprocessing.Process = None
        
    def find_pet(self, video_source, target_pet):
        print(f"启动寻宠进程: {target_pet}")
        p = multiprocessing.Process(
            target=background_pet_search_task,
            args=(video_source, self.model_path, target_pet),
            daemon=True
        )
        p.start()
        p.join() 
        print("寻宠进程已退出")

    def start_pet_tracking(self, video_source, target_pet):
        print(f"启动进程跟踪宠物: {target_pet}")
        if self._process is not None and self._process.is_alive():
            return
        _safe_remove_pid(self.pid_file)

        p = multiprocessing.Process(
            target=background_tracking_task,
            args=(video_source, self.model_path, target_pet, self.pid_file),
            daemon=True,
        )
        p.start()
        self._process = p
        try:
            with open(self.pid_file, "w") as f: f.write(str(p.pid))
        except OSError as e: 
            print(f"写入 PID 文件失败: {e}")

    def stop_pet_tracking(self):
        try: 
            self._terminate_process()
        finally:
            _safe_remove_pid(self.pid_file)
        set_motor(None, 0.0, 0.0)
        speak('已经关闭宠物跟踪进程')

    def _terminate_process(self):
        if self._process is not None:
            try:
                if self._process.is_alive():
                    self._process.terminate()
                    self._process.join(timeout=3.0)
                    if self._process.is_alive():
                        self._process.kill()
                        self._process.join(timeout=2.0)
            except Exception as e: pass
            finally: self._process = None
            return

        if not os.path.exists(self.pid_file): return
        try:
            pid_str = open(self.pid_file).read().strip()
            if not pid_str.isdigit(): return
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
