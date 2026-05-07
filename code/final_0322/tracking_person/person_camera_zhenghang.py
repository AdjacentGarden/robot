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
import random       
import datetime
from speaker import speak

warnings.filterwarnings("ignore")

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

DETECTOR_MODEL = "/home/test/openbot_test_zhenghang/model_0320/openbot_validify_code/openbot_model_tflite_and_onnx_0315_official/detector.onnx"
REID_MODEL     = "/home/test/openbot_test_zhenghang/model_0320/openbot_validify_code/openbot_model_tflite_and_onnx_0315_official/reid_model.onnx"
DET_CONF       = 0.45
COCO_LABELS: Dict[int, str] = {1: "person"}

try:
    from wheel_control import Board
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
        if not self.cap.isOpened(): raise RuntimeError(f"can not open camera: {src}")
        self._lock  = threading.Lock()
        self._stop  = threading.Event()
        self.ret, self.frame = self.cap.read()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while not self._stop.is_set():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self._lock:
                    self.ret   = ret
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
    def __init__(self, l, t, r, b):
        self.left, self.top, self.right, self.bottom = float(l), float(t), float(r), float(b)
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
            self._nm = {"num": "StatefulPartitionedCall:0", "scores": "StatefulPartitionedCall:1",
                        "classes": "StatefulPartitionedCall:2", "boxes": "StatefulPartitionedCall:3"}
        else: self._nm = None

    def _pre(self, bgr):
        img = cv2.resize(bgr, (self.INPUT_W, self.INPUT_H), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return (img.astype(np.float32) if "float" in self.inp_type else img.astype(np.uint8))[np.newaxis]

    def detect(self, bgr, W, H, cls="person") -> List[Detection]:
        outs = self.sess.run(None, {self.inp_name: self._pre(bgr)})
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
            if lbl != cls: continue
            ymin, xmin, ymax, xmax = boxes[i]
            x1 = max(0., min(float(xmin)*W, float(W))); y1 = max(0., min(float(ymin)*H, float(H)))
            x2 = max(0., min(float(xmax)*W, float(W))); y2 = max(0., min(float(ymax)*H, float(H)))
            if x2 > x1 and y2 > y1: res.append(Detection(lbl, sc, RectF(x1, y1, x2, y2)))
        return res

class OnnxReID:
    W = 128; H = 256; DIM = 512
    MEAN = np.array([123.675, 116.280, 103.530], dtype=np.float32)
    STD  = np.array([ 58.395,  57.120,  57.375], dtype=np.float32)
    def __init__(self, path):
        so = ort.SessionOptions(); so.intra_op_num_threads = 2; so.log_severity_level = 3
        self.sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
        self.inp  = self.sess.get_inputs()[0].name
        self.out  = self.sess.get_outputs()[0].name
        sh        = self.sess.get_inputs()[0].shape
        self.nchw = (len(sh) == 4 and sh[1] == 3)

    def run(self, crop):
        if crop is None or crop.size == 0: return np.zeros(self.DIM, dtype=np.float32)
        h, w = crop.shape[:2]
        sc = min(self.W / w, self.H / h)
        nw, nh = max(1, int(w * sc)), max(1, int(h * sc))
        r   = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)
        img = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        dx, dy = (self.W - nw) // 2, (self.H - nh) // 2
        img[dy:dy+nh, dx:dx+nw] = r
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = (img - self.MEAN) / self.STD
        t   = img.transpose(2, 0, 1)[np.newaxis] if self.nchw else img[np.newaxis]
        feat = self.sess.run([self.out], {self.inp: t})[0].squeeze()
        n    = float(np.linalg.norm(feat))
        return (feat / n).astype(np.float32) if n > 1e-9 else feat.astype(np.float32)

class TrackerState(Enum):
    IDLE = auto(); TRACKING = auto(); BUFFER_WAIT = auto(); SEARCHING = auto()

class DrawBox:
    def __init__(self, rect: RectF, score: float, is_target: bool):
        self.rect, self.score, self.is_target = rect, score, is_target

class MultiBoxTracker:
    MIN_SIZE      = 24.0; BASE_SPEED    = 0.40; STEERING_GAIN = 0.57
    DEAD_ZONE     = 0.10; MIN_STEER     = 0.15
    GALLERY_SIZE  = 16;   SIM_THRESHOLD = 0.55; UPDATE_INTERVAL = 10

    class _Cand:
        def __init__(self, rect, feat): self.rect, self.feat, self.score = rect, feat, 0.0

    def __init__(self, reid: OnnxReID):
        self.reid = reid
        self.currentState = TrackerState.IDLE
        self.gallery: List[np.ndarray] = []
        self.lastKnown    = None
        self.lastMoveDir  = 0.0
        self.framesSinceUpd = 0
        self.currentL = self.currentR = 0.0
        self.frameW, self.frameH = 640, 480
        self.drawBoxes: List[DrawBox] = []        
        self.lastSeenTime = time.time()  
        self.searchStartTime = 0.0       
        self.stationary_start_time = None 
        self.is_backing_up = False

    def _crop(self, bmp, rect):
        sh, sw = bmp.shape[:2]
        x, y   = max(0, int(rect.left)), max(0, int(rect.top))
        w, h   = min(int(rect.width()), sw - x), min(int(rect.height()), sh - y)
        return bmp[y:y+h, x:x+w] if w > 0 and h > 0 else None

    def _cos(self, f1, f2):
        n1, n2 = np.linalg.norm(f1), np.linalg.norm(f2)
        return float(np.dot(f1, f2) / (n1 * n2)) if n1 * n2 > 1e-9 else 0.0

    def trackResults(self, results: List[Detection], frame):
        if frame is not None:
            self.frameW, self.frameH = frame.shape[1], frame.shape[0]
        self.drawBoxes.clear()
        cands = []
        for r in results:
            if r.rect.width() < self.MIN_SIZE or r.rect.height() < self.MIN_SIZE: continue
            crop = self._crop(frame, r.rect)
            if crop is None: continue
            cands.append(self._Cand(r.rect, self.reid.run(crop)))

        if self.currentState == TrackerState.IDLE:
            best, bs = None, -1.0
            for c in cands:
                dx = c.rect.centerX() - self.frameW / 2.0
                dy = c.rect.centerY() - self.frameH / 2.0
                sc = (c.rect.width() * c.rect.height()) * (1.0 - math.sqrt(dx*dx + dy*dy) / self.frameW)
                if sc > bs: bs, best = sc, c
                self.drawBoxes.append(DrawBox(c.rect, 0.0, False))
            if best:
                self.gallery   = [best.feat.copy()]
                self.lastKnown = best.rect
                self.currentState = TrackerState.TRACKING
                self.lastSeenTime = time.time()
            return

        best, bd = None, -100.0
        for c in cands:
            c.score = max(self._cos(g, c.feat) for g in self.gallery) if self.gallery else 0.0
            dec = c.score
            if self.currentState == TrackerState.TRACKING and self.lastKnown:
                dec -= math.sqrt((c.rect.centerX() - self.lastKnown.centerX()) ** 2 + (c.rect.centerY() - self.lastKnown.centerY()) ** 2) / self.frameW * 0.5
            if dec > bd: bd, best = dec, c

        current_time = time.time()

        if best and best.score >= self.SIM_THRESHOLD:
            if self.lastKnown:
                dx = best.rect.centerX() - self.lastKnown.centerX()
                self.lastMoveDir = self.lastMoveDir * 0.8 + dx * 0.2
            self.lastKnown    = best.rect
            self.currentState = TrackerState.TRACKING
            self.lastSeenTime = current_time 
            self.framesSinceUpd += 1
            if self.framesSinceUpd >= self.UPDATE_INTERVAL:
                if len(self.gallery) >= self.GALLERY_SIZE: self.gallery.pop(0)
                self.gallery.append(best.feat.copy())
                self.framesSinceUpd = 0
            for c in cands: self.drawBoxes.append(DrawBox(c.rect, c.score, c is best))
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
                    self.gallery.clear()
                    self.lastKnown = None
            for c in cands: self.drawBoxes.append(DrawBox(c.rect, c.score, False))

    def updateTarget(self):
        tL = tR = 0.0
        if self.currentState == TrackerState.TRACKING and self.lastKnown:
            error = 1.0 - 2.0 * self.lastKnown.centerX() / float(self.frameW)
            area  = (self.lastKnown.width() * self.lastKnown.height()) / (self.frameW * self.frameH)
            hr    = self.lastKnown.height() / float(self.frameH)
            fwd = 0.0

            is_stationary = abs(self.currentL) < 0.05 and abs(self.currentR) < 0.05
            if is_stationary:
                if self.stationary_start_time is None: self.stationary_start_time = time.time()
            else: self.stationary_start_time = None

            if self.is_backing_up:
                if hr < 0.90: self.is_backing_up = False  
                else: fwd = -0.40
            else:
                if hr < 0.80 and area < 0.40:
                    if hr <= 0.50: raw_fwd = self.BASE_SPEED * 2.5
                    else:          raw_fwd = self.BASE_SPEED * ((0.80 - hr) / 0.30) * 2.5
                    fwd = max(0.0, raw_fwd)
                
                if self.stationary_start_time is not None and (time.time() - self.stationary_start_time) > 1.0:
                    if hr > 0.90: self.is_backing_up = True

            steer = 0.0
            if abs(error) > self.DEAD_ZONE:
                steer = error * self.STEERING_GAIN
                if 0  < steer < self.MIN_STEER:  steer =  self.MIN_STEER
                if 0  > steer > -self.MIN_STEER: steer = -self.MIN_STEER
            tL, tR = fwd - steer, fwd + steer
            
        elif self.currentState == TrackerState.BUFFER_WAIT:
            tL = tR = 0.0
            self.is_backing_up = False
            self.stationary_start_time = None
            
        elif self.currentState == TrackerState.SEARCHING:
            spd = 0.4
            tL  =  spd if self.lastMoveDir > 0 else -spd
            tR  = -spd if self.lastMoveDir > 0 else  spd
            self.is_backing_up = False
            self.stationary_start_time = None
            
        elif self.currentState == TrackerState.IDLE:
            tL = tR = 0.0
            self.is_backing_up = False
            self.stationary_start_time = None

        self.currentL = 0.5 * self.currentL + 0.5 * tL
        self.currentR = 0.5 * self.currentR + 0.5 * tR

        self.currentL = max(-1.0, min(1.0, self.currentL))
        self.currentR = max(-1.0, min(1.0, self.currentR))

        if abs(self.currentL) < 0.05: self.currentL = 0.0
        if abs(self.currentR) < 0.05: self.currentR = 0.0

        return self.currentL, self.currentR

def _spd_bar(canvas, cx, cy, speed, label):
    bh, bw = 100, 26; x0, y0 = cx-bw//2, cy-bh//2
    cv2.rectangle(canvas, (x0,y0), (x0+bw,y0+bh), (50,50,50), -1)
    fill = int(abs(speed)*bh/2); col = (50,220,50) if speed >= 0 else (50,50,220)
    if speed >= 0: cv2.rectangle(canvas, (x0+2,y0+bh//2-fill), (x0+bw-2,y0+bh//2), col, -1)
    else:          cv2.rectangle(canvas, (x0+2,y0+bh//2), (x0+bw-2,y0+bh//2+fill), col, -1)
    cv2.rectangle(canvas, (x0,y0), (x0+bw,y0+bh), (180,180,180), 1)
    cv2.line(canvas, (x0,cy), (x0+bw,cy), (255,255,255), 1)
    cv2.putText(canvas, label, (cx-8,y0+bh+16), cv2.FONT_HERSHEY_SIMPLEX, .44, (220,220,220), 1)
    cv2.putText(canvas, f"{speed:+.2f}", (cx-22,y0+bh+30), cv2.FONT_HERSHEY_SIMPLEX, .44, (220,220,220), 1)

def draw_tracking_ui(frame, tracker: MultiBoxTracker, target_name: str):
    vis = frame.copy(); H, W = vis.shape[:2]
    for db in tracker.drawBoxes:
        color = (0,255,0) if db.is_target else (0,255,255)
        x1,y1 = int(db.rect.left),int(db.rect.top); x2,y2 = int(db.rect.right),int(db.rect.bottom)
        cv2.rectangle(vis,(x1,y1),(x2,y2),color,3)
        cv2.putText(vis, f"{'Target:'+target_name if db.is_target else 'Other'}({db.score:.2f})",
                    (x1,max(20,y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    _spd_bar(vis,36,H-75,tracker.currentL,"L"); _spd_bar(vis,76,H-75,tracker.currentR,"R")
    for i,t in enumerate([f"Mode: Follow [{target_name}]",
                           f"State: {tracker.currentState.name}",
                           f"Gallery: {len(tracker.gallery)}/{tracker.GALLERY_SIZE}"]):
        cv2.putText(vis,t,(W-240,30+i*25),cv2.FONT_HERSHEY_SIMPLEX,0.6,(220,220,220),2,cv2.LINE_AA)
    return vis

def background_person_search_task(video_source, detector_path, target_name):
    board    = Board()
    detector = OnnxDetector(detector_path, conf=DET_CONF)
    cap   = None
    found = False
    time.sleep(2.0)

    try:
        cap = CameraReader(video_source)
        # print("开启 6 秒原地旋转搜索模式...")
        start_time = time.time()
        while cap.isOpened() and (time.time() - start_time) < 6.0:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            set_motor(board, speed_right=0.4, speed_left=-0.4)
            h, w  = frame.shape[:2]
            dets  = detector.detect(frame, w, h, cls="person")
            vis   = frame.copy()
            cv2.putText(vis, f"Searching for [{target_name}]...", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,165,255), 2)

            if dets:
                found = True
                set_motor(board, 0.0, 0.0)
                for det in dets:
                    x1,y1 = int(det.rect.left),int(det.rect.top)
                    x2,y2 = int(det.rect.right),int(det.rect.bottom)
                    cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,0),3)
                    cv2.putText(vis,"FOUND!",(x1,max(20,y1-10)), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                cv2.imshow("Person Search", vis)
                cv2.waitKey(800)
                break
            
            cv2.imshow("Person Search", vis)
            cv2.waitKey(1)

    finally:
        set_motor(board, 0.0, 0.0)
        if cap is not None:
            try: cap.release()
            except Exception: pass
        try:
            cv2.destroyAllWindows()
            for _ in range(10): cv2.waitKey(1) 
        except Exception: pass

    if found: 
        speak(f"{target_name}在这里")

def background_person_tracking_task(video_source, detector_path, reid_path, target_name, pid_file_path):
    is_running = True
    def handle_sigterm(signum, frame_obj):
        nonlocal is_running
        is_running = False  
    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT,  handle_sigterm)

    cap = board = None
    video_writer = None 
    
    record_dir = os.path.join(_CURRENT_DIR, "tracking_records")
    video_dir = os.path.join(record_dir, "videos")
    img_dir = os.path.join(record_dir, "screenshots")
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    
    session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(video_dir, f"track_{target_name}_{session_id}.mp4")

    try:
        board    = Board()
        detector = OnnxDetector(detector_path, conf=DET_CONF)
        reid     = OnnxReID(reid_path)       
        tracker  = MultiBoxTracker(reid)
        cap      = CameraReader(video_source)
        
        has_tracked = False  

        while cap.isOpened() and is_running:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            h, w  = frame.shape[:2]
            dets  = detector.detect(frame, w, h, cls="person")
            tracker.trackResults(dets, frame)
            L, R  = tracker.updateTarget()
            set_motor(board, speed_right=R, speed_left=L)

            vis = draw_tracking_ui(frame, tracker, target_name)

            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
                video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (w, h))
            
            if video_writer is not None:
                video_writer.write(vis) 
            if random.random() < 0.03:
                img_name = f"shot_{session_id}_{int(time.time() * 1000)}.jpg"
                cv2.imwrite(os.path.join(img_dir, img_name), vis)

            cv2.imshow("Person Follower", vis)
            key = cv2.waitKey(1) & 0xFF
            if key in [ord("3"), ord("e"), ord("q"), 27]:
                print("\n收到按键退出指令")
                break

            if tracker.currentState == TrackerState.TRACKING:
                has_tracked = True  
            elif tracker.currentState == TrackerState.IDLE and has_tracked:
                speak(f"{target_name}彻底跟丢了")
                print(f"{target_name}彻底跟丢了")
                break

    except Exception as e:
        print(f"异常: {e}")

    finally:
        set_motor(board, 0.0, 0.0)
        if video_writer is not None:
            video_writer.release()
            
        if cap is not None:
            try:   cap.release()
            except Exception: pass
        try:
            cv2.destroyAllWindows()
            for _ in range(10): cv2.waitKey(1)
        except Exception: pass
        _safe_remove_pid(pid_file_path)
        print("已退出，摄像头资源及视频文件已保存并释放")

class PersonSearchTrackingSystem:
    def __init__(self, detector_path=DETECTOR_MODEL, reid_path=REID_MODEL):
        self.detector_model = detector_path
        self.reid_model     = reid_path
        self.pid_file       = "/tmp/person_tracking_pid.txt"
        self._process: multiprocessing.Process = None

    def search_person(self, video_source, target_name):
        p = multiprocessing.Process(
            target=background_person_search_task,
            args=(video_source, self.detector_model, target_name),
            daemon=True
        )
        p.start()
        p.join() 

    def start_person_tracking(self, video_source, target_name):
        print(f"启动进程跟踪人物: {target_name}")
        if self._process is not None and self._process.is_alive():
            print("人物追踪任务已经在后台运行中")
            return
        _safe_remove_pid(self.pid_file)

        p = multiprocessing.Process(
            target=background_person_tracking_task,
            args=(video_source, self.detector_model, self.reid_model, target_name, self.pid_file),
            daemon=True,
        )
        p.start()
        self._process = p
        try:
            with open(self.pid_file, "w") as f: f.write(str(p.pid))
        except OSError as e: print(f" 写入 PID 文件失败: {e}")

    def stop_person_tracking(self):
        print("Kill 人物跟踪进程")
        try: self._terminate_process()
        finally: _safe_remove_pid(self.pid_file)  
        set_motor(None, 0.0, 0.0)
        speak("好的，已停止人物追踪")

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
