import os
import sys
import cv2
import numpy as np
import time
import math
import signal
import subprocess
import threading
from enum import Enum, auto
from typing import Dict, List
import warnings
from speaker import speak

from rknn3lite.api import RKNN3Lite

warnings.filterwarnings("ignore")


def _detect_gui_available() -> bool:
    gui_env = os.getenv("PET_CAMERA_GUI", "").strip().lower()
    if gui_env in {"0", "false", "off", "no"}:
        return False
    if gui_env in {"1", "true", "on", "yes"}:
        return True
    if not os.getenv("DISPLAY"):
        return False
    try:
        probe = subprocess.run(
            ["xdpyinfo"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=1.0,
            check=False,
        )
        return probe.returncode == 0
    except FileNotFoundError:
        return True
    except Exception:
        return False


GUI_AVAILABLE = _detect_gui_available()
if not GUI_AVAILABLE:
    print("[PetCamera] 未检测到可用图形显示环境，已启用无窗口模式")


def _show_frame(window_name: str, frame, delay: int = 1) -> int:
    if not GUI_AVAILABLE:
        return -1
    cv2.imshow(window_name, frame)
    if delay <= 0:
        return -1
    return cv2.waitKey(delay) & 0xFF


def _close_windows() -> None:
    if not GUI_AVAILABLE:
        return
    try:
        cv2.destroyAllWindows()
        for _ in range(10):
            cv2.waitKey(1)
    except Exception:
        pass


_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

# ! 
DETECTOR_MODEL = "/home/test/yolov8s/yolov8s_rknn3.rknn"
WEIGHT_MODEL   = "/home/test/yolov8s/yolov8s_rknn3.weight"

DET_CONF = 0.25
TRACK_SEARCH_TIMEOUT_SEC = 18.0
TRACK_RECORD_DURATION_SEC = 25.0
TRACK_SEARCH_SPIN_SPEED = 0.15
TRACK_OUTPUT_VIDEO_PATH = "/home/test/code/pet_tracking_record.mp4"


def _build_video_writer(frame_shape, output_path: str):
    h, w = frame_shape[:2]
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h))
    return writer if writer.isOpened() else None

COCO80_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]  # dog = index 16

COCO_91 = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
    16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep",
    21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe",
    27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase",
    34: "frisbee", 35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite",
    39: "baseball bat", 40: "baseball glove", 41: "skateboard", 42: "surfboard",
    43: "tennis racket", 44: "bottle", 46: "wine glass", 47: "cup",
    48: "fork", 49: "knife", 50: "spoon", 51: "bowl", 52: "banana",
    53: "apple", 54: "sandwich", 55: "orange", 56: "broccoli", 57: "carrot",
    58: "hot dog", 59: "pizza", 60: "donut", 61: "cake", 62: "chair",
    63: "couch", 64: "potted plant", 65: "bed", 67: "dining table", 70: "toilet",
    72: "tv", 73: "laptop", 74: "mouse", 75: "remote", 76: "keyboard",
    77: "cell phone", 78: "microwave", 79: "oven", 80: "toaster",
    81: "sink", 82: "refrigerator", 84: "book", 85: "clock", 86: "vase",
    87: "scissors", 88: "teddy bear", 89: "hair drier", 90: "toothbrush"
}

try:
    from test3 import Board
except ImportError:
    print("\n底盘驱动导入失败，使用虚拟车轮")

    class Board:
        def set_motor_speed(self, speeds):
            pass


def set_motor(board, speed_right, speed_left, max_speed=140):
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
        try:
            with open(pid_file, "w") as f:
                f.write("")
        except OSError:
            pass

class CameraReader:
    def __init__(self, src=0, width=640, height=480):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头流: {src}")

    def read(self):
        ret, frame = self.cap.read()
        if ret and frame is not None:
            return ret, cv2.flip(frame, 1)
        return ret, frame

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        if self.cap.isOpened():
            self.cap.release()


class RectF:
    def __init__(self, left, top, right, bottom):
        self.left, self.top = float(left), float(top)
        self.right, self.bottom = float(right), float(bottom)

    def width(self):
        return max(0.0, self.right - self.left)

    def height(self):
        return max(0.0, self.bottom - self.top)

    def centerX(self):
        return (self.left + self.right) * 0.5

    def centerY(self):
        return (self.top + self.bottom) * 0.5


class Detection:
    def __init__(self, title, confidence, rect: RectF):
        self.title, self.confidence, self.rect = title, float(confidence), rect

class RKNNDetector:
    INPUT_W = 640
    INPUT_H = 640

    def __init__(self, path, conf=0.25, core_mask=4):
        self.conf = conf

        _mask_map = {1: 0x01, 2: 0x02, 3: 0x04, 4: 0x07}
        self._rknn_core = _mask_map.get(core_mask, 0x01)

        self.rknn = RKNN3Lite()

        if not os.path.exists(path):
            raise FileNotFoundError(path)
        if not os.path.exists(WEIGHT_MODEL):
            raise FileNotFoundError(WEIGHT_MODEL)

        ret = self.rknn.load_rknn(path, WEIGHT_MODEL)
        if ret != 0:
            raise RuntimeError(f"load_rknn failed: {ret}")

        ids = self.rknn.get_devices_id()
        if not ids:
            raise RuntimeError("没有找到 RKNN 设备")

        ret = self.rknn.init_runtime(target="rk1828", core_mask=0x01, device_id=ids[0])
        if ret != 0:
            raise RuntimeError(f"init_runtime failed: {ret}")

    def _preprocess(self, bgr: np.ndarray):
        oh, ow = bgr.shape[:2]
        scale  = min(self.INPUT_W / ow, self.INPUT_H / oh)
        nw     = int(ow * scale)
        nh     = int(oh * scale)

        resized  = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas   = np.full((self.INPUT_H, self.INPUT_W, 3), 114, dtype=np.uint8)
        pad_left = (self.INPUT_W - nw) // 2
        pad_top  = (self.INPUT_H - nh) // 2
        canvas[pad_top:pad_top + nh, pad_left:pad_left + nw] = resized

        inp = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        inp = np.expand_dims(inp, 0)          # (1,H,W,3) uint8
        return inp, scale, pad_left, pad_top

    def _postprocess(self, outputs, scale, pad_left, pad_top,
                     orig_w, orig_h, target_cls_ids: set):
        raw = np.array(outputs[0], dtype=np.float32)
        if raw.ndim == 3 and raw.shape[0] == 1:
            raw = raw[0]                     # (512, 6)

        valid = np.any(raw != 0, axis=1)
        raw   = raw[valid]
        if len(raw) == 0:
            return []

        scores  = raw[:, 0]
        cls_ids = raw[:, 1].astype(np.int32)
        x1_lb   = raw[:, 2]
        y1_lb   = raw[:, 3]
        x2_lb   = raw[:, 4]
        y2_lb   = raw[:, 5]

        mask = (scores >= self.conf) & np.isin(cls_ids, list(target_cls_ids))
        if not mask.any():
            return []

        scores  = scores[mask]
        cls_ids = cls_ids[mask]
        x1_lb   = x1_lb[mask]
        y1_lb   = y1_lb[mask]
        x2_lb   = x2_lb[mask]
        y2_lb   = y2_lb[mask]

        x1 = np.clip((x1_lb - pad_left) / scale, 0, orig_w)
        y1 = np.clip((y1_lb - pad_top)  / scale, 0, orig_h)
        x2 = np.clip((x2_lb - pad_left) / scale, 0, orig_w)
        y2 = np.clip((y2_lb - pad_top)  / scale, 0, orig_h)

        valid2 = (x2 > x1) & (y2 > y1)
        results = []
        for i in np.where(valid2)[0]:
            cls_name = (COCO80_NAMES[cls_ids[i]]
                        if cls_ids[i] < len(COCO80_NAMES)
                        else f"class_{cls_ids[i]}")
            results.append(Detection(
                title=cls_name,
                confidence=float(scores[i]),
                rect=RectF(float(x1[i]), float(y1[i]),
                           float(x2[i]), float(y2[i]))
            ))
        return results

    def detect(self, bgr: np.ndarray, W: int, H: int,
               target_classes: List[str]) -> List[Detection]:
        target_cls_ids = {
            i for i, name in enumerate(COCO80_NAMES)
            if name in target_classes
        }
        if not target_cls_ids:
            return []

        inp, scale, pad_left, pad_top = self._preprocess(bgr)
        outputs = self.rknn.inference(inputs=[inp])
        if outputs is None:
            return []

        return self._postprocess(outputs, scale, pad_left, pad_top,
                                 W, H, target_cls_ids)

    def release(self):
        try:
            self.rknn.release()
        except Exception:
            pass


# =============================================================
# 以下全部保持原样，一字未改
# =============================================================

class TrackerState(Enum):
    IDLE = auto()
    TRACKING = auto()
    BUFFER_WAIT = auto()
    SEARCHING = auto()


class DrawBox:
    def __init__(self, rect: RectF, title: str, score: float, is_target: bool):
        self.rect, self.title, self.score, self.is_target = rect, title, score, is_target


class MultiBoxTracker:
    MIN_SIZE = 24.0
    BASE_SPEED = 0.35
    STEERING_GAIN = 0.60
    DEAD_ZONE = 0.07
    MIN_STEER = 0.15
    RAMP_STEP = 0.27

    def __init__(self):
        self.currentState = TrackerState.IDLE
        self.lastKnownLocation: RectF = None
        self.lastMoveDirection = 0.0
        self.currentLeftSpeed = 0.0
        self.currentRightSpeed = 0.0
        self.frameWidth = 640
        self.frameHeight = 480
        self.drawBoxes: List[DrawBox] = []

        self.lastSeenTime = time.time()
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
                dx = det.rect.centerX() - self.frameWidth / 2.0
                dy = det.rect.centerY() - self.frameHeight / 2.0
                sc = (det.rect.width() * det.rect.height()) * (1.0 - math.sqrt(dx * dx + dy * dy) / self.frameWidth)
                if sc > best_sc:
                    best_sc, best = sc, det
                self.drawBoxes.append(DrawBox(det.rect, det.title, det.confidence, False))
            if best is not None:
                self.lastKnownLocation = best.rect
                self.currentState = TrackerState.TRACKING
                self.lastSeenTime = time.time()
            return

        best, min_d = None, float("inf")
        for det in valid:
            if self.lastKnownLocation is not None:
                dx = det.rect.centerX() - self.lastKnownLocation.centerX()
                dy = det.rect.centerY() - self.lastKnownLocation.centerY()
                dist = math.sqrt(dx * dx + dy * dy)
                # ! hyper parameter to control the distance restrict
                if dist < min_d and dist < self.frameWidth * 0.3:
                    min_d, best = dist, det
            else:
                best = det

        current_time = time.time()

        if best is not None:
            if self.lastKnownLocation is not None:
                dx = best.rect.centerX() - self.lastKnownLocation.centerX()
                self.lastMoveDirection = self.lastMoveDirection * 0.8 + dx * 0.2
            self.lastKnownLocation = best.rect
            self.currentState = TrackerState.TRACKING
            self.lastSeenTime = current_time
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
            error = 1.0 - 2.0 * self.lastKnownLocation.centerX() / float(self.frameWidth)
            abs_e = abs(error)

            area = (self.lastKnownLocation.width() * self.lastKnownLocation.height()) / (self.frameWidth * self.frameHeight)
            hr = self.lastKnownLocation.height() / float(self.frameHeight)
            fwd = 0.0

            is_stationary = abs(self.currentLeftSpeed) < 0.05 and abs(self.currentRightSpeed) < 0.05
            if is_stationary:
                if self.stationary_start_time is None:
                    self.stationary_start_time = time.time()
            else:
                self.stationary_start_time = None

            if self.is_backing_up:
                # ! height restrict
                if hr < 0.80:
                    self.is_backing_up = False
                else:
                    fwd = -self.BASE_SPEED
            else:
                if hr < 0.80 and area < 0.40:
                    if hr <= 0.50:
                        raw_fwd = self.BASE_SPEED * 1
                    else:
                        raw_fwd = self.BASE_SPEED * ((0.80 - hr) / 0.30) * 1
                    fwd = max(0.0, raw_fwd)

                if self.stationary_start_time is not None and (time.time() - self.stationary_start_time) > 1.0:
                    if hr > 0.80 or area > 0.55:
                        self.is_backing_up = True

            dz = (self.DEAD_ZONE * 3.0 + area * 0.05) if fwd == 0.0 else self.DEAD_ZONE
            steer = 0.0

            if abs_e > dz:
                steer = error * self.STEERING_GAIN * 0.40
                if fwd < 0.0:
                    steer = -steer
                ms = self.MIN_STEER * 2.0 if fwd == 0.0 else self.MIN_STEER
                if 0 < steer < ms:
                    steer = ms
                if 0 > steer > -ms:
                    steer = -ms
            tL, tR = fwd - steer, fwd + steer

        elif self.currentState == TrackerState.BUFFER_WAIT:
            tL = tR = 0.0
            self.is_backing_up = False
            self.stationary_start_time = None

        elif self.currentState == TrackerState.SEARCHING:
            spd = 0.2
            tL = spd if self.lastMoveDirection > 0 else -spd
            tR = -spd if self.lastMoveDirection > 0 else spd
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

        if abs(self.currentLeftSpeed)  < 0.05:
            self.currentLeftSpeed  = 0.0
        if abs(self.currentRightSpeed) < 0.05:
            self.currentRightSpeed = 0.0

        return self.currentLeftSpeed, self.currentRightSpeed


def _spd_bar(canvas, cx, cy, speed, label):
    bh, bw = 100, 26
    x0, y0 = cx - bw // 2, cy - bh // 2
    cv2.rectangle(canvas, (x0, y0), (x0 + bw, y0 + bh), (50, 50, 50), -1)
    fill = int(abs(speed) * bh / 2)
    col = (50, 220, 50) if speed >= 0 else (50, 50, 220)
    if speed >= 0:
        cv2.rectangle(canvas, (x0 + 2, y0 + bh // 2 - fill), (x0 + bw - 2, y0 + bh // 2), col, -1)
    else:
        cv2.rectangle(canvas, (x0 + 2, y0 + bh // 2), (x0 + bw - 2, y0 + bh // 2 + fill), col, -1)
    cv2.rectangle(canvas, (x0, y0), (x0 + bw, y0 + bh), (180, 180, 180), 1)
    cv2.line(canvas, (x0, cy), (x0 + bw, cy), (255, 255, 255), 1)
    cv2.putText(canvas, label, (cx - 8, y0 + bh + 16), cv2.FONT_HERSHEY_SIMPLEX, .44, (220, 220, 220), 1)
    cv2.putText(canvas, f"{speed:+.2f}", (cx - 22, y0 + bh + 30), cv2.FONT_HERSHEY_SIMPLEX, .44, (220, 220, 220), 1)


def draw_tracking_ui(frame, tracker: MultiBoxTracker, target_pet: str):
    vis = frame.copy()
    H, W = vis.shape[:2]
    for db in tracker.drawBoxes:
        color = (0, 255, 0) if db.is_target else (0, 255, 255)
        status = "Target" if db.is_target else "Other"
        x1, y1 = int(db.rect.left), int(db.rect.top)
        x2, y2 = int(db.rect.right), int(db.rect.bottom)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
        cv2.putText(vis, f"{db.title}|{status}({db.score:.2f})",
                    (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    _spd_bar(vis, 36, H - 75, tracker.currentLeftSpeed, "L")
    _spd_bar(vis, 76, H - 75, tracker.currentRightSpeed, "R")
    for i, t in enumerate([f"Mode: Follow [{target_pet}]", f"State: {tracker.currentState.name}"]):
        cv2.putText(vis, t, (W - 240, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2, cv2.LINE_AA)
    return vis


def background_pet_search_task(video_source, model_path, target_pet, tts_mp_q=None, board=None, detector=None):
    import speaker
    speaker.init_mp_queue(tts_mp_q)
    if board is None:
        board = Board()
    owns_detector = detector is None
    if detector is None:
        detector = RKNNDetector(model_path, conf=DET_CONF, core_mask=4)
    cap = None
    found = False

    pet_dict = {"cat": "小猫", "dog": "小狗"}
    pet_name = pet_dict.get(target_pet, "宠物")
    target_classes = [target_pet] if target_pet in {"cat", "dog"} else ["cat", "dog"]

    try:
        cap = CameraReader(video_source)
        start_time = time.time()
        while cap.isOpened() and (time.time() - start_time) < 100000.0:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            h, w = frame.shape[:2]
            dets = detector.detect(frame, w, h, target_classes=target_classes)
            vis = frame.copy()
            cv2.putText(vis, f"Searching for [{pet_name}]...", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

            if dets:
                found = True
                set_motor(board, 0.0, 0.0)
                for det in dets:
                    x1, y1 = int(det.rect.left), int(det.rect.top)
                    x2, y2 = int(det.rect.right), int(det.rect.bottom)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(vis, "FOUND", (x1, max(20, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                _show_frame("Pet Search", vis, 800)
                break

            _show_frame("Pet Search", vis, 1)
            # ! searching speed
            set_motor(board, speed_right=0.2, speed_left=-0.2)

    finally:
        set_motor(board, 0.0, 0.0)
        if owns_detector and detector is not None:
            detector.release()
        if found:
            speak(f"这里有一只{pet_name}")
        else:
            speak(f"抱歉，我没有发现{pet_name}")
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        _close_windows()

        time.sleep(0.2)


def background_tracking_task(video_source, model_path, target_pet, pid_file_path, tts_mp_q=None, board=None, detector=None):
    import speaker
    with open(r'/home/test/code/final_0418/llm/pet_tracking_result.txt', 'w') as f:
        f.write(f"failure")
    speaker.init_mp_queue(tts_mp_q)

    is_running = True

    def handle_sigterm(signum, frame_obj):
        nonlocal is_running
        is_running = False

    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, handle_sigterm)
        signal.signal(signal.SIGINT, handle_sigterm)

    pet_dict = {"cat": "小猫", "dog": "小狗"}
    pet_name = pet_dict.get(target_pet, "宠物")
    target_classes = [target_pet] if target_pet in {"cat", "dog"} else ["cat", "dog"]
    cap = None
    owns_detector = detector is None
    video_writer = None
    recording_start_time = None

    try:
        if board is None:
            board = Board()
        if detector is None:
            detector = RKNNDetector(model_path, conf=DET_CONF, core_mask=4)
        tracker = MultiBoxTracker()
        cap = CameraReader(video_source)

        has_tracked = False
        start_time = time.time()

        while cap.isOpened() and is_running:
            now = time.time()
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            h, w = frame.shape[:2]
            dets = detector.detect(frame, w, h, target_classes=target_classes)
            tracker.trackResults(dets, frame)

            vis = draw_tracking_ui(frame, tracker, target_pet)
            key = _show_frame("Pet Follower", vis, 1)

            if not has_tracked:
                if tracker.currentState == TrackerState.TRACKING:
                    has_tracked = True
                    recording_start_time = now
                    if os.path.exists(TRACK_OUTPUT_VIDEO_PATH):
                        try:
                            os.remove(TRACK_OUTPUT_VIDEO_PATH)
                        except OSError:
                            pass
                    video_writer = _build_video_writer(vis.shape, TRACK_OUTPUT_VIDEO_PATH)
                    if video_writer is None:
                        raise RuntimeError(f"无法创建视频文件: {TRACK_OUTPUT_VIDEO_PATH}")
                    print(f"已锁定{pet_name}，开始录制视频: {TRACK_OUTPUT_VIDEO_PATH}")
                else:
                    set_motor(board, speed_right=TRACK_SEARCH_SPIN_SPEED, speed_left=-TRACK_SEARCH_SPIN_SPEED)
                    if now - start_time > TRACK_SEARCH_TIMEOUT_SEC:
                        speak("对不起，我没找到宠物")
                        print("寻找超时，未找到目标宠物，跟踪进程结束")
                        break
                    if key in [ord("x"), ord("e"), ord("q"), 27]:
                        print("\n收到按键退出指令")
                        break
                    continue

            L, R = tracker.updateTarget()
            if tracker.currentState == TrackerState.IDLE:
                set_motor(board, speed_right=TRACK_SEARCH_SPIN_SPEED, speed_left=-TRACK_SEARCH_SPIN_SPEED)
            else:
                set_motor(board, speed_right=R, speed_left=L)

            if video_writer is not None:
                video_writer.write(vis)

            if recording_start_time is not None and (now - recording_start_time) >= TRACK_RECORD_DURATION_SEC:
                print(f"视频录制完成（{TRACK_RECORD_DURATION_SEC:.0f}s）: {TRACK_OUTPUT_VIDEO_PATH}")
                with open(r'/home/test/code/final_0418/llm/pet_tracking_result.txt', 'w') as f:
                    f.write(f"success")
                break

            if key in [ord("x"), ord("e"), ord("q"), 27]:
                print("\n收到按键退出指令")
                break

    except Exception as e:
        print(f"异常: {e}")

    finally:
        set_motor(board, 0.0, 0.0)
        if owns_detector and detector is not None:
            detector.release()
        if video_writer is not None:
            try:
                video_writer.release()
            except Exception:
                pass
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        _close_windows()

        _safe_remove_pid(pid_file_path)
        time.sleep(0.2)


class PetTrackingSystem():
    def __init__(self, model_path=DETECTOR_MODEL):
        self.model_path = model_path
        self.pid_file = "/tmp/pet_tracking_pid.txt"
        self._process = None
        self.motor_board = None
        self.detector = None

    def set_motor_board(self, board):
        self.motor_board = board

    def preload(self):
        self._ensure_detector()

    def _ensure_detector(self):
        if self.detector is None:
            self.detector = RKNNDetector(self.model_path, conf=DET_CONF, core_mask=4)
        return self.detector

    def find_pet(self, video_source, target_pet):
        print(f"同步启动寻宠任务: {target_pet}")
        import speaker
        background_pet_search_task(
            video_source,
            self.model_path,
            target_pet,
            speaker._mp_q,
            self.motor_board,
            self._ensure_detector(),
        )
        print("寻宠任务已退出")

    def start_pet_tracking(self, video_source, target_pet):
        print(f"同步启动宠物跟踪任务: {target_pet}")
        if self._process is not None and self._process.is_alive():
            return
        _safe_remove_pid(self.pid_file)
        import speaker
        background_tracking_task(
            video_source,
            self.model_path,
            target_pet,
            self.pid_file,
            speaker._mp_q,
            self.motor_board,
            self._ensure_detector(),
        )

    def stop_pet_tracking(self):
        try:
            self._terminate_process()
        finally:
            _safe_remove_pid(self.pid_file)
        set_motor(self.motor_board, 0.0, 0.0)
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
            except Exception:
                pass
            finally:
                self._process = None
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
                except ProcessLookupError:
                    pass
            except ProcessLookupError:
                pass
            except PermissionError:
                pass
        except Exception:
            pass
