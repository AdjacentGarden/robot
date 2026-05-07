import os
import sys
import cv2
import numpy as np
import time
import math
import signal
import threading
import multiprocessing
from enum import Enum, auto
from typing import Dict, List
import warnings
from speaker import speak
import queue

from rknnlite.api import RKNNLite

warnings.filterwarnings("ignore")

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

DETECTOR_MODEL = "/home/test/code/final_0322/model/detect_v2.rknn"
# ! target threshold
DET_CONF = 0.40

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
    from wheel_control import Board
except ImportError:
    print("\n底盘驱动导入失败，使用虚拟车轮")

    class Board:
        def set_motor_speed(self, speeds):
            pass


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
                    self.frame = cv2.flip(self.frame, 1)
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


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def iou_one_to_many(box, boxes):
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, xx2 - xx1)
    inter_h = np.maximum(0.0, yy2 - yy1)
    inter = inter_w * inter_h

    area1 = np.maximum(0.0, (box[2] - box[0])) * np.maximum(0.0, (box[3] - box[1]))
    area2 = np.maximum(0.0, (boxes[:, 2] - boxes[:, 0])) * np.maximum(0.0, (boxes[:, 3] - boxes[:, 1]))
    union = area1 + area2 - inter + 1e-9
    return inter / union


def nms(boxes, scores, iou_thresh=0.6, top_k=100):
    if len(boxes) == 0:
        return []

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0 and len(keep) < top_k:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        ious = iou_one_to_many(boxes[i], boxes[order[1:]])
        inds = np.where(ious <= iou_thresh)[0]
        order = order[inds + 1]

    return keep


def ensure_nchw(arr):
    arr = np.array(arr)

    if arr.ndim != 4:
        raise RuntimeError(f"Expected 4D tensor, got {arr.shape}")

    if arr.shape[1] > 4 and arr.shape[2] <= 128 and arr.shape[3] <= 128:
        return arr

    if arr.shape[3] > 4 and arr.shape[1] <= 128 and arr.shape[2] <= 128:
        return np.transpose(arr, (0, 3, 1, 2))

    return arr


def collect_raw_predictions(outputs):
    tensors = []
    for i, out in enumerate(outputs):
        x = ensure_nchw(out).astype(np.float32)
        _, ch, h, w = x.shape
        tensors.append({
            "idx": i,
            "arr": x,
            "ch": ch,
            "h": h,
            "w": w,
        })

    groups = {}
    for t in tensors:
        groups.setdefault((t["h"], t["w"]), []).append(t)

    box_list = []
    cls_list = []
    feature_map_shapes = []

    sorted_keys = sorted(groups.keys(), key=lambda x: x[0] * x[1], reverse=True)

    for key in sorted_keys:
        items = groups[key]
        if len(items) != 2:
            raise RuntimeError(f"Feature map {key} expected 2 tensors, got {len(items)}")

        items = sorted(items, key=lambda x: x["ch"])
        box = items[0]["arr"]
        cls = items[1]["arr"]

        _, box_ch, h, w = box.shape
        _, cls_ch, h2, w2 = cls.shape

        if (h, w) != (h2, w2):
            raise RuntimeError(f"Shape mismatch: box={box.shape}, cls={cls.shape}")

        if box_ch % 4 != 0:
            raise RuntimeError(f"Box channel not divisible by 4: {box.shape}")

        anchors_per_loc = box_ch // 4
        if cls_ch % anchors_per_loc != 0:
            raise RuntimeError(f"Class channel mismatch: box={box.shape}, cls={cls.shape}")

        num_classes_with_bg = cls_ch // anchors_per_loc

        box = np.transpose(box, (0, 2, 3, 1)).reshape(-1, 4)
        cls = np.transpose(cls, (0, 2, 3, 1)).reshape(-1, num_classes_with_bg)

        box_list.append(box)
        cls_list.append(cls)
        feature_map_shapes.append((h, w))

    all_boxes = np.concatenate(box_list, axis=0)
    all_logits = np.concatenate(cls_list, axis=0)
    return all_boxes, all_logits, feature_map_shapes


def generate_ssd_anchors(feature_map_shapes, min_scale=0.2, max_scale=0.95):
    num_layers = len(feature_map_shapes)
    scales = np.linspace(min_scale, max_scale, num_layers).tolist()
    aspect_ratios = [1.0, 2.0, 0.5, 3.0, 0.3333]

    all_anchors = []

    for layer_idx, (fm_h, fm_w) in enumerate(feature_map_shapes):
        if layer_idx == 0:
            layer_box_specs = [
                (0.1, 1.0),
                (scales[layer_idx], 2.0),
                (scales[layer_idx], 0.5),
            ]
        else:
            s = scales[layer_idx]
            s_next = 1.0 if layer_idx == num_layers - 1 else scales[layer_idx + 1]
            layer_box_specs = [(s, ar) for ar in aspect_ratios]
            layer_box_specs.append((np.sqrt(s * s_next), 1.0))

        for y in range(fm_h):
            cy = (y + 0.5) / fm_h
            for x in range(fm_w):
                cx = (x + 0.5) / fm_w
                for scale, ar in layer_box_specs:
                    ratio_sqrt = np.sqrt(ar)
                    h = scale / ratio_sqrt
                    w = scale * ratio_sqrt

                    ymin = cy - h / 2.0
                    xmin = cx - w / 2.0
                    ymax = cy + h / 2.0
                    xmax = cx + w / 2.0
                    all_anchors.append([ymin, xmin, ymax, xmax])

    return np.array(all_anchors, dtype=np.float32)


def decode_boxes(rel_codes, anchors, y_scale=10.0, x_scale=10.0, h_scale=5.0, w_scale=5.0):
    ymin_a, xmin_a, ymax_a, xmax_a = anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3]

    ya = (ymin_a + ymax_a) / 2.0
    xa = (xmin_a + xmax_a) / 2.0
    ha = ymax_a - ymin_a
    wa = xmax_a - xmin_a

    ty = rel_codes[:, 0] / y_scale
    tx = rel_codes[:, 1] / x_scale
    th = rel_codes[:, 2] / h_scale
    tw = rel_codes[:, 3] / w_scale

    ycenter = ty * ha + ya
    xcenter = tx * wa + xa
    h = np.exp(th) * ha
    w = np.exp(tw) * wa

    ymin = ycenter - h / 2.0
    xmin = xcenter - w / 2.0
    ymax = ycenter + h / 2.0
    xmax = xcenter + w / 2.0

    decoded = np.stack([ymin, xmin, ymax, xmax], axis=1)
    decoded = np.clip(decoded, 0.0, 1.0)
    return decoded


def postprocess(raw_box_encodings, raw_class_logits, feature_map_shapes, orig_w, orig_h,
                score_thresh=0.40, iou_thresh=0.6, max_total=100):
    anchors = generate_ssd_anchors(feature_map_shapes)

    if len(anchors) != len(raw_box_encodings):
        raise RuntimeError(
            f"Anchor count mismatch: anchors={len(anchors)}, preds={len(raw_box_encodings)}"
        )

    decoded_boxes = decode_boxes(raw_box_encodings, anchors)
    probs = sigmoid(raw_class_logits)

    if probs.shape[1] <= 1:
        raise RuntimeError(f"Unexpected class dimension: {probs.shape}")

    probs_fg = probs[:, 1:]
    final_dets = []

    for cls_idx in range(probs_fg.shape[1]):
        cls_id = cls_idx + 1
        scores = probs_fg[:, cls_idx]
        keep = np.where(scores >= score_thresh)[0]
        if keep.size == 0:
            continue

        boxes_norm = decoded_boxes[keep]
        scores_kept = scores[keep]

        boxes_xyxy = np.stack([
            boxes_norm[:, 1] * orig_w,
            boxes_norm[:, 0] * orig_h,
            boxes_norm[:, 3] * orig_w,
            boxes_norm[:, 2] * orig_h,
        ], axis=1)

        keep_nms = nms(boxes_xyxy, scores_kept, iou_thresh=iou_thresh, top_k=max_total)

        for j in keep_nms:
            x1, y1, x2, y2 = boxes_xyxy[j]
            final_dets.append({
                "class_id": cls_id,
                "class_name": COCO_91.get(cls_id, f"class_{cls_id}"),
                "score": float(scores_kept[j]),
                "box": [int(x1), int(y1), int(x2), int(y2)],
            })

    final_dets.sort(key=lambda x: x["score"], reverse=True)
    return final_dets[:max_total]


def preprocess_for_rknn(image_bgr, input_h=300, input_w=300):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.uint8)
    x = np.expand_dims(x, axis=0)  # NHWC
    return x


class RKNNDetector:
    INPUT_W = 300
    INPUT_H = 300

    def __init__(self, path, conf=0.40, core_mask=4):
        self.conf = conf
        self.rknn_lite = RKNNLite()

        if not os.path.exists(path):
            raise FileNotFoundError(path)

        ret = self.rknn_lite.load_rknn(path)
        if ret != 0:
            raise RuntimeError(f"load_rknn failed: {ret}")

        if core_mask == 1:
            ret = self.rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        elif core_mask == 2:
            ret = self.rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_1)
        elif core_mask == 3:
            ret = self.rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_2)
        elif core_mask == 4:
            ret = self.rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
        else:
            ret = self.rknn_lite.init_runtime()

        if ret != 0:
            raise RuntimeError(f"init_runtime failed: {ret}")

    def detect(self, bgr, W, H, target_classes: List[str]) -> List[Detection]:
        inp = preprocess_for_rknn(bgr, self.INPUT_H, self.INPUT_W)
        outputs = self.rknn_lite.inference(inputs=[inp], data_format=['nhwc'])

        raw_box_encodings, raw_class_logits, feature_map_shapes = collect_raw_predictions(outputs)
        dets = postprocess(
            raw_box_encodings,
            raw_class_logits,
            feature_map_shapes,
            orig_w=W,
            orig_h=H,
            score_thresh=self.conf,
            iou_thresh=0.6,
            max_total=100,
        )

        res = []
        for det in dets:
            if det["class_name"] not in target_classes:
                continue
            x1, y1, x2, y2 = det["box"]
            if x2 > x1 and y2 > y1:
                res.append(
                    Detection(
                        det["class_name"],
                        det["score"],
                        RectF(x1, y1, x2, y2),
                    )
                )
        return res

    def release(self):
        try:
            self.rknn_lite.release()
        except Exception:
            pass


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
    DEAD_ZONE = 0.03
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

        self.currentLeftSpeed = 0.5 * self.currentLeftSpeed + 0.5 * tL
        self.currentRightSpeed = 0.5 * self.currentRightSpeed + 0.5 * tR

        self.currentLeftSpeed = max(-1.0, min(1.0, self.currentLeftSpeed))
        self.currentRightSpeed = max(-1.0, min(1.0, self.currentRightSpeed))

        if abs(self.currentLeftSpeed) < 0.05:
            self.currentLeftSpeed = 0.0
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


def background_pet_search_task(video_source, model_path, target_pet, tts_mp_q=None):
    import speaker
    speaker.init_mp_queue(tts_mp_q)
    board = Board()
    detector = RKNNDetector(model_path, conf=DET_CONF, core_mask=4)
    cap = None
    found = False

    pet_dict = {"cat": "小猫", "dog": "小狗"}
    pet_name = pet_dict.get(target_pet, target_pet)

    try:
        cap = CameraReader(video_source)
        start_time = time.time()
        while cap.isOpened() and (time.time() - start_time) < 6.0:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            h, w = frame.shape[:2]
            dets = detector.detect(frame, w, h, target_classes=[target_pet])
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
                cv2.imshow("Pet Search", vis)
                cv2.waitKey(800)
                break

            cv2.imshow("Pet Search", vis)
            cv2.waitKey(1)
            # ! searching speed
            set_motor(board, speed_right=0.2, speed_left=-0.2)

    finally:
        set_motor(board, 0.0, 0.0)
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
        try:
            cv2.destroyAllWindows()
            for _ in range(10):
                cv2.waitKey(1)
        except Exception:
            pass

        time.sleep(0.2)


def background_tracking_task(video_source, model_path, target_pet, pid_file_path, tts_mp_q=None):
    import speaker
    speaker.init_mp_queue(tts_mp_q)

    is_running = True

    def handle_sigterm(signum, frame_obj):
        nonlocal is_running
        is_running = False

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)

    pet_dict = {"cat": "小猫", "dog": "小狗"}
    pet_name = pet_dict.get(target_pet, "宠物")
    cap = None
    board = None
    detector = None

    try:
        board = Board()
        detector = RKNNDetector(model_path, conf=DET_CONF, core_mask=4)
        tracker = MultiBoxTracker()
        cap = CameraReader(video_source)

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
                break
            elif not has_tracked and (time.time() - start_time) > 5.0:
                speak(f"寻找超时，附近没有发现{pet_name}，已自动停止")
                break

    except Exception as e:
        print(f"异常: {e}")

    finally:
        set_motor(board, 0.0, 0.0)
        if detector is not None:
            detector.release()
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        try:
            cv2.destroyAllWindows()
            for _ in range(10):
                cv2.waitKey(1)
        except Exception:
            pass

        _safe_remove_pid(pid_file_path)
        time.sleep(0.2)


class PetTrackingSystem():
    def __init__(self, model_path=DETECTOR_MODEL):
        self.model_path = model_path
        self.pid_file = "/tmp/pet_tracking_pid.txt"
        self._process: multiprocessing.Process = None

    def find_pet(self, video_source, target_pet):
        print(f"启动寻宠进程: {target_pet}")
        import speaker
        ctx = multiprocessing.get_context('spawn')
        if speaker._mp_q is None:
            speaker.init_mp_queue(ctx.Queue())

        p = ctx.Process(
            target=background_pet_search_task,
            args=(video_source, self.model_path, target_pet, speaker._mp_q),
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

        import speaker
        ctx = multiprocessing.get_context('spawn')
        if speaker._mp_q is None:
            speaker.init_mp_queue(ctx.Queue())

        p = ctx.Process(
            target=background_tracking_task,
            args=(video_source, self.model_path, target_pet, self.pid_file, speaker._mp_q),
            daemon=True,
        )
        p.start()
        self._process = p
        try:
            with open(self.pid_file, "w") as f:
                f.write(str(p.pid))
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