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
import random
import datetime

from rknnlite.api import RKNNLite

warnings.filterwarnings("ignore")

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

DETECTOR_MODEL = "/home/test/openbot_test_zhenghang/model_0320/openbot_validify_code/final_0322/data/convert/detect/detect_v2.rknn"
REID_MODEL = "/home/test/openbot_test_zhenghang/model_0319/reid_model/reid_mmse.rknn"

DET_CONF = 0.30
RKNN_CORE_MASK = 4  # 0:auto, 1:NPU_CORE_0, 2:NPU_CORE_1, 3:NPU_CORE_2, 4:NPU_CORE_0_1_2

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
    class Board:
        def set_motor_speed(self, speeds):
            pass
        def enable_reception(self):
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
                    self.frame = cv2.flip(frame, -1)
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
    def __init__(self, l, t, r, b):
        self.left, self.top, self.right, self.bottom = float(l), float(t), float(r), float(b)

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


def preprocess_for_rknn(image_bgr, input_h=300, input_w=300):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.uint8)
    x = np.expand_dims(x, axis=0)
    return x


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
            "idx": i, "arr": x, "ch": ch, "h": h, "w": w,
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


def postprocess(raw_box_encodings, raw_class_logits, feature_map_shapes, orig_w, orig_h,
                score_thresh=0.3, iou_thresh=0.6, max_total=100):
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


def draw_detections(image_bgr, detections):
    out = image_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = f'{det["class_name"]} {det["score"]:.2f}'

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_text = max(y1, th + 4)
        cv2.rectangle(out, (x1, y_text - th - 4), (x1 + tw, y_text + baseline - 4), (0, 255, 0), -1)
        cv2.putText(out, label, (x1, y_text - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return out


class RknnDetector:
    INPUT_W = 300
    INPUT_H = 300

    def __init__(self, path, conf=DET_CONF, core_mask=RKNN_CORE_MASK):
        self.conf = conf
        self.path = path
        self.core_mask = core_mask
        self.rknn = RKNNLite()

        if not os.path.exists(path):
            raise FileNotFoundError(f"RKNN 模型不存在: {path}")

        print("--> load_rknn")
        ret = self.rknn.load_rknn(path)
        if ret != 0:
            raise RuntimeError(f"加载 RKNN 模型失败, 错误码: {ret}")

        print("--> init_runtime")
        if self.core_mask == 1:
            ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        elif self.core_mask == 2:
            ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_1)
        elif self.core_mask == 3:
            ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_2)
        elif self.core_mask == 4:
            ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
        else:
            ret = self.rknn.init_runtime()

        if ret != 0:
            raise RuntimeError(f"初始化 RKNN runtime 失败, 错误码: {ret}")

    def _pre(self, bgr):
        image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image_rgb, (self.INPUT_W, self.INPUT_H), interpolation=cv2.INTER_LINEAR)
        x = resized.astype(np.uint8)
        x = np.expand_dims(x, axis=0)
        return x

    def detect(self, bgr, W, H, cls="person") -> List[Detection]:
        x = self._pre(bgr)
        outputs = self.rknn.inference(inputs=[x], data_format=['nhwc'])

        if outputs is None or len(outputs) == 0:
            return []

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
            if det["class_name"] != cls:
                continue
            x1, y1, x2, y2 = det["box"]
            if x2 > x1 and y2 > y1:
                res.append(Detection(det["class_name"], det["score"], RectF(x1, y1, x2, y2)))
        return res

    def release(self):
        if hasattr(self, "rknn"):
            self.rknn.release()


class RknnReID:
    W = 128
    H = 256
    DIM = 512

    def __init__(self, path, core_mask=0):
        self.rknn = RKNNLite()
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"RKNN ReID 模型不存在: {path}")

        ret = self.rknn.load_rknn(path)
        if ret != 0:
            raise RuntimeError(f"加载 RKNN ReID 模型失败, 错误码: {ret}")

        if core_mask == 1:
            ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        elif core_mask == 2:
            ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_1)
        elif core_mask == 3:
            ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_2)
        elif core_mask == 4:
            ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
        else:
            ret = self.rknn.init_runtime()

        if ret != 0:
            raise RuntimeError(f"初始化 RKNN ReID runtime 失败, 错误码: {ret}")

    def run(self, crop):
        if crop is None or crop.size == 0:
            return np.zeros(self.DIM, dtype=np.float32)
        
        h, w = crop.shape[:2]
        sc = min(self.W / w, self.H / h)
        nw, nh = max(1, int(w * sc)), max(1, int(h * sc))
        r = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)
        img = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        dx, dy = (self.W - nw) // 2, (self.H - nh) // 2
        img[dy:dy + nh, dx:dx + nw] = r
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        arr = np.transpose(img, (2, 0, 1))
        data = np.expand_dims(arr, axis=0)

        outputs = self.rknn.inference(inputs=[data], data_format='nchw')
        feat = outputs[0][0]
        
        n = float(np.linalg.norm(feat))
        return (feat / n).astype(np.float32) if n > 1e-9 else feat.astype(np.float32)

    def release(self):
        if hasattr(self, "rknn"):
            self.rknn.release()


class TrackerState(Enum):
    IDLE = auto()
    TRACKING = auto()
    BUFFER_WAIT = auto()
    SEARCHING = auto()


class DrawBox:
    def __init__(self, rect: RectF, score: float, is_target: bool):
        self.rect, self.score, self.is_target = rect, score, is_target


class MultiBoxTracker:
    MIN_SIZE = 24.0
    BASE_SPEED = 0.40
    STEERING_GAIN = 0.57
    DEAD_ZONE = 0.10
    MIN_STEER = 0.15
    GALLERY_SIZE = 16
    SIM_THRESHOLD = 0.55
    UPDATE_INTERVAL = 10

    class _Cand:
        def __init__(self, rect, feat):
            self.rect, self.feat, self.score = rect, feat, 0.0

    def __init__(self, reid: RknnReID):
        self.reid = reid
        self.currentState = TrackerState.IDLE
        self.gallery: List[np.ndarray] = []
        self.lastKnown = None
        self.lastMoveDir = 0.0
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
        x, y = max(0, int(rect.left)), max(0, int(rect.top))
        w, h = min(int(rect.width()), sw - x), min(int(rect.height()), sh - y)
        return bmp[y:y + h, x:x + w] if w > 0 and h > 0 else None

    def _cos(self, f1, f2):
        n1, n2 = np.linalg.norm(f1), np.linalg.norm(f2)
        return float(np.dot(f1, f2) / (n1 * n2)) if n1 * n2 > 1e-9 else 0.0

    def _is_valid_candidate(self, rect: RectF) -> bool:
        if self.currentState not in [TrackerState.TRACKING, TrackerState.BUFFER_WAIT] or self.lastKnown is None:
            return True

        prev_h = self.lastKnown.height()
        curr_h = rect.height()
        if curr_h <= prev_h * 0.4 or curr_h >= prev_h * 2.5:
            return False

        pred_cx = self.lastKnown.centerX() + self.lastMoveDir
        pred_cy = self.lastKnown.centerY()

        dx = rect.centerX() - pred_cx
        dy = rect.centerY() - pred_cy
        dist = math.hypot(dx, dy)

        max_dist = max(self.frameW * 0.3, self.lastKnown.width() * 2.5)

        if dist > max_dist:
            return False

        return True

    def trackResults(self, results: List[Detection], frame):
        if frame is not None:
            self.frameW, self.frameH = frame.shape[1], frame.shape[0]
        self.drawBoxes.clear()
        cands = []

        for r in results:
            if r.rect.width() < self.MIN_SIZE or r.rect.height() < self.MIN_SIZE:
                continue

            if not self._is_valid_candidate(r.rect):
                self.drawBoxes.append(DrawBox(r.rect, 0.0, False))
                continue

            crop = self._crop(frame, r.rect)
            if crop is None:
                continue
            cands.append(self._Cand(r.rect, self.reid.run(crop)))

        if self.currentState == TrackerState.IDLE:
            # 原有的盲目寻找逻辑：寻找最大最中心的目标
            best, bs = None, -1.0
            for c in cands:
                dx = c.rect.centerX() - self.frameW / 2.0
                dy = c.rect.centerY() - self.frameH / 2.0
                sc = (c.rect.width() * c.rect.height()) * (1.0 - math.sqrt(dx * dx + dy * dy) / self.frameW)
                if sc > bs:
                    bs, best = sc, c
                self.drawBoxes.append(DrawBox(c.rect, 0.0, False))
            if best:
                self.gallery = [best.feat.copy()]
                self.lastKnown = best.rect
                self.currentState = TrackerState.TRACKING
                self.lastSeenTime = time.time()
            return

        best, bd = None, -100.0
        for c in cands:
            c.score = max(self._cos(g, c.feat) for g in self.gallery) if self.gallery else 0.0
            dec = c.score
            if self.currentState == TrackerState.TRACKING and self.lastKnown:
                dec -= math.sqrt((c.rect.centerX() - self.lastKnown.centerX()) ** 2 +
                                 (c.rect.centerY() - self.lastKnown.centerY()) ** 2) / self.frameW * 0.5
            if dec > bd:
                bd, best = dec, c

        current_time = time.time()

        if best and best.score >= self.SIM_THRESHOLD:
            if self.lastKnown:
                dx = best.rect.centerX() - self.lastKnown.centerX()
                self.lastMoveDir = self.lastMoveDir * 0.8 + dx * 0.2
            self.lastKnown = best.rect
            self.currentState = TrackerState.TRACKING
            self.lastSeenTime = current_time
            self.framesSinceUpd += 1
            if self.framesSinceUpd >= self.UPDATE_INTERVAL:
                if len(self.gallery) >= self.GALLERY_SIZE:
                    self.gallery.pop(0)
                self.gallery.append(best.feat.copy())
                self.framesSinceUpd = 0
            for c in cands:
                self.drawBoxes.append(DrawBox(c.rect, c.score, c is best))
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
            for c in cands:
                self.drawBoxes.append(DrawBox(c.rect, c.score, False))

    def updateTarget(self):
        tL = tR = 0.0
        if self.currentState == TrackerState.TRACKING and self.lastKnown:
            error = 1.0 - 2.0 * self.lastKnown.centerX() / float(self.frameW)
            area = (self.lastKnown.width() * self.lastKnown.height()) / (self.frameW * self.frameH)
            hr = self.lastKnown.height() / float(self.frameH)
            fwd = 0.0

            is_stationary = abs(self.currentL) < 0.05 and abs(self.currentR) < 0.05

            if is_stationary:
                if self.stationary_start_time is None:
                    self.stationary_start_time = time.time()
            else:
                self.stationary_start_time = None

            if self.is_backing_up:
                if hr < 0.90:
                    self.is_backing_up = False
                else:
                    fwd = -0.40
            else:
                if hr < 0.80 and area < 0.40:
                    if hr <= 0.50:
                        raw_fwd = self.BASE_SPEED * 2.5
                    else:
                        raw_fwd = self.BASE_SPEED * ((0.80 - hr) / 0.30) * 2.5
                    fwd = max(0.0, raw_fwd)

                if self.stationary_start_time is not None and (time.time() - self.stationary_start_time) > 0.5:
                    if hr > 0.90:
                        self.is_backing_up = True

            steer = 0.0
            if abs(error) > self.DEAD_ZONE:
                steer = error * self.STEERING_GAIN
                if 0 < steer < self.MIN_STEER:
                    steer = self.MIN_STEER
                if 0 > steer > -self.MIN_STEER:
                    steer = -self.MIN_STEER
            
            # 还原为原先的正号 (测试普通视频不需要反转修正)
            tL, tR = fwd - steer, fwd + steer

        elif self.currentState == TrackerState.BUFFER_WAIT:
            tL = tR = 0.0
            self.is_backing_up = False
            self.stationary_start_time = None

        elif self.currentState == TrackerState.SEARCHING:
            spd = 0.4
            tL = spd if self.lastMoveDir > 0 else -spd
            tR = -spd if self.lastMoveDir > 0 else spd
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

        if abs(self.currentL) < 0.05:
            self.currentL = 0.0
        if abs(self.currentR) < 0.05:
            self.currentR = 0.0

        return self.currentL, self.currentR


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


def draw_tracking_ui(frame, tracker: MultiBoxTracker, target_name: str):
    vis = frame.copy()
    H, W = vis.shape[:2]
    for db in tracker.drawBoxes:
        color = (0, 255, 0) if db.is_target else (0, 255, 255)
        x1, y1 = int(db.rect.left), int(db.rect.top)
        x2, y2 = int(db.rect.right), int(db.rect.bottom)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
        cv2.putText(vis, f"{'Target:' + target_name if db.is_target else 'Other'}({db.score:.2f})",
                    (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    _spd_bar(vis, 36, H - 75, tracker.currentL, "L")
    _spd_bar(vis, 76, H - 75, tracker.currentR, "R")
    for i, t in enumerate([f"Mode: Follow [{target_name}]",
                           f"State: {tracker.currentState.name}",
                           f"Gallery: {len(tracker.gallery)}/{tracker.GALLERY_SIZE}"]):
        cv2.putText(vis, t, (W - 240, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2, cv2.LINE_AA)
    return vis


def background_person_search_task(video_source, detector_path, target_name, tts_mp_q=None):
    try:
        import speaker
        speaker.init_mp_queue(tts_mp_q)
    except ImportError:
        pass

    board = Board()
    detector = RknnDetector(detector_path, conf=DET_CONF, core_mask=RKNN_CORE_MASK)
    cap = None
    found = False
    time.sleep(2.0)

    try:
        cap = CameraReader(video_source)
        start_time = time.time()
        while cap.isOpened() and (time.time() - start_time) < 6.0:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            set_motor(board, speed_right=0.4, speed_left=-0.4)
            h, w = frame.shape[:2]
            dets = detector.detect(frame, w, h, cls="person")
            vis = frame.copy()
            cv2.putText(vis, f"Searching for [{target_name}]...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

            if dets:
                found = True
                set_motor(board, 0.0, 0.0)
                for det in dets:
                    x1, y1 = int(det.rect.left), int(det.rect.top)
                    x2, y2 = int(det.rect.right), int(det.rect.bottom)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(vis, "FOUND!", (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Person Search", vis)
                cv2.waitKey(800)
                break

            cv2.imshow("Person Search", vis)
            cv2.waitKey(1)

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

    if found:
        try:
            speaker.speak(f"{target_name}在这里")
        except NameError:
            pass


def background_person_tracking_task(video_source, detector_path, reid_path, target_name, pid_file_path, tts_mp_q=None):
    try:
        import speaker
        speaker.init_mp_queue(tts_mp_q)
    except ImportError:
        pass

    is_running = True

    def handle_sigterm(signum, frame_obj):
        nonlocal is_running
        is_running = False

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)

    cap = board = detector = reid = None
    video_writer = None

    record_dir = os.path.join(_CURRENT_DIR, "tracking_records")
    video_dir = os.path.join(record_dir, "videos")
    img_dir = os.path.join(record_dir, "screenshots")
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(video_dir, f"track_{target_name}_{session_id}.mp4")

    try:
        board = Board()
        detector = RknnDetector(detector_path, conf=DET_CONF, core_mask=RKNN_CORE_MASK)
        
        reid = RknnReID(reid_path, core_mask=RKNN_CORE_MASK)
        tracker = MultiBoxTracker(reid)
        
        cap = CameraReader(video_source)

        has_tracked = False

        while cap.isOpened() and is_running:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            h, w = frame.shape[:2]
            dets = detector.detect(frame, w, h, cls="person")
            tracker.trackResults(dets, frame)
            L, R = tracker.updateTarget()
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
                try:
                    speaker.speak(f"{target_name}彻底跟丢了")
                except NameError:
                    pass
                print(f"{target_name}彻底跟丢了")
                break

    except Exception as e:
        print(f"异常: {e}")

    finally:
        set_motor(board, 0.0, 0.0)
        
        if detector is not None:
            detector.release()
        if reid is not None:
            reid.release()

        if video_writer is not None:
            video_writer.release()

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
        print("已退出，摄像头资源及视频文件已保存并释放")


class PersonSearchTrackingSystem:
    def __init__(self, detector_path=DETECTOR_MODEL, reid_path=REID_MODEL):
        self.detector_model = detector_path
        self.reid_model = reid_path
        self.pid_file = "/tmp/person_tracking_pid.txt"
        self._process: multiprocessing.Process = None

    def search_person(self, video_source, target_name):
        try:
            import speaker
            ctx = multiprocessing.get_context('spawn')
            if getattr(speaker, '_mp_q', None) is None:
                speaker.init_mp_queue(ctx.Queue())
            mq = speaker._mp_q
        except ImportError:
            mq = None

        p = multiprocessing.Process(
            target=background_person_search_task,
            args=(video_source, self.detector_model, target_name, mq),
            daemon=False
        )
        p.start()
        p.join()

    def start_person_tracking(self, video_source, target_name):
        print(f"启动进程跟踪人物: {target_name}")
        if self._process is not None and self._process.is_alive():
            print("人物追踪任务已经在后台运行中")
            return
        _safe_remove_pid(self.pid_file)

        try:
            import speaker
            ctx = multiprocessing.get_context('spawn')
            if getattr(speaker, '_mp_q', None) is None:
                speaker.init_mp_queue(ctx.Queue())
            mq = speaker._mp_q
        except ImportError:
            mq = None

        p = multiprocessing.Process(
            target=background_person_tracking_task,
            args=(video_source, self.detector_model, self.reid_model, target_name, self.pid_file, mq),
            daemon=False,
        )
        p.start()
        self._process = p
        try:
            with open(self.pid_file, "w") as f:
                f.write(str(p.pid))
        except OSError as e:
            print(f" 写入 PID 文件失败: {e}")

    def stop_person_tracking(self):
        print("Kill 人物跟踪进程")
        try:
            self._terminate_process()
        finally:
            _safe_remove_pid(self.pid_file)
        set_motor(None, 0.0, 0.0)
        try:
            import speaker
            speaker.speak("好的，已停止人物追踪")
        except Exception:
            pass

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


if __name__ == '__main__':
    # ⚠️ 请在这里修改为你用来测试的输入视频路径
    INPUT_VIDEO_PATH = "/home/test/openbot_test_zhenghang/model_0320/openbot_validify_code/final_0322/tracking_person/test_video.mp4"   
    OUTPUT_VIDEO_PATH = "tracked_output.mp4"

    if not os.path.exists(INPUT_VIDEO_PATH):
        print(f"找不到输入视频: {INPUT_VIDEO_PATH}，请确认路径是否正确。")
        sys.exit(1)

    print("\n" + "="*50)
    print(f"正在启动独立视频测试模式...")
    print(f"输入视频: {INPUT_VIDEO_PATH}")
    print(f"输出视频: {OUTPUT_VIDEO_PATH}")
    print("="*50 + "\n")

    # 初始化 NPU 模型
    detector = RknnDetector(DETECTOR_MODEL, conf=DET_CONF, core_mask=RKNN_CORE_MASK)
    reid = RknnReID(REID_MODEL, core_mask=RKNN_CORE_MASK)
    tracker = MultiBoxTracker(reid)

    # 打开视频流
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    # 配置视频写入
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

    frame_count = 0
    start_time = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1

            # 1. 人体检测
            dets = detector.detect(frame, width, height, cls="person")
            
            # 2. ReID 行人追踪 (内部使用最明显的人物自动初始化)
            tracker.trackResults(dets, frame)
            
            # 3. 计算底盘指令（仅作为显示参考）
            tracker.updateTarget()
            
            # 4. 绘制结果画框和状态 UI
            vis = draw_tracking_ui(frame, tracker, target_name="TestTarget")
            
            # 写入结果视频
            out_video.write(vis)
            
            # 本地窗口实时显示（如果在无桌面环境下运行，请注释掉 cv2.imshow）
            cv2.imshow("Tracking Test", vis)
            if cv2.waitKey(1) & 0xFF == 27:  # 按 ESC 退出
                print("手动终止测试")
                break
            
            if frame_count % 30 == 0:
                print(f"已处理 {frame_count} 帧...")

    finally:
        print("\n测试结束，正在释放资源...")
        cap.release()
        out_video.release()
        detector.release()
        reid.release()
        cv2.destroyAllWindows()
        
        elapsed = time.time() - start_time
        print(f"处理完成！共耗时 {elapsed:.2f} 秒，处理 {frame_count} 帧。")
        print(f"✅ 结果视频已保存至: {os.path.abspath(OUTPUT_VIDEO_PATH)}")