import os
import cv2
import numpy as np
import time
import math
from dataclasses import dataclass
from typing import List, Tuple
import sys

from rknn3lite.api import RKNN3Lite


RKNN_MODEL = "/home/test/yolov8s/yolov8s_rknn3.rknn"
WEIGHT_MODEL = "/home/test/yolov8s/yolov8s_rknn3.weight"

VIDEO_PATH = "/home/test/code/final_0322/tracking_person/test_video.mp4"
OUTPUT_PATH = "/home/test/code/final_0322/tracking_person/yolo_result.mp4"

INPUT_SIZE = 640
CONF_THRES = 0.25
IOU_THRES = 0.45
REG_MAX = 16
NC = 80
PERSON_CLASS = 0


@dataclass
class Rect:
    x: float
    y: float
    width: float
    height: float


@dataclass
class Object:
    rect: Rect
    label: int
    conf: float


# =========================
# 基础函数
# =========================
def sigmoid(x: float) -> float:
    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def clamp(val: float, min_val: float = 0.0, max_val: float = 1280.0) -> float:
    return max(min_val, min(val, max_val))


def softmax(src: List[float], length: int) -> float:
    alpha = max(src[:length])
    exps = [math.exp(x - alpha) for x in src[:length]]
    denominator = sum(exps)
    if denominator == 0:
        return 0.0
    dst = [exp_val / denominator for exp_val in exps]
    dis_sum = sum(i * dst[i] for i in range(length))
    return dis_sum


# =========================
# IoU / NMS
# =========================
def get_iou_value(rect1: Rect, rect2: Rect) -> float:
    xx1 = max(rect1.x, rect2.x)
    yy1 = max(rect1.y, rect2.y)
    xx2 = min(rect1.x + rect1.width - 1, rect2.x + rect2.width - 1)
    yy2 = min(rect1.y + rect1.height - 1, rect2.y + rect2.height - 1)

    inter_width = max(0, xx2 - xx1 + 1)
    inter_height = max(0, yy2 - yy1 + 1)

    inter_area = inter_width * inter_height
    union_area = rect1.width * rect1.height + rect2.width * rect2.height - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def nms_boxes(boxes: List[Rect], scores: List[float], score_threshold: float, nms_threshold: float) -> List[int]:
    objects = [Object(rect=boxes[i], conf=scores[i], label=i) for i in range(len(boxes))]
    objects.sort(key=lambda x: x.conf, reverse=True)

    indices = []
    while objects:
        current = objects.pop(0)
        if current.conf < score_threshold:
            continue
        indices.append(current.label)
        objects = [obj for obj in objects if get_iou_value(current.rect, obj.rect) <= nms_threshold]
    return indices


def non_max_suppression(
    proposals: List[Object],
    results: List[Object],
    orin_h: int,
    orin_w: int,
    dh: float = 0,
    dw: float = 0,
    ratio_h: float = 1.0,
    ratio_w: float = 1.0,
    conf_thres: float = 0.5,
    iou_thres: float = 0.5
):
    bboxes = [obj.rect for obj in proposals]
    scores = [obj.conf for obj in proposals]
    labels = [obj.label for obj in proposals]

    indices = nms_boxes(bboxes, scores, conf_thres, iou_thres)

    for idx in indices:
        bbox = bboxes[idx]

        x0 = (bbox.x - dw) / ratio_w
        y0 = (bbox.y - dh) / ratio_h
        x1 = (bbox.x + bbox.width - dw) / ratio_w
        y1 = (bbox.y + bbox.height - dh) / ratio_h

        x0 = clamp(x0, 0, orin_w)
        y0 = clamp(y0, 0, orin_h)
        x1 = clamp(x1, 0, orin_w)
        y1 = clamp(y1, 0, orin_h)

        if x1 <= x0 or y1 <= y0:
            continue

        obj = Object(
            rect=Rect(x=x0, y=y0, width=x1 - x0, height=y1 - y0),
            label=labels[idx],
            conf=scores[idx]
        )
        results.append(obj)


# =========================
# 预处理
# =========================
def preprocess_image(frame: np.ndarray, target_size: int = 640, mode: str = 'letterbox'):
    """
    与你成功代码保持一致：
    - resize: 直接拉伸到目标大小
    - letterbox: 保持原始比例并补黑边
    """
    original_h, original_w = frame.shape[:2]

    ratio_w = target_size / original_w
    ratio_h = target_size / original_h

    if mode == 'resize':
        resized_img = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        dw, dh = 0, 0
    elif mode == 'letterbox':
        scale = min(ratio_w, ratio_h)
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)
        ratio_h = scale
        ratio_w = scale

        resized_img = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        dw = target_size - new_w
        dh = target_size - new_h

        top, bottom = dh // 2, dh - (dh // 2)
        left, right = dw // 2, dw - (dw // 2)

        resized_img = cv2.copyMakeBorder(
            resized_img,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )
    else:
        raise ValueError("Invalid mode. Choose either 'resize' or 'letterbox'.")

    resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

    mean = np.array([0, 0, 0], dtype=np.float32)
    norm = np.array([0.0039215686], dtype=np.float32)
    normalized_img = (resized_img.astype(np.float32) - mean) * norm

    normalized_img = np.transpose(normalized_img, (2, 0, 1))
    img = np.expand_dims(normalized_img, axis=0)

    return img, ratio_w, ratio_h, dw, dh, original_w, original_h


# =========================
# 调试输出
# =========================
def debug_output_shapes(outputs):
    print("[DEBUG] 模型输出共", len(outputs), "个张量：")
    for i, o in enumerate(outputs):
        arr = np.array(o)
        print(f"  outputs[{i}] shape={arr.shape}  dtype={arr.dtype}  min={arr.min():.3f}  max={arr.max():.3f}")
        if arr.ndim == 3 and arr.shape[-1] == 6:
            print("  [DEBUG] 前5行数据：")
            print(arr[0, :5, :])


# =========================
# YOLOv8 三尺度输出适配
# =========================
def to_hwc(feat, nc=NC, reg_max=REG_MAX):
    """
    把输出统一成 (1, H, W, C)
    兼容：
    - (1, C, H, W)
    - (1, H, W, C)
    - (C, H, W)
    - (H, W, C)
    """
    feat = np.array(feat, dtype=np.float32)
    expected_c = nc + 4 * reg_max

    if feat.ndim == 4:
        if feat.shape[-1] == expected_c:
            return feat
        if feat.shape[1] == expected_c:
            return np.transpose(feat, (0, 2, 3, 1))
    elif feat.ndim == 3:
        if feat.shape[0] == expected_c:
            return np.transpose(feat, (1, 2, 0))[np.newaxis, ...]
        if feat.shape[-1] == expected_c:
            return feat[np.newaxis, ...]
        if feat.shape[2] == expected_c:
            return feat[np.newaxis, ...]
    raise ValueError(f"Unexpected feature map shape: {feat.shape}, expected channel={expected_c}")


def generate_proposals(stride: int, feat_mat: np.ndarray, prob_threshold: float) -> List[Object]:
    objects = []
    feat_mat = to_hwc(feat_mat, NC, REG_MAX)

    num_grid_y, num_grid_x, num_w = feat_mat.shape[1], feat_mat.shape[2], feat_mat.shape[3]
    num_class = num_w - 4 * REG_MAX

    for i in range(num_grid_y):
        for j in range(num_grid_x):
            matat = feat_mat[0, i, j, :]

            class_scores = matat[:num_class]
            class_index = int(np.argmax(class_scores))
            class_score = sigmoid(float(class_scores[class_index]))

            if class_score >= prob_threshold:
                boxes_mat_ptr = matat[num_class:]

                x0 = j + 0.5 - softmax(boxes_mat_ptr[:REG_MAX].tolist(), REG_MAX)
                y0 = i + 0.5 - softmax(boxes_mat_ptr[REG_MAX:2 * REG_MAX].tolist(), REG_MAX)
                x1 = j + 0.5 + softmax(boxes_mat_ptr[2 * REG_MAX:3 * REG_MAX].tolist(), REG_MAX)
                y1 = i + 0.5 + softmax(boxes_mat_ptr[3 * REG_MAX:4 * REG_MAX].tolist(), REG_MAX)

                x0 *= stride
                y0 *= stride
                x1 *= stride
                y1 *= stride

                obj = Object(
                    rect=Rect(x=x0, y=y0, width=x1 - x0, height=y1 - y0),
                    label=class_index,
                    conf=class_score
                )
                objects.append(obj)

    return objects


def post_process_yolov8(
    outputs: List[np.ndarray],
    original_shape: tuple,
    ratio_w: float,
    ratio_h: float,
    dw: float,
    dh: float,
    conf_thres: float = 0.5,
    iou_thres: float = 0.5
) -> List[Object]:
    proposals = []
    strides = [8, 16, 32]

    normalized_feats = [to_hwc(o, NC, REG_MAX) for o in outputs]

    if len(normalized_feats) >= 3:
        for stride, feat_mat in zip(strides, normalized_feats[:3]):
            objects_stride = generate_proposals(stride, feat_mat, conf_thres)
            proposals.extend(objects_stride)
    else:
        for stride, feat_mat in zip(strides, normalized_feats):
            objects_stride = generate_proposals(stride, feat_mat, conf_thres)
            proposals.extend(objects_stride)

    img_h, img_w = original_shape

    results = []
    non_max_suppression(
        proposals,
        results,
        orin_h=img_h,
        orin_w=img_w,
        dh=dh / 2,
        dw=dw / 2,
        ratio_h=ratio_h,
        ratio_w=ratio_w,
        conf_thres=conf_thres,
        iou_thres=iou_thres
    )
    return results


# =========================
# 6D 输出适配： (1, 512, 6)
# =========================
def post_process_6d(
    output: np.ndarray,
    original_shape: tuple,
    ratio_w: float,
    ratio_h: float,
    dw: float,
    dh: float,
    conf_thres: float = 0.5,
    iou_thres: float = 0.5
) -> List[Object]:
    """
    适配输出形状为 (1, N, 6) 或 (N, 6) 的模型。
    尝试兼容两种常见格式：
    - [x1, y1, x2, y2, score, class_id]
    - [cx, cy, w, h, score, class_id]

    这里做了自动判断：
    1. 如果四个坐标看起来像左上右下，则按 xyxy
    2. 否则按 cxcywh
    3. 如果坐标明显在 0~1 或 0~2 范围，会先按 INPUT_SIZE 放大
    """
    arr = np.array(output, dtype=np.float32)

    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    elif arr.ndim != 2:
        raise ValueError(f"Unsupported 6D output shape: {arr.shape}")

    if arr.shape[1] != 6:
        raise ValueError(f"Expected last dim = 6, but got {arr.shape}")

    img_h, img_w = original_shape
    candidates: List[Object] = []

    for row in arr:
        if np.allclose(row, 0):
            continue

        a, b, c, d, score, cls_id = row[:6].tolist()

        if score > 1.0 or score < 0.0:
            score = sigmoid(float(score))

        if score < conf_thres:
            continue

        coords = [a, b, c, d]
        if max(abs(v) for v in coords) <= 2.0:
            a *= INPUT_SIZE
            b *= INPUT_SIZE
            c *= INPUT_SIZE
            d *= INPUT_SIZE

        # 先按 xyxy 试一次
        use_xyxy = False
        if c > a and d > b:
            if (c - a) > 1 and (d - b) > 1:
                use_xyxy = True

        if use_xyxy:
            x1, y1, x2, y2 = a, b, c, d
        else:
            cx, cy, w, h = a, b, c, d
            x1 = cx - w / 2.0
            y1 = cy - h / 2.0
            x2 = cx + w / 2.0
            y2 = cy + h / 2.0

        x0 = (x1 - dw) / ratio_w
        y0 = (y1 - dh) / ratio_h
        x1 = (x2 - dw) / ratio_w
        y1 = (y2 - dh) / ratio_h

        x0 = clamp(x0, 0, img_w)
        y0 = clamp(y0, 0, img_h)
        x1 = clamp(x1, 0, img_w)
        y1 = clamp(y1, 0, img_h)

        if x1 <= x0 or y1 <= y0:
            continue

        candidates.append(
            Object(
                rect=Rect(x=x0, y=y0, width=x1 - x0, height=y1 - y0),
                label=int(cls_id),
                conf=float(score)
            )
        )

    if not candidates:
        return []

    bboxes = [obj.rect for obj in candidates]
    scores = [obj.conf for obj in candidates]
    labels = [obj.label for obj in candidates]

    indices = nms_boxes(bboxes, scores, conf_thres, iou_thres)

    results = []
    for idx in indices:
        results.append(
            Object(
                rect=bboxes[idx],
                label=labels[idx],
                conf=scores[idx]
            )
        )

    return results


# =========================
# 自动后处理入口
# =========================
def post_process(
    outputs: List[np.ndarray],
    original_shape: tuple,
    ratio_w: float,
    ratio_h: float,
    dw: float,
    dh: float,
    conf_thres: float = 0.5,
    iou_thres: float = 0.5
) -> List[Object]:
    """
    自动适配两类输出：
    1. (1, 512, 6) 或 (512, 6)
    2. YOLOv8 三尺度输出
    """
    if len(outputs) == 1:
        arr = np.array(outputs[0], dtype=np.float32)

        if arr.ndim in (2, 3) and arr.shape[-1] == 6:
            return post_process_6d(
                arr,
                original_shape=original_shape,
                ratio_w=ratio_w,
                ratio_h=ratio_h,
                dw=dw,
                dh=dh,
                conf_thres=conf_thres,
                iou_thres=iou_thres
            )

    try:
        return post_process_yolov8(
            outputs,
            original_shape=original_shape,
            ratio_w=ratio_w,
            ratio_h=ratio_h,
            dw=dw,
            dh=dh,
            conf_thres=conf_thres,
            iou_thres=iou_thres
        )
    except Exception as e:
        print(f"❌ YOLOv8 后处理失败: {e}")
        return []


# =========================
# 可视化
# =========================
def draw(frame, detections):
    for obj in detections:
        if obj.label != PERSON_CLASS:
            continue

        x0, y0 = int(obj.rect.x), int(obj.rect.y)
        x1, y1 = int(obj.rect.x + obj.rect.width), int(obj.rect.y + obj.rect.height)

        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"person {obj.conf:.2f}",
            (x0, max(20, y0 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
    return frame


# =========================
# 主函数
# =========================
def main():
    rknn = RKNN3Lite()

    ret = rknn.load_rknn(RKNN_MODEL, WEIGHT_MODEL)
    if ret != 0:
        print("❌ load_rknn failed")
        return

    device_id_list = rknn.get_devices_id()
    if not device_id_list:
        print("❌ 没有找到可用设备")
        rknn.release()
        return

    device_id = device_id_list[0]
    ret = rknn.init_runtime(target="rk1820", core_mask=0x01, device_id=device_id)
    if ret != 0:
        print("❌ init_runtime failed")
        rknn.release()
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ 视频打开失败")
        rknn.release()
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 20.0

    ret_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ret_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (ret_w, ret_h))
    if not out.isOpened():
        print("❌ 输出视频创建失败")
        cap.release()
        rknn.release()
        return

    print("🚀 开始推理...")
    frame_idx = 0
    t_start = time.time()
    debug_done = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("✅ 视频读取完毕")
                break

            h, w = frame.shape[:2]

            img, ratio_w, ratio_h, dw, dh, original_w, original_h = preprocess_image(
                frame,
                target_size=INPUT_SIZE,
                mode='letterbox'
            )

            outputs = rknn.inference(inputs=[img])

            if outputs is None:
                print(f"❌ 第 {frame_idx} 帧推理失败")
                break

            if not debug_done:
                debug_output_shapes(outputs)
                debug_done = True

            detections = post_process(
                outputs,
                original_shape=(original_h, original_w),
                ratio_w=ratio_w,
                ratio_h=ratio_h,
                dw=dw,
                dh=dh,
                conf_thres=CONF_THRES,
                iou_thres=IOU_THRES
            )

            frame = draw(frame, detections)
            out.write(frame)

            frame_idx += 1
            if frame_idx % 20 == 0:
                elapsed = time.time() - t_start
                avg_fps = frame_idx / elapsed if elapsed > 0 else 0
                print(f"   已处理 {frame_idx} 帧, 平均 FPS: {avg_fps:.2f}")

    except KeyboardInterrupt:
        print("\n🛑 用户手动停止")
    finally:
        cap.release()
        out.release()
        rknn.release()
        print("✅ 完成")


if __name__ == "__main__":
    main()