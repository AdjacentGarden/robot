import argparse
from pathlib import Path

import cv2
import numpy as np
from rknnlite.api import RKNNLite


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
    """
    对应你的转换配置:
      mean_values: [127.5, 127.5, 127.5]
      std_values : [127.5, 127.5, 127.5]
      quant_img_RGB2BGR: false

    所以这里只做:
      BGR -> RGB
      resize
      HWC uint8
    归一化交给 RKNN Runtime
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.uint8)
    x = np.expand_dims(x, axis=0)   # NHWC
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

    # already NCHW
    if arr.shape[1] > 4 and arr.shape[2] <= 128 and arr.shape[3] <= 128:
        return arr

    # maybe NHWC
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/test/openbot_test_zhenghang/model_0320/openbot_validify_code/final_0322/data/convert/detect/detect_v2.rknn")
    parser.add_argument("--image", default="./person.png")
    parser.add_argument("--score", type=float, default=0.3)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--output", default="./result_rk3588.jpg")
    parser.add_argument("--core-mask", type=int, default=4,
                        help="0:auto, 1:NPU_CORE_0, 2:NPU_CORE_1, 3:NPU_CORE_2, 4:NPU_CORE_0_1_2")
    args = parser.parse_args()

    if not Path(args.model).exists():
        raise FileNotFoundError(args.model)
    if not Path(args.image).exists():
        raise FileNotFoundError(args.image)

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")
    orig_h, orig_w = image.shape[:2]

    rknn_lite = RKNNLite()

    print("--> load_rknn")
    ret = rknn_lite.load_rknn(args.model)
    if ret != 0:
        raise RuntimeError(f"load_rknn failed: {ret}")

    print("--> init_runtime")
    if args.core_mask == 1:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    elif args.core_mask == 2:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_1)
    elif args.core_mask == 3:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_2)
    elif args.core_mask == 4:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    else:
        ret = rknn_lite.init_runtime()

    if ret != 0:
        raise RuntimeError(f"init_runtime failed: {ret}")

    x = preprocess_for_rknn(image, 300, 300)
    print(f"Input shape: {x.shape}, dtype={x.dtype}")

    outputs = rknn_lite.inference(inputs=[x], data_format=['nhwc'])

    print(f"Output tensors: {len(outputs)}")
    for i, out in enumerate(outputs):
        arr = np.array(out)
        print(f"[{i}] shape={arr.shape}, dtype={arr.dtype}, min={arr.min():.4f}, max={arr.max():.4f}")

    raw_box_encodings, raw_class_logits, feature_map_shapes = collect_raw_predictions(outputs)

    print(f"Feature map shapes: {feature_map_shapes}")
    print(f"Raw boxes shape: {raw_box_encodings.shape}")
    print(f"Raw logits shape: {raw_class_logits.shape}")

    detections = postprocess(
        raw_box_encodings,
        raw_class_logits,
        feature_map_shapes,
        orig_w=orig_w,
        orig_h=orig_h,
        score_thresh=args.score,
        iou_thresh=args.iou,
        max_total=100,
    )

    print(f"Detections: {len(detections)}")
    for det in detections[:20]:
        print(det)

    vis = draw_detections(image, detections)
    cv2.imwrite(args.output, vis)
    print(f"Saved result to: {args.output}")

    rknn_lite.release()


if __name__ == "__main__":
    main()
