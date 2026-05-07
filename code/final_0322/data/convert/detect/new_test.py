import os
import time
import cv2
import numpy as np
import onnxruntime
from rknnlite.api import RKNNLite

RKNN_MODEL = "./detect_v2.rknn"
ONNX_MODEL = "/home/test/openbot_test_zhenghang/model_0320/openbot_validify_code/final_0322/data/convert/detect/detector_v2.onnx"
IMAGE = "./cat.png"

def preprocess(image_bgr, input_h=300, input_w=300):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.uint8)
    x = np.expand_dims(x, axis=0)   # NHWC (Shape: 1, H, W, 3)
    return x

def list_core_masks():
    masks = {}
    for name in dir(RKNNLite):
        if "NPU_CORE" in name:
            masks[name] = getattr(RKNNLite, name)
    return masks

def try_rknn(mask_name, mask_value, x, repeat=10):
    rknn = RKNNLite()
    ret = rknn.load_rknn(RKNN_MODEL)
    if ret != 0:
        print(f"RKNN [{mask_name}]: load_rknn failed -> {ret}")
        return

    try:
        ret = rknn.init_runtime(core_mask=mask_value)
        if ret != 0:
            print(f"RKNN [{mask_name}]: init_runtime failed -> {ret}")
            return

        # warmup
        _ = rknn.inference(inputs=[x], data_format=['nhwc'])

        ts = []
        out_shapes = None
        for _ in range(repeat):
            t0 = time.perf_counter()
            outputs = rknn.inference(inputs=[x], data_format=['nhwc'])
            t1 = time.perf_counter()
            ts.append((t1 - t0) * 1000.0)
            if out_shapes is None:
                out_shapes = [np.array(o).shape for o in outputs]

        print(f"RKNN [{mask_name}]: OK, avg={sum(ts)/len(ts):.2f} ms, outputs={out_shapes}")

    except Exception as e:
        print(f"RKNN [{mask_name}]: exception -> {e}")
    finally:
        rknn.release()

def try_onnx(onnx_path, x_nhwc, repeat=10):
    print("\n" + "="*40)
    print("Testing ONNX on CPU...")
    print("="*40)
    
    if not os.path.exists(onnx_path):
        print(f"ONNX CPU: Model not found at {onnx_path}")
        return
        
    try:
        # 初始化 ONNX Runtime (强制使用 CPU)
        options = onnxruntime.SessionOptions()
        # options.intra_op_num_threads = 4 # 可选：限制CPU线程数以测试单核/多核性能
        session = onnxruntime.InferenceSession(onnx_path, sess_options=options, providers=['CPUExecutionProvider'])
        
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        
        # ONNX 通常期望 float32 类型
        x_onnx = x_nhwc.astype(np.float32)
        
        # 很多 ONNX 模型导出时输入格式是 NCHW (1, 3, H, W)
        # 这里的简单判断：如果第二维是 3，且当前 x_onnx 最后一维是 3，则进行 NHWC -> NCHW 转换
        if len(input_shape) == 4 and input_shape[1] == 3 and x_onnx.shape[3] == 3:
            x_onnx = np.transpose(x_onnx, (0, 3, 1, 2)) 
            
        # warmup
        _ = session.run(None, {input_name: x_onnx})
        
        ts = []
        out_shapes = None
        for _ in range(repeat):
            t0 = time.perf_counter()
            outputs = session.run(None, {input_name: x_onnx})
            t1 = time.perf_counter()
            ts.append((t1 - t0) * 1000.0)
            if out_shapes is None:
                out_shapes = [np.array(o).shape for o in outputs]
                
        print(f"ONNX CPU: OK, avg={sum(ts)/len(ts):.2f} ms, outputs={out_shapes}")
        
    except Exception as e:
        print(f"ONNX CPU exception -> {e}")

def main():
    print("Detected core-mask constants:")
    masks = list_core_masks()
    for k, v in masks.items():
        print(f"  {k} = {v}")

    image = cv2.imread(IMAGE)
    if image is None:
        raise FileNotFoundError(f"Image not found: {IMAGE}")
    
    # 预处理获取 NHWC, uint8 格式的数据
    x = preprocess(image)

    # 1. 测试 RKNN 在不同 NPU 核心上的表现
    print("\n" + "="*40)
    print("Testing RKNN on NPU masks...")
    print("="*40)
    for k, v in masks.items():
        try_rknn(k, v, x, repeat=10)

    print("\nTesting RKNN default init_runtime() without core_mask...")
    rknn = RKNNLite()
    if rknn.load_rknn(RKNN_MODEL) == 0:
        if rknn.init_runtime() == 0:
            _ = rknn.inference(inputs=[x], data_format=['nhwc'])
            ts = []
            for _ in range(10):
                t0 = time.perf_counter()
                _ = rknn.inference(inputs=[x], data_format=['nhwc'])
                t1 = time.perf_counter()
                ts.append((t1 - t0) * 1000.0)
            print(f"RKNN [default]: OK, avg={sum(ts)/len(ts):.2f} ms")
        rknn.release()

    # 2. 测试 ONNX 在 CPU 上的表现
    try_onnx(ONNX_MODEL, x, repeat=10)

if __name__ == "__main__":
    main()