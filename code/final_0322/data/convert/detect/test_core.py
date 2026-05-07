import time
import cv2
import numpy as np
from rknnlite.api import RKNNLite

MODEL = "./detect_v2.rknn"
IMAGE = "./cat.png"

def preprocess(image_bgr, input_h=300, input_w=300):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.uint8)
    x = np.expand_dims(x, axis=0)   # NHWC
    return x

def list_core_masks():
    masks = {}
    for name in dir(RKNNLite):
        if "NPU_CORE" in name:
            masks[name] = getattr(RKNNLite, name)
    return masks

def try_one(mask_name, mask_value, x, repeat=10):
    rknn = RKNNLite()
    ret = rknn.load_rknn(MODEL)
    if ret != 0:
        print(f"{mask_name}: load_rknn failed -> {ret}")
        return

    try:
        ret = rknn.init_runtime(core_mask=mask_value)
        if ret != 0:
            print(f"{mask_name}: init_runtime failed -> {ret}")
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

        print(f"{mask_name}: OK, avg={sum(ts)/len(ts):.2f} ms, outputs={out_shapes}")

    except Exception as e:
        print(f"{mask_name}: exception -> {e}")
    finally:
        rknn.release()

def main():
    print("Detected core-mask constants:")
    masks = list_core_masks()
    for k, v in masks.items():
        print(f"  {k} = {v}")

    image = cv2.imread(IMAGE)
    if image is None:
        raise FileNotFoundError(IMAGE)
    x = preprocess(image)

    print("\nTesting masks...")
    for k, v in masks.items():
        try_one(k, v, x, repeat=10)

    print("\nTesting default init_runtime() without core_mask...")
    rknn = RKNNLite()
    ret = rknn.load_rknn(MODEL)
    if ret != 0:
        print(f"default: load_rknn failed -> {ret}")
        return
    ret = rknn.init_runtime()
    if ret != 0:
        print(f"default: init_runtime failed -> {ret}")
        return
    _ = rknn.inference(inputs=[x], data_format=['nhwc'])
    ts = []
    for _ in range(10):
        t0 = time.perf_counter()
        _ = rknn.inference(inputs=[x], data_format=['nhwc'])
        t1 = time.perf_counter()
        ts.append((t1 - t0) * 1000.0)
    print(f"default: OK, avg={sum(ts)/len(ts):.2f} ms")
    rknn.release()

if __name__ == "__main__":
    main()
