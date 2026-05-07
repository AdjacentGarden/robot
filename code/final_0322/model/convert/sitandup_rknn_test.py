import os
import cv2
import numpy as np
from rknnlite.api import RKNNLite

# ================= 路径配置 =================
IMAGE_PATH = "/home/test/openbot_test_zhenghang/model_0320/openbot_validify_code/final_0322/model/convert/sitandup_image/image.png"
MODEL_PATH = "/home/test/openbot_test_zhenghang/model_0320/openbot_validify_code/final_0322/model/convert/sitandup.rknn"
OUTPUT_PATH = "verify_result_rknn.png"
# ============================================


def preprocess_for_rknn(cv_img, resize=(224, 224)):
    """
    保持和原 ONNX 流程一致的输入语义：
    1. cv2.imread -> BGR
    2. resize 到 224x224
    3. 不手工做 mean/std 归一化（因为已经写进 RKNN config）
    4. 转成 NCHW 4维输入
    """
    img = cv2.resize(cv_img, resize).astype(np.uint8)   # 保持 BGR
    img = np.transpose(img, (2, 0, 1))                  # HWC -> CHW
    img = np.expand_dims(img, axis=0)                   # CHW -> NCHW
    img = np.ascontiguousarray(img)
    return img


def get_peak(fm):
    """
    从单张热力图中提取峰值点，返回归一化 (y, x)
    """
    idx = np.unravel_index(np.argmax(fm), fm.shape)
    return [(idx[0] + 0.5) / fm.shape[0], (idx[1] + 0.5) / fm.shape[1]]


def load_rknn_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到 RKNN 模型文件: {model_path}")

    print(f"正在加载 RKNN 模型: {model_path} ...")
    rknn = RKNNLite()

    ret = rknn.load_rknn(model_path)
    if ret != 0:
        raise RuntimeError(f"load_rknn 失败，返回码: {ret}")

    print("正在初始化 RKNN 运行时 ...")
    try:
        ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
    except Exception:
        ret = rknn.init_runtime()

    if ret != 0:
        raise RuntimeError(f"init_runtime 失败，返回码: {ret}")

    print("RKNN 模型加载成功。")
    return rknn


def main():
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"找不到图片文件: {IMAGE_PATH}")

    rknn = None
    try:
        rknn = load_rknn_model(MODEL_PATH)

        print(f"正在读取图片: {IMAGE_PATH} ...")
        ori_img = cv2.imread(IMAGE_PATH)
        if ori_img is None:
            raise ValueError("图片读取失败，请检查图片格式是否正确或是否损坏。")
        h, w = ori_img.shape[:2]

        # 1. 预处理
        input_data = preprocess_for_rknn(ori_img)
        print(f"输入 shape: {input_data.shape}, dtype: {input_data.dtype}")

        # 2. RKNN 推理
        # 注意：这里显式指定 NCHW
        outputs = rknn.inference(inputs=[input_data], data_format=['nchw'])
        if outputs is None or len(outputs) == 0:
            raise RuntimeError("RKNN 推理失败，没有输出。")

        raw_output = outputs[0]
        print(f"原始输出 shape: {raw_output.shape}, dtype: {raw_output.dtype}")

        # 预期输出形状一般是 [1, 3, H, W]
        if raw_output.ndim != 4 or raw_output.shape[0] != 1:
            raise ValueError(f"输出形状异常，期望类似 [1, 3, H, W]，实际为: {raw_output.shape}")

        heatmaps = raw_output[0]   # [3, H, W]

        # 3. 从热力图中提取峰值坐标
        p_head = get_peak(heatmaps[0])
        p_knee = get_peak(heatmaps[1])
        p_crotch = get_peak(heatmaps[2])

        # 4. 还原到原图像素坐标 (X, Y)
        head_px = (int(p_head[1] * w), int(p_head[0] * h))
        knee_px = (int(p_knee[1] * w), int(p_knee[0] * h))
        crotch_px = (int(p_crotch[1] * w), int(p_crotch[0] * h))

        print("-" * 30)
        print("检测结果坐标 (X, Y):")
        print(f"头部 (Head)   : {head_px}")
        print(f"膝盖 (Knee)   : {knee_px}")
        print(f"胯部 (Crotch) : {crotch_px}")
        print("-" * 30)

        # 5. 可视化绘制
        draw_img = ori_img.copy()

        cv2.line(draw_img, head_px, crotch_px, (255, 255, 0), 3)  # 青色：头到胯
        cv2.line(draw_img, crotch_px, knee_px, (0, 255, 255), 3)  # 黄色：胯到膝

        cv2.circle(draw_img, head_px, 8, (0, 0, 255), -1)         # 红色：头
        cv2.circle(draw_img, crotch_px, 8, (255, 0, 0), -1)       # 蓝色：胯
        cv2.circle(draw_img, knee_px, 8, (0, 255, 0), -1)         # 绿色：膝

        cv2.putText(draw_img, "Head", (head_px[0] + 10, head_px[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(draw_img, "Crotch", (crotch_px[0] + 10, crotch_px[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(draw_img, "Knee", (knee_px[0] + 10, knee_px[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imwrite(OUTPUT_PATH, draw_img)
        print(f"可视化验证图已成功保存为: {OUTPUT_PATH}")
        print("你可以打开该图片查看量化后的 RKNN 模型预测是否准确。")

    finally:
        if rknn is not None:
            try:
                rknn.release()
            except Exception:
                pass


if __name__ == "__main__":
    main()
