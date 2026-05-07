import os
import cv2
import numpy as np
import onnxruntime

# ================= 路径配置 =================
IMAGE_PATH = "/home/test/openbot_test_zhenghang/model_0320/openbot_validify_code/final_0322/model/convert/sitandup_image/image.png"
MODEL_PATH = "/home/test/openbot_test_zhenghang/model_0320/openbot_validify_code/final_0322/model/convert/sitandup_model.onnx"
OUTPUT_PATH = "verify_result.png"  # 验证结果会保存在当前目录下
# ============================================

def situp_preproc(cv_img, resize=(224, 224),
                  mean=(103.53, 116.28, 123.675),
                  std=(57.375, 57.12, 58.395)):
    """与主程序保持完全一致的预处理逻辑"""
    img = cv2.resize(cv_img, resize).astype(np.float32)
    img = (img - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    return np.expand_dims(img.transpose(2, 0, 1), axis=0)

def main():
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"找不到图片文件: {IMAGE_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"找不到模型文件: {MODEL_PATH}")

    print(f"正在加载模型: {MODEL_PATH} ...")
    # 2. 初始化 ONNX Runtime Session
    so = onnxruntime.SessionOptions()
    so.intra_op_num_threads = 1
    session = onnxruntime.InferenceSession(MODEL_PATH, sess_options=so)
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    print(f"正在读取图片: {IMAGE_PATH} ...")
    # 3. 读取并备份原图
    ori_img = cv2.imread(IMAGE_PATH)
    if ori_img is None:
        raise ValueError("图片读取失败，请检查图片格式是否正确或是否损坏。")
    h, w = ori_img.shape[:2]

    # 4. 预处理与模型推理
    input_data = situp_preproc(ori_img)
    preds = session.run([output_name], {input_name: input_data})
    
    # 获取热力图 (shape: [3, H, W])
    heatmaps = preds[0][0] 

    # 5. 从热力图中提取峰值坐标 (返回归一化的 y, x)
    def get_peak(fm):
        idx = np.unravel_index(np.argmax(fm), fm.shape)
        # 注意：idx[0] 是 y轴 (高度方向), idx[1] 是 x轴 (宽度方向)
        return [(idx[0] + 0.5) / fm.shape[0], (idx[1] + 0.5) / fm.shape[1]]

    p_head = get_peak(heatmaps[0])
    p_knee = get_peak(heatmaps[1])
    p_crotch = get_peak(heatmaps[2])

    # 6. 将归一化坐标还原到原图的像素级坐标 (X, Y)
    head_px = (int(p_head[1] * w), int(p_head[0] * h))
    knee_px = (int(p_knee[1] * w), int(p_knee[0] * h))
    crotch_px = (int(p_crotch[1] * w), int(p_crotch[0] * h))

    print("-" * 30)
    print(f"检测结果坐标 (X, Y):")
    print(f"头部 (Head)   : {head_px}")
    print(f"膝盖 (Knee)   : {knee_px}")
    print(f"胯部 (Crotch) : {crotch_px}")
    print("-" * 30)

    # 7. 可视化绘制
    draw_img = ori_img.copy()
    
    # 绘制连接线
    cv2.line(draw_img, head_px, crotch_px, (255, 255, 0), 3)  # 青色：头到胯
    cv2.line(draw_img, crotch_px, knee_px, (0, 255, 255), 3)  # 黄色：胯到膝盖

    # 绘制关键点圆圈
    cv2.circle(draw_img, head_px, 8, (0, 0, 255), -1)     # 红色：头
    cv2.circle(draw_img, crotch_px, 8, (255, 0, 0), -1)   # 蓝色：胯
    cv2.circle(draw_img, knee_px, 8, (0, 255, 0), -1)     # 绿色：膝盖

    # 绘制文字标签
    cv2.putText(draw_img, "Head", (head_px[0] + 10, head_px[1]), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(draw_img, "Crotch", (crotch_px[0] + 10, crotch_px[1]), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(draw_img, "Knee", (knee_px[0] + 10, knee_px[1]), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 8. 保存结果 (在无头服务器环境下，imwrite 比 imshow 更安全)
    cv2.imwrite(OUTPUT_PATH, draw_img)
    print(f"可视化验证图已成功保存为: {OUTPUT_PATH}")
    print("你可以打开该图片查看模型预测是否准确！")

if __name__ == "__main__":
    main()