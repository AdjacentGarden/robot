import cv2
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1
import os

# ================= 1. 填入你指定的绝对路径 =================
MODEL_PATH = '/home/test/openbot_test_zhenghang/model_0320/openbot_validify_code/final_0322/model/convert/20180402-114759-vggface2.pt'
FACE1_PATH = '/home/test/openbot_test_zhenghang/model_0320/openbot_validify_code/final_0322/model/convert/face_image/face1.png'
FACE2_PATH = '/home/test/openbot_test_zhenghang/model_0320/openbot_validify_code/final_0322/model/convert/face_image/face2.png'

# ================= 2. 核心提取与比对逻辑 =================
def load_face_model(ckpt_path):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"❌ 找不到模型权重文件: {ckpt_path}")
        
    print(f"🔄 正在加载模型权重...")
    model = InceptionResnetV1(pretrained=None)
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=False)
    model.eval()
    print("✅ 模型加载完毕！")
    return model

def extract_feature(image_path, model):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ 找不到图片文件: {image_path}")
        
    # 1. OpenCV 读取图片
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"❌ 图片读取失败，请检查文件是否损坏: {image_path}")
        
    # 2. 严格对齐你项目里的预处理逻辑
    resized = cv2.resize(img, (160, 160))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).float()
    tensor = (tensor - 127.5) / 128.0

    # 3. 提取 512 维特征
    with torch.no_grad():
        vector = model(tensor).cpu().numpy().flatten()
    
    return vector

def calculate_similarity(vec1, vec2):
    # 余弦相似度计算
    dot_product = np.dot(vec1, vec2)
    norm_v1 = np.linalg.norm(vec1)
    norm_v2 = np.linalg.norm(vec2)
    
    if norm_v1 < 1e-9 or norm_v2 < 1e-9:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)

# ================= 3. 运行验证 =================
if __name__ == '__main__':
    print("\n" + "="*50)
    print("🎯 FaceNet 模型绝对精度验证")
    print("="*50)
    
    try:
        # 加载模型
        face_model = load_face_model(MODEL_PATH)
        
        # 提取特征
        print(f"\n📷 提取特征 1: {os.path.basename(FACE1_PATH)}")
        vec1 = extract_feature(FACE1_PATH, face_model)
        
        print(f"📷 提取特征 2: {os.path.basename(FACE2_PATH)}")
        vec2 = extract_feature(FACE2_PATH, face_model)
        
        # 计算相似度
        sim = calculate_similarity(vec1, vec2)
        
        print("\n" + "="*50)
        print(f"🧠 余弦相似度得分: {sim:.4f}")
        print("="*50)
        
        if sim >= 0.7:
            print("✅ 结论: 相似度 >= 0.7，模型判定为【同一个人】！模型功能完全正常。")
        else:
            print("❌ 结论: 相似度 < 0.7，模型判定为【不同的人】。")
            print("💡 提示: 如果这两张确实是同一个人，可能是其中一张脸极度模糊、大角度侧脸或被严重遮挡。")
            
    except Exception as e:
        print(f"\n💥 运行发生异常: {e}")