#!/usr/bin/env python3
"""
MediaPipe Pose模型下载脚本
用于下载MediaPipe Pose预训练模型到本地目录
"""

import os
import urllib.request
import zipfile
import shutil

def download_mediapipe_pose_models():
    """下载MediaPipe Pose模型文件到本地"""
    
    # 创建模型目录
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mediapipe_models')
    pose_model_dir = os.path.join(models_dir, 'pose')
    
    os.makedirs(pose_model_dir, exist_ok=True)
    
    # MediaPipe Pose模型URL（这些是MediaPipe官方提供的模型文件）
    model_urls = {
        'pose_landmark_lite': 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
        'pose_landmark_full': 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task',
        'pose_landmark_heavy': 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task'
    }
    
    print("开始下载MediaPipe Pose模型...")
    
    for model_name, url in model_urls.items():
        model_path = os.path.join(pose_model_dir, f'{model_name}.task')
        
        if os.path.exists(model_path):
            print(f"模型 {model_name} 已存在，跳过下载")
            continue
            
        print(f"正在下载 {model_name}...")
        
        try:
            # 下载模型文件
            urllib.request.urlretrieve(url, model_path)
            print(f"成功下载 {model_name}")
        except Exception as e:
            print(f"下载 {model_name} 失败: {e}")
    
    print(f"模型下载完成，保存在目录: {pose_model_dir}")
    
    # 创建README文件说明模型用途
    readme_content = """# MediaPipe Pose 模型文件说明

## 模型文件说明
- `pose_landmark_lite.task`: 轻量级姿态检测模型，适合移动设备和实时应用
- `pose_landmark_full.task`: 完整版姿态检测模型，精度更高
- `pose_landmark_heavy.task`: 高精度姿态检测模型，适合需要最高精度的场景

## 使用方法
在代码中通过指定 `model_asset_path` 参数来使用本地模型：

```python
import mediapipe as mp

# 使用本地模型
pose = mp.solutions.pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    smooth_segmentation=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_asset_path='mediapipe_models/pose/pose_landmark_full.task'
)
```
"""
    
    with open(os.path.join(pose_model_dir, 'README.md'), 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    return pose_model_dir

if __name__ == "__main__":
    download_mediapipe_pose_models()