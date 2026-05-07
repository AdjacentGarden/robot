import os
import sys
import requests
from datetime import datetime

# ===================== 视频上传功能 =====================
def upload_pet_tracking_video():
    """
    上传宠物追踪视频到服务器
    """
    # 接口地址
    url = "http://119.91.146.142:8882/api/video/upload" # url = "http://119.91.146.142:8882/api/video/upload"
    
    # 请求参数
    code = "pet_tracking_robot"  # 设备唯一编码
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 视频开始时间
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 视频结束时间
    
    # 视频文件路径
    video_file_path = "/home/test/code/pet_tracking_record.mp4"
    
    # 检查文件是否存在
    if not os.path.exists(video_file_path):
        print(f"❌ 视频文件不存在: {video_file_path}")
        return False
    
    # 构建请求数据
    data = {
        "code": code,
        "startTime": start_time,
        "endTime": end_time
    }
    
    # 构建文件数据
    try:
        files = {
            "file": ("pet_tracking.mp4", open(video_file_path, "rb"), "video/mp4")
        }
        
        # 发送请求
        print("📤 开始上传宠物追踪视频...")
        print(f"请求URL: {url}")
        print(f"视频文件: {video_file_path}")
        
        response = requests.get(url, data=data, files=files)
        
        # 打印响应结果
        print(f"响应状态码: {response.status_code}")
        print(f"响应内容: {response.text}")
        
        if response.status_code == 200:
            print("✅ 视频上传成功")
            return True
        else:
            print("❌ 视频上传失败")
            return False
            
    except Exception as e:
        print(f"❌ 视频上传请求失败: {str(e)}")
        return False
    # finally:
        # 不需要手动关闭文件，requests库会自动处理

# ===================== 新视频上传功能 =====================
def upload_pet_video_to_app_server():
    """
    上传视频到指定服务器并返回视频URL
    """
    # 填你【云端电脑的IP】
    CLOUD_HOST = "100.86.247.77"  # 换成另一台电脑的内网IP
    PORT = 8000
    
    video_file_path = "/home/test/code/pet_tracking_record.mp4"
    
    # 检查文件是否存在
    if not os.path.exists(video_file_path):
        print(f"❌ 视频文件不存在: {video_file_path}")
        return None
    
    print("📤 开始上传宠物追踪视频到指定服务器...")
    video_upload_url = f"http://{CLOUD_HOST}:{PORT}/upload"
    print(f"视频上传URL: {video_upload_url}")
    print(f"视频文件: {video_file_path}")
    
    try:
        # 构建文件数据
        files = {"file": open(video_file_path, "rb")}
        
        # 发送请求
        video_response = requests.post(video_upload_url, files=files)
        
        # 打印响应结果
        print(f"响应状态码: {video_response.status_code}")
        print(f"响应内容: {video_response.text}")
        
        if video_response.status_code == 200:
            data = video_response.json()
            print("✅ 视频上传到指定服务器成功")
            print("📹 可用videoUrl：", data["videoUrl"])
            # 返回视频URL
            return data["videoUrl"]
        else:
            print("❌ 视频上传到指定服务器失败")
            return None
            
    except Exception as e:
        print(f"❌ 视频上传请求失败: {str(e)}")
        return None