import cv2
import numpy as np
import os
import time

# ==========================================
# 标定核心配置区 (请务必根据实际情况修改)
# ==========================================
CAMERA_ID = '/dev/video21'  # 你的摄像头设备号
CHESSBOARD_SIZE = (11, 8)    # 棋盘格内角点数量 (横向格数-1, 纵向格数-1)。常见的 10x7 格子，填 (9, 6)
SQUARE_SIZE = 0.020         # 【极度重要】物理世界中一个小黑方块的边长，单位：米。这里假设是 25mm。

SAVE_DIR = "./calib_images" # 抓拍图片的保存目录
CALIB_FILE = "camera_params.npz" # 最终标定结果保存的文件名

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def main():
    # 1. 准备真实的 3D 世界坐标
    # 格式如: (0,0,0), (0.025,0,0), (0.05,0,0) ...
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    # 用于存储所有成功抓拍图片的 3D 物理点和 2D 像素点
    objpoints = [] 
    imgpoints = [] 

    # 亚像素级角点提取的迭代终止条件 (最大迭代 30 次或精度达到 0.001)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    print("======================================================")
    print(" 📷 工业级相机标定程序已启动！")
    print(f" ⚠️ 当前设定棋盘格: {CHESSBOARD_SIZE[0]}x{CHESSBOARD_SIZE[1]} 内角点, 方块边长: {SQUARE_SIZE*1000} mm")
    print("------------------------------------------------------")
    print(" [操作指南]")
    print(" 1. 将打印好的棋盘格贴在【绝对平整】的硬板上。")
    print(" 2. 在摄像头前变换距离、上下左右倾斜角度。")
    print(" 3. 画面出现彩色连线时，按键盘 'C' 键抓拍。")
    print(" 4. 抓拍 15~30 张后，按键盘 'Q' 键开始自动计算！")
    print("======================================================")

    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print(f"❌ 无法打开摄像头 {CAMERA_ID}，请检查设备连线或权限！")
        return

    captured_count = 0
    img_shape = None

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
            
        # 翻转画面以便于人眼观察（可选，如果不习惯可以注释掉）
        frame = cv2.flip(frame, -1) 
        frame = cv2.flip(frame, 1) 
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if img_shape is None:
            img_shape = gray.shape[::-1] # 获取 (width, height)

        # 寻找棋盘格内角点
        # 增加自适应阈值和图像归一化，提高识别率
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        ret_corners, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, flags)
        
        vis = frame.copy()
        
        if ret_corners:
            # 画出角点
            cv2.drawChessboardCorners(vis, CHESSBOARD_SIZE, corners, ret_corners)
            cv2.putText(vis, "Ready to Capture! Press 'C'", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(vis, "Finding corners...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
        cv2.putText(vis, f"Captured: {captured_count} / 20+", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        cv2.imshow('Camera Calibration', vis)
        
        key = cv2.waitKey(1) & 0xFF
        
        # 按下 'C' 键且识别到了角点
        if key == ord('c') and ret_corners:
            # 对找到的角点进行亚像素级精确化
            corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            imgpoints.append(corners_subpix)
            objpoints.append(objp)
            
            # 保存原图留底
            cv2.imwrite(os.path.join(SAVE_DIR, f"calib_{captured_count:02d}.jpg"), frame)
            
            captured_count += 1
            print(f"✅ 成功抓拍第 {captured_count} 张！(请换个角度继续)")
            
            # 屏幕闪烁提示
            cv2.imshow('Camera Calibration', np.ones_like(frame)*255)
            cv2.waitKey(100)
            
        # 按下 'Q' 或 'ESC' 退出并开始计算
        elif key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    if captured_count < 10:
        print("\n❌ 抓拍图片太少（少于10张），无法保证标定精度，已取消计算。")
        return

    print("\n⏳ 正在进行矩阵解算，请稍候...")
    
    # 2. 执行标定核心算法
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)
    
    # 3. 计算重投影误差 (检验标定质量)
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    total_error = mean_error / len(objpoints)

    # 4. 打印并保存结果
    print("\n🎉 标定完成！以下是你的摄像头专属内参：")
    print("=" * 60)
    print("CAMERA_MATRIX = np.array([")
    print(f"    [{mtx[0][0]:.4f}, {mtx[0][1]:.4f}, {mtx[0][2]:.4f}],")
    print(f"    [{mtx[1][0]:.4f}, {mtx[1][1]:.4f}, {mtx[1][2]:.4f}],")
    print(f"    [{mtx[2][0]:.4f}, {mtx[2][1]:.4f}, {mtx[2][2]:.4f}]")
    print("], dtype=np.float32)")
    print()
    print(f"DIST_COEFFS = np.array([{dist[0][0]:.5f}, {dist[0][1]:.5f}, {dist[0][2]:.5f}, {dist[0][3]:.5f}, {dist[0][4]:.5f}], dtype=np.float32)")
    print("=" * 60)
    
    print(f"\n🎯 标定评级 [重投影误差]: {total_error:.4f} 像素")
    if total_error < 0.2:
        print("   🌟 完美！你的标定非常精准。")
    elif total_error < 0.5:
        print("   ✅ 优秀！可直接用于高精度 3D 导航。")
    elif total_error < 1.0:
        print("   ⚠️ 及格。能用，但可能不够精准，如果回仓有偏差建议重新标定。")
    else:
        print("   ❌ 极差！误差过大，可能是棋盘格不平整或尺寸填错，强烈建议重新抓拍。")

    # 保存为 npz 文件，方便以后直接 load
    np.savez(CALIB_FILE, camera_matrix=mtx, dist_coeffs=dist)
    print(f"\n💾 参数已备份至本地文件: {CALIB_FILE}")

if __name__ == "__main__":
    main()
