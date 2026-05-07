# import cv2
# import time

# def open_camera(cam_id):
#     cap = cv2.VideoCapture(cam_id)
#     if not cap.isOpened():
#         print(f"[错误] 无法打开摄像头 {cam_id}")
#         return None
#     print(f"[成功] 摄像头 {cam_id} 已打开")
#     return cap

# def main():
#     # 👉 改这里（你之前是 40 或 41）
#     cam_id = 40

#     cap = open_camera(cam_id)
#     if cap is None:
#         return

#     # 可选：设置分辨率
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#     prev_time = time.time()

#     print("按 q 退出")

#     while True:
#         ret, frame = cap.read()
#         frame = cv2.flip(frame, -1)  # 水平翻转，适合自拍摄像头
#         if not ret:
#             print("[错误] 读取帧失败")
#             break

#         # 计算 FPS
#         curr_time = time.time()
#         fps = 1 / (curr_time - prev_time)
#         prev_time = curr_time

#         # 画信息
#         cv2.putText(
#             frame,
#             f"CamID: {cam_id} FPS: {fps:.2f}",
#             (20, 40),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             1,
#             (0, 255, 0),
#             2
#         )

#         # 显示窗口（需要 VNC 正常授权）
#         cv2.imshow("Camera Test", frame)

#         # 按 q 退出
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()

import cv2
import time

def open_camera(cam_id):
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"[错误] 无法打开摄像头 {cam_id}")
        return None
    print(f"[成功] 摄像头 {cam_id} 已打开")
    return cap

def main():
    cam_id = 40
    duration = 30  
    output_path = "output.mp4"

    cap = open_camera(cam_id)
    if cap is None:
        return

    width = 640
    height = 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 20.0 

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"开始录制 {duration} 秒视频，保存为 {output_path}")

    start_time = time.time()
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[错误] 读取帧失败")
            break

        out.write(frame)

        curr_time = time.time()
        show_fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # 显示信息
        # cv2.putText(
        #     frame,
        #     f"Recording... FPS: {show_fps:.2f}",
        #     (20, 40),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     1,
        #     (0, 0, 255),
        #     2
        # )

        # cv2.imshow("Recording", frame)

        # 时间控制：30秒自动结束
        if time.time() - start_time > duration:
            print("录制完成")
            break

        # 手动退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("手动停止")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"视频已保存到: {output_path}")

if __name__ == "__main__":
    main()