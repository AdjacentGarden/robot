import os
import sys
import torch
import cv2

_CURR_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_CURR_DIR))

def _p(*rel):
    return os.path.join(_ROOT, *rel)

if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pet.pet_camera import PetTrackingSystem
from face.face_camera import FaceRecognitionSystem
from tracking_person.person_camera import PersonSearchTrackingSystem
from movement_count.sitandup_camera import SitupCountingSystem
from hand.script import GestureRecognitionSystem
from tracking_person.wheel_control import Board

CAMERA_ID = '/dev/video21'
_DETECTOR_PATH = _p('model', 'detector.onnx')
_REID_PATH     = _p('model', 'reid_model.onnx')
_FACE_MODEL    = _p('model', '20180402-114759-vggface2.pt')
_HAND_MODEL    = _p('model', 'hand_model_999.pth')
_SITUP_MODEL   = _p('model', 'sitandup_model.onnx')
_FACE_DB       = _p('test_faces.db')

print("初始化模块\n")

pet_system      = PetTrackingSystem(model_path=_DETECTOR_PATH)
face_system     = FaceRecognitionSystem(db_path=_FACE_DB, model_path=_FACE_MODEL)
person_sys      = PersonSearchTrackingSystem(_DETECTOR_PATH, _REID_PATH)
gesture_system  = GestureRecognitionSystem(model_path=_HAND_MODEL)
sitandup_system = SitupCountingSystem(model_path=_SITUP_MODEL)

print("所有视觉模块挂载完毕\n")


def _force_release_camera():
    """
    每个需要打开摄像头的任务执行前调用。
    通过原生 cv2 短暂打开再立即释放，清除上一个任务异常退出后
    残留的设备占用，确保下一个任务能正常拿到摄像头。
    """
    try:
        cap = cv2.VideoCapture(CAMERA_ID)
        if cap.isOpened():
            cap.release()
            print("[资源清理] 摄像头已强制释放")
        else:
            print("[资源清理] 摄像头未被占用，无需释放")
    except Exception as e:
        print(f"[资源清理] 释放摄像头时出错（可忽略）: {e}")


# ──────────────────────────────────────────────
# 宠物系统
# ──────────────────────────────────────────────

def find_pet(target: str):
    _force_release_camera()
    print(f"底层实现 开始旋转寻找宠物 {target}")
    pet_system.find_pet(CAMERA_ID, target)

def start_pet_tracking(target: str):
    _force_release_camera()
    print(f"底层实现 启动进程跟踪宠物 {target}")
    pet_system.start_pet_tracking(CAMERA_ID, target)

def stop_pet_tracking():
    # 停止操作不打开摄像头，不需要释放
    print("底层实现 终止宠物跟踪进程")
    pet_system.stop_pet_tracking()


# ──────────────────────────────────────────────
# 人脸系统
# ──────────────────────────────────────────────

def register_face(name: str):
    _force_release_camera()
    print(f"底层实现 录入人脸 {name}")
    face_system.register_face(CAMERA_ID, name)

def recognize_face():
    _force_release_camera()
    print("底层实现 持续五秒识别当前人脸")
    face_system.recognize_face(CAMERA_ID)


# ──────────────────────────────────────────────
# 人物系统
# ──────────────────────────────────────────────

def search_person(name: str):
    _force_release_camera()
    print(f"底层实现 开始搜寻人物 {name}")
    person_sys.search_person(CAMERA_ID, name)

def start_person_tracking(name: str):
    _force_release_camera()
    print(f"底层实现 启动进程跟踪人物 {name}")
    person_sys.start_person_tracking(CAMERA_ID, name)

def stop_person_tracking():
    # 停止操作不打开摄像头，不需要释放
    print("底层实现 终止人物跟踪进程")
    person_sys.stop_person_tracking()


# ──────────────────────────────────────────────
# 手势系统
# ──────────────────────────────────────────────

def identify_gesture():
    _force_release_camera()
    print("底层实现 开启手势识别窗口")
    gesture_system.identify_gesture(camera_id=CAMERA_ID)


# ──────────────────────────────────────────────
# 仰卧起坐系统
# ──────────────────────────────────────────────

def start_situp_counting():
    _force_release_camera()
    print("底层实现 启动仰卧起坐计数后台进程")
    sitandup_system.start_counting(CAMERA_ID)

def query_situp_progress():
    # 查询进度只读文件，不打开摄像头，不需要释放
    print("底层实现 读取文件中的实时个数并播报")
    sitandup_system.query_progress()

def stop_and_summarize():
    # 停止操作不打开摄像头，不需要释放
    print("底层实现 终止计数进程并汇总播报")
    sitandup_system.stop_and_summarize()


# ──────────────────────────────────────────────
# 本地测试菜单
# ──────────────────────────────────────────────

if __name__ == "__main__":
    board = Board()             
    board.enable_reception()   
    print("START...")
    menu = """
========================================
 Robot Central Control System - Integration Test Menu
========================================
 [Pet System - pet_camera.py]
   [1] Find cat              find_pet('cat')
   [2] Start tracking dog    start_pet_tracking('dog')
   [3] Stop pet tracking     stop_pet_tracking()

 [Face System - face_camera.py]
   [4] Register face         register_face(name)
   [5] Recognize face        recognize_face()

 [Person System - person_camera.py]
   [6] Search person         search_person(name)
   [7] Start tracking person start_person_tracking(name)
   [8] Stop person tracking  stop_person_tracking()

 [Gesture System - hand/script.py]
   [9] Recognize gesture     identify_gesture()

 [Exercise Counting - sitandup_camera.py]
   [a] Start counting        start_situp_counting()
   [b] Query progress        query_situp_progress()
   [c] Stop and summarize    stop_and_summarize()

   [e] Exit
========================================"""
    print(menu)

    while True:
        cmd = input("\n请输入指令 ").strip().lower()
        if cmd == 'e':
            print("退出系统")
            break
        elif cmd == '1':
            find_pet('cat')
        elif cmd == '2':
            start_pet_tracking('dog')
        elif cmd == '3':
            stop_pet_tracking()
        elif cmd == '4':
            name = input("请输入要注册的姓名 ").strip()
            if name:
                register_face(name)
            else:
                print("姓名不能为空，取消注册")
        elif cmd == '5':
            recognize_face()
        elif cmd == '6':
            name = input("请输入目标姓名 ").strip()
            if name:
                search_person(name)
            else:
                print("姓名不能为空")
        elif cmd == '7':
            name = input("请输入目标姓名 ").strip()
            if name:
                start_person_tracking(name)
            else:
                print("姓名不能为空")
        elif cmd == '8':
            stop_person_tracking()
        elif cmd == '9':
            identify_gesture()
        elif cmd == 'a':
            start_situp_counting()
        elif cmd == 'b':
            query_situp_progress()
        elif cmd == 'c':
            stop_and_summarize()
        else:
            print("无效指令，请重新输入")
