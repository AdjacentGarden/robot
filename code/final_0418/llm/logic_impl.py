import os
import sys
import torch
import time
import multiprocessing
# from speaker import speak

try:
    multiprocessing.set_start_method('fork', force=True)
except RuntimeError:
    pass

_ROOT = os.path.dirname(os.path.abspath(__file__))
_FACE_DB = os.getenv("FACE_DB_PATH", os.path.join(_ROOT, "faces.db"))
def _p(*rel):
    return os.path.join(_ROOT, *rel)

if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pet_camera import PetTrackingSystem
from face_camera import FaceRecognitionSystem
from movement_count_2.pullup_camera_base_fps_1 import PullupCountingSystem
from movement_count_2.pushup_camera_base_fps_1 import PushupCountingSystem
from movement_count_2.squat_camera_base_fps_1 import SquatCountingSystem
from test3 import Board


CAMERA_ID = os.getenv("FACE_CAMERA_ID", "/dev/video40")
_FACE_MODEL = os.getenv("FACE_MODEL_PATH", _p("model", "20180402-114759-vggface2.pt"))
pet_system = PetTrackingSystem()
face_system = FaceRecognitionSystem(db_path=_FACE_DB, model_path=_FACE_MODEL)
pullup_system = PullupCountingSystem()
pushup_system = PushupCountingSystem()
squat_system = SquatCountingSystem()

def set_pet_motor_board(board):
    pet_system.set_motor_board(board)

def _read_and_delete(filepath):
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            os.remove(filepath)
            return content if content else None
        except Exception:
            return None
    return None

def find_pet(target: str): 
    pet_system.find_pet(CAMERA_ID, target)

def start_pet_tracking(target: str): 
    pet_system.start_pet_tracking(CAMERA_ID, target)

def stop_pet_tracking(): 
    pet_system.stop_pet_tracking()

def register_face(name: str): 
    return face_system.register_face(CAMERA_ID, name)

def start_face_enrollment():
    return face_system.start_face_enrollment(CAMERA_ID)

def confirm_face_name(name: str):
    return face_system.confirm_face_name(name)

def recognize_face(): 
    return face_system.recognize_face(CAMERA_ID)

def start_pullup_counting():
    pullup_system.start_counting(CAMERA_ID)

def query_pullup_progress():
    pullup_system.query_progress()

def stop_pullup_and_summarize():
    pullup_system.stop_and_summarize()

def start_pushup_counting():
    pushup_system.start_counting(CAMERA_ID)

def query_pushup_progress():
    pushup_system.query_progress()

def stop_pushup_and_summarize():
    pushup_system.stop_and_summarize()

def start_squat_counting():
    squat_system.start_counting(CAMERA_ID)

def query_squat_progress():
    squat_system.query_progress()

def stop_squat_and_summarize():
    squat_system.stop_and_summarize()

if __name__ == "__main__":
    board = Board()             
    board.enable_reception()   
    print("START...")
    menu = """
========================================
 Robot Central Control System - Integration Test Menu
========================================
 [Pet System - pet_camera.py]
   [1] Find cat        find_pet('cat')
   [2] Start tracking dog      start_pet_tracking('dog')
   [3] Stop pet tracking    stop_pet_tracking()

 [Face System - face_camera.py]
   [4] Register face        register_face(name)
   [5] Recognize face        recognize_face()

 [Person System - person_camera.py]
   [6] Search person        search_person(name)
   [7] Start tracking person    start_person_tracking(name)
   [s] Specific person tracking
   [8] Stop person tracking    stop_person_tracking()

 [Gesture System - hand/script.py]
   [9] Recognize gesture        identify_gesture()

  [Exercise Counting - sitandup_camera.py]
    [a] Start counting        start_situp_counting()
    [b] Query progress        query_situp_progress()
    [c] Stop and summarize      stop_and_summarize()

  [Exercise Counting - pullup_camera.py]
    [j] Start pull-up counting     start_pullup_counting()
    [k] Query pull-up progress     query_pullup_progress()
    [l] Stop pull-up and summarize stop_pullup_and_summarize()

  [Exercise Counting - pushup_camera.py]
    [m] Start push-up counting     start_pushup_counting()
    [n] Query push-up progress     query_pushup_progress()
    [o] Stop push-up and summarize stop_pushup_and_summarize()

  [Exercise Counting - squat_camera.py]
    [p] Start squat counting       start_squat_counting()
    [r] Query squat progress       query_squat_progress()
    [t] Stop squat and summarize   stop_squat_and_summarize()

  [QR Navigation - qr_camera.py]
    [d] Navigate to QR Code   start_qr_navigation(target)
    [f] Stop QR Navigation    stop_qr_navigation()

 [ArUco 3D Docking - aruco_camera.py]
   [y] Navigate to ArUco ID   start_aruco_navigation(id)
   [u] Stop ArUco Nav         stop_aruco_navigation()

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
            name = input("请输入要注册的姓名: ").strip()
            if name:
                ok = register_face(name)
                print("人脸录入成功" if ok else "人脸录入失败，请正视摄像头并重试")
            else:
                print("姓名不能为空")
        elif cmd == '5':    
            result = recognize_face()
            if isinstance(result, dict):
                status = result.get("status")
                if status == "matched":
                    print(f"识别成功：{result.get('name')}")
                elif status == "unknown":
                    print("识别完成：检测到人脸，但数据库中无匹配")
                elif status == "no_face":
                    print("识别失败：未检测到人脸")
                elif status == "empty_db":
                    print("识别失败：当前数据库无人脸，请先注册")
                elif status == "error":
                    print(f"识别异常：{result.get('error')}")
        elif cmd == 'j':
            start_pullup_counting()
        elif cmd == 'k':
            query_pullup_progress()
        elif cmd == 'l':
            stop_pullup_and_summarize()
        elif cmd == 'm':
            start_pushup_counting()
        elif cmd == 'n':    
            query_pushup_progress()
        elif cmd == 'o':
            stop_pushup_and_summarize()
        elif cmd == 'p':
            start_squat_counting()
        elif cmd == 'r':
            query_squat_progress()
        elif cmd == 't':
            stop_squat_and_summarize()
        else: 
            print("无效指令,请重新输入")
