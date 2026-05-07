import os
import sys
import torch
import time
import multiprocessing
from speaker import speak

try:
    multiprocessing.set_start_method('fork', force=True)
except RuntimeError:
    pass

_ROOT = os.path.dirname(os.path.abspath(__file__))
_CODE_ROOT = os.path.dirname(_ROOT)
_FINAL_0418_ROOT = os.path.join(_CODE_ROOT, "final_0418")

def _p(*rel):
    return os.path.join(_ROOT, *rel)

if _FINAL_0418_ROOT not in sys.path:
    sys.path.insert(0, _FINAL_0418_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from function.pet import PetTrackingSystem
from face.face_camera import FaceRecognitionSystem
from tracking_person.person_camera_refine import PersonSearchTrackingSystem
from movement_count.sitandup_camera import SitupCountingSystem
from movement_count_2.pullup_camera import PullupCountingSystem
from movement_count_2.pushup_camera import PushupCountingSystem
from movement_count_2.squat_camera import SquatCountingSystem
from hand.script import GestureRecognitionSystem
from tracking_person.wheel_control import Board
# ! qr code navigation
from qr_nav.qr_camera import QRNavigationSystem
# ! aruco code navigation
from aruco_nav.aruco_back import ArucoNavigationSystem

CAMERA_ID = '/dev/video21'
_DETECTOR_PATH = _p('model', 'detect_v2.rknn')
_REID_PATH = _p('model', 'reid_mmse.rknn')
_FACE_MODEL = _p('model', '20180402-114759-vggface2.pt')
_HAND_MODEL = _p('model', 'hand_model_999.pth')
_SITUP_MODEL = _p('model', 'sitandup.rknn')
_FACE_DB = _p('faces.db')

pet_system = PetTrackingSystem(model_path=_DETECTOR_PATH)
face_system = FaceRecognitionSystem(db_path=_FACE_DB, model_path=_FACE_MODEL)
person_sys = PersonSearchTrackingSystem(_DETECTOR_PATH, _REID_PATH)
gesture_system = GestureRecognitionSystem(model_path=_HAND_MODEL)
sitandup_system = SitupCountingSystem(model_path=_SITUP_MODEL)
pullup_system = PullupCountingSystem()
pushup_system = PushupCountingSystem()
squat_system = SquatCountingSystem()
qr_nav_system = QRNavigationSystem()
aruco_nav_system = ArucoNavigationSystem()

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
    face_system.register_face(CAMERA_ID, name)

def recognize_face(): 
    face_system.recognize_face(CAMERA_ID)

def search_person(name: str): 
    person_sys.search_person(CAMERA_ID, name)

def start_person_tracking(name: str):
    person_sys.start_person_tracking(CAMERA_ID, name)

def stop_person_tracking():
    person_sys.stop_person_tracking()

def identify_gesture(): 
    gesture_system.identify_gesture(camera_id=CAMERA_ID)

def start_situp_counting(): 
    sitandup_system.start_counting(CAMERA_ID)

def query_situp_progress(): 
    sitandup_system.query_progress()

def stop_and_summarize(): 
    sitandup_system.stop_and_summarize()

# ! new movement 1 
def start_pullup_counting():
    pullup_system.start_counting(CAMERA_ID)

def query_pullup_progress():
    pullup_system.query_progress()

def stop_pullup_and_summarize():
    pullup_system.stop_and_summarize()

# ! new movement 2
def start_pushup_counting():
    pushup_system.start_counting(CAMERA_ID)

def query_pushup_progress():
    pushup_system.query_progress()

def stop_pushup_and_summarize():
    pushup_system.stop_and_summarize()

# ! new movement 3
def start_squat_counting():
    squat_system.start_counting(CAMERA_ID)

def query_squat_progress():
    squat_system.query_progress()

def stop_squat_and_summarize():
    squat_system.stop_and_summarize()

def start_qr_navigation(target: str):
    qr_nav_system.start_qr_navigation(CAMERA_ID, target)

def stop_qr_navigation():
    qr_nav_system.stop_qr_navigation()

# ! new function part
def start_aruco_navigation(target_id: int):
    aruco_nav_system.start_aruco_navigation(CAMERA_ID, target_id)
def stop_aruco_navigation():
    aruco_nav_system.stop_aruco_navigation()


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
            name = input("请输入要注册的姓名 ").strip()
            if name:
                register_face(name)
            else:
                print("姓名不能为空 取消注册")
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
        elif cmd == 's':
            name = input("请输入你想专属追踪的数据库姓名(确保已注册): ").strip()
            if name:
                person_sys.start_specific_person_tracking(CAMERA_ID, _FACE_DB, _FACE_MODEL, name)
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
        elif cmd == 'd':
            target = input("请输入要寻找的二维码内容(如: Dock_A): ").strip()
            if target:
                start_qr_navigation(target)
            else:
                print("内容不能为空")
        elif cmd == 'f':
            stop_qr_navigation()

        # ================= ArUco 调用分支 =================
        elif cmd == 'y':
            target = input("请输入要寻找的 ArUco 码 ID (如 0 或 1): ").strip()
            if target.isdigit():
                print(f"\n🚀 正在下发 3D 对齐回仓任务，目标 ID: {target}...")
                start_aruco_navigation(int(target))
            else:
                print("⚠️ 目标 ID 必须是数字！取消任务。")
        elif cmd == 'u':
            print("\n🛑 正在强制停止 ArUco 回仓任务...")
            stop_aruco_navigation()
        else:
            print("无效指令,请重新输入")
