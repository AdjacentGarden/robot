import os
import sys
import time
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pet.pet_camera                  import PetTrackingSystem
from face.face_camera                import FaceRecognitionSystem
from tracking_person.person_camera   import PersonSearchTrackingSystem
from movement_count.sitandup_camera  import SitupCountingSystem
from hand.script                     import GestureRecognitionSystem
from tracking_person.wheel_control import Board
CAMERA_ID = "/dev/video21"

def _p(*rel):
    return os.path.join(_ROOT, *rel)

_DETECTOR_PATH = _p("model", "detector.onnx")
_REID_PATH     = _p("model", "reid_model.onnx")
_FACE_MODEL    = _p("model", "20180402-114759-vggface2.pt")
_HAND_MODEL    = _p("model", "hand_model_999.pth")
_SITUP_MODEL   = _p("model", "sitandup_model.onnx")
_FACE_DB       = _p("test_faces.db")
def _check_models():
    required = {
        "检测器模型":     _DETECTOR_PATH,
        "ReID 模型":      _REID_PATH,
        "人脸识别模型":   _FACE_MODEL,
        "手势识别模型":   _HAND_MODEL,
        "仰卧起坐模型":   _SITUP_MODEL,
    }
    missing = [f"{name}: {path}" for name, path in required.items()
               if not os.path.exists(path)]
    if missing:
        print("⚠️  以下模型文件缺失，对应功能将无法使用：")
        for m in missing:
            print(f"   · {m}")

_check_models()

print("⏳ 正在初始化全局中控视觉模块...")

pet_system = PetTrackingSystem(model_path=_DETECTOR_PATH)

face_system = FaceRecognitionSystem(
    db_path    = _FACE_DB,
    model_path = _FACE_MODEL,
)

person_sys = PersonSearchTrackingSystem(
    detector_path = _DETECTOR_PATH,
    reid_path     = _REID_PATH,
)

gesture_system = GestureRecognitionSystem(model_path=_HAND_MODEL)

sitandup_system = SitupCountingSystem(model_path=_SITUP_MODEL)

print("✅ 所有视觉模块挂载完毕\n")


def find_pet(target: str):
    """宠物寻找（同步阻塞，旋转 6 秒，找到即停）"""
    print(f"[底层实现] 开始旋转寻找宠物: {target}")
    pet_system.find_pet(CAMERA_ID, target)


def start_pet_tracking(target: str):
    """启动宠物跟随后台进程（异步）"""
    print(f"[底层实现] 启动进程跟踪宠物: {target}")
    pet_system.start_pet_tracking(CAMERA_ID, target)


def stop_pet_tracking():
    """
    停止宠物跟随进程。
    内部：SIGTERM → join(3s) → SIGKILL → join(2s) 确认死亡
    摄像头由后台进程 finally 块释放，stop 返回后摄像头已可用。
    """
    print("[底层实现] 终止宠物跟踪进程")
    pet_system.stop_pet_tracking()


def register_face(name: str):
    """人脸录入（同步阻塞，最多 5 秒采集）"""
    print(f"[底层实现] 录入人脸: {name}")
    face_system.register_face(CAMERA_ID, name)


def recognize_face():
    """人脸识别（同步阻塞，最多 5 秒）"""
    print("[底层实现] 持续 5s 识别当前人脸")
    face_system.recognize_face(CAMERA_ID)


def search_person(name: str):
    """自主寻人（同步阻塞：2s 等待 + 6s 旋转搜索）"""
    print(f"[底层实现] 开始搜寻人物: {name}")
    person_sys.search_person(CAMERA_ID, name)


def start_person_tracking(name: str):
    """启动人物跟随后台进程（异步）"""
    print(f"[底层实现] 启动进程跟踪人物: {name}")
    person_sys.start_person_tracking(CAMERA_ID, name)


def stop_person_tracking():
    """
    停止人物跟随进程。
    内部：三段式终止，摄像头由后台进程 finally 块释放。
    """
    print("[底层实现] 终止人物跟踪进程")
    person_sys.stop_person_tracking()


def identify_gesture():
    """手势识别（同步阻塞，最多 5 秒）"""
    print("[底层实现] 开启手势识别窗口")
    gesture_system.identify_gesture(camera_id=CAMERA_ID)


def start_situp_counting():
    """启动仰卧起坐计数后台进程（异步）"""
    print("[底层实现] 启动仰卧起坐计数后台进程")
    sitandup_system.start_counting(CAMERA_ID)


def query_situp_progress():
    """查询当前计数（读取临时文件，非阻塞）"""
    print("[底层实现] 读取文件中的实时个数并播报")
    sitandup_system.query_progress()


def stop_and_summarize():
    """终止计数进程并播报最终结果"""
    print("[底层实现] 终止计数进程并汇总播报")
    sitandup_system.stop_and_summarize()

if __name__ == "__main__":
    board = Board()            
    board.enable_reception()  
    print("START...")
    task_type = None
    task_parameter = None
    # ! audio wakeup mechanism
    while True:
        try:
            wkup = board.get_wkup() # 获取语音唤醒状态
            if wkup is not None:
                print("唤醒状态:", wkup)
            time.sleep(0.01) # 短暂休眠，防止CPU占用过高
            # ! audio("我在")
            # ! audio_file = record_audio(5)
            # ! text_file = audio2text(audio)
            # ! task_type, task_parameter = llm_invoke(text_file)
            if task_type == 'find_pet':
                find_pet(task_parameter)
            elif task_type == 'start_pet_tracking':
                start_pet_tracking(task_parameter)
            elif task_type == 'stop_pet_tracking':
                stop_pet_tracking(task_parameter)
            elif task_type == 'register_face':
                register_face(task_parameter)
            elif task_type == 'recognize_face':
                recognize_face()
            elif task_type == 'search_person':
                search_person(task_parameter)
            elif task_type == 'start_person_tracking':
                start_person_tracking(task_parameter)
            elif task_type == 'stop_person_tracking':
                stop_person_tracking()
            elif task_type == 'identify_gesture':
                identify_gesture()
            elif task_type == 'start_situp_counting':
                start_situp_counting()
            elif task_type == 'query_situp_progress':
                query_situp_progress()
            elif task_type == 'stop_and_summarize':
                stop_and_summarize()
            else:
                pass
        except KeyboardInterrupt:
            break