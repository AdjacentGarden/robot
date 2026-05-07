import time
import threading
import subprocess
import socket
import os
import re
import importlib.util
import numpy as np
from flask import Flask, request, jsonify
# from flask_cors import CORS
from final_0418.llm.main_robot import RobotAssistant
from final_0418.llm import speaker
from final_0418.llm import local_tts
from final_0418.llm import toolmain as llm_toolmain
from final_0418.function.control import Board, set_household
import final_0418.function.ROS2control
from final_0418.function.ROS2control import ROS2NavigationController
import final_0418.function.videoUpLoad
from final_0418.function.videoUpLoad import upload_pet_tracking_video,upload_pet_video_to_app_server
import final_0418.llm.logic_impl
from final_0418.llm.logic_impl import start_pet_tracking

FACE_CAMERA_ID = os.getenv("FACE_CAMERA_ID", "/dev/video40")
FACE_MODEL_PATH = os.getenv("FACE_MODEL_PATH", "/home/test/code/final_0418/llm/model/20180402-114759-vggface2.pt")
FACE_DB_PATH = os.getenv("FACE_DB_PATH", "/home/test/code/final_0418/llm/faces.db")
os.environ["FACE_CAMERA_ID"] = FACE_CAMERA_ID
os.environ["FACE_MODEL_PATH"] = FACE_MODEL_PATH
os.environ["FACE_DB_PATH"] = FACE_DB_PATH
os.environ.setdefault("FACE_CAMERA_SHOW_WINDOW", "0")
os.environ.setdefault("FACE_CAMERA_USE_SUBPROCESS", "0")

EXTERNAL_IMU_HELPER_PATH = "/home/test/imu_test.py"
DEFAULT_HEAD_STATE = 1
FACE_HEAD_STATE = 2
FITNESS_LEVEL_HEAD_STATE = 1
FITNESS_UP_HEAD_STATE = 2
AMBIGUOUS_VOICE_REPLY = "我没有听清您是要下指令还是想聊天，请先唤醒我，再说清楚一点。"
current_head_state = 1

app = Flask(__name__)

app_has_command = False
app_device_type = ""
app_device_command = ""
app_point_name = ""
app_x = None
app_y = None
app_yaw = 0.0
app_request_payload = {}
app_command_lock = threading.RLock()

video_url = None  

# 搜索状态跟踪
search_status = "fail"  # 初始状态为fail，可能的值：success, fail, finding


def _is_mapping_stop_command(device: str, cmd: str) -> bool:
    return device == "mapping" and cmd == "stop"


def _is_mapping_running() -> bool:
    controller = globals().get("ros2_controller")
    if controller is None:
        return False
    checker = getattr(controller, "is_mapping_active", None)
    if callable(checker):
        return bool(checker(refresh=False))
    return bool(getattr(controller, "mapping_started", False))


def _current_execution_reason():
    manager = globals().get("task_manager")
    if manager is None:
        return None
    getter = getattr(manager, "current_execution_reason", None)
    if callable(getter):
        return getter()
    return manager.current_busy_reason()


def _get_app_command_block_reason(device: str, cmd: str):
    if _is_mapping_stop_command(device, cmd) and _is_mapping_running():
        return None
    return _current_execution_reason()


def _store_app_command(device: str, cmd: str, *, point_name="", x=None, y=None, yaw=0.0, payload=None):
    global app_has_command, app_device_type, app_device_command
    global app_point_name, app_x, app_y, app_yaw, app_request_payload
    with app_command_lock:
        if app_has_command:
            return False
        app_device_type = device
        app_device_command = cmd
        app_point_name = point_name
        app_x = x
        app_y = y
        app_yaw = yaw
        app_request_payload = dict(payload or {})
        app_has_command = True
        return True


def _has_pending_app_command() -> bool:
    with app_command_lock:
        return bool(app_has_command)


def _consume_pending_app_command():
    global app_has_command
    with app_command_lock:
        if not app_has_command:
            return None
        command = {
            "device_type": app_device_type,
            "command": app_device_command,
            "point_name": app_point_name,
            "x": app_x,
            "y": app_y,
            "yaw": app_yaw,
            "payload": dict(app_request_payload) if isinstance(app_request_payload, dict) else {},
        }
        app_has_command = False
        return command


FAN_PRIORITY = {"on": 3, "off": 3, "start": 2, "turn": 1, "high":1, "medium":1, "low":1}
fan_cached_cmd = None
fan_executed_priorities = set()

DEFAULT_MOVE_STEPS = 1
SEVERAL_MOVE_STEPS = 3
MAX_MOVE_STEPS = 6
MOVE_INTER_STEP_PAUSE_SEC = 0.12
VOICE_REARM_COOLDOWN_SEC = 0.8
MOVE_PROFILES = {
    "forward": {"speed_right": 0.16, "speed_left": 0.16, "duration": 0.42, "label": "向前"},
    "backward": {"speed_right": -0.16, "speed_left": -0.16, "duration": 0.42, "label": "向后"},
    "left": {"speed_right": -0.16, "speed_left": 0.16, "duration": 0.28, "label": "向左"},
    "right": {"speed_right": 0.16, "speed_left": -0.16, "duration": 0.28, "label": "向右"},
}
CHINESE_STEP_NUMBERS = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
}

DEVICE_COMMAND_MAP = {
    "feeder": ["start"],
    "light": ["on", "off"],
    "fan": ["on", "off", "start", "turn", "high", "medium", "low"],  # 新增测试用风速指令
    "search": ["start", "pause", "continue", "stop"],  # 搜索功能指令
    "mapping": ["start", "stop", "status"],  # 建图功能
    "navigation": ["fan", "feeder", "light", "stop", "status"],  # 导航功能（导航到风扇、投食机、灯的位置）
    "goal": ["set"],  # 设置目标点
    "points": ["list", "delete"],  # 命名点管理
    "system": ["status", "stop"],  # 系统状态和控制
    "move": ["forward", "backward", "left", "right", "up", "down"],  # 新增移动控制指令
    "video": ["upload"]  # 新增视频上传指令
}

DEVICE_STATE_MAP = {
    "feeder": {"start": 0},
    "light": {
        "on": 1, 
        "off": 2
    },
    "fan": {
        "on": 3, "off": 3,
        "start": 4,
        "turn": 5,
        "high": 5, "medium": 5, "low": 5 
    },
    "search": {
        "start": None, 
        "pause": None, 
        "continue": None,
        "stop": None
    },
    "mapping": {
        "start": None, 
        "stop": None, 
        "status": None
    },  # 建图功能
    "navigation": {
        "fan": None, 
        "feeder": None, 
        "light": None,
        "stop": None,
        "status": None
    },  # 导航功能（导航到风扇、投食机、灯的位置）
    "goal": {"set": None},  # 设置目标点
    "points": {
        "list": None, 
        "delete": None
        },  # 命名点管理
    "system": {
        "status": None, 
        "stop": None
    },  # 系统状态和控制
    "move": {
        "forward": None, 
        "backward": None, 
        "left": None, 
        "right": None,
        "up": None,
        "down": None
    },  # 移动控制功能
    "video": {
        "upload": None
    }  # 视频上传功能
}

@app.route('/api/device/command', methods=['POST'])
def send_device_command():
    global fan_cached_cmd
    req_data = request.get_json()

    # 参数校验
    if not req_data or not all(k in req_data for k in ['device_type', 'code', 'command']):
        return jsonify({"code":400,"message":"参数缺失"}), 400

    device = req_data["device_type"]
    cmd = req_data["command"]

    # 设备/命令校验 - 特殊处理搜索设备的check命令
    if device != "search" or cmd != "check":
        if device not in DEVICE_COMMAND_MAP or cmd not in DEVICE_COMMAND_MAP[device]:
            return jsonify({"code":400,"message":"不支持的设备或命令"}), 400

    
    # 特殊参数处理
    point_name = req_data.get("point_name", "")
    x = req_data.get("x")
    y = req_data.get("y")
    yaw = req_data.get("yaw", 0.0)
    
    # 风扇优先级处理（高优先级总是可以执行，低优先级需要高优先级已执行）
    if device == "fan":
        current_priority = FAN_PRIORITY[cmd]
        
        # 高优先级命令总是可以执行
        if current_priority == 3:  # 最高优先级
            final_cmd = cmd
            print(f"✅ 风扇命令 '{cmd}' (优先级{current_priority}) 允许执行，高优先级总是可执行")
        else:
            # 检查是否所有更高优先级都已经执行过
            higher_priorities_executed = all(priority in fan_executed_priorities 
                                           for priority in range(current_priority + 1, 4))
            
            if higher_priorities_executed:
                # 所有更高优先级已执行，允许执行当前命令
                final_cmd = cmd
                print(f"✅ 风扇命令 '{cmd}' (优先级{current_priority}) 允许执行，更高优先级已执行")
            else:
                # 有更高优先级未执行，拒绝当前命令
                final_cmd = None
                print(f"❌ 风扇命令 '{cmd}' (优先级{current_priority}) 被拒绝，有更高优先级未执行")
    else:
        final_cmd = cmd
        fan_cached_cmd = None

    # ✅ 修改3：强化打印，清晰展示【接口收到的完整指令】
    print(f"\n=====================================")
    print(f"📥 接口已接收指令：设备={device}，命令={final_cmd}")
    print(f"=====================================\n")

    if final_cmd is None:
        return jsonify({
            "code": 409,
            "message": "当前命令未满足执行条件，已拒绝",
            "data": {
                "device_type": device,
                "code": req_data.get("code", ""),
                "command": cmd
            }
        }), 409
    
    # 特殊处理搜索设备的check命令
    if device == "search" and cmd == "check":
        global search_status
        print(f"🔍 搜索状态查询：设备={device}，命令={cmd}，状态={search_status}")
        return jsonify({
            "code": 200,
            "message": "获取搜索状态成功",
            "data": {
                "code": req_data.get("code", ""),
                "status": search_status
            }
        }), 200

    block_reason = _get_app_command_block_reason(device, final_cmd)
    if block_reason is not None:
        return jsonify({
            "code": 409,
            "message": f"机器人当前{block_reason}，不能接收新的APP执行指令",
            "data": {
                "device_type": device,
                "code": req_data.get("code", ""),
                "command": final_cmd,
                "busy_reason": block_reason
            }
        }), 409

    if _is_mapping_stop_command(device, final_cmd) and _is_mapping_running():
        manager = globals().get("task_manager")
        if manager is not None:
            manager.begin_local_task("APP停止建图中")
        try:
            speaker.speak("建图完成了哦，我要去玩了")
            print("🛑 APP立即停止建图功能")
            success = ros2_controller.stop_all()
        finally:
            if manager is not None:
                manager.end_local_task()
        status_code = 200 if success else 500
        return jsonify({
            "code": status_code,
            "message": "停止建图指令执行成功" if success else "停止建图指令执行失败",
            "data": {
                "device_type": device,
                "code": req_data.get("code", ""),
                "command": final_cmd
            }
        }), status_code

    # 根据设备类型返回不同的成功消息
    if device == "video" and cmd == "upload":
        # 处理视频上传命令
        print("📤 开始处理视频上传命令...")
        # 检查全局变量video_url是否已经存在
        global video_url
        manager = globals().get("task_manager")
        if manager is not None:
            manager.begin_local_task("视频上传中")
        if video_url:
            print("✅ 视频URL已存在，直接返回")
            speaker.speak("上传成功")
            if manager is not None:
                manager.end_local_task()
            return jsonify({
                "code": 200,
                "message": "视频上传指令执行成功",
                "data": {
                    "device_type": device,
                    "code": req_data.get("code", ""),
                    "command": final_cmd,
                    "video_url": video_url
                }
            }), 200
        else:
            # 上传视频到指定服务器
            try:
                video_url = upload_pet_video_to_app_server()
                if video_url:
                    print("✅ 视频上传成功，返回URL")
                    speaker.speak("上传成功")
                    return jsonify({
                        "code": 200,
                        "message": "视频上传指令执行成功",
                        "data": {
                            "device_type": device,
                            "code": req_data.get("code", ""),
                            "command": final_cmd,
                            "video_url": video_url
                        }
                    }), 200
                else:
                    print("❌ 视频上传失败")
                    return jsonify({
                        "code": 500,
                        "message": "视频上传失败",
                        "data": {
                            "device_type": device,
                            "code": req_data.get("code", ""),
                            "command": final_cmd
                        }
                    }), 500
            finally:
                if manager is not None:
                    manager.end_local_task()

    if not _store_app_command(device, final_cmd, point_name=point_name, x=x, y=y, yaw=yaw, payload=req_data):
        return jsonify({
            "code": 409,
            "message": "已有APP指令等待执行，当前指令已拒绝",
            "data": {
                "device_type": device,
                "code": req_data.get("code", ""),
                "command": final_cmd
            }
        }), 409

    # 根据设备类型返回不同的成功消息
    if device == "search":
        command_messages = {
            "start": "搜索宠物指令执行成功",
            "pause": "暂停搜索指令执行成功", 
            "continue": "继续搜索指令执行成功",
            "stop": "停止搜索指令执行成功"
        }
        message = command_messages.get(final_cmd, "指令执行成功")
        return jsonify({
            "code": 200,
            "message": message,
            "data": {
                "device_type": device,
                "code": req_data.get("code", ""),
                "command": final_cmd
            }
        }), 200
    elif device == "feeder":
        # 投食机专用返回格式
        command_messages = {
            "start": "投食指令发送成功"
        }
        message = command_messages.get(final_cmd, "指令执行成功")
        return jsonify({
            "code": 200,
            "message": message,
            "data": {
                "device_type": device,
                "code": req_data.get("code", ""),
                "command": final_cmd
            }
        }), 200
    elif device == "light":
        # 灯具专用返回格式
        command_messages = {
            "on": "开灯指令发送成功",
            "off": "关灯指令发送成功"
        }
        message = command_messages.get(final_cmd, "指令执行成功")
        return jsonify({
            "code": 200,
            "message": message,
            "data": {
                "device_type": device,
                "code": req_data.get("code", ""),
                "command": final_cmd
            }
        }), 200
    else:
        return jsonify({"code":200,"message":"指令已接收"}), 200

# 搜索宠物状态接口
@app.route('/api/device/status', methods=['POST'])
def get_device_status():
    global search_status
    req_data = request.get_json()

    # 参数校验
    if not req_data or not all(k in req_data for k in ['device_type', 'code', 'command']):
        return jsonify({"code":400,"message":"参数缺失"}), 400
    
    device = req_data["device_type"]
    cmd = req_data["command"]
    device_code = req_data["code"]
    
    # 只支持搜索设备的状态查询
    if device != "search" or cmd != "check":
        return jsonify({"code":400,"message":"不支持的设备或命令"}), 400
    
    print(f"🔍 状态查询接口：设备={device}，命令={cmd}，状态={search_status}")
    
    # 返回搜索状态
    return jsonify({
        "code": 200,
        "message": "获取搜索状态成功",
        "data": {
            "code": device_code,
            "status": search_status
        }
    }), 200

# 后台运行Flask服务
# APP_PORT_CANDIDATES = [8080, 8081, 8082]
APP_PORT_CANDIDATES = [8082]

def _is_port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.connect_ex(("127.0.0.1", port)) != 0


def _pick_app_port() -> int:
    for p in APP_PORT_CANDIDATES:
        if _is_port_free(p):
            return p
    return APP_PORT_CANDIDATES[0]


def run_flask_server(port: int):
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

MIC_DEVICE = "hw:rockchipi2sdmic"
MIC_RAW_SAMPLE_RATE = 48000
MIC_CHANNELS = 2
MIC_SELECT_CHANNEL = 0
MIC_SOFT_GAIN = 2.0
SPEAKER_DEVICE = "hw:rockchiptas6424"


def _play_wav_with_hardware_test_device(path: str, volume: int = 140):
    cmd = [
        "aplay",
        "-D", SPEAKER_DEVICE,
        path,
        "-c", "2",
        "-r", "48000",
        "-q",
    ]
    subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


local_tts._play_wav = _play_wav_with_hardware_test_device


def get_mic_stream(chunk_size=512, sample_rate=16000):
    target_duration = chunk_size / sample_rate
    raw_frames_per_chunk = int(MIC_RAW_SAMPLE_RATE * target_duration)
    bytes_per_chunk = raw_frames_per_chunk * MIC_CHANNELS * 2

    cmd = [
        "arecord",
        "-D", MIC_DEVICE,
        "-r", str(MIC_RAW_SAMPLE_RATE),
        "-c", str(MIC_CHANNELS),
        "-f", "S16_LE",
        "-t", "raw",
        "-q",
        "-"
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )

    print(f"麦克风已打开: {MIC_DEVICE}")
    print('start recording...')

    try:
        chunk_counter = 0
        while True:
            buffer = bytearray()
            while len(buffer) < bytes_per_chunk:
                piece = proc.stdout.read(bytes_per_chunk - len(buffer))
                if not piece:
                    err = proc.stderr.read().decode(errors="ignore")
                    print("[arecord 退出]")
                    print("returncode =", proc.poll())
                    print("stderr =", err)
                    return
                buffer.extend(piece)

            audio = np.frombuffer(buffer, dtype=np.int16).reshape(-1, MIC_CHANNELS)
            mono = audio[:, MIC_SELECT_CHANNEL].astype(np.float32)

            if MIC_SOFT_GAIN != 1.0:
                mono *= MIC_SOFT_GAIN
                np.clip(mono, -32768, 32767, out=mono)

            mono = mono.astype(np.int16)
            downsampled = mono[:: MIC_RAW_SAMPLE_RATE // sample_rate]

            if downsampled.size != chunk_size:
                downsampled = downsampled[:chunk_size]
                if downsampled.size < chunk_size:
                    downsampled = np.pad(downsampled, (0, chunk_size - downsampled.size), mode="constant")

            chunk_counter += 1
            if chunk_counter % 40 == 0:
                level = float(np.abs(downsampled).mean())
                print(f"[MIC] level={level:.1f}")

            yield downsampled.tobytes()

    finally:
        print('exit recording')
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=1.0)
        except Exception as e:
            print(f"[MIC] 关闭录音进程失败: {e}")


def _set_chassis_motion(speed_right: float, speed_left: float, max_speed: int = 100):
    board.set_motor_speed([
        [1, float(max_speed * speed_right)],
        [2, float(max_speed * speed_left * -1)],
    ])


def _stop_chassis_motion():
    try:
        _set_chassis_motion(0.0, 0.0)
    except Exception as e:
        print(f"[Chassis] 停车失败: {e}")


def _extract_step_count(text: str) -> int:
    arabic = re.search(r"([1-9])\s*步", text)
    if arabic:
        return min(int(arabic.group(1)), MAX_MOVE_STEPS)

    chinese = re.search(r"([一二两三四五六])\s*步", text)
    if chinese:
        return min(CHINESE_STEP_NUMBERS[chinese.group(1)], MAX_MOVE_STEPS)

    if "几步" in text:
        return SEVERAL_MOVE_STEPS
    if any(token in text for token in ("走一走", "走走", "挪一挪", "动一动", "移动一下", "走一下")):
        return DEFAULT_MOVE_STEPS
    return DEFAULT_MOVE_STEPS


def _parse_chassis_move_command(text: str):
    normalized = re.sub(r"\s+", "", str(text or ""))
    if not normalized:
        return None

    direction_aliases = {
        "forward": ("向前", "往前", "朝前", "前面", "前"),
        "backward": ("向后", "往后", "朝后", "后面", "后"),
        "left": ("向左", "往左", "朝左", "左边", "左"),
        "right": ("向右", "往右", "朝右", "右边", "右"),
    }
    direction = None
    for key, aliases in direction_aliases.items():
        if any(alias in normalized for alias in aliases):
            direction = key
            break

    if direction is None:
        return None
    if not any(token in normalized for token in ("走", "挪", "动", "移动")):
        return None

    return {"direction": direction, "steps": _extract_step_count(normalized)}


def _parse_mapping_voice_command(text: str):
    normalized = re.sub(r"\s+", "", str(text or ""))
    if not normalized:
        return None

    controller = globals().get("ros2_controller")
    mapping_running = bool(controller is not None and getattr(controller, "mapping_started", False))
    mapping_keywords = ("建图", "地图", "见图", "简图")
    has_mapping_keyword = any(token in normalized for token in mapping_keywords)
    stop_keywords = ("结束", "停止", "关闭", "退出")
    status_keywords = ("状态", "情况", "进度", "怎么样")
    start_keywords = ("开始", "启动", "打开", "进入")

    # 建图进行中时，允许直接说“结束”“停止”“退出”，也兼容常见 ASR 把“建图”识别成“地图/见图”。
    if any(token in normalized for token in stop_keywords):
        if has_mapping_keyword or mapping_running:
            return "stop"

    if any(token in normalized for token in status_keywords):
        if has_mapping_keyword or mapping_running:
            return "status"

    if not has_mapping_keyword:
        return None

    if any(token in normalized for token in start_keywords):
        return "start"
    if any(token in normalized for token in stop_keywords):
        return "stop"
    if any(token in normalized for token in status_keywords):
        return "status"
    return None


def _wait_for_tts_queue():
    tts_queue = getattr(globals().get("robot"), "tts_queue", None)
    if tts_queue is not None:
        tts_queue.wait_until_done()


def _load_external_imu_module():
    spec = importlib.util.spec_from_file_location("external_pet_head_imu", EXTERNAL_IMU_HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载步进电机控制文件: {EXTERNAL_IMU_HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HeadPoseManager:
    def __init__(self):
        self._imu_module = None
        self._imu_thread = None
        self._imu_stop_event = None
        self._imu_ready = False
        self._imu_ready_lock = threading.Lock()
        self._pose_lock = threading.RLock()
        self._current_state = DEFAULT_HEAD_STATE
        self._fitness_session_token = 0

    def ensure_imu_ready(self) -> bool:
        if self._imu_ready:
            return True

        with self._imu_ready_lock:
            if self._imu_ready:
                return True

            try:
                self._imu_module = _load_external_imu_module()
                self._imu_stop_event = threading.Event()
                self._imu_thread = threading.Thread(
                    target=self._imu_module.imu_thread_func,
                    args=(self._imu_stop_event,),
                    daemon=True,
                    name="external-head-imu",
                )
                self._imu_thread.start()
                time.sleep(3.0)
                self._imu_ready = True
                print("[HeadPose] 外部 IMU 步进电机控制已接入")
            except Exception as e:
                self._imu_ready = False
                print(f"[HeadPose] 步进电机控制初始化失败: {e}")

        return self._imu_ready

    def set_head_state(self, state: int, reason: str = "") -> bool:
        if not self.ensure_imu_ready():
            return False

        with self._pose_lock:
            if self._current_state == state:
                return True
            try:
                self._imu_module.set_head(int(state))
                self._current_state = int(state)
                if reason:
                    print(f"[HeadPose] 头部状态 -> {state} ({reason})")
                else:
                    print(f"[HeadPose] 头部状态 -> {state}")
                return True
            except Exception as e:
                print(f"[HeadPose] 设置头部状态失败: state={state}, error={e}")
                return False

    def ensure_default_pose(self, reason: str = "") -> bool:
        return self.set_head_state(DEFAULT_HEAD_STATE, reason=reason or "恢复平视")

    def move_for_face_task(self, reason: str = "") -> bool:
        return self.set_head_state(FACE_HEAD_STATE, reason=reason or "人脸任务抬头")

    def move_for_fitness_task(self, exercise: str, reason: str = "") -> bool:
        exercise = str(exercise or "").strip().lower()
        target_state = FITNESS_UP_HEAD_STATE if exercise in {"pullup", "squat"} else FITNESS_LEVEL_HEAD_STATE
        return self.set_head_state(target_state, reason=reason or f"{exercise}任务准备")

    def run_blocking_task_with_pose(self, enter_state: int, fn, *args, task_name: str = "", **kwargs):
        self.set_head_state(enter_state, reason=f"{task_name}开始前调整姿态" if task_name else "阻塞任务开始")
        try:
            return fn(*args, **kwargs)
        finally:
            self.ensure_default_pose(reason=f"{task_name}结束后恢复平视" if task_name else "阻塞任务结束")

    def begin_fitness_session(self, exercise: str, system_getter):
        exercise = str(exercise or "").strip().lower()
        with self._pose_lock:
            self._fitness_session_token += 1
        self.move_for_fitness_task(exercise, reason=f"{exercise}计数启动前调整姿态")

    def end_fitness_session(self, exercise: str = ""):
        with self._pose_lock:
            self._fitness_session_token += 1
        self.ensure_default_pose(reason=f"{exercise or '运动'}任务结束后恢复平视")

    def _monitor_fitness_session(self, token: int, exercise: str, system_getter):
        wait_deadline = time.time() + 5.0
        process_seen = False

        while True:
            with self._pose_lock:
                if token != self._fitness_session_token:
                    return

            system = system_getter()
            process = getattr(system, "_process", None) if system is not None else None
            if process is not None:
                process_seen = True
                if not process.is_alive():
                    break
            elif process_seen or time.time() >= wait_deadline:
                break

            time.sleep(0.5)

        with self._pose_lock:
            if token != self._fitness_session_token:
                return
            self._fitness_session_token += 1

        self.ensure_default_pose(reason=f"{exercise}计数自动结束后恢复平视")


head_pose_manager = HeadPoseManager()


def _get_situp_system():
    return getattr(final_0418.llm.logic_impl, "situp_system", None)


def _ensure_situp_support():
    logic_module = final_0418.llm.logic_impl
    if getattr(logic_module, "situp_system", None) is None:
        try:
            module = __import__("final_0418.llm.movement_count_2.sitandup_camera", fromlist=["SitupCountingSystem"])
            situp_cls = getattr(module, "SitupCountingSystem")
            logic_module.situp_system = situp_cls()
            print("[Bootstrap] 仰卧起坐计数系统已动态接入")
        except Exception as e:
            print(f"[Bootstrap] 仰卧起坐计数系统接入失败: {e}")
            return

    situp_system = getattr(logic_module, "situp_system", None)
    if situp_system is None:
        return

    if not callable(getattr(logic_module, "start_situp_counting", None)):
        def start_situp_counting():
            situp_system.start_counting(FACE_CAMERA_ID)

        logic_module.start_situp_counting = start_situp_counting

    if not callable(getattr(logic_module, "query_situp_progress", None)):
        def query_situp_progress():
            situp_system.query_progress()

        logic_module.query_situp_progress = query_situp_progress

    if not callable(getattr(logic_module, "stop_and_summarize", None)):
        def stop_and_summarize():
            situp_system.stop_and_summarize()

        logic_module.stop_and_summarize = stop_and_summarize


def _install_head_pose_hooks():
    logic_module = final_0418.llm.logic_impl
    _ensure_situp_support()

    if getattr(logic_module, "_head_pose_hooks_installed", False):
        return

    original_register_face = getattr(logic_module, "register_face", None)
    if callable(original_register_face):
        def wrapped_register_face(name: str):
            return head_pose_manager.run_blocking_task_with_pose(
                FACE_HEAD_STATE,
                original_register_face,
                name,
                task_name="人脸注册",
            )

        logic_module.register_face = wrapped_register_face

    original_start_face_enrollment = getattr(logic_module, "start_face_enrollment", None)
    if callable(original_start_face_enrollment):
        def wrapped_start_face_enrollment():
            return head_pose_manager.run_blocking_task_with_pose(
                FACE_HEAD_STATE,
                original_start_face_enrollment,
                task_name="人脸录入",
            )

        logic_module.start_face_enrollment = wrapped_start_face_enrollment

    original_recognize_face = getattr(logic_module, "recognize_face", None)
    if callable(original_recognize_face):
        def wrapped_recognize_face():
            return head_pose_manager.run_blocking_task_with_pose(
                FACE_HEAD_STATE,
                original_recognize_face,
                task_name="人脸识别",
            )

        logic_module.recognize_face = wrapped_recognize_face
        # 建图停止和完全停止/
    fitness_hook_specs = [
        ("situp", "start_situp_counting", "query_situp_progress", "stop_and_summarize", _get_situp_system),
        ("pushup", "start_pushup_counting", "query_pushup_progress", "stop_pushup_and_summarize", lambda: getattr(logic_module, "pushup_system", None)),
        ("pullup", "start_pullup_counting", "query_pullup_progress", "stop_pullup_and_summarize", lambda: getattr(logic_module, "pullup_system", None)),
        ("squat", "start_squat_counting", "query_squat_progress", "stop_squat_and_summarize", lambda: getattr(logic_module, "squat_system", None)),
    ]

    for exercise, start_name, query_name, stop_name, system_getter in fitness_hook_specs:
        original_start = getattr(logic_module, start_name, None)
        if callable(original_start):
            def make_start_wrapper(fn, exercise_name, getter):
                def wrapped_start():
                    head_pose_manager.begin_fitness_session(exercise_name, getter)
                    return fn()

                return wrapped_start

            setattr(logic_module, start_name, make_start_wrapper(original_start, exercise, system_getter))

        original_query = getattr(logic_module, query_name, None)
        if callable(original_query):
            def make_query_wrapper(fn, exercise_name, getter):
                def wrapped_query():
                    head_pose_manager.move_for_fitness_task(exercise_name, reason=f"{exercise_name}计数查询前调整姿态")
                    try:
                        return fn()
                    finally:
                        system = getter()
                        process = getattr(system, "_process", None) if system is not None else None
                        if process is None or not process.is_alive():
                            head_pose_manager.ensure_default_pose(reason=f"{exercise_name}查询结束后恢复平视")

                return wrapped_query

            setattr(logic_module, query_name, make_query_wrapper(original_query, exercise, system_getter))

        original_stop = getattr(logic_module, stop_name, None)
        if callable(original_stop):
            def make_stop_wrapper(fn, exercise_name):
                def wrapped_stop():
                    try:
                        return fn()
                    finally:
                        head_pose_manager.end_fitness_session(exercise_name)

                return wrapped_stop

            setattr(logic_module, stop_name, make_stop_wrapper(original_stop, exercise))

    logic_module._head_pose_hooks_installed = True
    print("[Bootstrap] 已安装头部姿态控制钩子")


COMMAND_KEYWORDS = (
    "开灯", "关灯", "投食", "喂食", "风扇", "导航", "建图", "搜索", "找", "跟踪",
    "识别", "录入", "记住", "数", "计数", "仰卧起坐", "俯卧撑", "引体向上", "下蹲", "深蹲",
    "向前", "向后", "向左", "向右", "往前", "往后", "往左", "往右", "移动", "走", "停下",
)
CHAT_KEYWORDS = (
    "你好", "你是谁", "你叫什么", "在吗", "早上好", "晚上好", "谢谢", "讲个笑话",
    "你多大", "你会什么", "今天天气", "聊天", "聊聊", "介绍一下你自己",
)
AMBIGUOUS_KEYWORDS = (
    "那个", "这个", "一下", "来一下", "帮我一下", "开始吧", "停一下", "继续吧", "弄一下",
)


def _classify_voice_intent(agent, query: str, image_url=None, force_vl=False) -> str:
    query = str(query or "").strip()
    if not query:
        return "unclear"
    if image_url or force_vl:
        return "command"
    if getattr(agent, "awaiting_face_full_name", False):
        return "command"

    normalized = re.sub(r"\s+", "", query.lower())
    if normalized in {"理想同学", "在吗", "你好", "你好啊", "嗨", "hi", "hello"}:
        return "chat"

    has_command_kw = any(token in query for token in COMMAND_KEYWORDS)
    has_chat_kw = any(token in query for token in CHAT_KEYWORDS)
    has_ambiguous_kw = any(token in query for token in AMBIGUOUS_KEYWORDS)

    if has_command_kw and not has_chat_kw and not has_ambiguous_kw:
        return "command"
    if has_chat_kw and not has_command_kw:
        return "chat"
    if has_ambiguous_kw and not has_command_kw:
        return "unclear"

    try:
        msg_obj = agent._call_llm(
            [{
                "role": "user",
                "content": (
                    "请判断下面这句话属于哪一类，只能输出 command、chat、unclear 三选一。\n"
                    "command: 明确要机器人执行动作、控制设备、启动识别/导航/计数等指令。\n"
                    "chat: 日常闲聊、问答、寒暄。\n"
                    "unclear: 无法可靠区分，或者信息不足。\n"
                    f"用户句子：{query}"
                ),
            }],
            tools=None,
            use_vl=False,
        )
        decision = str(getattr(msg_obj, "content", "") or "").strip().lower()
    except Exception as e:
        print(f"[IntentGate] 意图分类失败，按 unclear 处理: {e}")
        return "unclear"

    if "command" in decision:
        return "command"
    if "chat" in decision:
        return "chat"
    if "unclear" in decision:
        return "unclear"
    return "unclear"


class VoiceTaskManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._local_task_stack = []
        self._voice_cooldown_until = 0.0

    def begin_local_task(self, name: str):
        with self._lock:
            self._local_task_stack.append(name)

    def end_local_task(self):
        with self._lock:
            if self._local_task_stack:
                self._local_task_stack.pop()

    def mark_voice_cooldown(self, seconds: float = VOICE_REARM_COOLDOWN_SEC):
        with self._lock:
            self._voice_cooldown_until = max(self._voice_cooldown_until, time.time() + seconds)

    def _current_local_task(self):
        with self._lock:
            return self._local_task_stack[-1] if self._local_task_stack else None

    def _in_voice_cooldown(self):
        with self._lock:
            return time.time() < self._voice_cooldown_until

    def _is_pet_tracking_running(self) -> bool:
        pet_system = getattr(final_0418.llm.logic_impl, "pet_system", None)
        process = getattr(pet_system, "_process", None)
        return bool(process is not None and process.is_alive())

    def _is_fitness_task_running(self) -> bool:
        logic_module = final_0418.llm.logic_impl
        systems = [
            getattr(logic_module, "situp_system", None),
            getattr(logic_module, "pullup_system", None),
            getattr(logic_module, "pushup_system", None),
            getattr(logic_module, "squat_system", None),
        ]
        for system in systems:
            process = getattr(system, "_process", None)
            if process is not None and process.is_alive():
                return True
        return False

    def _is_navigation_or_mapping_running(self) -> bool:
        controller = globals().get("ros2_controller")
        if controller is None:
            return False
        return bool(getattr(controller, "navigation_started", False) or getattr(controller, "mapping_started", False))

    def current_busy_reason(self):
        execution_reason = self.current_execution_reason()
        if execution_reason:
            return execution_reason
        if self._in_voice_cooldown():
            return "语音冷却中"
        return None

    def current_execution_reason(self):
        local_task = self._current_local_task()
        if local_task:
            return local_task
        if self._is_pet_tracking_running():
            return "宠物跟踪中"
        if self._is_fitness_task_running():
            return "运动计数中"
        if self._is_navigation_or_mapping_running():
            return "导航或建图中"
        return None

    def is_voice_locked(self) -> bool:
        return self.current_busy_reason() is not None


task_manager = VoiceTaskManager()


def _can_open_mapping_voice_session() -> bool:
    controller = globals().get("ros2_controller")
    if controller is None:
        return False
    checker = getattr(controller, "is_mapping_active", None)
    if callable(checker):
        return bool(checker(refresh=False))
    return bool(getattr(controller, "mapping_started", False))


def _get_mapping_start_block_reason():
    local_task = task_manager._current_local_task()
    if local_task:
        return local_task
    if task_manager._is_pet_tracking_running():
        return "宠物跟踪中"
    if task_manager._is_fitness_task_running():
        return "运动计数中"

    controller = globals().get("ros2_controller")
    if controller is None:
        return "导航控制器未初始化"
    sync_status = getattr(controller, "sync_status", None)
    if callable(sync_status):
        sync_status(force=True)
    if getattr(controller, "mapping_started", False):
        return "建图中"
    if getattr(controller, "navigation_started", False):
        return "导航中"
    return None


def execute_mapping_voice_command(action: str):
    ensure_runtime_initialized()
    head_pose_manager.ensure_default_pose(reason="建图语音控制保持平视")

    if action == "status":
        if ros2_controller.is_mapping_active(refresh=True):
            speaker.speak("我正在建图。")
        else:
            speaker.speak("我现在没有在建图。")
        _wait_for_tts_queue()
        task_manager.mark_voice_cooldown()
        return

    if action == "start":
        block_reason = _get_mapping_start_block_reason()
        if block_reason == "建图中":
            speaker.speak("我已经在建图了。")
            _wait_for_tts_queue()
            task_manager.mark_voice_cooldown()
            return
        if block_reason is not None:
            speaker.speak(f"现在还不能开始建图，我当前{block_reason}。")
            _wait_for_tts_queue()
            task_manager.mark_voice_cooldown()
            return

        speaker.speak("好的，我开始建图。")
        _wait_for_tts_queue()
        task_manager.begin_local_task("建图启动中")
        try:
            success = ros2_controller.start_mapping()
        finally:
            task_manager.end_local_task()
            task_manager.mark_voice_cooldown()

        if success:
            speaker.speak("建图已经开始了。")
        else:
            speaker.speak("建图启动失败了。")
        _wait_for_tts_queue()
        return

    if action == "stop":
        if not ros2_controller.is_mapping_active(refresh=True):
            if ros2_controller.is_navigation_active(refresh=False):
                speaker.speak("我现在没有在建图，当前是在导航。")
            else:
                speaker.speak("我现在没有在建图。")
            _wait_for_tts_queue()
            task_manager.mark_voice_cooldown()
            return

        speaker.speak("好的，我来结束建图。")
        _wait_for_tts_queue()
        task_manager.begin_local_task("建图停止中")
        try:
            success = ros2_controller.stop_all()
        finally:
            task_manager.end_local_task()
            task_manager.mark_voice_cooldown()

        if success:
            speaker.speak("建图已经结束了。")
        else:
            speaker.speak("结束建图失败了。")
        _wait_for_tts_queue()


def execute_chassis_move(direction: str, steps: int):
    ensure_runtime_initialized()
    head_pose_manager.ensure_default_pose(reason="底盘移动保持平视")
    profile = MOVE_PROFILES[direction]
    steps = max(1, min(int(steps), MAX_MOVE_STEPS))
    task_manager.begin_local_task(f"底盘移动: {profile['label']}{steps}步")
    try:
        for _ in range(steps):
            _set_chassis_motion(profile["speed_right"], profile["speed_left"])
            time.sleep(profile["duration"])
            _stop_chassis_motion()
            time.sleep(MOVE_INTER_STEP_PAUSE_SEC)
    finally:
        _stop_chassis_motion()
        task_manager.end_local_task()
        task_manager.mark_voice_cooldown()


def install_agent_fast_path(agent):
    original_run_workflow = agent.run_workflow

    def wrapped_run_workflow(query, image_url=None, force_vl=False):
        mapping_cmd = _parse_mapping_voice_command(query)
        if getattr(agent, "_mapping_stop_only_session", False):
            if mapping_cmd == "stop":
                execute_mapping_voice_command(mapping_cmd)
            else:
                speaker.speak("我正在建图，你可以说结束建图。")
                _wait_for_tts_queue()
                task_manager.mark_voice_cooldown()
            agent.expect_followup = False
            return ""

        if mapping_cmd is not None:
            execute_mapping_voice_command(mapping_cmd)
            agent.expect_followup = False
            return ""

        move_cmd = _parse_chassis_move_command(query)
        if move_cmd is not None:
            direction = move_cmd["direction"]
            steps = move_cmd["steps"]
            label = MOVE_PROFILES[direction]["label"]
            speaker.speak(f"好的，我先{label}走{steps}步。")
            _wait_for_tts_queue()
            execute_chassis_move(direction, steps)
            agent.expect_followup = False
            return ""

        if _can_open_mapping_voice_session():
            speaker.speak("我正在建图，你可以说结束建图。")
            _wait_for_tts_queue()
            task_manager.mark_voice_cooldown()
            agent.expect_followup = False
            return ""

        intent = _classify_voice_intent(agent, query, image_url=image_url, force_vl=force_vl)
        if intent == "unclear":
            speaker.speak(AMBIGUOUS_VOICE_REPLY)
            _wait_for_tts_queue()
            task_manager.mark_voice_cooldown()
            agent.expect_followup = False
            return ""

        return original_run_workflow(query, image_url=image_url, force_vl=force_vl)

    agent.run_workflow = wrapped_run_workflow
    print("[Bootstrap] 已安装语音动作指令快速通道和意图分流")

class RuntimeBootstrap:
    def __init__(self, controller_cli_path: str):
        self.controller_cli_path = controller_cli_path
        self.logic_module = final_0418.llm.logic_impl
        self.tool_module = llm_toolmain
        self.robot = None
        self.board = None
        self.ros2_controller = None
        self.preloaded_components = {}
        self._bootstrap()

    def _bootstrap(self):
        print("开始预实例化语音、视觉、导航和工具组件")
        _install_head_pose_hooks()
        self.robot = RobotAssistant()
        speaker.init(self.robot.tts_queue)
        self.preloaded_components["robot_assistant"] = self.robot
        self.preloaded_components["vad"] = getattr(self.robot, "vad", None)
        self.preloaded_components["asr_model"] = getattr(self.robot, "asr_model", None)
        self.preloaded_components["llm_agent"] = getattr(self.robot, "agent", None)
        self.preloaded_components["tts_engine"] = getattr(self.robot.agent, "tts", None)

        self.board = Board()
        self.board.enable_reception()
        self.preloaded_components["board"] = self.board
        self.logic_module.set_pet_motor_board(self.board)

        self.ros2_controller = ROS2NavigationController(self.controller_cli_path)
        self.preloaded_components["ros2_navigation_controller"] = self.ros2_controller

        _ensure_situp_support()
        self.preloaded_components["pet_tracking_system"] = getattr(self.logic_module, "pet_system", None)
        self.preloaded_components["face_recognition_system"] = getattr(self.logic_module, "face_system", None)
        self.preloaded_components["situp_counting_system"] = getattr(self.logic_module, "situp_system", None)
        self.preloaded_components["pullup_counting_system"] = getattr(self.logic_module, "pullup_system", None)
        self.preloaded_components["pushup_counting_system"] = getattr(self.logic_module, "pushup_system", None)
        self.preloaded_components["squat_counting_system"] = getattr(self.logic_module, "squat_system", None)
        self.preloaded_components["tool_instances"] = getattr(self.tool_module, "tools_instances", None)

        self._validate_required_components()
        self._warmup_optional_components()
        self._print_summary()

    def _validate_required_components(self):
        required_names = [
            "robot_assistant",
            "vad",
            "asr_model",
            "llm_agent",
            "tts_engine",
            "board",
            "ros2_navigation_controller",
            "pet_tracking_system",
            "face_recognition_system",
            "pullup_counting_system",
            "pushup_counting_system",
            "squat_counting_system",
            "tool_instances",
        ]
        missing = [name for name in required_names if self.preloaded_components.get(name) is None]
        if missing:
            raise RuntimeError(f"启动期预实例化失败，缺少组件: {', '.join(missing)}")

    def _warmup_optional_components(self):
        print("[Bootstrap] 开始执行可选 warmup，这一步允许更慢一些")
        self._warmup_pet_detector()
        self._warmup_pose_detectors()
        self._warmup_face_model()

    def _warmup_pet_detector(self):
        try:
            self.logic_module.pet_system.preload()
            self.preloaded_components["pet_detector"] = self.logic_module.pet_system.detector
            print("[Bootstrap] 宠物检测 RKNN warmup 完成")
        except Exception as e:
            print(f"[Bootstrap] 宠物检测 warmup 跳过: {e}")

    def _warmup_pose_detectors(self):
        systems = [
            ("pullup_detector", getattr(self.logic_module, "pullup_system", None)),
            ("pushup_detector", getattr(self.logic_module, "pushup_system", None)),
            ("squat_detector", getattr(self.logic_module, "squat_system", None)),
            ("situp_detector", getattr(self.logic_module, "situp_system", None)),
        ]
        for key, system in systems:
            if system is None or not callable(getattr(system, "preload_detector", None)):
                continue
            try:
                system.preload_detector()
                self.preloaded_components[key] = system.detector
                print(f"[Bootstrap] {key} preload 瀹屾垚")
            except Exception as e:
                print(f"[Bootstrap] {key} preload 璺宠繃: {e}")
        return
        detector_specs = [
            ("引体向上", "final_0418.llm.movement_count_2.pullup_camera_base_fps_1", "PullupDet"),
            ("俯卧撑", "final_0418.llm.movement_count_2.pushup_camera_base_fps_1", "PushupDet"),
            ("深蹲", "final_0418.llm.movement_count_2.squat_camera_base_fps_1", "SquatDet"),
        ]
        for label, module_name, class_name in detector_specs:
            try:
                module = __import__(module_name, fromlist=[class_name])
                detector_cls = getattr(module, class_name)
                detector = detector_cls()
                detector.release()
                print(f"[Bootstrap] {label}检测器 warmup 完成")
            except Exception as e:
                print(f"[Bootstrap] {label}检测器 warmup 跳过: {e}")

    def _warmup_face_model(self):
        try:
            self.logic_module.face_system.preload_model()
            self.preloaded_components["face_embedding_model"] = self.logic_module.face_system.model
            print("[Bootstrap] 人脸特征模型 warmup 完成")
        except Exception as e:
            print(f"[Bootstrap] 人脸模型 warmup 跳过: {e}")

    def _print_summary(self):
        print("[Bootstrap] 预实例化完成，后续语音调用直接复用以下实例:")
        for name in [
            "robot_assistant",
            "vad",
            "asr_model",
            "llm_agent",
            "tts_engine",
            "board",
            "ros2_navigation_controller",
            "pet_tracking_system",
            "face_recognition_system",
            "pullup_counting_system",
            "pushup_counting_system",
            "squat_counting_system",
            "tool_instances",
        ]:
            obj = self.preloaded_components[name]
            print(f"  - {name}: {type(obj).__name__}")


def ensure_runtime_initialized():
    if runtime is None or robot is None or board is None or ros2_controller is None:
        raise RuntimeError("运行时尚未完成预实例化")


# 初始化ROS2导航控制器
# controller_cli_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ros2_ws", "src", "demo", "controller_cli.py")
# controller_cli_path = "/home/test/code/ros2_ws/src/demo/controller_cli.py"
controller_cli_path = "/home/test/ros2_ws/src/demo/controller_cli.py"
runtime = RuntimeBootstrap(controller_cli_path)
robot = runtime.robot
board = runtime.board
ros2_controller = runtime.ros2_controller
install_agent_fast_path(robot.agent)
head_pose_manager.ensure_default_pose(reason="系统启动后默认平视")

class Config:
    ENCODER_PATH = "./model/encoder-epoch-99-avg-1.rknn"
    DECODER_PATH = "./model/decoder-epoch-99-avg-1.rknn"
    JOINER_PATH = "./model/joiner-epoch-99-avg-1.rknn"
    VOCAB_PATH = "./model/vocab.txt"
    LLM_MODE = "cloud"
    LLM_LOCAL = "http://localhost:8081/v1"
    API_KEY = "sk-ff528950477e421999763986692ce67e"
    MIN_SILENCE_MS = 400
    MIN_SEGMENT_SEC = 0.4
    CHUNK_SIZE = 512
    SAMPLE_RATE = 16000

def execute_received_command(command=None):
    if command is None:
        command = _consume_pending_app_command()
    if command is None:
        return
    task_label = f"APP指令执行中: {command.get('device_type')}/{command.get('command')}"
    task_manager.begin_local_task(task_label)
    try:
        return _execute_received_command_impl(command)
    finally:
        task_manager.end_local_task()


def _execute_received_command_impl(command):
    ensure_runtime_initialized()
    app_device_type = command["device_type"]
    app_device_command = command["command"]
    app_point_name = command.get("point_name", "")
    app_x = command.get("x")
    app_y = command.get("y")
    app_yaw = command.get("yaw", 0.0)
    req_data = dict(command.get("payload") or {})
    head_pose_manager.ensure_default_pose(reason="远程指令执行前保持平视")
    
    print(f"\n=====================================")
    print(f"执行接口指令：设备={app_device_type}，命令={app_device_command}")
    if app_point_name:
        print(f"目标点: {app_point_name}")
    if app_x is not None and app_y is not None:
        print(f"坐标: x={app_x}, y={app_y}, yaw={app_yaw}")
    print(f"=====================================\n")
    
    if app_device_type == "search":
        global search_status

        if app_device_command == "start":
            search_status = "finding"
            speaker.speak("我要去找找小狗狗在哪里")
            print("执行搜索功能：启动导航并导航到目标点")
            
            search_target = "feeder"
            # search_target = "b"
            
            # 暂时注释
            # success = ros2_controller.navigate_to_point(search_target)
            success = True  # 模拟导航成功，实际使用时请取消注释上面一行
            

            if success:
                # print(f"✅ 搜索功能已启动，导航到目标点: {search_target}")
                # print("🕒 等待导航到达目标点...")
                # speaker.speak("我想想，先去狗狗小窝那里看看吧")
                # # 未结束就开始执行下面过程了
                # with open(r'/home/test/code/final_0418/function/pet_navigation_result.txt', 'r', encoding='utf-8') as f:
                #     result_n = f.read()  # 读取全部文本

                # while result_n != "success":
                #     print("⏳ 导航中...")
                #     time.sleep(2)  # 每2秒检查一次
                #     with open(r'/home/test/code/final_0418/function/pet_navigation_result.txt', 'r', encoding='utf-8') as f:
                #         result_n = f.read()  # 读取全部文本
                    
                # print("导航完成啦")
                # success_0 = ros2_controller.stop_all()
                # time.sleep(8)  # 确保命令发送和状态更新有足够时间
                # if success_0:
                #     print("搜索功能已停止")
                #     # 更新搜索状态为fail（用户主动停止）
                #     search_status = "fail"
                # else:
                #     print("❌ 搜索功能停止失败")
                # with open(r'/home/test/code/final_0418/function/pet_navigation_result.txt', 'w', encoding='utf-8') as f:
                #     f.write("failure")
                
                # 启动宠物追踪功能，默认追踪猫
                print("🐾 启动宠物追踪功能...")
                try:
                    speaker.speak("启动宠物追踪")
                    start_pet_tracking('dog')  
                    print("✅ 宠物追踪功能已启动")
                    
                    with open(r'/home/test/code/final_0418/llm/pet_tracking_result.txt', 'r', encoding='utf-8') as f:
                        result = f.read()  # 读取全部文本

                    while result != "success":
                        print("⏳ 等待宠物追踪完成...")
                        time.sleep(2)  # 每2秒检查一次
                        with open(r'/home/test/code/final_0418/llm/pet_tracking_result.txt', 'r', encoding='utf-8') as f:
                            result = f.read()  # 读取全部文本
                    
                    print("✅ 宠物追踪已完成")
                    with open(r'/home/test/code/final_0418/llm/pet_tracking_result.txt', 'w', encoding='utf-8') as f:
                        f.write("failure")

                    # # 停止宠物追踪
                    # print("🛑 停止宠物追踪...")
                    # from final_0418.llm.logic_impl import stop_pet_tracking
                    # stop_pet_tracking()
                    # print("✅ 宠物追踪已停止")
                    # 更新搜索状态为success
                    search_status = "success"
                    
                    # 上传追踪视频
                    print("📤 开始上传宠物追踪视频...")
                    upload_success = upload_pet_tracking_video()
                    # upload_success = True
                    if upload_success:
                        speaker.speak("正在上传宠物视频")
                        # print("✅ 宠物追踪视频上传完成")
                        # # 调用新的视频上传函数
                        # global video_url
                        # video_url = upload_pet_video_to_app_server()
                        # if video_url:
                        #     print(f"✅ 视频上传到APP服务器成功，URL: {video_url}")
                        # else:
                        #     print("❌ 视频上传到APP服务器失败")
                        # 更新搜索状态为success
                        search_status = "success"

                except Exception as e:
                    # 更新搜索状态为fail
                    search_status = "fail"
                    speaker.speak("搜索失败了，别灰心")
                    print(f"❌ 宠物追踪功能启动失败: {e}")
            else:
                # 更新搜索状态为fail
                search_status = "fail"
                speaker.speak("搜索失败了，别灰心")
                print(f"搜索功能启动失败")
        
        elif app_device_command == "pause":
            print("暂停搜索功能")
            success = ros2_controller.stop_all()
            if success:
                print("搜索功能已暂停")
                search_status = "fail"
            else:
                print("搜索功能暂停失败")

        elif app_device_command == "continue":
            print("继续搜索功能")
            # 更新搜索状态为finding
            search_status = "finding"
            # 继续功能可以重新启动导航或恢复之前的导航状态
            success = ros2_controller.start_navigation()
            if success:
                # 更新搜索状态为finding
                search_status = "finding"
                print("搜索功能已继续")
            else:
                print("搜索功能继续失败")
        elif app_device_command == "stop":
            print("停止搜索功能")
            success = ros2_controller.stop_all()
            if success:
                print("搜索功能已停止")
                search_status = "fail"
            else:
                print("搜索功能停止失败")
        return
    
    # 建图功能
    if app_device_type == "mapping":
        if app_device_command == "start":
            block_reason = _get_mapping_start_block_reason()
            if block_reason == "建图中":
                speaker.speak("我已经在建图了")
                print("建图已在运行，忽略重复启动")
                return
            if block_reason is not None:
                speaker.speak(f"现在还不能开始建图，我当前{block_reason}")
                print(f"建图启动被拦截：{block_reason}")
                return
            speaker.speak("我要去建图看看我的新家啦，可能有点久，完成了我会告诉你的")
            print("启动建图功能")
            task_manager.begin_local_task("建图启动中")
            try:
                success = ros2_controller.start_mapping()
                if success:
                    print("✅ 建图已启动")
                else:
                    print("❌ 建图启动失败")
            finally:
                task_manager.end_local_task()
        elif app_device_command == "stop":
            speaker.speak("建图完成了哦，我要去玩了")
            print("🛑 停止建图功能")
            task_manager.begin_local_task("建图停止中")
            try:
                success = ros2_controller.stop_all()
                if success:
                    print("✅ 建图已停止")
                else:
                    print("❌ 建图停止失败")
            finally:
                task_manager.end_local_task()
        elif app_device_command == "status":
            print("获取建图状态")
            success = ros2_controller.get_status()
            if success:
                print("✅ 状态获取成功")
            else:
                print("❌ 状态获取失败")
        return
    
    # 导航功能（导航到风扇、投食机、灯的位置）
    if app_device_type == "navigation":
        if app_device_command == "stop":
            print("停止导航功能")
            success = ros2_controller.stop_all()
            if success:
                speaker.speak("导航已经停止了")
                print("✅ 导航已停止")
            else:
                speaker.speak("导航停止失败了")
                print("❌ 导航停止失败")
            return

        if app_device_command == "status":
            print("📊 获取导航状态")
            success = ros2_controller.get_status()
            if success:
                print("✅ 导航状态获取成功")
            else:
                print("❌ 导航状态获取失败")
            return

        if app_device_command in ["fan", "feeder", "light"]:
            target_point = app_device_command  # 使用命令本身作为目标点名称

            print(f"导航到{target_point}位置")
            MAP_CN = {"fan": "风扇", "feeder": "投食机", "light": "台灯"}
            if ros2_controller.is_mapping_active(refresh=True):
                speaker.speak("我正在建图，暂时不能导航")
                print("导航被拦截：建图进行中")
                return
            # MAP_EN = {"fan": "a", "feeder": "b", "light": "c"}
            # success = ros2_controller.navigate_to_point(MAP_EN[target_point])
            success = ros2_controller.navigate_to_point(target_point)

            if success:
                speaker.speak(f"开始导航到{MAP_CN[target_point]}")
                print(f"已导航到{target_point}位置")
                # 未结束就开始执行下面过程了
                # with open(r'/home/test/code/final_0418/function/pet_navigation_result.txt', 'r', encoding='utf-8') as f:
                #     result_w = f.read()  # 读取全部文本

                # while result_w != "success":
                #     print("⏳ 导航中...")
                #     time.sleep(2)  # 每2秒检查一次
                #     with open(r'/home/test/code/final_0418/function/pet_navigation_result.txt', 'r', encoding='utf-8') as f:
                #         result_w = f.read()  # 读取全部文本
                    
                # speaker.speak("导航已成功")
                # with open(r'/home/test/code/final_0418/function/pet_navigation_result.txt', 'w', encoding='utf-8') as f:
                #     f.write("failure")
            else:
                speaker.speak(f"我迷路了")
                print(f"❌ 导航到{target_point}失败")
        return
    
    # 目标点设置功能（通过命名点实现）
    if app_device_type == "goal":
        if app_device_command == "set":
            if ros2_controller.is_mapping_active(refresh=True):
                speaker.speak("我正在建图，暂时不能导航")
                print("目标点导航被拦截：建图进行中")
                return
            if app_x is not None and app_y is not None:
                print(f"🎯 导航到坐标: x={app_x}, y={app_y}, yaw={app_yaw}")
                success = ros2_controller.navigate_to_pose(app_x, app_y, app_yaw)
            else:
                if not app_point_name:
                    print("❌ 设置目标点需要提供point_name参数，或提供x/y坐标")
                    return
                print(f"🎯 导航到命名点: {app_point_name}")
                success = ros2_controller.navigate_to_point(app_point_name)
            if success:
                print("✅ 已发送目标点导航")
            else:
                print("目标点导航失败")
        return
    
    # 命名点管理功能
    if app_device_type == "points":
        if app_device_command == "list":
            print("列出所有命名点")
            success = ros2_controller.list_points()
            if success:
                print("命名点列表获取成功")
            else:
                print("命名点列表获取失败")
        elif app_device_command == "delete":
            if not app_point_name:
                print("删除命名点需要提供point_name参数")
                return
            print(f"删除命名点: {app_point_name}")
            success = ros2_controller.delete_point(app_point_name)
            if success:
                print("命名点删除成功")
            else:
                print("命名点删除失败")
        return
    
    # 系统控制功能
    if app_device_type == "system":
        if app_device_command == "stop":
            print("停止所有系统功能")
            success = ros2_controller.stop_all()
            if success:
                print("所有系统功能已停止")
            else:
                print("系统停止失败")
        elif app_device_command == "status":
            print("获取系统状态")
            success = ros2_controller.get_status()
            if success:
                print("系统状态获取成功")
            else:
                print("系统状态获取失败")
        return

    # 移动控制功能
    if app_device_type == "move":
        if app_device_command in ["forward", "backward", "left", "right"]:
            direction = app_device_command
            # 默认步数为1步，可以通过APP参数传递步数
            steps = 3
            
            # 如果有步数参数，使用接口线程提前缓存的普通字典
            if 'steps' in req_data and req_data['steps'] is not None:
                try:
                    steps = max(1, min(int(req_data['steps']), MAX_MOVE_STEPS))
                except (TypeError, ValueError):
                    print(f"⚠️ 非法 steps 参数，使用默认值 {steps}: {req_data.get('steps')}")
            
            print(f"🚗 执行移动控制：方向={direction}，步数={steps}")
            
            # 语音提示
            direction_labels = {
                "forward": "向前",
                "backward": "向后", 
                "left": "向左",
                "right": "向右"
            }
            # speaker.speak(f"好的，我{direction_labels[direction]}走{steps}步")
            _wait_for_tts_queue()
            
            # 执行移动
            try:
                execute_chassis_move(direction, steps)
                # speaker.speak(f"我{direction_labels[direction]}走了{steps}步")
                print(f"✅ 移动控制完成：{direction_labels[direction]}{steps}步")
            except Exception as e:
                print(f"❌ 移动控制失败：{e}")
                # speaker.speak("移动失败了，请检查底盘状态")
        return
    
    # 直接使用Board对象的方法控制硬件
    if app_device_type in DEVICE_STATE_MAP and app_device_command in DEVICE_STATE_MAP[app_device_type]:
        state = DEVICE_STATE_MAP[app_device_type][app_device_command]
        if app_device_type == "light" and app_device_command == "on":
            speaker.speak("看我用魔法棒点亮灯")
        elif app_device_type == "light" and app_device_command == "off":
            speaker.speak("我要把灯吹灭了")
        elif app_device_type == "feeder" and app_device_command == "start":
            speaker.speak("小狗狗要吃饭了哦")
        elif app_device_type == "fan":
            if app_device_command == "on":
                speaker.speak("风扇开啦，凉爽来袭")
            elif app_device_command == "off":
                speaker.speak("风扇关啦，安静模式")
            elif app_device_command == "start":
                speaker.speak("风扇转起来了哦")
            elif app_device_command == "turn":
                speaker.speak("风扇转头啦")
        print(f"调用set_household函数，state参数={state}")
        try:
            # 使用APP指定的目标点，如果没有则使用默认
            # search_target = app_point_name if app_point_name else "search_point"
            # success = ros2_controller.navigate_to_point(search_target)
            success = True  # 模拟导航成功，实际使用时请取消注释上面两行
            if success:
                board.set_household(state)
                print(f"硬件控制指令已发送：设备={app_device_type}，命令={app_device_command}，state={state}")
            
                if app_device_type == "fan":
                    priority = FAN_PRIORITY[app_device_command]
                    fan_executed_priorities.add(priority)
                    print(f"记录风扇优先级 {priority} 已执行")
                
        except Exception as e:
            print(f"硬件控制失败：{e}")
    else:
        print(f"未找到对应的硬件控制映射：设备={app_device_type}，命令={app_device_command}")

def run_voice_interaction_once(mapping_stop_only: bool = False):
    ensure_runtime_initialized()
    head_pose_manager.ensure_default_pose(reason="语音交互前保持平视")
    print('before get_mic_stream')
    mic_stream = get_mic_stream(Config.CHUNK_SIZE, Config.SAMPLE_RATE)
    print('after get_mic_stream, before robot.run_once')
    task_manager.begin_local_task("语音交互中")
    previous_mapping_stop_only = getattr(robot.agent, "_mapping_stop_only_session", False)
    robot.agent._mapping_stop_only_session = bool(mapping_stop_only)
    try:
        robot.run_once(mic_stream)
    finally:
        robot.agent._mapping_stop_only_session = previous_mapping_stop_only
        task_manager.end_local_task()
        task_manager.mark_voice_cooldown()
        close_stream = getattr(mic_stream, "close", None)
        if callable(close_stream):
            close_stream()

if __name__ == "__main__":
    app_port = _pick_app_port()
    if app_port != 8082:
        print(f" 8082端口被占用，接口改用 {app_port}")
    threading.Thread(target=run_flask_server, args=(app_port,), daemon=True).start()
    print(f"服务启动：APP指令接口已启用，端口={app_port}")
    speaker.speak("我醒啦")
    while True:
        try:
            wkup = board.get_wkup()
            app_trigger = _has_pending_app_command()
            voice_locked = task_manager.is_voice_locked()
            busy_reason = task_manager.current_busy_reason()

            if app_trigger and wkup is None:
                # speaker.speak("不要怕，我要远程控制了")
                execute_received_command() 
                # mic_stream = get_mic_stream(MIC_DEVICE_INDEX, Config.CHUNK_SIZE, Config.SAMPLE_RATE)
                # robot.run_once(mic_stream)

            elif not app_trigger and wkup is not None and voice_locked:
                if _can_open_mapping_voice_session():
                    speaker.speak("我在建图，你可以直接说结束建图。")
                    time.sleep(1.0)
                    run_voice_interaction_once(mapping_stop_only=True)
                else:
                    print(f"[VoiceGate] 当前{busy_reason}，忽略本次唤醒")
                    time.sleep(0.05)

            elif not app_trigger and wkup is not None:
                speaker.speak("我在哟")
                time.sleep(1.0)
                run_voice_interaction_once()

            elif app_trigger and wkup is not None and voice_locked:
                print(f"[VoiceGate] 当前{busy_reason}，忽略语音唤醒，仅执行远程指令")
                execute_received_command()

            elif app_trigger and wkup is not None:
                speaker.speak("我在哟")
                execute_received_command()
                print("[VoiceGate] 远程指令执行完成，本次唤醒不自动进入语音交互")

            else:
                time.sleep(0.01)

        except KeyboardInterrupt:
            break
