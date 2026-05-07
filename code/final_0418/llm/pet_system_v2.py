# 仅基于原始代码修改：APP指令接口正常使用 + 风扇优先级 + 四种场景
import os
import sys
import time
import threading
from flask import Flask, request, jsonify
from main_robot import RobotAssistant, get_mic_stream, list_audio_devices
import speaker

_HOST_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _HOST_ROOT not in sys.path:
    sys.path.insert(0, _HOST_ROOT)
from test3 import Board

# 统一人脸模块配置（logic_impl.py / face_camera.py 会读取这些环境变量）
FACE_CAMERA_ID = os.getenv("FACE_CAMERA_ID", "/dev/video40")
FACE_MODEL_PATH = os.getenv("FACE_MODEL_PATH", os.path.join(os.path.dirname(__file__), "model", "20180402-114759-vggface2.pt"))
FACE_DB_PATH = os.getenv("FACE_DB_PATH", os.path.join(os.path.dirname(__file__), "faces.db"))
os.environ["FACE_CAMERA_ID"] = FACE_CAMERA_ID
os.environ["FACE_MODEL_PATH"] = FACE_MODEL_PATH
os.environ["FACE_DB_PATH"] = FACE_DB_PATH
os.environ.setdefault("FACE_CAMERA_SHOW_WINDOW", "0")
os.environ.setdefault("FACE_CAMERA_USE_SUBPROCESS", "0")

# ===================== Flask APP服务（接口正常接收+使用指令） =====================
app = Flask(__name__)

# 全局变量：存储APP接收的指令（核心：让主线程使用接口数据）
app_has_command = False
app_device_type = ""
app_device_command = ""

# 风扇优先级：通电/断电(最高) > 转动(次) > 转头/风速(最低)
# ✅ 修改1：补充测试用的high命令优先级，匹配你的测试代码
FAN_PRIORITY = {"on": 3, "off": 3, "start": 2, "turn": 1, "high":1, "medium":1, "low":1}
fan_cached_cmd = None

# 设备命令配置
# ✅ 修改2：添加风扇high命令，匹配你的测试代码，避免报错
DEVICE_COMMAND_MAP = {
    "feeder": ["start"],
    "light": ["on", "off"],
    "fan": ["on", "off", "start", "turn", "high", "medium", "low"],  # 新增测试用风速指令
    "search": ["start"]
}

# 核心接口：被APP调用，接收并存储指令
@app.route('/api/device/command', methods=['POST'])
def send_device_command():
    global app_has_command, app_device_type, app_device_command, fan_cached_cmd
    req_data = request.get_json()

    # 参数校验
    if not req_data or not all(k in req_data for k in ['device_type', 'code', 'command']):
        return jsonify({"code":400,"message":"参数缺失"}), 400
    
    device = req_data["device_type"]
    cmd = req_data["command"]
    
    # 设备/命令校验
    if device not in DEVICE_COMMAND_MAP or cmd not in DEVICE_COMMAND_MAP[device]:
        return jsonify({"code":400,"message":"不支持的设备或命令"}), 400

    # 风扇优先级处理（高优先级覆盖低优先级）
    if device == "fan":
        if fan_cached_cmd is None or FAN_PRIORITY[cmd] >= FAN_PRIORITY[fan_cached_cmd]:
            fan_cached_cmd = cmd
        final_cmd = fan_cached_cmd
    else:
        final_cmd = cmd
        fan_cached_cmd = None

    app_device_type = device
    app_device_command = final_cmd
    app_has_command = True

    print(f"\n=====================================")
    print(f"📥 接口已接收指令：设备={device}，命令={final_cmd}")
    print(f"=====================================\n")
    return jsonify({"code":200,"message":"指令已接收"}), 200

def run_flask_server():
    app.run(host='0.0.0.0', port=8080, debug=False, use_reloader=False)


list_audio_devices()
MIC_DEVICE_INDEX = 6
robot = RobotAssistant()
board = Board()
board.enable_reception()

class Config:
    ENCODER_PATH = "./model/encoder-epoch-99-avg-1.rknn"
    DECODER_PATH = "./model/decoder-epoch-99-avg-1.rknn"
    JOINER_PATH = "./model/joiner-epoch-99-avg-1.rknn"
    VOCAB_PATH = "./model/vocab.txt"
    LLM_MODE = "local"
    LLM_LOCAL = "http://localhost:8081/v1"
    API_KEY = "sk-ff528950477e421999763986692ce67e"
    MIN_SILENCE_MS = 400
    MIN_SEGMENT_SEC = 0.4
    CHUNK_SIZE = 512
    SAMPLE_RATE = 16000

def execute_received_command():
    print(f"\n=====================================")
    print(f"执行接口指令：设备={app_device_type}，命令={app_device_command}")
    print(f"=====================================\n")

# ===================== 主程序：四种场景 =====================
if __name__ == "__main__":
    # 启动Flask线程，接口开始工作
    threading.Thread(target=run_flask_server, daemon=True).start()
    print("服务启动：APP指令接口已启用，send_device_command可正常使用")
    while True:
        try:
            wkup = board.get_wkup()
            app_trigger = app_has_command

            # 场景1：APP发起调用，wkup未调用 → 执行指令+机器人
            if app_trigger and wkup is None:
                print("🔍 场景1：APP指令触发")
                execute_received_command() 
                # mic_stream = get_mic_stream(MIC_DEVICE_INDEX, Config.CHUNK_SIZE, Config.SAMPLE_RATE)
                # robot.run_once(mic_stream)
                app_has_command = False  # 重置指令

            # 场景2：APP未调用，wkup发起调用 → 原始逻辑
            elif not app_trigger and wkup is not None:
                print("🔍 场景2：硬件唤醒触发")
                speaker.speak("我在")
                time.sleep(1.0)
                mic_stream = get_mic_stream(MIC_DEVICE_INDEX, Config.CHUNK_SIZE, Config.SAMPLE_RATE)
                robot.run_once(mic_stream)

            # 场景3：APP+wkup都调用 → 先到先处理
            elif app_trigger and wkup is not None:
                print("🔍 场景3：双触发，执行APP指令")
                execute_received_command()  # ✅ 使用接口接收的指令
                mic_stream = get_mic_stream(MIC_DEVICE_INDEX, Config.CHUNK_SIZE, Config.SAMPLE_RATE)
                robot.run_once(mic_stream)
                app_has_command = False

            # 场景4：都未调用 → 静默等待
            else:
                time.sleep(0.01)

        except KeyboardInterrupt:
            break
