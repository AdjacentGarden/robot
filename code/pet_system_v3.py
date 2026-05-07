# 仅基于原始代码修改：APP指令接口正常使用 + 风扇优先级 + 四种场景
import time
import threading
from flask import Flask, request, jsonify
# from final_0418.llm.main_robot import RobotAssistant, get_mic_stream, list_audio_devices
# from final_0418.llm import speaker
# from ..test3 import Board
from final_0418.function.control import Board, set_household

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
# 记录已执行过的优先级，用于控制低优先级命令的执行条件
fan_executed_priorities = set()

# 设备命令配置
# ✅ 修改2：添加风扇high命令，匹配你的测试代码，避免报错
DEVICE_COMMAND_MAP = {
    "feeder": ["start"],
    "light": ["on", "off"],
    "fan": ["on", "off", "start", "turn", "high", "medium", "low"],  # 新增测试用风速指令
    "search": ["start"]
}

# 设备命令与set_household函数state参数的映射关系
# state: 0-投食机, 1-灯开, 2-灯关, 3-风扇电源控制, 4-风扇使能控制, 5-风扇转动控制
DEVICE_STATE_MAP = {
    "feeder": {"start": 0},      # 投食机触发一次投食功能
    "light": {"on": 1, "off": 2},  # 灯开/关
    "fan": {
        "on": 3, "off": 3,      # 风扇电源控制
        "start": 4,              # 风扇使能控制
        "turn": 5,               # 风扇转动控制
        "high": 5, "medium": 5, "low": 5  # 风速控制映射到转动控制
    }
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

    # 存储指令 → 主线程会读取使用！！！
    app_device_type = device
    app_device_command = final_cmd
    app_has_command = True

    # ✅ 修改3：强化打印，清晰展示【接口收到的完整指令】
    print(f"\n=====================================")
    print(f"📥 接口已接收指令：设备={device}，命令={final_cmd}")
    print(f"=====================================\n")
    return jsonify({"code":200,"message":"指令已接收"}), 200

# 后台运行Flask服务
def run_flask_server():
    app.run(host='0.0.0.0', port=8080, debug=False, use_reloader=False)

# ===================== 原始代码（无任何新增功能） =====================
# list_audio_devices()
MIC_DEVICE_INDEX = 6
# robot = RobotAssistant()
# 初始化硬件控制板
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

# 执行APP指令（核心：使用send_device_command接收的指令）
def execute_received_command():
    global app_device_type, app_device_command, fan_executed_priorities
    
    # ✅ 修改4：强化执行打印，清晰展示【正在执行的指令】
    print(f"\n=====================================")
    print(f"🚀 执行接口指令：设备={app_device_type}，命令={app_device_command}")
    print(f"=====================================\n")
    
    # 直接使用Board对象的方法控制硬件
    if app_device_type in DEVICE_STATE_MAP and app_device_command in DEVICE_STATE_MAP[app_device_type]:
        state = DEVICE_STATE_MAP[app_device_type][app_device_command]
        print(f"🔧 调用set_household函数，state参数={state}")
        try:
            board.set_household(state)
            print(f"✅ 硬件控制指令已发送：设备={app_device_type}，命令={app_device_command}，state={state}")
            
            # 如果是风扇命令，记录已执行的优先级
            if app_device_type == "fan":
                priority = FAN_PRIORITY[app_device_command]
                fan_executed_priorities.add(priority)
                print(f"📝 记录风扇优先级 {priority} 已执行")
                
        except Exception as e:
            print(f"❌ 硬件控制失败：{e}")
    else:
        print(f"⚠️  未找到对应的硬件控制映射：设备={app_device_type}，命令={app_device_command}")

# ===================== 主程序：四种场景 =====================
if __name__ == "__main__":
    # 启动Flask线程，接口开始工作
    threading.Thread(target=run_flask_server, daemon=True).start()
    print("服务启动：APP指令接口已启用，send_device_command可正常使用")
    while True:
        try:
            # wkup = board.get_wkup()
            app_trigger = app_has_command

            # 场景1：APP发起调用，wkup未调用 → 执行指令+机器人
            if app_trigger:
                print("🔍 场景1：APP指令触发")
                execute_received_command() 
                # mic_stream = get_mic_stream(MIC_DEVICE_INDEX, Config.CHUNK_SIZE, Config.SAMPLE_RATE)
                # robot.run_once(mic_stream)
                app_has_command = False  # 重置指令

            # # 场景1：APP发起调用，wkup未调用 → 执行指令+机器人
            # if app_trigger and wkup is None:
            #     print("🔍 场景1：APP指令触发")
            #     execute_received_command() 
            #     # mic_stream = get_mic_stream(MIC_DEVICE_INDEX, Config.CHUNK_SIZE, Config.SAMPLE_RATE)
            #     # robot.run_once(mic_stream)
            #     app_has_command = False  # 重置指令

            # # 场景2：APP未调用，wkup发起调用 → 原始逻辑
            # elif not app_trigger and wkup is not None:
            #     print("🔍 场景2：硬件唤醒触发")
            #     speaker.speak("我在")
            #     time.sleep(1.0)
            #     mic_stream = get_mic_stream(MIC_DEVICE_INDEX, Config.CHUNK_SIZE, Config.SAMPLE_RATE)
            #     robot.run_once(mic_stream)

            # # 场景3：APP+wkup都调用 → 先到先处理
            # elif app_trigger and wkup is not None:
            #     print("🔍 场景3：双触发，执行APP指令")
            #     execute_received_command()  # ✅ 使用接口接收的指令
            #     mic_stream = get_mic_stream(MIC_DEVICE_INDEX, Config.CHUNK_SIZE, Config.SAMPLE_RATE)
            #     robot.run_once(mic_stream)
            #     app_has_command = False

            # 场景4：都未调用 → 静默等待
            else:
                time.sleep(0.01)

        except KeyboardInterrupt:
            break