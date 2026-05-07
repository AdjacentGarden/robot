import time
import threading
import subprocess
import socket
import os
import numpy as np
from flask import Flask, request, jsonify
from final_0418.llm.main_robot import RobotAssistant
from final_0418.llm import speaker
from final_0418.llm import local_tts
from final_0418.function.control import Board, set_household
import final_0418.function.ROS2control
from final_0418.function.ROS2control import ROS2NavigationController
import final_0418.function.videoUpLoad
from final_0418.function.videoUpLoad import upload_pet_tracking_video
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

app = Flask(__name__)

app_has_command = False
app_device_type = ""
app_device_command = ""

FAN_PRIORITY = {"on": 3, "off": 3, "start": 2, "turn": 1, "high":1, "medium":1, "low":1}
fan_cached_cmd = None
fan_executed_priorities = set()

DEVICE_COMMAND_MAP = {
    "feeder": ["start"],
    "light": ["on", "off"],
    "fan": ["on", "off", "start", "turn", "high", "medium", "low"],  # 新增测试用风速指令
    "search": ["start", "stop", "continue"],  # 搜索功能指令
    "mapping": ["start", "stop", "status"],  # 建图功能
    "navigation": ["start", "stop", "status"],  # 导航功能
    "goal": ["set"],  # 设置目标点
    "points": ["list", "delete"],
    "system": ["status", "stop"]
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
    },
    "search": {"start": None, "stop": None, "continue": None},  # 搜索功能需要特殊处理
    "mapping": {"start": None, "stop": None, "status": None},  # 建图功能
    "navigation": {"start": None, "stop": None, "status": None},  # 导航功能
    "goal": {"set": None},  # 设置目标点
    "points": {"list": None, "delete": None},  # 命名点管理
    "system": {"status": None, "stop": None}  # 系统状态和控制
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
    
    # 特殊参数处理
    point_name = req_data.get("point_name", "")
    x = req_data.get("x")
    y = req_data.get("y")
    yaw = req_data.get("yaw", 0.0)
    
    # 存储额外参数到全局变量
    global app_point_name, app_x, app_y, app_yaw
    app_point_name = point_name
    app_x = x
    app_y = y
    app_yaw = yaw

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
        stderr=subprocess.PIPE,   # 不要吞掉
        bufsize=0,
    )

    print(f">>> 麦克风已打开: {MIC_DEVICE} (arecord)")
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

robot = RobotAssistant()
speaker.init(robot.tts_queue)
board = Board()
board.enable_reception()

# 初始化ROS2导航控制器
# controller_cli_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ros2_ws", "src", "demo", "controller_cli.py")
# controller_cli_path = "/home/test/code/ros2_ws/src/demo/controller_cli.py"
controller_cli_path = "/home/test/ros2_ws/src/demo/controller_cli.py"
ros2_controller = ROS2NavigationController(controller_cli_path)

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
    global app_point_name, app_x, app_y, app_yaw
    
    # ✅ 修改4：强化执行打印，清晰展示【正在执行的指令】
    print(f"\n=====================================")
    print(f"🚀 执行接口指令：设备={app_device_type}，命令={app_device_command}")
    if app_point_name:
        print(f"目标点: {app_point_name}")
    if app_x is not None and app_y is not None:
        print(f"坐标: x={app_x}, y={app_y}, yaw={app_yaw}")
    print(f"=====================================\n")
    
    # 特殊处理：搜索功能（集成导航控制）
    if app_device_type == "search":
        if app_device_command == "start":
            speaker.speak("我要去找找小狗狗在哪里")
            print("🔍 执行搜索功能：启动导航并导航到目标点")
            
            # 使用APP指定的目标点，如果没有则使用默认
            search_target = "search_point"
            
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
                    
                # print("✅ 导航完成啦")
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
                    
                    # 上传追踪视频
                    print("📤 开始上传宠物追踪视频...")
                    upload_success = upload_pet_tracking_video()
                    if upload_success:
                        speaker.speak("可以看到宠物的视频啦")
                        print("✅ 宠物追踪视频上传完成")
                    else:
                        speaker.speak("真讨厌，网络故障啦")
                        print("❌ 宠物追踪视频上传失败")

                except Exception as e:
                    speaker.speak("搜索失败，我大概是病了")
                    print(f"❌ 宠物追踪功能启动失败: {e}")
            else:
                speaker.speak("搜索失败，我大概是病了")
                print(f"❌ 搜索功能启动失败")
        elif app_device_command == "stop":
            print("🛑 停止搜索功能")
            success = ros2_controller.stop_all()
            if success:
                print("✅ 搜索功能已停止")
            else:
                print("❌ 搜索功能停止失败")
        elif app_device_command == "continue":
            print("▶️  继续搜索功能")
            # 继续功能可以重新启动导航或恢复之前的导航状态
            success = ros2_controller.start_navigation()
            if success:
                print("✅ 搜索功能已继续")
            else:
                print("❌ 搜索功能继续失败")
        return
    
    # 建图功能
    if app_device_type == "mapping":
        if app_device_command == "start":
            speaker.speak("我要去建图看看我的新家啦，可能有点久，完成了我会告诉你的")
            print("🗺️  启动建图功能")
            success = ros2_controller.start_mapping()
            if success:
                print("✅ 建图已启动")
            else:
                print("❌ 建图启动失败")
        elif app_device_command == "stop":
            speaker.speak("建图完成了哦，我要去玩了")
            print("🛑 停止建图功能")
            success = ros2_controller.stop_all()
            if success:
                print("✅ 建图已停止")
            else:
                print("❌ 建图停止失败")
        elif app_device_command == "status":
            print("📊 获取建图状态")
            success = ros2_controller.get_status()
            if success:
                print("✅ 状态获取成功")
            else:
                print("❌ 状态获取失败")
        return
    
    # 导航功能
    if app_device_type == "navigation":
        if app_device_command == "start":
            print("🧭 启动导航功能")
            success = ros2_controller.start_navigation()
            if success:
                print("✅ 导航已启动")
            else:
                print("❌ 导航启动失败")
        elif app_device_command == "stop":
            print("🛑 停止导航功能")
            success = ros2_controller.stop_all()
            if success:
                print("✅ 导航已停止")
            else:
                print("❌ 导航停止失败")
        elif app_device_command == "status":
            print("📊 获取导航状态")
            success = ros2_controller.get_status()
            if success:
                print("✅ 状态获取成功")
            else:
                print("❌ 状态获取失败")
        return
    
    # 目标点设置功能（通过命名点实现）
    if app_device_type == "goal":
        if app_device_command == "set":
            if not app_point_name:
                print("❌ 设置目标点需要提供point_name参数")
                return
            print(f"🎯 导航到命名点: {app_point_name}")
            success = ros2_controller.navigate_to_point(app_point_name)
            if success:
                print(f"✅ 已导航到目标点: {app_point_name}")
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
        print(f"🔧 调用set_household函数，state参数={state}")
        try:
            # 使用APP指定的目标点，如果没有则使用默认
            # search_target = app_point_name if app_point_name else "search_point"
            # success = ros2_controller.navigate_to_point(search_target)
            success = True  # 模拟导航成功，实际使用时请取消注释上面两行
            if success:
                board.set_household(state)
                print(f"硬件控制指令已发送：设备={app_device_type}，命令={app_device_command}，state={state}")
            
                # 如果是风扇命令，记录已执行的优先级
                if app_device_type == "fan":
                    priority = FAN_PRIORITY[app_device_command]
                    fan_executed_priorities.add(priority)
                    print(f"记录风扇优先级 {priority} 已执行")
                
        except Exception as e:
            print(f"硬件控制失败：{e}")
    else:
        print(f"未找到对应的硬件控制映射：设备={app_device_type}，命令={app_device_command}")

def run_voice_interaction_once():
    print('before get_mic_stream')
    mic_stream = get_mic_stream(Config.CHUNK_SIZE, Config.SAMPLE_RATE)
    print('after get_mic_stream, before robot.run_once')
    try:
        robot.run_once(mic_stream)
    finally:
        close_stream = getattr(mic_stream, "close", None)
        if callable(close_stream):
            close_stream()

if __name__ == "__main__":  
    app_port = _pick_app_port()
    if app_port != 8082:
        print(f" 8082端口被占用，接口改用 {app_port}")
    threading.Thread(target=run_flask_server, args=(app_port,), daemon=True).start()
    print(f"服务启动：APP指令接口已启用，端口={app_port}")
    # speaker.speak("我醒啦。是新成员呢，我要录入一下吗？需要的话，请唤醒我吧")
    speaker.speak("我醒啦")

    while True:
        try:
            wkup = board.get_wkup()
            app_trigger = app_has_command

            if app_trigger and wkup is None:
                speaker.speak("不要怕哦，我要远程控制啦")
                execute_received_command() 
                # mic_stream = get_mic_stream(MIC_DEVICE_INDEX, Config.CHUNK_SIZE, Config.SAMPLE_RATE)
                # robot.run_once(mic_stream)
                app_has_command = False

            elif not app_trigger and wkup is not None:
                speaker.speak("我在哟")
                time.sleep(1.0)
                run_voice_interaction_once()

            elif app_trigger and wkup is not None:
                speaker.speak("我在哟")
                execute_received_command()
                run_voice_interaction_once()
                app_has_command = False

            else:
                time.sleep(0.01)

        except KeyboardInterrupt:
            break
