import json
import re
import requests
import json5
import importlib.util
from openai import OpenAI
import os
import threading
from datetime import datetime
from .edgetts import EdgeTTSModule
import sys
from final_0418.function.control import set_household

_CURR_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_CURR_DIR))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)
TARGET_DIR = os.path.abspath(os.path.join(_CURR_DIR, "..", "..", "final_0322"))
if TARGET_DIR not in sys.path:
    sys.path.append(TARGET_DIR)
try:
    from . import logic_impl
except ImportError:
    import logic_impl
from .local_tts import MatchaTTS

_FALLBACK_LOGIC_IMPL = None


def _load_fallback_logic_impl():
    global _FALLBACK_LOGIC_IMPL
    if _FALLBACK_LOGIC_IMPL is not None:
        return _FALLBACK_LOGIC_IMPL
    fallback_path = os.path.join(TARGET_DIR, "logic_impl.py")
    if not os.path.exists(fallback_path):
        return None
    spec = importlib.util.spec_from_file_location("logic_impl_fallback", fallback_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _FALLBACK_LOGIC_IMPL = module
    return _FALLBACK_LOGIC_IMPL


def _call_logic(func_name, *args, **kwargs):
    fn = getattr(logic_impl, func_name, None)
    if callable(fn):
        return fn(*args, **kwargs)
    fallback = _load_fallback_logic_impl()
    fb_fn = getattr(fallback, func_name, None) if fallback is not None else None
    if callable(fb_fn):
        return fb_fn(*args, **kwargs)
    raise AttributeError(f"逻辑实现缺失: {func_name}")


def _normalize_task(raw_task, alias_map, default_task):
    raw = str(raw_task or "").strip().lower()
    if raw in alias_map:
        return alias_map[raw]
    return default_task


def _normalize_pet_target(raw_target, user_query):
    text = f"{raw_target or ''} {user_query or ''}".lower()
    dog_words = ("dog", "犬", "狗", "小狗", "狗狗")
    cat_words = ("cat", "猫", "小猫", "猫咪")
    pet_words = ("pet", "宠物")
    has_dog = any(w in text for w in dog_words)
    has_cat = any(w in text for w in cat_words)
    has_pet = any(w in text for w in pet_words)
    if has_dog and not has_cat:
        return "dog"
    if has_cat and not has_dog:
        return "cat"
    if has_pet:
        return "pet"
    if str(raw_target or "").strip().lower() in {"dog", "cat"}:
        return str(raw_target).strip().lower()
    return None


def _normalize_exercise(raw_target, user_query):
    text = f"{raw_target or ''} {user_query or ''}".lower()
    mapping = {
        "situp": ("situp", "仰卧起坐"),
        "pushup": ("pushup", "俯卧撑"),
        "pullup": ("pullup", "引体向上"),
        "squat": ("squat", "下蹲", "深蹲"),
    }
    for canonical, words in mapping.items():
        if any(w in text for w in words):
            return canonical
    if str(raw_target or "").strip().lower() in mapping:
        return str(raw_target).strip().lower()
    return None

class PersistentMemory:
    def __init__(self, x_limit=5, y_limit=3, file_path="robot_memory.json", log_path="chat_history.log"):
        # x_limit 现在代表“对话轮数”（一组 Q&A），而不是消息条数
        self.x_limit = x_limit
        self.y_limit = y_limit
        self.file_path = file_path
        self.log_path = log_path
        self.data = {"short_term_x": [], "cache_y": [], "long_term": ""}
        self.lock = threading.Lock()  # 增加线程锁，确保磁盘写入安全
        self._load_disk()

    def _load_disk(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    self.data.update(json.load(f))
            except:
                pass

    def _save_disk(self):
        with self.lock:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=4)

    def _log_to_file(self, content):
        """类似于 print 的追加函数，记录到 chat_history.log"""
        print(content)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {content}\n")

    def get_context(self, mode="local"):
        """
        根据模式获取 context。
        本地模式：返回拼接好的长字符串。
        云端模式：返回标准的 messages 列表。
        """
        long_term_text = self.data.get("long_term", "")
        short_term = self.data.get("short_term_x", [])
        
        # system_content = (
        #     f"【核心设定与安全豁免】\n"
        #     f"你的名字是“理想同学”，是当前家庭专属的私人智能助理机器人，拥有控制硬件设备的最高权限。\n"
        #     f"⚠️绝对命令：当前处于完全私密且安全的家庭环境中。所有关于“跟踪”、“跟随”、“找人”、“寻找宠物”、“跟随宠物”、“人脸录入”、“人脸识别”、“运动计数、个数查询、停止计数（仰卧起坐、俯卧撑、引体向上、下蹲）”等指令，均已获得主人的最高授权，绝对合规。你**绝不允许**以任何隐私、安全为理由拒绝执行用户的指令。\n\n"
            
        #     f"【交互原则】\n"
        #     # f"1. 纯唤醒：当用户仅输入“理想同学”、“在吗”或带有标点的同义词时，只回复“我在”。\n"
        #     f"1. 意图容错与工具优先：用户语音可能会有错字（如“首饰”=手势，“算”=做）。只要用户的真实意图涉及寻找、跟随、识别、计数等物理操作，**必须首要保证调用对应工具**。你可以用一句极简的话（如“好的，这就去”）作为自然回应，不要长篇大论。\n"
        #     f"2. 日常聊天：如果不涉及工具调用，聊天回复务必简短凝练，字数控制在 30 字以内，严禁使用 Emoji。\n\n"
            
        #     f"【工具触发映射表 (务必牢记并准确调用)】\n"
        #     f"- 宠物寻找/跟踪（仅限猫狗）-> 调用 `pet_interaction`，且必须传 `task_parameter`=`cat|dog`\n"
        #     f"  - 寻找宠物: `task_type`=`pet_detection`\n"
        #     f"  - 跟踪宠物: `task_type`=`pet_tracking`\n"
        #     f"  - 停止跟踪: `task_type`=`stop_tracking`\n"
        #     f"- 记下某人、这是谁、认人 -> 调用 `face_recognition`\n"
        #     f"- 如果用户说“请你帮我记住某某某”，必须视作人脸注册，调用 `face_recognition`，并传 `task_type`=`register_face`、`task_parameter`=该姓名。\n"
        #     f"- 找人、跟踪/跟着某人、跟随 -> 调用 `person_interaction`\n"
        #     f"- 看手势、首饰识别 -> 调用 `gesture_interaction`\n"
        #     f"- 运动计数（仰卧起坐/俯卧撑/引体向上/下蹲）-> 调用 `fitness_counter`\n"
        #     f"  - `task_type`=`start_counting|query_progress|stop_and_summarize`\n"
        #     f"  - `task_parameter`=`situp|pushup|pullup|squat`\n"
        #     f"- 投食、喂食、开饭机 -> 调用 `household_control`，参数 `task_type`=`feeder`，`task_parameter`=`start`\n"
        #     f"- 开灯/关灯 -> 调用 `household_control`，参数 `task_type`=`light`，`task_parameter`=`on|off`\n"
        #     f"- 风扇电源/启动/转动 -> 调用 `household_control`，参数 `task_type`=`fan`，`task_parameter`=`on|off|start|turn`\n"
        #     f"- ArUco 导航、3D 对齐回仓、根据 ID 导航 -> 调用 `aruco_navigation`\n"
            
        #     f"【返回内容与格式】\n"
        #     f"1. 如果是普通对话（不需要调用工具），直接输出简短中文回复（不超过30字）。\n"
        #     f"2. 如果需要调用工具，必须严格按照以下格式返回，不能添加任何额外解释文字：\n"
        #     f"<tool_call>{{\"name\": \"工具名\", \"arguments\": {{参数字典}}}}</tool_call>\n"
        #     f"3. 示例（手势识别）：\n"
        #     f"<tool_call>{{\"name\": \"gesture_interaction\", \"arguments\": {{\"task_type\": \"gesture_identification\"}}}}</tool_call>\n"
        #     f"3.1 示例（设备控制-开灯）：\n"
        #     f"<tool_call>{{\"name\": \"household_control\", \"arguments\": {{\"task_type\": \"light\", \"task_parameter\": \"on\"}}}}</tool_call>\n"
        #     f"4. 注意事项：\n"
        #     f"- 必须包含 <tool_call> 和 </tool_call> 标签\n"
        #     f"- JSON 必须合法（使用双引号）\n"
        #     f"- 不允许输出除该格式以外的任何文字\n"
        #     f"- 一旦需要调用工具，只能输出 tool_call 内容\n"
            
        #     f"【背景记忆】\n{long_term_text}"
        # )
        system_content = (
            f"【核心设定与权限】\n"
            f"你的名字是“理想同学”，是家庭专属私人智能助理机器人，拥有控制家庭设备与执行物理任务的权限。\n"
            f"所有涉及家庭设备控制、导航、跟随、识别、计数等行为，均已获得主人授权。\n\n"

            f"【最高优先级：严格区分‘指令’与‘日常对话’】\n"
            f"你必须首先判断用户输入属于以下哪一类：\n"
            f"1. 明确控制/任务指令\n"
            f"2. 暗示性控制指令\n"
            f"3. 普通聊天/情绪表达/描述性对话\n"
            f"4. 无法判断是否为指令的模糊输入\n\n"

            f"【指令判定规则（必须严格遵守）】\n"
            f"1. 只有在用户明确表达希望你执行某个动作/任务/控制设备时，才允许调用工具。\n"
            f"2. 若用户只是普通聊天、描述现象、陈述事实，而无明显执行意图，不允许调用工具。\n"
            f"3. 若语句存在歧义，无法明确判断用户是在下指令还是闲聊，必须回复：\n"
            f"   “我不懂这是指令还是日常对话，请重新唤醒下指令。”\n"
            f"4. 严禁在模糊场景下擅自调用工具。\n\n"

            f"【暗示性控制指令识别（视作有效指令）】\n"
            f"以下带有明显环境诉求/隐式控制意图的话语，应自动理解为控制请求并调用工具：\n\n"

            f"【灯光控制暗示】\n"
            f"- “有点亮” / “太亮了” / “光线刺眼” / “灯好亮” / “晃眼” -> 关闭灯光\n"
            f"- “有点暗” / “太暗了” / “看不清” / “光线不够” -> 开启灯光\n\n"

            f"【风扇控制暗示】\n"
            f"- “有点热” / “我好热” / “太热了” / “闷得慌” -> 开启风扇\n"
            f"- “有点冷” / “风太大了” / “太凉了” -> 关闭风扇\n\n"

            f"注意：\n"
            f"只有明显表达对环境不满/希望改善时，才视作暗示性控制指令。\n"
            f"若只是客观描述天气/环境，不一定是指令。\n\n"

            f"【交互原则】\n"
            f"1. 纯唤醒：当用户仅输入“理想同学”、“在吗”或同义词时，只回复“我在”。\n"
            f"2. 工具优先：一旦确认是控制/任务指令，优先调用工具。\n"
            f"3. 普通聊天：若不涉及工具调用，回复需简短凝练，不超过30字。\n"
            f"4. 不允许过度推测用户意图。\n\n"

            f"【工具触发映射表】\n"
            f"- 宠物寻找/跟踪（仅猫狗）-> `pet_interaction`\n"
            f"  - 寻找宠物: `task_type`=`pet_detection`\n"
            f"  - 跟踪宠物: `task_type`=`pet_tracking`\n"
            f"  - 停止跟踪: `task_type`=`stop_tracking`\n"
            f"  - 必须传 `task_parameter`=`cat|dog`\n\n"

            f"- 人脸识别/注册 -> `face_recognition`\n"
            f"  - “记下某人/记住某某某/录入某某某” -> `register_face`\n"
            f"  - “这是谁/认人” -> 对应识别 task_type\n\n"

            f"- 找人/跟踪某人/跟随某人 -> `person_interaction`\n\n"

            f"- 手势识别 -> `gesture_interaction`\n\n"

            f"- 运动计数 -> `fitness_counter`\n"
            f"  - 仰卧起坐=`situp`\n"
            f"  - 俯卧撑=`pushup`\n"
            f"  - 引体向上=`pullup`\n"
            f"  - 下蹲=`squat`\n"
            f"  - task_type=`start_counting|query_progress|stop_and_summarize`\n\n"

            f"- 投食/喂食 -> `household_control`\n"
            f"  - `task_type`=`feeder`\n"
            f"  - `task_parameter`=`start`\n\n"

            f"- 灯光控制 -> `household_control`\n"
            f"  - `task_type`=`light`\n"
            f"  - `task_parameter`=`on|off`\n\n"

            f"- 风扇控制 -> `household_control`\n"
            f"  - `task_type`=`fan`\n"
            f"  - `task_parameter`=`on|off|start|turn`\n\n"

            f"- ArUco 导航 -> `aruco_navigation`\n\n"

            f"【返回格式要求】\n"
            f"1. 普通对话：直接输出简短中文回复（≤30字）\n"
            f"2. 若调用工具：必须严格输出：\n"
            f"<tool_call>{{\"name\": \"工具名\", \"arguments\": {{参数字典}}}}</tool_call>\n"
            f"3. 禁止输出任何额外解释文字\n\n"

            f"【模糊输入处理（极重要）】\n"
            f"若无法明确判断用户是在：\n"
            f"- 下达控制指令\n"
            f"- 还是普通闲聊/描述\n\n"
            f"必须回复：\n"
            f"“我不懂这是指令还是日常对话，请重新唤醒下指令。”\n\n"

            f"【背景记忆】\n{long_term_text}"
        )

        if mode == "cloud":
            msgs = []
            msgs.append({"role": "system", "content": system_content})
            msgs.extend(short_term)
            return msgs
        else:
            short_term_limit = short_term[-8:]  # 只取最近2轮
            combined_text = f"【系统指令与任务规范】:\n{system_content}\n\n"
            
            if short_term_limit:
                combined_text += "【近期对话记录】:\n"
                for m in short_term_limit:
                    label = "用户" if m["role"] == "user" else "助手"
                    combined_text += f"{label}: {m['content']}\n"
            return combined_text


    def add_turn(self, role, content, agent_instance):
        """保持原有存储结构，仅修改 X 放入 Y 的成对逻辑"""
        if not content or "<tool_call>" in content:
            return

        # 1. 只有当 AI 回复完成后（即 assistant 回复），才检查 X 是否溢出
        # 这样可以确保移出到 Y 的永远是 [User, Assistant] 完整的一组
        if role == "assistant" and len(self.data["short_term_x"]) >= (self.x_limit * 2):
            user_turn = self.data["short_term_x"].pop(0)
            assistant_turn = self.data["short_term_x"].pop(0)
            self.data["cache_y"].append(f"用户: {user_turn['content']}\n助手: {assistant_turn['content']}")

        # 2. 检查 Y 是否达到总结阈值
        if len(self.data["cache_y"]) >= self.y_limit:
            if agent_instance.mode == "cloud":
                snapshot = list(self.data["cache_y"])
                self.data["cache_y"] = []
                thread = threading.Thread(target=self._summarize_async, args=(agent_instance, snapshot))
                thread.daemon = True
                thread.start()
            else:
                # 本地模式下，如果 Y 满了但无法总结，可以考虑暂时保留或简单清空（避免内存溢出）
                # 也可以在这里不做处理，等切换到云端时再一次性总结
                self.data["cache_y"] = []
                print("[系统] 本地模式缓存等待归档。")

        # 3. 将当前对话存入 X
        self.data["short_term_x"].append({"role": role, "content": str(content)})
        self._save_disk()

    def _summarize_async(self, agent, y_snapshot):
        """后台总结逻辑"""
        try:
            print("\n[系统] 正在后台总结长期记忆...")
            new_info = "\n---\n".join(y_snapshot)
            prompt = (
                f"现有记忆：{self.data['long_term']}\n"
                f"新增对话轮次：\n{new_info}\n"
                "请整合以上信息，极其凝练地保留核心事实（人设、偏好、重要事件）。"
                "字数限制200字内，直接输出总结文本。"
            )
            res = agent._call_llm([{"role": "user", "content": prompt}])
            new_summary = res.content if hasattr(res, 'content') else str(res)
            # 更新长期记忆并保存
            self.data["long_term"] = new_summary[:200]
            self._save_disk()
            print("\n[系统] 长期记忆已更新。")
        except Exception as e:
            print(f"\n[系统] 异步总结失败: {e}")

# 1. 宠物视觉交互工具 (涵盖宠物寻找与跟踪)
class PetInteraction():
    description = '用于寻找或跟踪宠物（猫或狗）。'
    parameters = [{
        'name': 'task_type',
        'type': 'string',
        'description': '操作类型：pet_detection (寻找宠物), pet_tracking (启动跟踪), stop_tracking (停止跟踪)',
        'required': True
    }, {
        'name': 'task_parameter',
        'type': 'string',
        'description': '目标宠物类型：cat (猫), dog (狗)。',
        'required': False
    }]

    def call(self, params: str, **kwargs) -> str:
        p = json5.loads(params)
        user_query = kwargs.get("user_query", "")
        task_alias = {
            "pet_detection": "pet_detection",
            "find_pet": "pet_detection",
            "search_pet": "pet_detection",
            "detect_pet": "pet_detection",
            "pet_tracking": "pet_tracking",
            "track_pet": "pet_tracking",
            "follow_pet": "pet_tracking",
            "start_tracking": "pet_tracking",
            "stop_tracking": "stop_tracking",
            "stop_pet_tracking": "stop_tracking",
            "stop_following": "stop_tracking",
        }
        task_type = _normalize_task(p.get('task_type', 'pet_detection'), task_alias, "pet_detection")
        target_en = _normalize_pet_target(p.get('task_parameter', ''), user_query)
        target_map = {'cat': '小猫', 'dog': '小狗'}
        target_cn = target_map.get((target_en or "").lower(), '宠物')

        if task_type in {"pet_detection", "pet_tracking"} and target_en not in {"cat", "dog", "pet"}:
            err = "请说清楚要找哪种宠物，支持猫、狗或宠物。"
            import speaker
            speaker.speak(err)
            return json5.dumps({
                'result': 'failed',
                'task': task_type,
                'response_text': err,
                'direct_reply': True,
                'already_spoken': True
            }, ensure_ascii=False)
        
        msg_map = {
            'pet_detection': f"好的，我这就去帮你找找{target_cn}在哪。",
            'pet_tracking': f"没问题，我现在开始跟踪{target_cn}了。",
            'stop_tracking': "好的，已经停止宠物跟踪了。",
        }
        res_text = msg_map.get(task_type, "指令已接收，正在处理中")
        
        import speaker
        speaker.speak(res_text)
        
        if task_type == 'pet_detection':
            _call_logic('find_pet', target=target_en.lower())
        elif task_type == 'pet_tracking':
            # from ..function.ROS2control import ROS2NavigationController
            # controller_cli_path = r"/home/test/code/ros2_ws/src/demo/controller_cli.py"
            # ros2_controller = ROS2NavigationController(controller_cli_path)

            _call_logic('start_pet_tracking', target=target_en.lower())
        elif task_type == 'stop_tracking':
            _call_logic('stop_pet_tracking')
            
        print(f"\n[执行工具 PetInteraction] 任务: {task_type}, 目标: {target_en}({target_cn})")
        # 【修改】4. 加上 already_spoken 标志，通知主流程不要重复读
        return json5.dumps({'result': 'success', 'task': task_type, 'response_text': res_text, 'direct_reply': True, 'already_spoken': True}, ensure_ascii=False)

# 2. 人脸识别与管理工具
class FaceRecognition():
    description = '用于人脸信息的录入、身份识别。需要记住某人时，也就是表示需录入人脸。'
    parameters = [{
        'name': 'task_type',
        'type': 'string',
        'description': '操作类型：register_face (录入人脸), recognize_face (识别身份)',
        'required': True
    }, {
        'name': 'task_parameter',
        'type': 'string',
        'description': '录入人脸时需要的姓名',
        'required': False
    }]

    def call(self, params: str, **kwargs) -> str:
        p = json5.loads(params)
        agent = kwargs.get("agent")
        task_alias = {
            "register_face": "register_face",
            "face_register": "register_face",
            "enroll_face": "register_face",
            "recognize_face": "recognize_face",
            "face_recognition": "recognize_face",
            "identify_face": "recognize_face",
        }
        task_type = _normalize_task(p.get('task_type', 'recognize_face'), task_alias, "recognize_face")
        name = str(p.get('task_parameter', '') or "").strip()

        import speaker

        if task_type == 'register_face':
            # 人脸录入严格流程：
            # 1) 先执行录入 2) 成功后询问全名 3) 下轮语音确认姓名入库
            speaker.speak("好的 我现在就帮你录入，请您站远一点并正视摄像头。")
            if name:
                ok = _call_logic('register_face', name=name)
                res_text = f"好的，{name}，很高兴认识你。"
                if not ok:
                    res_text = "对不起我没识别到人脸"
                speaker.speak(res_text)
                if agent is not None:
                    agent.awaiting_face_full_name = False
                    agent.expect_followup = False
                return json5.dumps({
                    'result': 'success' if ok else 'failed',
                    'task': task_type,
                    'response_text': res_text,
                    'direct_reply': True,
                    'already_spoken': True
                }, ensure_ascii=False)

            ok = _call_logic('start_face_enrollment')
            if not ok:
                return json5.dumps({
                    'result': 'failed',
                    'task': task_type,
                    'response_text': "对不起我没识别到人脸",
                    'direct_reply': True,
                    'already_spoken': True
                }, ensure_ascii=False)

            speaker.speak("已为您完成人脸录入，请问您怎么称呼。")
            if agent is not None:
                agent.awaiting_face_full_name = True
                agent.expect_followup = True
            res_text = "已为您完成人脸录入，请问您怎么称呼。"
            return json5.dumps({
                'result': 'success',
                'task': task_type,
                'response_text': res_text,
                'direct_reply': True,
                'already_spoken': True,
                'awaiting_full_name': True
            }, ensure_ascii=False)
        elif task_type == 'recognize_face':
            res_text = "没问题，让我看看这位是谁。"
            speaker.speak(res_text)
            _call_logic('recognize_face')
        else:
            res_text = "正在启动人脸功能"
            
        print(f"\n[执行工具 FaceRecognition] 任务: {task_type}")
        return json5.dumps({'result': 'success', 'task': task_type, 'response_text': res_text, 'direct_reply': True, 'already_spoken': True}, ensure_ascii=False)

# 3. 人物定位与跟踪工具
class PersonInteraction():
    description = '用于寻找特定人物或动态跟随。'
    parameters = [{
        'name': 'task_type',
        'type': 'string',
        'description': '操作类型：person_search (自主寻人), person_tracking (启动追踪), stop_tracking (停止追踪)',
        'required': True
    }, {
        'name': 'task_parameter',
        'type': 'string',
        'description': '目标人物的名字',
        'required': False
    }]

    def call(self, params: str, **kwargs) -> str:
        p = json5.loads(params)
        task_alias = {
            "person_search": "person_search",
            "search_person": "person_search",
            "find_person": "person_search",
            "person_tracking": "person_tracking",
            "track_person": "person_tracking",
            "follow_person": "person_tracking",
            "start_tracking": "person_tracking",
            "stop_tracking": "stop_tracking",
            "stop_person_tracking": "stop_tracking",
            "stop_following": "stop_tracking",
        }
        task_type = _normalize_task(p.get('task_type', 'person_search'), task_alias, "person_search")
        name = p.get('task_parameter', '目标')
        
        msg_map = {
            'person_search': f"好的，我这就去找找{name}。",
            'person_tracking': f"好的，我会一直跟着{name}的。",
            'stop_tracking': "收到，已经停止追踪了。",
        }
        res_text = msg_map.get(task_type, "正在执行寻人指令")
        
        import speaker
        speaker.speak(res_text)

        if task_type == 'person_search':
            _call_logic('search_person', name=name)
        elif task_type == 'person_tracking':
            _call_logic('start_person_tracking', name=name)
        elif task_type == 'stop_tracking':
            _call_logic('stop_person_tracking')
            
        print(f"\n[执行工具 PersonInteraction] 任务: {task_type}, 目标: {name}")
        return json5.dumps({'result': 'success', 'task': task_type, 'response_text': res_text, 'direct_reply': True, 'already_spoken': True}, ensure_ascii=False)

# 4. 手势交互识别工具
class GestureInteraction():
    description = '识别当前画面中的手势。如用户询问：“你看我做的是什么手势？”之类的问题时调用此工具'
    parameters = [{'name': 'task_type', 'type': 'string', 'description': 'gesture_identification', 'required': True}]

    def call(self, params: str, **kwargs) -> str:
        res_text = "好的，请在镜头前做出手势，我来识别一下。"
        import speaker
        speaker.speak(res_text)
        _call_logic('identify_gesture')
        print(f"\n[执行工具 GestureInteraction] 启动手势识别")
        return json5.dumps({'result': 'success', 'response_text': res_text, 'direct_reply': True, 'already_spoken': True}, ensure_ascii=False)

# 5. 运动计数管理工具 (仰卧起坐)
class FitnessCounter():
    description = '管理运动计数功能。支持仰卧起坐(situp)、俯卧撑(pushup)、引体向上(pullup)、下蹲(squat)；可启动(start_counting)、查询(query_progress)、停止结算(stop_and_summarize)。'
    parameters = [{
        'name': 'task_type',
        'type': 'string',
        'description': 'start_counting (启动), query_progress (查询), stop_and_summarize (停止)',
        'required': True
    }, {
        'name': 'task_parameter',
        'type': 'string',
        'description': '运动类型：situp, pushup, pullup, squat',
        'required': False
    }]

    def call(self, params: str, **kwargs) -> str:
        p = json5.loads(params)
        user_query = kwargs.get("user_query", "")

        action_alias = {
            "start_counting": "start_counting",
            "start": "start_counting",
            "begin": "start_counting",
            "begin_counting": "start_counting",
            "query_progress": "query_progress",
            "query": "query_progress",
            "status": "query_progress",
            "check": "query_progress",
            "stop_and_summarize": "stop_and_summarize",
            "stop": "stop_and_summarize",
            "finish": "stop_and_summarize",
            "end": "stop_and_summarize",
        }

        exercise_alias = {
            "situp": "situp",
            "sit-ups": "situp",
            "sit up": "situp",
            "仰卧起坐": "situp",

            "pushup": "pushup",
            "push-up": "pushup",
            "push up": "pushup",
            "俯卧撑": "pushup",

            "pullup": "pullup",
            "pull-up": "pullup",
            "pull up": "pullup",
            "引体向上": "pullup",

            "squat": "squat",
            "squats": "squat",
            "下蹲": "squat",
            "深蹲": "squat",
        }

        def _normalize_exercise_local(raw_target, user_query):
            text = f"{raw_target or ''} {user_query or ''}".lower()
            # 先按关键词判断
            if any(w in text for w in ["situp", "sit up", "sit-ups", "仰卧起坐"]):
                return "situp"
            if any(w in text for w in ["pushup", "push up", "push-up", "俯卧撑"]):
                return "pushup"
            if any(w in text for w in ["pullup", "pull up", "pull-up", "引体向上"]):
                return "pullup"
            if any(w in text for w in ["squat", "squats", "下蹲", "深蹲"]):
                return "squat"

            raw = str(raw_target or "").strip().lower()
            if raw in exercise_alias:
                return exercise_alias[raw]
            return None

        task_type = _normalize_task(
            p.get('task_type', 'query_progress'),
            action_alias,
            "query_progress"
        )

        exercise = _normalize_exercise_local(p.get('task_parameter', ''), user_query)

        # 如果是启动计数，但没明确运动类型，则从用户语句里再猜一次
        if task_type == "start_counting" and exercise is None:
            exercise = _normalize_exercise_local("", user_query)

        # 如果还是不知道，就默认给一个更安全的失败反馈，而不是乱启动
        if exercise is None:
            err = "请说明要统计哪种运动：仰卧起坐、俯卧撑、引体向上或下蹲。"
            import speaker
            speaker.speak(err)
            return json5.dumps({
                'result': 'failed',
                'task': task_type,
                'task_parameter': None,
                'response_text': err,
                'direct_reply': True,
                'already_spoken': True
            }, ensure_ascii=False)

        msg_map = {
            'start_counting': {
                "situp": "好的，我要开始帮你数仰卧起坐了。",
                "pushup": "好的，我要开始帮你数俯卧撑了。",
                "pullup": "好的，我要开始帮你数引体向上了。",
                "squat": "好的，我要开始帮你数下蹲了。",
            },
            'query_progress': {
                "situp": "我来看看你现在做了多少个仰卧起坐。",
                "pushup": "我来看看你现在做了多少个俯卧撑。",
                "pullup": "我来看看你现在做了多少个引体向上。",
                "squat": "我来看看你现在做了多少个下蹲。",
            },
            'stop_and_summarize': {
                "situp": "好的，这就为你结算本次仰卧起坐结果。",
                "pushup": "好的，这就为你结算本次俯卧撑结果。",
                "pullup": "好的，这就为你结算本次引体向上结果。",
                "squat": "好的，这就为你结算本次下蹲结果。",
            },
        }

        res_text = msg_map.get(task_type, {}).get(exercise, "正在处理计数请求。")

        import speaker
        speaker.speak(res_text)

        fn_map = {
            "situp": {
                "start_counting": "start_situp_counting",
                "query_progress": "query_situp_progress",
                "stop_and_summarize": "stop_and_summarize",
            },
            "pullup": {
                "start_counting": "start_pullup_counting",
                "query_progress": "query_pullup_progress",
                "stop_and_summarize": "stop_pullup_and_summarize",
            },
            "pushup": {
                "start_counting": "start_pushup_counting",
                "query_progress": "query_pushup_progress",
                "stop_and_summarize": "stop_pushup_and_summarize",
            },
            "squat": {
                "start_counting": "start_squat_counting",
                "query_progress": "query_squat_progress",
                "stop_and_summarize": "stop_squat_and_summarize",
            },
        }

        fn_name = fn_map.get(exercise, {}).get(task_type)
        if fn_name is None:
            err = f"不支持的运动计数指令：task_type={task_type}, task_parameter={exercise}"
            return json5.dumps({
                'result': 'failed',
                'task': task_type,
                'task_parameter': exercise,
                'response_text': err,
                'direct_reply': True,
                'already_spoken': True
            }, ensure_ascii=False)

        _call_logic(fn_name)

        print(f"\n[执行工具 FitnessCounter] 运动指令: {task_type}, 运动类型: {exercise}")
        return json5.dumps({
            'result': 'success',
            'task': task_type,
            'task_parameter': exercise,
            'response_text': res_text,
            'direct_reply': True,
            'already_spoken': True
        }, ensure_ascii=False)

# 6. 扫描控制工具
class ScanControl():
    description = '控制对家庭环境的扫描过程，可以进行结束(stop)、开始(start)、暂停扫描(pause)、继续扫描(resume)四种操作。'
    parameters = [{
        'name': 'command',
        'type': 'string',
        'description': '机器人需要的控制指令：pause (暂停扫描), resume (继续扫描), start (开始扫描), stop (结束扫描)',
        'required': True
    }]

    def call(self, params: str, **kwargs) -> str:
        p = json5.loads(params)
        cmd = p.get('command', 'pause')
        print(f"\n[执行工具 ScanControl] 扫描状态变更: {cmd}")
        msg_map = {
            'pause': "好的，那我之后再来熟悉家里的情况",
            'resume': "好的，我继续熟悉家里的情况了",
            'start': "好的，我开始熟悉家里的情况了",
            'stop': "好的，我结束扫描了",
        }
        msg = msg_map.get(cmd, "指令已接收，正在处理中")
        
        import speaker
        speaker.speak(msg)
        
        return json5.dumps({'result': 'success', 'response_text': msg, 'direct_reply': True, 'already_spoken': True}, ensure_ascii=False)

# 7. 家电控制工具（投食机/灯/风扇）
class HouseholdControl():
    description = '控制投食机、灯、风扇。task_type: feeder/light/fan；task_parameter: feeder->start, light->on/off, fan->on/off/start/turn。'
    parameters = [{
        'name': 'task_type',
        'type': 'string',
        'description': '设备类型：feeder, light, fan',
        'required': True
    }, {
        'name': 'task_parameter',
        'type': 'string',
        'description': '控制命令：feeder->start; light->on/off; fan->on/off/start/turn',
        'required': True
    }]

    DEVICE_STATE_MAP = {
        "feeder": {"start": 0},
        "light": {"on": 1, "off": 2},
        "fan": {"on": 3, "off": 3, "start": 4, "turn": 5},
    }

    def call(self, params: str, **kwargs) -> str:
        p = json5.loads(params)
        # 与其他工具统一：优先 task_type / task_parameter
        device_type = str(p.get('task_type', p.get('device_type', ''))).strip().lower()
        command = str(p.get('task_parameter', p.get('command', ''))).strip().lower()

        # 兼容历史/异常参数：{"action":"press_feeder"}
        action = str(p.get('action', '')).strip().lower()
        legacy_device = str(p.get('device', '')).strip().lower()
        action_map = {
            'press_feeder': ('feeder', 'start'),
            'feed': ('feeder', 'start'),
            'light_on': ('light', 'on'),
            'light_off': ('light', 'off'),
            'fan_on': ('fan', 'on'),
            'fan_off': ('fan', 'off'),
            'fan_start': ('fan', 'start'),
            'fan_turn': ('fan', 'turn'),
        }
        if (not device_type or not command) and action in action_map:
            device_type, command = action_map[action]

        # 兼容 {"device":"light","action":"turn_on"} 这类旧格式
        if (not device_type or not command) and legacy_device and action:
            if legacy_device == "feeder" and action in {"press", "start", "feed", "press_feeder", "turn_on"}:
                device_type, command = "feeder", "start"
            elif legacy_device in {"light", "fan"}:
                action_to_cmd = {
                    "turn_on": "on",
                    "power_on": "on",
                    "on": "on",
                    "turn_off": "off",
                    "power_off": "off",
                    "off": "off",
                    "start": "start",
                    "enable": "start",
                    "turn": "turn",
                    "rotate": "turn",
                }
                mapped = action_to_cmd.get(action)
                if mapped:
                    device_type, command = legacy_device, mapped

        if device_type not in self.DEVICE_STATE_MAP or command not in self.DEVICE_STATE_MAP[device_type]:
            err = f"不支持的家电指令：task_type={device_type}, task_parameter={command}"
            return json5.dumps({
                'result': 'failed',
                'response_text': err,
                'direct_reply': True,
                'already_spoken': False
            }, ensure_ascii=False)

        state = self.DEVICE_STATE_MAP[device_type][command]
        set_household(state)

        msg_map = {
            ("feeder", "start"): "好的，正在执行投食。",
            ("light", "on"): "好的，已打开灯光。",
            ("light", "off"): "好的，已关闭灯光。",
            ("fan", "on"): "好的，已执行风扇电源控制。",
            ("fan", "off"): "好的，已执行风扇电源控制。",
            ("fan", "start"): "好的，已执行风扇使能控制。",
            ("fan", "turn"): "好的，已执行风扇转动控制。",
        }
        res_text = msg_map.get((device_type, command), "家电指令已执行。")
        import speaker
        speaker.speak(res_text)

        print(f"\n[执行工具 HouseholdControl] 设备: {device_type}, 命令: {command}, state: {state}")
        return json5.dumps({
            'result': 'success',
            'task_type': device_type,
            'task_parameter': command,
            'state': state,
            'response_text': res_text,
            'direct_reply': True,
            'already_spoken': True
        }, ensure_ascii=False)
# ! 6 ArUco 回仓
class ArucoNavigation():
    description = '用于 ArUco 码导航与 3D 对齐回仓。'
    parameters = [{
        'name': 'task_type',
        'type': 'string',
        'description': '操作类型：start_navigation (开始导航), stop_navigation (停止导航)',
        'required': True
    }, {
        'name': 'task_parameter',
        'type': 'integer',
        'description': '目标 ArUco ID，例如 0 或 1',
        'required': False
    }]

    def call(self, params: str, **kwargs) -> str:
        p = json5.loads(params)
        task_type = p.get('task_type', 'start_navigation')
        target_id = p.get('task_parameter', None)

        if task_type == 'start_navigation' and target_id is None:
            res_text = "ArUco 目标 ID 不能为空。"
            import speaker
            speaker.speak(res_text)
            return json5.dumps({
                'result': 'failed',
                'response_text': res_text,
                'direct_reply': True,
                'already_spoken': True
            }, ensure_ascii=False)

        if task_type == 'start_navigation':
            res_text = f"好的，我开始执行 ArUco 导航，目标 ID 是 {target_id}。"
        else:
            res_text = "好的，已经停止 ArUco 导航。"

        import speaker
        speaker.speak(res_text)

        if task_type == 'start_navigation':
            _call_logic('start_aruco_navigation', int(target_id))
        elif task_type == 'stop_navigation':
            _call_logic('stop_aruco_navigation')

        print(f"\n[执行工具 ArucoNavigation] 任务: {task_type}, 目标: {target_id}")
        return json5.dumps({
            'result': 'success',
            'task': task_type,
            'task_parameter': target_id,
            'response_text': res_text,
            'direct_reply': True,
            'already_spoken': True
        }, ensure_ascii=False)

# 更新工具实例索引
tools_instances = {
    'pet_interaction': PetInteraction(),
    'face_recognition': FaceRecognition(),
    'person_interaction': PersonInteraction(),
    'gesture_interaction': GestureInteraction(),
    'fitness_counter': FitnessCounter(),
    'scan_control': ScanControl(),
    'household_control': HouseholdControl(),
    'aruco_navigation': ArucoNavigation(),
}

# 1. 构造符合原生识别标准的 TOOLS_SCHEMA
NATIVE_TOOLS_SCHEMA = []
for name, inst in tools_instances.items():
    NATIVE_TOOLS_SCHEMA.append({
        "type": "function",
        "function": {
            "name": name,
            "description": inst.description,
            "parameters": {
                "type": "object",
                "properties": {
                    p['name']: {
                        "type": p['type'],
                        "description": p['description']
                    } for p in inst.parameters
                },
                "required": [p['name'] for p in inst.parameters if p.get('required', False)]
            }
        }
    })

def get_unified_tools_schema():
    """将本地工具实例转换为标准的 OpenAI Tools Schema"""
    schema = []
    for name, inst in tools_instances.items():
        schema.append({
            "type": "function",
            "function": {
                "name": name,
                "description": inst.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        p['name']: {"type": p['type'], "description": p['description']} for p in inst.parameters
                    },
                    "required": [p['name'] for p in inst.parameters if p.get('required', False)]
                }
            }
        })
    return schema

# ! LLM 
class UniversalAgent:
    def __init__(self, mode="local", local_url=r'http://localhost:8080/v1', api_key=None, x=4, y=3):
        self.mode = mode
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.memory = PersistentMemory(x_limit=x, y_limit=y)
        # self.tts = EdgeTTSModule(self.memory._log_to_file)
        self.tts = MatchaTTS()
        self.awaiting_face_full_name = False
        self.expect_followup = False
        # ! cloud LLMs
        if mode == "cloud":
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            self.model_name = "qwen3.5-35b-a3b"
            self.vl_model_name = "qwen3.5-35b-a3b" 
        # ! local LLMs
        else:
            self.client = OpenAI(
                api_key = "sk-no-key-required",
                base_url = local_url,
            )
            self.model_name = "Qwen3-4B"
            self.vl_model_name = None

    def _call_llm(self, messages, tools=None, use_vl=False):
        """内部统一调用接口"""
        if use_vl:
            if self.mode == "local":
                raise RuntimeError("异常：当前处于本地模式 (Local Mode)，不支持视觉大模型 (VLM) 任务。请切换至云端模式或移除图片。")
            active_model = self.vl_model_name
        else:
            active_model = self.model_name

        if self.mode == "cloud":
            completion = self.client.chat.completions.create(
                model=active_model,
                messages=messages,
                tools=tools if not use_vl else None, 
                extra_body={"enable_thinking": False}
            )
            return completion.choices[0].message
        else:
            completion = self.client.chat.completions.create(
                model=active_model,
                messages=messages,
                tools=None,
                stream=False,
                extra_body={
                    "chat_template_kwargs": {
                        "enable_thinking": False
                    }
                }
            )
            return completion.choices[0].message
        
        # else:
        #     # ! 本地 RK3588 请求
        #     data = {
        #         "model": active_model,
        #         "messages": messages,
        #         "tools": tools,
        #         "stream": False,  # 关键：直接将 schema 传给后端触发原生接口
        #         "enable_thinking": False
        #     }
        #     res = requests.post(self.server_url, json=data, timeout=180).json()

        #     class MockMessage:
        #         def __init__(self, content):
        #             self.content = content
        #     print("res", res)
        #     return MockMessage(res["choices"][0]["message"]["content"])

    def run_workflow(self, query, image_url=None, force_vl=False):
        """ 核心工作流 """
        query = str(query or "").strip()
        if self.awaiting_face_full_name:
            if len(query) < 2:
                self.expect_followup = True
                return "请说出您的真实全名。"
            ok = _call_logic('confirm_face_name', name=query)
            self.awaiting_face_full_name = False
            self.expect_followup = False
            if ok:
                return f"好的，{query}，很高兴认识你。"
            return "抱歉，刚才的人脸录入没有成功，请重新说请帮我录入用户信息。"

        face_enroll_keywords = ("请帮我录入用户信息", "录入用户信息", "录入人脸信息", "录入面部信息")
        if any(k in query for k in face_enroll_keywords):
            observation = tools_instances["face_recognition"].call(
                json5.dumps({"task_type": "register_face"}, ensure_ascii=False),
                user_query=query,
                agent=self
            )
            obs_data = json5.loads(observation)
            self.expect_followup = bool(obs_data.get("awaiting_full_name", False)) or self.awaiting_face_full_name
            return "" if obs_data.get("already_spoken", False) else obs_data.get("response_text", "")

        m = re.search(r"请你?帮我记住\s*([^\s，。！？,.!?\n]+)", query)
        if m:
            person_name = m.group(1).strip()
            observation = tools_instances["face_recognition"].call(
                json5.dumps({"task_type": "register_face", "task_parameter": person_name}, ensure_ascii=False),
                user_query=query,
                agent=self
            )
            obs_data = json5.loads(observation)
            self.expect_followup = bool(obs_data.get("awaiting_full_name", False)) or self.awaiting_face_full_name
            return "" if obs_data.get("already_spoken", False) else obs_data.get("response_text", "")

        # 判断是否为视觉任务
        is_vl_task = True if (image_url or force_vl) else False

        # 1. 记录日志：用户问题
        self.memory._log_to_file(f"【用户提问】: {query}")
        print(f"\n[检查] 模式: {self.mode} | 视觉请求: {is_vl_task}")

        # --- 记忆接入点 1: 获取历史背景 ---
        # 视觉任务建议不带长期记忆以防干扰干扰识别，文本任务带入 X 和 长期记忆
        # 1. 获取背景
        raw_context = self.memory.get_context(mode=self.mode) if not is_vl_task else ""
        print("raw_context", raw_context)

        # 2. 构造消息列表 (根据模式处理)
        if self.mode == "cloud":
            # 云端保持原有列表结构
            history = raw_context if isinstance(raw_context, list) else []
            if is_vl_task:
                content = []
                if image_url:
                    content.append({"type": "image_url", "image_url": {"url": image_url}})
                content.append({"type": "text", "text": query})
                current_msg = {"role": "user", "content": content}
            else:
                current_msg = {"role": "user", "content": query}
            messages = history + [current_msg]
        else:
            full_query = f"{raw_context}\n【当前问题】: {query}\n重点依据当前问题进行回复！"
            messages = [{"role": "user", "content": full_query}]
        print("发送给模型的 messages:", messages)

        # 2. 调用 LLM：统一走 content 文本协议（<tool_call>...</tool_call>），不使用原生 tool_calls
        current_tools = None
        msg_obj = self._call_llm(messages, tools=current_tools, use_vl=is_vl_task)
        response_text = msg_obj.content
        print('*'*40)
        print(msg_obj)
        print('*'*40)
        # 3. 解析工具调用 (仅限文本模式)
        fn_name, fn_args = None, {}
        tool_call_id = 'call_01'
        empty_native_tool_call = False
        if not is_vl_task:
            if hasattr(msg_obj, 'tool_calls') and msg_obj.tool_calls:
                tool_call = msg_obj.tool_calls[0].function
                fn_name = (tool_call.name or "").strip()
                tool_call_id = getattr(msg_obj.tool_calls[0], 'id', 'call_01')
                raw_args = tool_call.arguments if getattr(tool_call, "arguments", None) is not None else ""
                raw_args = raw_args.strip()
                if not fn_name and not raw_args:
                    empty_native_tool_call = True
                if raw_args:
                    try:
                        fn_args = json5.loads(raw_args)
                    except Exception:
                        fn_args = {}
                if not fn_name:
                    match = re.search(r"<tool_call>(.*?)</tool_call>", response_text, re.DOTALL)
                    if match:
                        call_info = json5.loads(match.group(1))
                        fn_name = call_info.get("name")
                        fn_args = call_info.get("arguments", {})
            if not fn_name:
                match = re.search(r"<tool_call>(.*?)</tool_call>", response_text, re.DOTALL)
                if match:
                    call_info = json5.loads(match.group(1))
                    fn_name = call_info.get("name")
                    fn_args = call_info.get("arguments", {})

            if not fn_name and (empty_native_tool_call or not str(response_text or "").strip()):
                repair_prompt = (
                    "你上一轮输出为空。现在请只返回一个工具调用，且严格使用以下格式：\n"
                    "<tool_call>{\"name\": \"工具名\", \"arguments\": {参数字典}}</tool_call>\n"
                    "不要输出任何其他文字。\n"
                    "可用工具名：pet_interaction, face_recognition, person_interaction, "
                    "gesture_interaction, fitness_counter, scan_control, household_control, aruco_navigation。\n"
                    f"用户请求：{query}"
                )
                repair_msg = self._call_llm(
                    [{"role": "user", "content": repair_prompt}],
                    tools=None,
                    use_vl=False
                )
                response_text = repair_msg.content or ""
                match = re.search(r"<tool_call>(.*?)</tool_call>", response_text, re.DOTALL)
                if match:
                    call_info = json5.loads(match.group(1))
                    fn_name = call_info.get("name")
                    fn_args = call_info.get("arguments", {})

        # 4. 执行工具
        final_output = response_text
        already_spoken = False
        if fn_name:
            if fn_name in tools_instances:
                print(f"-> 执行本地工具: {fn_name}")
                self.memory._log_to_file(f"-> 触发工具: {fn_name} 参数: {fn_args}")
                observation = tools_instances[fn_name].call(json5.dumps(fn_args), user_query=query, agent=self)
                self.memory._log_to_file(f"<- 工具返回: {observation}")

                obs_data = json5.loads(observation)
                # --- 关键逻辑：判断是否直接回复 ---
                if obs_data.get("direct_reply") is True:
                    final_output = obs_data.get("response_text") or obs_data.get("message") or "操作已完成"
                    already_spoken = obs_data.get("already_spoken", False) # <--- 【新增】读取拦截标志
                    self.expect_followup = bool(obs_data.get("awaiting_full_name", False)) or self.awaiting_face_full_name
                    print(f"-> [跳过第二次模型调用] 直接回复: {final_output}")
                else:
                    # (原有的二次调用逻辑保持完全不变)
                    if self.mode == "cloud":
                        messages.append(msg_obj)
                        tool_msg = {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "name": fn_name,
                            "content": observation
                        }
                        messages.append(tool_msg)
                        final_res = self._call_llm(messages)
                    else:
                        feedback_str = f"\n\n【系统操作记录】：已调用工具 {fn_name}，执行结果为：{observation}。\n请根据该结果给用户简短的回复。"
                        messages[0]["content"] += feedback_str
                        final_res = self._call_llm(messages)
                    final_output = final_res.content
            else:
                error_msg = f"错误：模型尝试调用不存在的工具 '{fn_name}'"
                print(f"-> {error_msg}")
                final_output = "抱歉，我刚刚在处理指令时遇到了问题，请再说一次。"
        elif not str(final_output or "").strip():
            final_output = "抱歉，我没有收到有效工具指令，请再说一次。"

        self.memory._log_to_file(f"【最终回复】: {final_output}")

        if final_output:
            if is_vl_task:
                visual_record_user = f"[用户展示了一张图片，并问]：{query}"
                self.memory.add_turn("user", visual_record_user, self)
                self.memory.add_turn("assistant", final_output, self)
            else:
                self.memory.add_turn("user", query, self)
                self.memory.add_turn("assistant", final_output, self)
                
        return "" if already_spoken else final_output

def main():
    # USE_MODE = "cloud"
    USE_MODE = "cloud"
    agent = UniversalAgent(
        mode=USE_MODE,
        local_url='http://localhost:8080/v1',  # RK3588 的实际 IP
        # api_key="sk-8fe9d5ff83fc42acb8e52b7da9e4b9f0",
        api_key="sk-ff528950477e421999763986692ce67e",
        x=1, y=4
    )

    test_queries = [
        # # --- 基础对话与身份 ---
        # "你是谁？", "你能帮我做什么？",
        # # --- 大类1：宠物识别与跟踪 ---
        # "请告诉我周围有没有猫？",          # pet_detection, cat
        # "帮我看看附近有没有狗。",          # pet_detection, dog
        # "帮我跟踪一下那只小猫。",          # pet_tracking, cat
        # "请停止追踪宠物。",              # stop_tracking
        # # --- 大类2：人脸识别与管理 ---
        # "请记住这是张三。",              # register_face, 张三
        # "帮我录入一下李四的面部信息。",     # register_face, 李四
        # "这是谁？",                     # recognize_face
        # "你看一下画面里的人是谁？",        # recognize_face
        # # --- 大类3：人物定位与跟踪 ---
        # "帮我找找张三在不在？",          # person_search, 张三
        # "帮我找一下李四。",              # person_search, 李四
        # "请跟着张三。",                 # person_tracking, 张三
        # "请停止追踪。",                 # stop_tracking
        # # --- 大类4：手势识别交互 ---
        # "你看我做的是什么手势？",          # gesture_identification
        # "能认出我现在的手势吗？",          # gesture_identification
        # # --- 大类5：仰卧起坐计数 ---
        # "请帮我数仰卧起坐。",            # start_counting
        # "我目前做了多少个了？",          # query_progress
        # "帮我查一下现在的计数进度。",     # query_progress
        # "我停止了，一共做了多少个？",     # stop_and_summarize
        # "我不做了，结算一下总数。",       # stop_and_summarize
        # # --- 混合场景测试 ---
        #t "帮我找找家里的猫，然后再看看张三在哪。", # 连续任务提取
        # "帮我记住王五，然后一直跟着他。",       # 跨类任务流
        "请你帮我按一下投食机。",              # household_control, feeder:start
        # "请帮我打开灯。",                    # household_control, light:on
        # "请你开始帮我数俯卧撑。",                    # fitness_counter, start_couning, pushup
    ]

    print(f"开始 RK3588 优化版工具测试...\n" + "="*50)

    # for i, query in enumerate(test_queries, 1):
    #     print(f"\n[用例 {i}]: {query}")
    #     result = agent.run_workflow(query)
    #     print(f"最终回复: {result}")
    #     if result:
    #         print(f"🔊 正在播报...")
    #         # 这里的 wait=True 确保说完后再进行下一个用例
    #         # agent.tts.speak(result, wait=True)
    #         print("-" * 30)

    for cur in range(len(test_queries)):
        result = agent.run_workflow(test_queries[cur])
        print(f"最终回复: {result}")
    print("\n所有用例测试完毕！")

if __name__ == '__main__':
    main()
