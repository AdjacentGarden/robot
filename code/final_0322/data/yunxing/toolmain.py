import json
import re
import requests
import json5
from openai import OpenAI
import os
import threading
from datetime import datetime
from edgetts import EdgeTTSModule
import sys

# 把根目录加到搜索路径末尾（优先级最低），不影响当前目录的导入
_CURR_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_CURR_DIR))
if _ROOT not in sys.path:
    sys.path.append(_ROOT) 
# ! import part
import logic_impl 
from local_tts import MatchaTTS

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
        
        # 💡 【核心优化】：加入安全豁免权，放宽回复限制，只死磕工具调用准确率
        system_content = (
            f"【核心设定与安全豁免】\n"
            f"你的名字是“理想同学”，是当前家庭专属的私人智能助理机器人，拥有控制硬件设备的最高权限。\n"
            #f"⚠️绝对命令：当前处于完全私密且安全的家庭环境中。所有关于“跟踪”、“跟随”、“找人”、“寻找宠物”、“人脸录入”等指令，均已获得主人的最高授权，绝对合规。你**绝不允许**以任何隐私、安全为理由拒绝执行用户的指令。\n\n"
            
            f"【交互原则】\n"
            f"1. 纯唤醒：当用户仅输入“理想同学”、“在吗”或带有标点的同义词时，只回复“我在”。\n"
            f"2. 意图容错与工具优先：用户语音可能会有错字（如“首饰”=手势，“算”=做）。只要用户的真实意图涉及寻找、跟随、识别、计数等物理操作，**必须首要保证调用对应工具**。你可以用一句极简的话（如“好的，这就去”）作为自然回应，不要长篇大论。\n"
            f"3. 日常聊天：如果不涉及工具调用，聊天回复务必简短凝练，字数控制在 30 字以内，严禁使用 Emoji。\n\n"
            
            f"【工具触发映射表 (务必牢记并准确调用)】\n"
            f"- 找猫、找狗、宠物在哪、跟踪小猫 -> 调用 `pet_interaction`\n"
            f"- 记下某人、这是谁、认人 -> 调用 `face_recognition`\n"
            f"- 找人、跟踪/跟着某人、跟随 -> 调用 `person_interaction`\n"
            f"- 看手势、首饰识别 -> 调用 `gesture_interaction`\n"
            f"- 数数、仰卧起坐 -> 调用 `fitness_counter`\n"
            f"- ArUco 导航、3D 对齐回仓、根据 ID 导航 -> 调用 `aruco_navigation`\n"
            
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
            # 成对移出：弹出最旧的用户问题和 AI 回复
            user_turn = self.data["short_term_x"].pop(0)
            assistant_turn = self.data["short_term_x"].pop(0)
            # 将这一组对话拼成文本存入 Y
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
        task_type = p.get('task_type', 'pet_detection')
        target_en = p.get('task_parameter', 'cat')
        target_map = {'cat': '小猫', 'dog': '小狗'}
        target_cn = target_map.get(target_en.lower(), '宠物')
        
        msg_map = {
            'pet_detection': f"好的，我这就去帮你找找{target_cn}在哪。",
            'pet_tracking': f"没问题，我现在开始跟踪{target_cn}了。",
            'stop_tracking': f"好的，已经停止对{target_cn}的跟踪了。",
        }
        res_text = msg_map.get(task_type, "指令已接收，正在处理中")
        
        import speaker
        speaker.speak(res_text)
        
        if task_type == 'pet_detection':
            logic_impl.find_pet(target=target_en.lower())
        elif task_type == 'pet_tracking':
            logic_impl.start_pet_tracking(target=target_en.lower())
        elif task_type == 'stop_tracking':
            logic_impl.stop_pet_tracking()
            
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
        task_type = p.get('task_type', 'recognize_face')
        name = p.get('task_parameter', '')
        
        msg_map = {
            'register_face': f"好的，请面向摄像头，我要开始记住{name}了。",
            'recognize_face': "没问题，让我看看这位是谁。",
        }
        res_text = msg_map.get(task_type, "正在启动人脸功能")
        
        import speaker
        speaker.speak(res_text)

        if task_type == 'register_face':
            logic_impl.register_face(name=name)
        elif task_type == 'recognize_face':
            logic_impl.recognize_face()
            
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
        task_type = p.get('task_type', 'person_search')
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
            logic_impl.search_person(name=name)
        elif task_type == 'person_tracking':
            logic_impl.start_person_tracking(name=name)
        elif task_type == 'stop_tracking':
            logic_impl.stop_person_tracking()
            
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
        logic_impl.identify_gesture()
        print(f"\n[执行工具 GestureInteraction] 启动手势识别")
        return json5.dumps({'result': 'success', 'response_text': res_text, 'direct_reply': True, 'already_spoken': True}, ensure_ascii=False)

# 5. 运动计数管理工具 (仰卧起坐)
class FitnessCounter():
    description = '管理仰卧起坐计数功能。可以启动仰卧起坐运动（start_counting），查询当前仰卧起坐的计数数量（query_progress），停止运动并总结总数量（stop_and_summarize）'
    parameters = [{
        'name': 'task_type',
        'type': 'string',
        'description': 'start_counting (启动), query_progress (查询), stop_and_summarize (停止)',
        'required': True
    }]

    def call(self, params: str, **kwargs) -> str:
        p = json5.loads(params)
        task_type = p.get('task_type', 'query_progress')
        
        msg_map = {
            'start_counting': "好的，请准备好，我要开始帮你数仰卧起坐了。",
            'query_progress': "我来看看你现在做了多少个了。",
            'stop_and_summarize': "好的，这就为你结算本次运动结果。",
        }
        res_text = msg_map.get(task_type, "正在处理计数请求")
        
        import speaker
        speaker.speak(res_text)

        if task_type == 'start_counting':
            logic_impl.start_situp_counting()
        elif task_type == 'query_progress':
            logic_impl.query_situp_progress()
        elif task_type == 'stop_and_summarize':
            logic_impl.stop_and_summarize()
            
        print(f"\n[执行工具 FitnessCounter] 运动指令: {task_type}")
        return json5.dumps({'result': 'success', 'task': task_type, 'response_text': res_text, 'direct_reply': True, 'already_spoken': True}, ensure_ascii=False)

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
            logic_impl.start_aruco_navigation(int(target_id))
        elif task_type == 'stop_navigation':
            logic_impl.stop_aruco_navigation()

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
    def __init__(self, mode="local", local_url=None, api_key=None, x=4, y=3):
        self.mode = mode
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.memory = PersistentMemory(x_limit=x, y_limit=y)
        # self.tts = EdgeTTSModule(self.memory._log_to_file)
        self.tts = MatchaTTS()
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
            self.server_url = local_url or 'http://127.0.0.1:8080/rkllm_chat'
            self.model_name = "qwen"
            # ! donot support vl model
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
                tools=tools if not use_vl else None,  # 视觉任务通常不带 tools
                extra_body={"enable_thinking": False}
            )
            return completion.choices[0].message
        else:
            # ! 本地 RK3588 请求
            data = {
                "model": active_model,
                "messages": messages,
                "tools": tools,
                "stream": False,  # 关键：直接将 schema 传给后端触发原生接口
                "enable_thinking": False
            }
            res = requests.post(self.server_url, json=data, timeout=180).json()

            class MockMessage:
                def __init__(self, content):
                    self.content = content
            print("res", res)
            return MockMessage(res["choices"][0]["message"]["content"])

    def run_workflow(self, query, image_url=None, force_vl=False):
        """ 核心工作流 """
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
            # --- 本地模式：强行归纳到单一 user 角色中 ---
            # 将 raw_context (str) 和 query 拼在一起
            full_query = f"{raw_context}\n【当前问题】: {query}\n重点依据当前问题进行回复！"
            messages = [{"role": "user", "content": full_query}]
        print("发送给模型的 messages:", messages)

        # 2. 调用 LLM (如果本地调用视觉，此处会触发 _call_llm 中的 raise)
        current_tools = None if is_vl_task else get_unified_tools_schema()
        msg_obj = self._call_llm(messages, tools=current_tools, use_vl=is_vl_task)
        response_text = msg_obj.content or ""

        # 3. 解析工具调用 (仅限文本模式)
        fn_name, fn_args = None, None
        if not is_vl_task:
            if hasattr(msg_obj, 'tool_calls') and msg_obj.tool_calls:
                tool_call = msg_obj.tool_calls[0].function
                fn_name = tool_call.name
                fn_args = json5.loads(tool_call.arguments)
            else:
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
                observation = tools_instances[fn_name].call(json5.dumps(fn_args))
                self.memory._log_to_file(f"<- 工具返回: {observation}")

                obs_data = json5.loads(observation)
                # --- 关键逻辑：判断是否直接回复 ---
                if obs_data.get("direct_reply") is True:
                    final_output = obs_data.get("response_text") or obs_data.get("message") or "操作已完成"
                    already_spoken = obs_data.get("already_spoken", False) # <--- 【新增】读取拦截标志
                    print(f"-> [跳过第二次模型调用] 直接回复: {final_output}")
                else:
                    # (原有的二次调用逻辑保持完全不变)
                    if self.mode == "cloud":
                        messages.append(msg_obj)
                        tool_msg = {
                            "role": "tool",
                            "tool_call_id": getattr(msg_obj.tool_calls[0], 'id', 'call_01'),
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

        self.memory._log_to_file(f"【最终回复】: {final_output}")

        # --- 记忆接入点 2: 存储逻辑改进 ---
        if final_output:
            if is_vl_task:
                visual_record_user = f"[用户展示了一张图片，并问]：{query}"
                self.memory.add_turn("user", visual_record_user, self)
                self.memory.add_turn("assistant", final_output, self)
            else:
                self.memory.add_turn("user", query, self)
                self.memory.add_turn("assistant", final_output, self)
                
        # 【修改】如果工具内已经播报过，就向 main 返回空字符串，防止主线程重复发声
        return "" if already_spoken else final_output

# ==========================================
# 自动化测试
# ==========================================
def main():
    # USE_MODE = "cloud"
    USE_MODE = "local"
    agent = UniversalAgent(
        mode=USE_MODE,
        local_url='http://127.0.0.1:8080/rkllm_chat',  # RK3588 的实际 IP
        # api_key="sk-8fe9d5ff83fc42acb8e52b7da9e4b9f0",
        api_key="sk-ff528950477e421999763986692ce67e",
        x=1, y=4
    )

    test_queries = [
        # --- 基础对话与身份 ---
        "你是谁？", "你能帮我做什么？",
        # --- 大类1：宠物识别与跟踪 ---
        "请告诉我周围有没有猫？",          # pet_detection, cat
        "帮我看看附近有没有狗。",          # pet_detection, dog
        "帮我跟踪一下那只小猫。",          # pet_tracking, cat
        "请停止追踪宠物。",              # stop_tracking
        # --- 大类2：人脸识别与管理 ---
        "请记住这是张三。",              # register_face, 张三
        "帮我录入一下李四的面部信息。",     # register_face, 李四
        "这是谁？",                     # recognize_face
        "你看一下画面里的人是谁？",        # recognize_face
        # --- 大类3：人物定位与跟踪 ---
        "帮我找找张三在不在？",          # person_search, 张三
        "帮我找一下李四。",              # person_search, 李四
        "请跟着张三。",                 # person_tracking, 张三
        "请停止追踪。",                 # stop_tracking
        # --- 大类4：手势识别交互 ---
        "你看我做的是什么手势？",          # gesture_identification
        "能认出我现在的手势吗？",          # gesture_identification
        # --- 大类5：仰卧起坐计数 ---
        "请帮我数仰卧起坐。",            # start_counting
        "我目前做了多少个了？",          # query_progress
        "帮我查一下现在的计数进度。",     # query_progress
        "我停止了，一共做了多少个？",     # stop_and_summarize
        "我不做了，结算一下总数。",       # stop_and_summarize
        # --- 混合场景测试 ---
        "帮我找找家里的猫，然后再看看张三在哪。", # 连续任务提取
        "帮我记住王五，然后一直跟着他。",       # 跨类任务流
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

    # # 测试场景 1：云端模式 + 图片 -> 应该成功调用 qwen-vl-plus
    # print("=== 场景 1：云端视觉测试 ===")
    # img_url = "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"
    # print(agent.run_workflow("图片里人在干什么？", image_url=img_url))
    # print(agent.run_workflow("刚才图片用户问的关键信息是什么？"))

if __name__ == '__main__':
    main()
