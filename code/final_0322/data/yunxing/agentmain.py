import json
import re
import requests
import json5

# ==========================================
# 第一部分：工具实例初始化（保持你的类定义不变，直接实例化）
# ==========================================
# (这里假设你已经包含了上面提到的 MemberManager, FeatureTutorial 等类定义)
# 1. 成员管理工具
class MemberManager():
    description = '管理家庭成员信息，包括添加、删除或更改成员身份。'
    parameters = [{
        'name': 'action',
        'type': 'string',
        'description': '操作类型：add (添加), delete (删除), update (更改)',
        'required': True
    }, {
        'name': 'name',
        'type': 'string',
        'description': '家庭成员的名字',
        'required': True
    }, {
        'name': 'role',
        'type': 'string',
        'description': '成员的身份或角色（如：爸爸、妈妈、哥哥）',
        'required': False
    }]

    def call(self, params: str, **kwargs) -> str:
        p = json5.loads(params)
        # 防御性获取：如果模型漏掉参数，赋予默认值或占位符
        action = p.get('action', 'add') 
        name = p.get('name', '未知姓名')
        role = p.get('role', '未指定')
        
        print(f"\n[执行工具 MemberManager] 操作: {action}, 姓名: {name}, 角色: {role}")
        return json5.dumps({'result': 'success', 'message': f"已尝试完成对成员 {name} 的{action}操作"}, ensure_ascii=False)

# 2. 系统教学工具
class FeatureTutorial():
    description = '向用户介绍机器人的各项功能及其使用方法。'
    parameters = [{
        'name': 'feature_name',
        'type': 'string',
        'description': '用户想要了解的功能名称',
        'required': True
    }]

    def call(self, params: str, **kwargs) -> str:
        p = json5.loads(params)
        # 防御性获取
        feature = p.get('feature_name', '通用功能')
        
        print(f"\n[执行工具 FeatureTutorial] 介绍功能: {feature}")
        return json5.dumps({'result': 'success', 'content': f"好的，我来介绍一下{feature}功能……"}, ensure_ascii=False)

# 3. 扫描控制工具
class ScanControl():
    description = '控制机器人对家庭环境的扫描过程（暂停、继续、重新开始）。'
    parameters = [{
        'name': 'command',
        'type': 'string',
        'description': '扫描指令：pause (暂停), resume (继续), restart (重新扫描)',
        'required': True
    }]

    def call(self, params: str, **kwargs) -> str:
        p = json5.loads(params)
        # 防御性获取：默认暂停以防误操作
        cmd = p.get('command', 'pause')
        
        print(f"\n[执行工具 ScanControl] 扫描状态变更: {cmd}")
        msg_map = {
            'pause': "好的，那我之后再来熟悉家里的情况",
            'resume': "好的，我继续熟悉家里的情况了",
            'restart': "好的，我重新开始熟悉家里的情况了"
        }
        msg = msg_map.get(cmd, "指令已接收，正在处理中")
        return json5.dumps({'result': 'success', 'response_text': msg}, ensure_ascii=False)

# 4. 投影与会议控制工具
class ProjectionControl():
    description = '控制投影设备的开关、画面缩放及会议幻灯片翻页。'
    parameters = [{
        'name': 'action',
        'type': 'string',
        'description': '投影操作：open (开启), close (关闭), zoom_in (放大), zoom_out (缩小), next_page (下一张幻灯片)',
        'required': True
    }, {
        'name': 'content',
        'type': 'string',
        'description': '投影的内容描述',
        'required': False
    }]

    def call(self, params: str, **kwargs) -> str:
        p = json5.loads(params)
        # 重点防御：针对用例 11 的 KeyError 修复
        action = p.get('action', 'open') 
        content = p.get('content', '无')
        
        print(f"\n[执行工具 ProjectionControl] 动作: {action}, 内容: {content}")
        return json5.dumps({'result': 'success', 'action_performed': action}, ensure_ascii=False)

# 5. 运动健身工具
class ExerciseManager():
    description = '管理陪练模式，包括开始特定的运动项目和结束运动。'
    parameters = [{
        'name': 'action',
        'type': 'string',
        'description': '运动操作：start (开始), stop (停止)',
        'required': True
    }, {
        'name': 'exercise_type',
        'type': 'string',
        'description': '运动类型',
        'required': False
    }]

    def call(self, params: str, **kwargs) -> str:
        p = json5.loads(params)
        # 防御性获取
        action = p.get('action', 'stop')
        ex_type = p.get('exercise_type', '当前项目')
        
        print(f"\n[执行工具 ExerciseManager] 动作: {action}, 运动项目: {ex_type}")
        return json5.dumps({'result': 'success'}, ensure_ascii=False)

# 6. 智能家居/场景模式工具
class SmartHomeControl():
    description = '根据用户的生理感受或状态切换环境模式（如制冷、睡眠模式）。'
    parameters = [{
        'name': 'mode',
        'type': 'string',
        'description': '场景模式：cooling (开启空调/热了), sleep (睡觉/困了)',
        'required': True
    }]

    def call(self, params: str, **kwargs) -> str:
        p = json5.loads(params)
        # 防御性获取
        mode = p.get('mode', 'cooling')
        
        print(f"\n[执行工具 SmartHomeControl] 激活模式: {mode}")
        return json5.dumps({'result': 'success', 'activated_mode': mode}, ensure_ascii=False)
# 实例化所有工具并建立索引
tools_instances = {
    'member_manager': MemberManager(),
    'feature_tutorial': FeatureTutorial(),
    'scan_control': ScanControl(),
    'projection_control': ProjectionControl(),
    'exercise_manager': ExerciseManager(),
    'smart_home_control': SmartHomeControl()
}

# 提取所有工具的参数描述，供模型参考
TOOLS_SCHEMA = []
for name, inst in tools_instances.items():
    TOOLS_SCHEMA.append({
        "type": "function",
        "function": {
            "name": name,
            "description": inst.description,
            "parameters": {
                "type": "object",
                "properties": {p['name']: {"type": p['type'], "description": p['description']} for p in inst.parameters},
                "required": [p['name'] for p in inst.parameters if p.get('required', False)]
            }
        }
    })

# ==========================================
# 第二部分：RK3588 优化逻辑封装
# ==========================================

class RK3588Agent:
    def __init__(self, server_url):
        self.server_url = server_url
        self.headers = {'Content-Type': 'application/json'}
        # 构建极简的工具描述字符串
        self.tools_prompt = "\n".join([
            f"- {name}: {obj.description} 参数: {[p['name'] for p in obj.parameters]}" 
            for name, obj in tools_instances.items()
        ])

    def chat(self, messages):
        data = {
            "model": "qwen",
            "messages": messages,
            "stream": False,
            "enable_thinking": False
        }
        try:
            response = requests.post(self.server_url, json=data, timeout=60)
            res_json = response.json()
            if "choices" in res_json and len(res_json["choices"]) > 0:
                return res_json["choices"][0]["message"]["content"]
            else:
                return f"None (后端无响应数据)"
        except Exception as e:
            return f"Error: {e}"

    def run_workflow(self, query):
        # 针对 RK3588 后端跳过 system 的问题，将指令和工具定义合并入第一条 user 消息
        instruction = (
            f"你是一个执行任务的机器人。可选工具如下：\n{self.tools_prompt}\n"
            "规则：若需调用工具，必须回复格式：<tool_call>{'name':'工具名', 'arguments':{...}}</tool_call>\n"
            f"用户请求：{query}"
        )
        
        messages = [{"role": "user", "content": instruction}]

        # 1. 意图识别
        response_content = self.chat(messages)
        print(f"模型原始输出: {response_content}")

        # 2. 提取工具调用
        match = re.search(r"<tool_call>(.*?)</tool_call>", response_content, re.DOTALL)
        
        if not match:
            return response_content

        # 3. 执行工具
        try:
            call_info = json5.loads(match.group(1))
            fn_name = call_info.get("name")
            fn_args = call_info.get("arguments", {})
            
            if fn_name in tools_instances:
                observation = tools_instances[fn_name].call(json5.dumps(fn_args))
                
                # 4. 反馈结果
                messages.append({"role": "assistant", "content": response_content})
                
                # 优化点：给模型一个非常强烈的指令，让它切换到“总结模式”，严禁再次输出 <tool_call>
                feedback_prompt = (
                    f"工具执行结果: {observation}。\n"
                    "注意：任务已完成，请不要再次调用工具。请直接用一句简短的中文告诉用户事情已经办好了。"
                )
                messages.append({"role": "user", "content": feedback_prompt})
                
                # 5. 第二次请求
                final_answer = self.chat(messages)
                
                # 再次防御：如果模型还是不听话输出了标签，我们手动清理掉，只给用户看干净的文字
                clean_answer = re.sub(r"<tool_call>.*?</tool_call>", "", final_answer, flags=re.DOTALL).strip()
                return clean_answer if clean_answer else "好的，我已经为您处理完毕了。"
            else:
                return f"未找到工具: {fn_name}"
        except Exception as e:
            return f"解析或执行错误: {e}"

# ==========================================
# 第三部分：自动化测试
# ==========================================

def main():
    RK_SERVER_URL = 'http://127.0.0.1:8080/rkllm_chat'
    agent = RK3588Agent(RK_SERVER_URL)

    test_queries = [
        "你是谁？",
        "添加一位家庭成员，名字叫张三，他是我的哥哥",          # member_manager (add + name + role)
        "把张三的身份改成亲戚",                              # member_manager (update)
        "删除成员张三",                                     # member_manager (delete)
        "我想了解一下你的运动陪练功能是怎么用的",                # feature_tutorial
        "我刚才看到你在扫描客厅，先暂停一下吧",                  # scan_control (pause)
        "好了，你可以继续扫描了",                             # scan_control (resume)
        "感觉地图不太准，重新扫描一次家里吧",                    # scan_control (restart)
        "帮我投一下手机上的视频会议",                          # projection_control (open + content)
        "画面太小了，调大一点",                               # projection_control (zoom_in)
        "帮我翻到下一张幻灯片",                               # projection_control (next_page)
        "投影可以关掉了",                                    # projection_control (close)
        "陪我做一组仰卧起坐",                                 # exercise_manager (start + type)
        "今天就练到这里吧，停下来",                            # exercise_manager (stop)
        "我有点热了",                                       # smart_home_control (cooling)
        "我准备去睡觉了，好困",                               # smart_home_control (sleep)
        "给我讲个笑话",
        "我有想出去玩了"
    ]

    print(f"开始 RK3588 优化版工具测试...\n" + "="*50)

    for i, query in enumerate(test_queries, 1):
        print(f"\n[用例 {i}]: {query}")
        result = agent.run_workflow(query)
        print(f"最终回复: {result}")
        print("-" * 30)

if __name__ == '__main__':
    main()