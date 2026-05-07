import os
import re
import threading
import subprocess
import requests
import dashscope

class QwenTTSModule:
    def __init__(self, api_key, log_func=None):
        self.api_key = api_key
        self.log = log_func
        dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'
        # 确保模型名称准确
        self.model = "qwen3-tts-instruct-flash" 
        self.voice = "Stella"
        self.pulse_env = "PULSE_SERVER=unix:/run/user/1000/pulse/native "

    def _speak_task(self, text):
        clean_text = re.sub(r'[*#>`\-]', '', text).replace('"', ' ')
        # 建议先用 /tmp 目录测试，排除磁盘权限问题
        save_dir = "/data/zipformer/tts"
        tmp_file = os.path.join(save_dir, "output_audio.wav")
        
        # 自动创建目录
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        try:
            print(f"[TTS] 正在请求阿里 API: {clean_text[:10]}...")
            response = dashscope.MultiModalConversation.call(
                model=self.model,
                api_key=self.api_key,
                text=clean_text,
                voice=self.voice,
                # instructions='语速快速，带有明显的上扬语调，适合介绍时尚产品。',
                # optimize_instructions=True,
            )

            if response.status_code == 200:
                # 提取音频 URL
                audio_url = response.output.audio.url
                print(f"[TTS] 获取到 URL: {audio_url}")
                
                # 下载音频
                audio_res = requests.get(audio_url)
                if audio_res.status_code == 200:
                    with open(tmp_file, 'wb') as f:
                        f.write(audio_res.content)
                    print(f"[TTS] 文件已保存至: {tmp_file} (大小: {len(audio_res.content)} bytes)")
                    
                    # 1. 自动初始化音量（防止重启后变回最大）
                    # 将数字增益降到 130，将硬件输出降到 22
                    init_vol_cmd = (
                        "amixer -c 1 cset numid=31 130 > /dev/null && "
                        "amixer -c 1 cset numid=34 22 > /dev/null"
                        "amixer -c 1 cset numid=35 22 > /dev/null"
                    )
                    subprocess.run(init_vol_cmd, shell=True)
                    # 播放指令
                    play_cmd = f"{self.pulse_env} aplay -D plughw:1,0 -q {tmp_file}"
                    subprocess.run(play_cmd, shell=True)
                else:
                    print(f"[TTS] 下载音频失败，状态码: {audio_res.status_code}")
            else:
                print(f"[TTS] API 报错信息: {response.message}")
                print(f"[TTS] 完整响应: {response}")
                
        except Exception as e:
            print(f"[TTS] 运行异常: {e}")

    def speak(self, text, wait=True):
        if not text: return
        t = threading.Thread(target=self._speak_task, args=(text,), daemon=True)
        t.start()
        # --- 核心修改：如果需要等待，就调用 join() ---
        if wait:
            t.join()
    
if __name__ == "__main__":
    # 替换为你真实的 API_KEY
    tts = QwenTTSModule(api_key="sk-8fe9d5ff83fc42acb8e52b7da9e4b9f0")
    
    print("--- 开始测试播报 ---")
    tts.speak("你好，我已经在 RK3588 上复活了，现在声音正常吗？")
    
    # 关键：阻塞主线程，等待后台线程完成下载和播放
    print("正在后台处理中，请不要关闭程序...")
    input("\n[按回车键结束测试]\n")