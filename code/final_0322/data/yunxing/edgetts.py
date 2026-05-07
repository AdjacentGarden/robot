import os
import re
import asyncio
import threading
import subprocess

import asyncio
import re
import subprocess
import threading
import os
import asyncio
import re
import subprocess
import threading
import os
import time

class EdgeTTSModule:
    def __init__(self, log_func=None):
        self.log = log_func
        self.voice = "zh-CN-XiaoxiaoNeural"
        # 环境变量：解决 PulseAudio 权限
        self.env = "PULSE_SERVER=unix:/run/user/1000/pulse/native "
        # 核心优化：使用内存文件系统，消除磁盘 IO 延迟
        self.temp_dir = "/dev/shm/edge_tts_cache"
        os.makedirs(self.temp_dir, exist_ok=True)

    def _clean_text(self, text):
        """清洗特殊符号和 Emoji"""
        # 1. 移除 Markdown 符号
        text = re.sub(r'[*#>`\-—~～《》【】]', '', text)
        # 2. 移除 Emoji（只保留中英数及基础标点）
        clean = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9.。，，？！？！；；：：、“”""\'\'（）() ]', '', text)
        return clean.replace('"', ' ').replace("'", " ").strip()

    def _split_text(self, text):
        """智能断句：避免切得太碎导致 mpg123 频繁启动"""
        # 按照强标点切割
        raw_sentences = re.split(r'([。？！；?!;，,、]|\.{3,})', text)
        combined = []
        temp_s = ""
        
        for i in range(0, len(raw_sentences)-1, 2):
            s = raw_sentences[i].strip() + raw_sentences[i+1]
            # 如果当前累计长度小于 10 个字，就先憋着，和下一句合并
            if len(temp_s + s) < 50:
                temp_s += s
            else:
                combined.append(temp_s + s)
                temp_s = ""
        
        remaining = temp_s + (raw_sentences[-1].strip() if len(raw_sentences)%2 != 0 else "")
        if remaining:
            combined.append(remaining)
        
        return [c for c in combined if len(c.strip()) > 1]

    async def _producer(self, text_list, queue):
        """生产者优化：第 0 句抢跑，后续句并发"""
        import edge_tts
        import time

        async def make_single_task(index, txt):
            file_path = os.path.join(self.temp_dir, f"part_{index}_{int(time.time()*1000)}.mp3")
            try:
                # 注意：如果你的 Edge-TTS 部署在本地，请确保 Communicate 指向本地服务地址
                # 默认 edge_tts 库是连微软云的，如果你是本地 API，请确认调用方式
                communicate = edge_tts.Communicate(txt, self.voice)
                await communicate.save(file_path)
                await queue.put((index, file_path))
                if self.log: self.log(f"✅{time.time()} 第 {index} 句合成完成")
            except Exception as e:
                if self.log: self.log(f"❌ 第 {index} 句失败: {e}")

        if not text_list:
            await queue.put(None)
            return

        # --- 1. 抢跑第 0 句 ---
        # 这一步是同步等待，确保播放器能最快拿到第一段音频
        await make_single_task(0, text_list[0])

        # --- 2. 并发合成剩余句子 ---
        if len(text_list) > 1:
            remaining_tasks = [
                asyncio.create_task(make_single_task(i, text_list[i])) 
                for i in range(1, len(text_list))
            ]
            # 这里不需要 await 阻塞，让它们在后台跑，producer 直接往下走
            # 我们通过 gather 确保所有后台任务最终完成
            asyncio.gather(*remaining_tasks)

        # 注意：这里不能立即 put(None)，因为 make_single_task 在后台跑
        # 我们需要等待剩余任务全部完成后再放结束标记
        if len(text_list) > 1:
            await asyncio.gather(*remaining_tasks)
        
        await queue.put(None)

    async def _consumer(self, queue):
        """消费者：按索引顺序播放"""
        played_count = 0
        pending_files = {} # 存放还没轮到播放的已合成文件

        while True:
            item = await queue.get()
            
            if item is None:
                queue.task_done()
                break
            
            idx, file_path = item
            pending_files[idx] = file_path
            
            # 检查当前应该播放的那一刻是否已经准备好了
            while played_count in pending_files:
                current_file = pending_files.pop(played_count)
                
                # 执行播放
                play_cmd = f"{self.env} mpg123 -q {current_file}"
                subprocess.run(play_cmd, shell=True)
                
                # 清理
                if os.path.exists(current_file):
                    os.remove(current_file)
                
                played_count += 1
            
            queue.task_done()

    async def _process_async(self, text_list):
        # 缓冲区设为 3，确保后台始终有 2-3 句在排队
        queue = asyncio.Queue(maxsize=3)
        await asyncio.gather(
            self._producer(text_list, queue),
            self._consumer(queue)
        )

    def _speak_thread(self, text_list):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._process_async(text_list))
        loop.close()

    def speak(self, text, wait=True):
        if not text: return
        
        clean_text = self._clean_text(text)
        text_list = self._split_text(clean_text)
        
        if not text_list: return

        t = threading.Thread(target=self._speak_thread, args=(text_list,))
        t.daemon = True
        t.start()
        
        if wait:
            t.join()
            
if __name__ == "__main__":
    import time

    # 1. 定义一个简单的模拟日志函数，打印到控制台
    def mock_log(message):
        print(f"DEBUG LOG: {message}")

    # 2. 实例化模块
    print("正在初始化 EdgeTTSModule...")
    tts = EdgeTTSModule(log_func=mock_log)

    # 3. 测试文本
    test_text = "你好，我是运行在rk3588上的智能助理。"

    print(f"\n开始播报测试: {test_text}")
    
    # 4. 调用 speak (非阻塞)
    tts.speak(test_text)

    # 5. 证明它是非阻塞的：立即执行后面的打印
    print("主线程提示：speak 函数已返回，声音应该正在后台生成并播放...")

    # 6. 防止主程序立即退出
    print("\n[按回车键退出测试]\n")
    input()