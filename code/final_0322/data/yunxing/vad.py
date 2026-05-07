import time
from collections import deque
from vad_module import SileroVAD

# 1. 初始化 VAD
vad = SileroVAD(threshold=0.5, min_silence_ms=1000)

# 2. 准备缓冲区
# 512个采样是31.25ms，16帧大约是500ms，足以覆盖说话前的语气和爆破音
pre_roll_buffer = deque(maxlen=16) 
recorded_audio = []
is_recording = False

print(">>> 机器人开始监听...")

while True:
    # 【假设】你从硬件读取 512 采样
    chunk = stream.read(512, exception_on_overflow=False)
    
    # 无论是否说话，先塞进环形缓冲区
    pre_roll_buffer.append(chunk)
    
    # 交给 VAD 处理
    is_speaking, is_final = vad.process(chunk)

    # 情况 A：检测到开始说话，但还没进入录音状态
    if is_speaking and not is_recording:
        print(">>> 检测到语音，开始保存...")
        is_recording = True
        # 【关键步骤】将预录缓冲区的内容全部“吐”出来作为开头
        recorded_audio.extend(list(pre_roll_buffer))
        pre_roll_buffer.clear() # 清空，防止重复

    # 情况 B：正在录音中
    elif is_recording:
        recorded_audio.append(chunk)
        
        # 情况 C：检测到用户说完（静默超时）
        if is_final:
            print("<<< 说话结束，保存音频中...")
            # 此时拼接所有 chunk，发送给同事的 ASR 模块
            full_wav_data = b"".join(recorded_audio)
            # save_to_file(full_wav_data) # 或者直接传给 ASR 函数
            
            # 重置状态，准备下一轮
            recorded_audio = []
            is_recording = False
            vad.reset()
            print(">>> 监听重置，等待下一条指令...")

    # (可选) 增加一个安全上限，比如录音超过 15 秒强制截断，防止背景噪音导致一直录音
    if len(recorded_audio) > 500: # 约 15 秒
         is_final = True