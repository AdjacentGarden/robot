import os
import wave
from collections import deque
from vad_module import SileroVAD
from audio_loader import get_audio_stream  # 导入解耦的读取函数

# --- 配置区 ---
INPUT_FILE = "test.wav"
OUTPUT_DIR = "vad_results"
MIN_SILENCE_MS = 800
MIN_SEGMENT_SEC = 0.4  # 【过滤】短于0.4秒的分段认为是杂音，丢弃
THRESHOLD = 0.5

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def save_wav(frames, filename):
    """保存标准 16k/16bit/Mono WAV"""
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"".join(frames))
    print(f"  >>> [文件保存成功]: {filename}")

def run_vad_process():
    # 1. 初始化
    vad = SileroVAD(min_silence_ms=MIN_SILENCE_MS, threshold=THRESHOLD)
    pre_roll_buffer = deque(maxlen=15) # 480ms 预录
    recorded_audio = []
    is_recording = False
    segment_count = 0

    print(f"开始 VAD 监听处理...")

    # 2. 获取模拟流 (以后换成麦克风只需要改这一行)
    stream = get_audio_stream(INPUT_FILE)

    for chunk in stream:
        pre_roll_buffer.append(chunk)
        
        # 执行 VAD
        is_speaking, is_final, prob = vad.process(chunk)

        # 状态机：开始说话
        if is_speaking and not is_recording:
            segment_count += 1
            print(f"\n[{segment_count}] 检测到人声...")
            is_recording = True
            recorded_audio.extend(list(pre_roll_buffer))
            pre_roll_buffer.clear()
        
        # 状态机：正在录制
        elif is_recording:
            recorded_audio.append(chunk)
            
            # 状态机：检测到结束
            if is_final:
                # --- 增加时长校验，防止 segment_2 这种空文件 ---
                duration = (len(recorded_audio) * 512) / 16000
                
                if duration < MIN_SEGMENT_SEC:
                    print(f"  ! 丢弃无效分段: 时长 {duration:.2f}s 过短 (疑似杂音)")
                else:
                    print(f"  * 判定结束: 时长 {duration:.2f}s")
                    filename = os.path.join(OUTPUT_DIR, f"segment_{segment_count}.wav")
                    save_wav(recorded_audio, filename)
                
                # 重置状态
                recorded_audio = []
                is_recording = False
                vad.reset()

    # 处理文件结束后的收尾
    if is_recording and len(recorded_audio) > 0:
        duration = (len(recorded_audio) * 512) / 16000
        if duration >= MIN_SEGMENT_SEC:
            save_wav(recorded_audio, os.path.join(OUTPUT_DIR, f"segment_{segment_count}_final.wav"))

    print("\n所有任务处理完毕。")

if __name__ == "__main__":
    run_vad_process()