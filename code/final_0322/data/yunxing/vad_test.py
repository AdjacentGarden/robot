import wave
import os
from collections import deque
from vad_module import SileroVAD
from pydub import AudioSegment
import io
# --- 配置区 ---
INPUT_WAV = "test.wav"  # 你的原始测试音频
OUTPUT_DIR = "vad_results"         # 切分后的存储目录
MIN_SILENCE_MS = 800              # 停顿超过1秒则切分
THRESHOLD = 0.5                    # 灵敏度

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def save_wav(frames, filename, sample_rate=16000):
    """保存音频切片"""
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2) # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(frames))
    print(f" [文件已保存]: {filename}")

def run_test():
    # 1. 初始化 VAD
    vad = SileroVAD(min_silence_ms=MIN_SILENCE_MS, threshold=THRESHOLD)
    
    # 2. 【修改处】：使用 pydub 加载并强制转换格式
    print(f"正在转换音频格式: {INPUT_WAV} ...")
    try:
        audio = AudioSegment.from_file(INPUT_WAV)
        # 强制转为 16000Hz, 单声道, 16-bit PCM
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        raw_bytes = audio.raw_data
    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    # 3. 变量初始化
    pre_roll_buffer = deque(maxlen=15) # 预录缓冲 (约480ms)
    recorded_audio = []
    is_recording = False
    segment_count = 0
    
    print(f"开始分析处理后的音频 (16k/Mono) ...")
    
    # 【修改处】：步长改为 1024 字节 (即 512 个采样点 * 2字节)
    chunk_samples = 512
    chunk_bytes_len = chunk_samples * 2
    
    for i in range(0, len(raw_bytes), chunk_bytes_len):
        data = raw_bytes[i : i + chunk_bytes_len]
        # 最后一帧如果长度不足 1024，补零对齐（VAD 要求严格 512 采样）
        if len(data) < chunk_bytes_len:
            data = data.ljust(chunk_bytes_len, b'\x00')
        
        # 预录缓冲始终保持
        pre_roll_buffer.append(data)
        
        # 核心 VAD 处理
        is_speaking, is_final, prob = vad.process(data)
        # 每隔一段打印一下概率，看看停顿处概率有没有掉下来
        if i % (chunk_bytes_len * 10) == 0:
            print(f"当前人声概率: {prob:.4f} | 录音状态: {is_recording} | 正在说话: {is_speaking}")
            
        # 状态机逻辑
        if is_speaking and not is_recording:
            # 检测到开始说话
            segment_count += 1
            print(f">>> 发现第 {segment_count} 段语音开始...")
            is_recording = True
            # 把预录的“开头”加进去，防止吞音
            recorded_audio.extend(list(pre_roll_buffer))
            pre_roll_buffer.clear()
            
        elif is_recording:
            recorded_audio.append(data)
            
            if is_final:
                # 检测到长停顿，切分文件
                print(f"<<< 第 {segment_count} 段语音结束（停顿超过 {MIN_SILENCE_MS}ms）")
                filename = os.path.join(OUTPUT_DIR, f"segment_{segment_count}.wav")
                save_wav(recorded_audio, filename)
                
                # 重置录音状态，准备下一段
                recorded_audio = []
                is_recording = False
                vad.reset()

    # 如果录音到文件末尾还没结束，强制保存最后一段
    if is_recording and recorded_audio:
        filename = os.path.join(OUTPUT_DIR, f"segment_{segment_count}_final.wav")
        save_wav(recorded_audio, filename)

    print("\n测试完成！请在 vad_results 文件夹查看切片。")

if __name__ == "__main__":
    run_test()