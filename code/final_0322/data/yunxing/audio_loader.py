import os
from pydub import AudioSegment

def get_audio_stream(file_path, chunk_samples=512):
    """
    将任意音频文件转换为 16k/16bit/单声道的字节流生成器
    chunk_samples: 每次返回的采样点数 (Silero 要求 512)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到音频文件: {file_path}")

    # 1. 转换格式
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    raw_bytes = audio.raw_data
    
    # 每个 chunk 的字节数 (16bit = 2 bytes)
    chunk_bytes_len = chunk_samples * 2
    
    # 2. 模拟流式输出 (Generator)
    for i in range(0, len(raw_bytes), chunk_bytes_len):
        chunk = raw_bytes[i : i + chunk_bytes_len]
        
        # 补零对齐
        if len(chunk) < chunk_bytes_len:
            chunk = chunk.ljust(chunk_bytes_len, b'\x00')
            
        yield chunk