import pyaudio
import numpy as np

p = pyaudio.PyAudio()
# 使用双声道 16000Hz 采样
stream = p.open(format=pyaudio.paInt16, channels=2, rate=16000, input=True, frames_per_buffer=1024)

print("--- 实时音量监测 (Ctrl+C 退出) ---")
try:
    while True:
        data = stream.read(1024, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16)
        # 分别计算左、右声道的最大振幅
        left = np.abs(audio_data[0::2]).max()
        right = np.abs(audio_data[1::2]).max()
        print(f"L: {'█' * (left // 1000)} ({left}) | R: {'█' * (right // 1000)} ({right})", end='\r')
except KeyboardInterrupt:
    print("\n监测结束")
    stream.stop_stream()
    stream.close()
    p.terminate()