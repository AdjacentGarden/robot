import subprocess
import os
import wave
import numpy as np

REC_FILE = "/home/test/48k_16bit_2chn_Music.wav"

def record_audio(duration=5):
    """
    使用 arecord 录音
    """
    cmd = [
        "arecord",
        "-D", "hw:rockchipi2sdmic",
        "-r", "48000",
        "-c", "2",
        "-f", "S16_LE",
        "-d", str(duration),
        REC_FILE,
        "-vv"
    ]
    print(">>> 开始录音...")
    subprocess.run(cmd, check=True)
    print(">>> 录音完成")


def play_audio():
    """
    使用 aplay 播放
    """
    cmd = [
        "aplay",
        "-D", "hw:rockchiptas6424",
        REC_FILE,
        "-vv"
    ]
    print(">>> 开始播放...")
    subprocess.run(cmd, check=True)
    print(">>> 播放完成")


def analyze_audio():
    """
    检查录音是否有效（是否有声音）
    """
    if not os.path.exists(REC_FILE):
        print("[错误] 录音文件不存在")
        return

    with wave.open(REC_FILE, 'rb') as wf:
        n_channels = wf.getnchannels()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()

        audio_data = wf.readframes(n_frames)
        audio_np = np.frombuffer(audio_data, dtype=np.int16)

        volume = np.abs(audio_np).mean()

        print("\n====== 音频信息 ======")
        print(f"声道数: {n_channels}")
        print(f"采样率: {framerate}")
        print(f"帧数: {n_frames}")
        print(f"平均音量: {volume:.2f}")

        if volume < 50:
            print("⚠️ 基本没声音（可能麦克风没采到）")
        else:
            print("✅ 录音正常（检测到声音）")


if __name__ == "__main__":
    record_audio(duration=5)
    analyze_audio()
    play_audio()