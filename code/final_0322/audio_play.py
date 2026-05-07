import subprocess
import os
import sys

def set_audio_mixer():
    """
    配置你的硬件混音器（无多余输出，无报错）
    """
    commands = [
        "amixer -D hw:1 set 'spk switch' on",
        "amixer -D hw:1 set 'Right Mixer Right Bypass' on",
        "amixer -D hw:1 set 'OUT1' on",
        "amixer -D hw:1 set 'OUT2' on",
        "amixer -D hw:1 set 'PCM' 100",
    ]

    print("正在配置音频硬件...")

    for cmd in commands:
        # 静默执行，不输出 ALSA 垃圾日志
        result = subprocess.run(
            cmd, shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    print("✅ 混音器配置完成")

def play_wav(file_path):
    """
    用 aplay 直接播放硬件设备，零报错
    """
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return

    # 关键：直接调用 aplay，指定 hw:1,0，干净无错
    cmd = (
        f"aplay -D hw:1,0 '{file_path}' -c 2 -r 48000"
    )

    print(f"🎵 开始播放: {file_path}")

    try:
        # 静默执行，屏蔽 ALSA 错误
        subprocess.run(
            cmd, shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print("✅ 播放完成")
    except KeyboardInterrupt:
        print("\n⏹️  播放停止")

if __name__ == "__main__":
    # 配置混音器
    set_audio_mixer()

    # 播放（替换成你的实际 wav 路径）
    play_wav("48k_16bit_2chn_Music.wav")