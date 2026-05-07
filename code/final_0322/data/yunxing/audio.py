import subprocess
import os
import sys

def set_audio_mixer(volume: int = 100):
    """
    配置音频混音器
    :param volume: 音量值，范围 0-190
    """
    # --- 1. 音量参数合法性检查 ---
    if not (0 <= volume <= 190):
        print(f"音量值 {volume} 无效，必须在 0-190 之间")
        sys.exit(1)

    # --- 2. 定义命令列表 ---
    commands = [
        "amixer -D hw:1 set 'spk switch' on",
        "amixer -D hw:1 set 'Right Mixer Right Bypass' on",
        "amixer -D hw:1 set 'OUT1' on",
        "amixer -D hw:1 set 'OUT2' on",
        f"amixer -D hw:1 set 'PCM' {volume}",  # 动态插入音量值
    ]

    print(f"正在配置音频硬件 (音量: {volume})...")

    # --- 3. 静默执行命令 ---
    for cmd in commands:
        subprocess.run(
            cmd, shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    print("混音器配置完成")

def play_wav(file_path: str, volume: int = 100):
    """
    播放 WAV 文件
    :param file_path: WAV 文件路径
    :param volume: 播放音量 (0-190)
    """
    # 1. 先设置混音器（包含音量）
    set_audio_mixer(volume)

    # 2. 检查文件
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return

    # 3. 播放命令
    cmd = f"aplay -D hw:1,0 '{file_path}' -c 2 -r 48000"
    print(f"开始播放: {file_path}")

    try:
        subprocess.run(
            cmd, shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print("播放完成")
    except KeyboardInterrupt:
        print("\n  播放停止")

if __name__ == "__main__":
    # ==========================================
    # 在这里修改音量值 (0-190)
    # ==========================================
    TARGET_VOLUME = 100  # 示例：设置为 100
    
    # 播放音频
    play_wav("48k_16bit_2chn_Music.wav", volume=TARGET_VOLUME)