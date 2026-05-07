import wave
with wave.open('segment_1.wav', 'rb') as f:
    print('采样率:', f.getframerate())
    print('声道数:', f.getnchannels())
    print('位深度:', f.getsampwidth() * 8, 'bit')
    print('总帧数:', f.getnframes())
    print('时长:', f.getnframes() / f.getframerate(), '秒')