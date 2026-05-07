import sherpa_onnx
import numpy as np
import time
import scipy.signal
import os
import wave
import tempfile
import subprocess
import librosa


# ──────────────────────────────────────────────
# 公共播放函数（替代 sounddevice，走 aplay 硬件通道）
# ──────────────────────────────────────────────

def _init_mixer(volume: int = 100):
    """初始化 RK3588 板载混音器"""
    commands = [
        "amixer -D hw:1 set 'spk switch' on",
        "amixer -D hw:1 set 'Right Mixer Right Bypass' on",
        "amixer -D hw:1 set 'OUT1' on",
        "amixer -D hw:1 set 'OUT2' on",
        f"amixer -D hw:1 set 'PCM' {volume}",
    ]
    for cmd in commands:
        subprocess.run(cmd, shell=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _play_wav(path: str, volume: int = 140):
    """
    用 aplay 播放 wav 文件（阻塞，播完才返回）。
    wav 文件必须是 48kHz / 16bit / 双声道。
    """
    _init_mixer(volume)
    cmd = f"aplay -D hw:1,0 '{path}' -c 2 -r 48000"
    subprocess.run(cmd, shell=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _save_wav(samples, sample_rate: int, path: str):
    """
    float32 音频 → 48kHz / 16bit / 双声道 wav 文件。
    匹配 aplay -D hw:1,0 -c 2 -r 48000 的要求。
    """
    target_sr = 48000

    # 1. 重采样到 48kHz
    if sample_rate != target_sr:
        num = int(len(samples) * target_sr / sample_rate)
        samples = scipy.signal.resample(samples, num)

    # 2. float32 → int16
    samples = np.clip(np.array(samples), -1.0, 1.0)
    samples_int16 = (samples * 32767).astype(np.int16)

    # 3. 单声道 → 双声道
    stereo = np.column_stack([samples_int16, samples_int16])

    # 4. 写文件
    with wave.open(path, 'w') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(target_sr)
        wf.writeframes(stereo.tobytes())


# ──────────────────────────────────────────────
# TTS 引擎 1：Piper VITS（中文 huayan）
# ──────────────────────────────────────────────

class LocalTTS:
    def __init__(self, model_dir):
        config = sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=f"{model_dir}/zh_CN-huayan-medium.onnx",
                tokens=f"{model_dir}/tokens.txt",
                data_dir=f"{model_dir}/espeak-ng-data",
            ),
            num_threads=4,
            debug=False,
        )
        self.tts = sherpa_onnx.OfflineTts(
            sherpa_onnx.OfflineTtsConfig(model=config)
        )
        self._tmp_wav = os.path.join(tempfile.gettempdir(), "tts_vits.wav")
        print("--- [本地 TTS] VITS Piper 引擎启动成功 ---")

    def speak(self, text, wait=True, speaker_id=0):
        start_time = time.time()

        audio = self.tts.generate(text, sid=speaker_id)
        if not audio or len(audio.samples) == 0:
            return

        latency = (time.time() - start_time) * 1000
        print(f"✅ VITS 合成完成! 采样率: {audio.sample_rate}Hz, 耗时: {latency:.2f} ms")

        _save_wav(audio.samples, audio.sample_rate, self._tmp_wav)
        _play_wav(self._tmp_wav)


# ──────────────────────────────────────────────
# TTS 引擎 2：ZipVoice 音色克隆
# ──────────────────────────────────────────────

class LocalTTS2:
    def __init__(self, model_dir):
        config = sherpa_onnx.OfflineTtsModelConfig(
            zipvoice=sherpa_onnx.OfflineTtsZipvoiceModelConfig(
                tokens=os.path.join(model_dir, "tokens.txt"),
                encoder=os.path.join(model_dir, "encoder.int8.onnx"),
                decoder=os.path.join(model_dir, "decoder.int8.onnx"),
                vocoder=os.path.join(model_dir, "vocos_24khz.onnx"),
                data_dir=os.path.join(model_dir, "espeak-ng-data"),
                lexicon=os.path.join(model_dir, "lexicon.txt"),
            ),
            num_threads=4,
            debug=False,
        )
        self.tts = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(model=config))
        self._tmp_wav = os.path.join(tempfile.gettempdir(), "tts_zipvoice.wav")

        # 预加载音色特征
        print("正在预加载音色特征...")
        refer_wav = os.path.join(model_dir, "test_wavs/news-female-2.wav")
        audio_samples, sample_rate = librosa.load(refer_wav, sr=None)
        self.generation_config = sherpa_onnx.GenerationConfig()
        self.generation_config.reference_audio = audio_samples
        self.generation_config.reference_sample_rate = sample_rate
        self.generation_config.reference_text = (
            "本台消息, 中共中央国务院, 近日印发关于构建数据基础制度, "
            "更好发挥数据要素作用的意见."
        )
        self.generation_config.num_steps = 4
        print("--- [本地 TTS] ZipVoice 引擎启动成功 ---")

    def speak(self, text, wait=True):
        start_time = time.time()

        audio = self.tts.generate(text, self.generation_config)
        if not audio or len(audio.samples) == 0:
            return

        latency = (time.time() - start_time) * 1000
        print(f"✅ ZipVoice 合成完成! 耗时: {latency:.2f} ms")

        _save_wav(audio.samples, audio.sample_rate, self._tmp_wav)
        _play_wav(self._tmp_wav)


# ──────────────────────────────────────────────
# TTS 引擎 3：Matcha（你当前在用的）
# ──────────────────────────────────────────────

class MatchaTTS:
    def __init__(self, model_dir=None):
        if model_dir is None:
            curr_dir = os.path.dirname(os.path.abspath(__file__))
            model_dir = os.path.abspath(
                os.path.join(curr_dir, "..", "..", "final_0322", "data", "yunxing", "models", "tts", "matcha-icefall-zh-en")
            )
        acoustic_model = os.path.join(model_dir, "model-steps-3.onnx")
        vocoder = os.path.join(model_dir, "vocos-16khz-univ.onnx")
        tokens = os.path.join(model_dir, "tokens.txt")
        lexicon = os.path.join(model_dir, "lexicon.txt")
        data_dir = os.path.join(model_dir, "espeak-ng-data")

        for p in (acoustic_model, vocoder, tokens, lexicon):
            if not os.path.exists(p):
                raise FileNotFoundError(f"TTS model file not found: {p}")

        config = sherpa_onnx.OfflineTtsModelConfig(
            matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
                acoustic_model=acoustic_model,
                vocoder=vocoder,
                tokens=tokens,
                lexicon=lexicon,
                data_dir=data_dir,
            ),
            num_threads=4,
            debug=False,
        )
        self.tts = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(model=config))
        self._tmp_wav = os.path.join(tempfile.gettempdir(), "tts_matcha.wav")
        print("--- [本地 TTS] Matcha 引擎启动成功 ---")

    def speak(self, text, wait=True, sid=0):
        start_time = time.time()

        audio = self.tts.generate(text, sid=sid)
        if not audio or len(audio.samples) == 0:
            return

        latency = (time.time() - start_time) * 1000
        print(f"✅ Matcha 合成完成! 采样率: {audio.sample_rate}Hz, 耗时: {latency:.2f} ms")

        _save_wav(audio.samples, audio.sample_rate, self._tmp_wav)
        _play_wav(self._tmp_wav)


# ──────────────────────────────────────────────
# 本地测试
# ──────────────────────────────────────────────

if __name__ == "__main__":
    MODEL_PATH = "/data/yunxing/models/tts/matcha-icefall-zh-en"
    tts = MatchaTTS(MODEL_PATH)
    tts.speak("你好，我是基于 Matcha 架构的语音助理。Nice to meet you!")
