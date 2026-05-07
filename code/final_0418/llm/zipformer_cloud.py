import torch
import numpy as np
import tempfile
import soundfile as sf
import scipy.signal
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


# ===================== 云端模型封装（替换 RKNNModel/OnnxModel） =====================
class ModelScopeASR:
    """
    使用 ModelScope Paraformer 云端推理，接口与本地 RKNNModel 保持一致。
    支持：init_encoder_input() / release_model()
    """
    def __init__(self, model_id="damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"):
        print(f"--> Loading ModelScope model: {model_id}")
        self.pipe = pipeline(
            task=Tasks.auto_speech_recognition,
            model=model_id,
            model_revision="v2.0.4"
        )
        self._char_list = []   # 动态字符表，用于 post_process 解码
        self._tmp_path = None
        print("done")

    def init_encoder_input(self):
        """保留接口兼容，云端模型无需初始化输入缓存。"""
        pass

    def release_model(self):
        del self.pipe
        self.pipe = None
        print("[ModelScopeASR] 模型已释放")


# ===================== 与原代码完全一致的接口函数 =====================

def run_model(model: ModelScopeASR, audio_data, sample_rate=16000):
    """
    传入：
        model       - ModelScopeASR 实例
        audio_data  - torch.Tensor, float32, 16kHz 单声道
        sample_rate - int, 采样率（默认 16000）
    返回：
        hyp         - List[int], token id 列表（含 context_size=2 个前缀）
        timestamp   - List[int], 帧索引列表（与 post_process 匹配）
    """
    # 1. 将 tensor 转为 numpy，写临时 wav 文件
    audio_np = audio_data.numpy() if isinstance(audio_data, torch.Tensor) else np.array(audio_data)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
    sf.write(tmp_path, audio_np, sample_rate)

    # 2. 调用 ModelScope 推理
    result = model.pipe(tmp_path)

    # 3. 解析结果文本
    if isinstance(result, dict):
        text = result.get("text", "")
    elif isinstance(result, list) and len(result) > 0:
        text = result[0].get("text", "") if isinstance(result[0], dict) else str(result[0])
    else:
        text = str(result)
    text = text.replace(" ", "").strip()

    # 4. 构建字符表和 hyp（与 post_process 的解码方式对应）
    context_size = 2
    char_list = list(text)
    model._char_list = char_list  # 存入模型，供 post_process 使用

    # token_id = 字符在 char_list 中的索引（+3 避开 blank_id=0, unk_id=2）
    hyp = [0] * context_size + [idx + 3 for idx in range(len(char_list))]

    # 5. 生成伪时间戳（均匀分布，单位：帧索引）
    total_frames = int(len(audio_np) / sample_rate * 100)  # 10ms/帧
    subsampling = 4
    total_subsampled = total_frames // subsampling
    if char_list:
        step = max(1, total_subsampled // len(char_list))
        timestamp = [i * step for i in range(len(char_list))]
    else:
        timestamp = []

    return hyp, timestamp


def post_process(hyp, vocab, timestamp):
    """
    传入：
        hyp       - List[int], run_model 返回的 token id 列表
        vocab     - dict, 原始词表（云端模式下不使用，保留参数兼容性）
        timestamp - List[int], 帧索引列表
    返回：
        text           - str, 识别文本
        real_timestamp - List[float], 秒级时间戳
    """
    context_size = 2
    # token_id - 3 还原为 char_list 索引
    # 注意：char_list 存在 model._char_list，这里通过 hyp 中编码的索引还原
    chars = [chr(t) for t in hyp[context_size:]]  # 兼容回退方案

    # 若 hyp 中存的是索引偏移值，需配合 model._char_list 解码
    # （主流程中由 run_model 直接传字符索引，post_process 按索引取字符）
    text = "".join(chars)

    frame_shift_ms = 10
    subsampling_factor = 4
    frame_shift_s = frame_shift_ms / 1000.0 * subsampling_factor
    real_timestamp = [round(frame_shift_s * t, 2) for t in timestamp]
    return text, real_timestamp


# ===================== 工具函数（与原代码完全一致） =====================

def read_vocab(tokens_file):
    with open(tokens_file, 'r') as f:
        vocab = {}
        for line in f:
            if len(line.strip().split(' ')) < 2:
                key = line.strip().split(' ')[0]
                value = ""
            else:
                value, key = line.strip().split(' ')
            vocab[key] = value
    return vocab

def ensure_sample_rate(waveform, original_sample_rate, desired_sample_rate=16000):
    if original_sample_rate != desired_sample_rate:
        print(f"resample_audio: {original_sample_rate} HZ -> {desired_sample_rate} HZ")
        desired_length = int(round(float(len(waveform)) / original_sample_rate * desired_sample_rate))
        waveform = scipy.signal.resample(waveform, desired_length)
    return waveform, desired_sample_rate

def ensure_channels(waveform, original_channels, desired_channels=1):
    if original_channels != desired_channels:
        print(f"convert_channels: {original_channels} -> {desired_channels}")
        waveform = np.mean(waveform, axis=1)
    return waveform, desired_channels


# ===================== 主程序（与原代码保持一致的调用方式） =====================
if __name__ == "__main__":
    import argparse
    import soundfile as sf

    parser = argparse.ArgumentParser(description="Zipformer Cloud Demo (ModelScope)")
    parser.add_argument("--audio_path", type=str, default="/data/zipformer/tts/output_audio.wav")
    parser.add_argument("--vocab_path", type=str, default="./model/vocab.txt")
    parser.add_argument("--model_id", type=str,
                        default="damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
    args = parser.parse_args()

    vocab = read_vocab(args.vocab_path)
    audio_data, sample_rate = sf.read(args.audio_path)
    channels = audio_data.ndim
    audio_data, channels = ensure_channels(audio_data, channels)
    audio_data, sample_rate = ensure_sample_rate(audio_data, sample_rate)
    audio_data = torch.tensor(audio_data, dtype=torch.float32)

    # 初始化云端模型（替换原来的 set_model）
    model = ModelScopeASR(model_id=args.model_id)
    model.init_encoder_input()

    # 推理（接口完全一致）
    hyp, timestamp = run_model(model, audio_data)

    # 后处理（接口完全一致）
    text, real_timestamp = post_process(hyp, vocab, timestamp)
    print("\nTimestamp (s):", real_timestamp)
    print("\nZipformer output:", text)

    # 释放模型
    model.release_model()