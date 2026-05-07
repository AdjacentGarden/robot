import os
import torch
import numpy as np
import threading
from collections import deque
import time
from pydub import AudioSegment

# 导入你的模块
from vad_module import SileroVAD
from zipformer import set_model, run_model, post_process, read_vocab
from toolmain import UniversalAgent

# --- 配置 ---
class Config:
    ENCODER_PATH = "./model/encoder-epoch-99-avg-1.rknn"
    DECODER_PATH = "./model/decoder-epoch-99-avg-1.rknn"
    JOINER_PATH = "./model/joiner-epoch-99-avg-1.rknn"
    VOCAB_PATH = "./model/vocab.txt"
    LLM_MODE = "cloud"
    LLM_LOCAL = 'http://127.0.0.1:8080/rkllm_chat'
    # API_KEY = "sk-8fe9d5ff83fc42acb8e52b7da9e4b9f0"
    API_KEY = "sk-ff528950477e421999763986692ce67e"
    MIN_SILENCE_MS = 800
    MIN_SEGMENT_SEC = 0.2


def convert_to_16k_mono(input_path, output_path="temp_16k.wav"):
    audio = AudioSegment.from_wav(input_path)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    
    # 补 1 秒静音，确保 VAD 能检测到语音结束
    silence = AudioSegment.silent(duration=1000, frame_rate=16000)
    audio = audio + silence
    
    audio.export(output_path, format="wav")
    print(f"转换完成：{input_path} → {output_path}")
    return output_path



class RobotAssistant:
    def __init__(self):
        print("--- [系统初始化] 机器人启动 (多线程防堆积模式) ---")
        self.vad = SileroVAD(min_silence_ms=Config.MIN_SILENCE_MS)
        self.pre_roll_buffer = deque(maxlen=15)
        self.recorded_audio = []
        self.is_recording = False
        self.is_busy = False

        # 初始化 ASR
        class Args:
            encoder_model_path = Config.ENCODER_PATH
            decoder_model_path = Config.DECODER_PATH
            joiner_model_path = Config.JOINER_PATH
            target = "rk3588"
            device_id = None

        self.asr_vocab = read_vocab(Config.VOCAB_PATH)
        self.asr_model = set_model(Args())
        self.asr_model.init_encoder_input()
        self.agent = UniversalAgent(
            mode=Config.LLM_MODE,
            local_url=Config.LLM_LOCAL,
            api_key=Config.API_KEY,
            x=1,
            y=4
        )

    def _think_and_speak_thread(self, audio_data_list):
        """后台线程：负责 ASR -> LLM -> TTS"""
        try:
            print("开始时间：", time.time())
            # 1. ASR 识别
            audio_bytes = b"".join(audio_data_list)
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            audio_tensor = torch.tensor(audio_np, dtype=torch.float32)

            hyp, timestamp = run_model(self.asr_model, audio_tensor, sample_rate=16000)
            text, _ = post_process(hyp, self.asr_vocab, timestamp)

            if text.strip():
                print(f"\n[用户]: {text}")
                # 2. LLM 思考
                response = self.agent.run_workflow(text)
                print(f"[机器人]: {response}")
                # 3. TTS 播报
                if response:
                    self.agent.tts.speak(response, wait=True)

        except Exception as e:
            print(f"[系统错误]: {e}")

        finally:
            print("\n>>> 回复完毕，清空残留数据并重新开始监听...")
            self.recorded_audio = []
            self.is_recording = False
            self.pre_roll_buffer.clear()
            self.vad.reset()
            self.is_busy = False

    def run_forever(self, stream_source):
        """主循环：永远不停止读取，确保 stream 里的旧数据被实时消耗掉"""
        print("\n>>> 机器人已就绪，请说话...")
        chunk_count = 0

        for chunk in stream_source:
            chunk_count += 1

            if self.is_busy:
                continue

            self.pre_roll_buffer.append(chunk)
            is_speaking, is_final, prob = self.vad.process(chunk)

            # 每 50 帧打印一次 VAD 状态，方便调试
            if chunk_count % 50 == 0:
                print(f"[DEBUG] chunk#{chunk_count} | prob={prob:.3f} | is_speaking={is_speaking} | is_final={is_final}")

            if is_speaking and not self.is_recording:
                print(f"[DEBUG] 检测到语音开始，chunk#{chunk_count}, prob={prob:.3f}")
                self.is_recording = True
                self.recorded_audio = list(self.pre_roll_buffer)
                self.pre_roll_buffer.clear()

            elif self.is_recording:
                self.recorded_audio.append(chunk)

                if is_final:
                    duration = (len(self.recorded_audio) * 512) / 16000
                    print(f"[DEBUG] is_final 触发，duration={duration:.2f}s")
                    if duration >= Config.MIN_SEGMENT_SEC:
                        print(f"\n[系统] 锁定监听，开启后台任务...")
                        self.is_busy = True

                        task_thread = threading.Thread(
                            target=self._think_and_speak_thread,
                            args=(list(self.recorded_audio),)
                        )
                        task_thread.start()
                    else:
                        print(f"[DEBUG] 片段太短（{duration:.2f}s），丢弃")
                        self.recorded_audio = []
                        self.is_recording = False
                        self.vad.reset()

        print(f"\n[DEBUG] 流读取完毕，共处理 {chunk_count} 个 chunk")


# --- 运行 ---
if __name__ == "__main__":
    from audio_loader import get_audio_stream

    assistant = RobotAssistant()

    WAV_PATH = "/home/test/openbot_test_zhenghang/model_0320/openbot_validify_code/final_0322/data/yunxing/vad_results/followzhenghang.wav"

    # 自动转换为 16kHz 单声道
    converted = convert_to_16k_mono(WAV_PATH)
    test_stream = get_audio_stream(converted)
    assistant.run_forever(test_stream)

    # 等待最后一个任务完成
    time.sleep(0.3)
    while assistant.is_busy:
        time.sleep(0.5)
    print(">>> 全部完成，程序退出")
