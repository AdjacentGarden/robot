import os
import torch
import numpy as np
import threading
from collections import deque
import time
import pyaudio
import sys
from final_0418.llm.vad_module import SileroVAD
from final_0418.llm.zipformer import set_model, run_model, post_process, read_vocab
from final_0418.llm.toolmain import UniversalAgent
_CURR_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_CURR_DIR))
if _ROOT not in sys.path:
    sys.path.append(_ROOT) 
from tts_queue import TTSQueue
import speaker
import multiprocessing

# ! config
class Config:
    MODEL_DIR = os.path.join(_CURR_DIR, "model")
    ENCODER_PATH = os.path.join(MODEL_DIR, "encoder-epoch-99-avg-1.rknn")
    DECODER_PATH = os.path.join(MODEL_DIR, "decoder-epoch-99-avg-1.rknn")
    JOINER_PATH = os.path.join(MODEL_DIR, "joiner-epoch-99-avg-1.rknn")
    VOCAB_PATH = os.path.join(MODEL_DIR, "vocab.txt")
    LLM_MODE = "cloud"
    LLM_LOCAL = "http://localhost:8080/v1"
    API_KEY = "sk-ff528950477e421999763986692ce67e"
    # ! minimum sentence interval
    MIN_SILENCE_MS = 400
    # ! minimum voice duration
    MIN_SEGMENT_SEC = 0.4
    # ! voice chunk size
    CHUNK_SIZE = 512
    SAMPLE_RATE = 16000

def list_audio_devices():
    """打印所有可用音频输入设备，方便选择编号"""
    p = pyaudio.PyAudio()
    print("\n========== 可用音频输入设备 ==========")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            print(f" [{i}] {info['name']} (采样率: {int(info['defaultSampleRate'])}Hz)")
    print("=======================================\n")
    p.terminate()

def get_mic_stream(device_index=None, chunk_size=512, sample_rate=16000):
    """
    实时麦克风音频流生成器
    device_index 对应关系（根据你的 arecord -l）：
    运行时会打印完整列表，以实际输出为准
    通常:
    板载 ES8388 (card 1) → pyaudio 里找名字含 ES8323 的
    USB 摄像头麦克风 (card 6) → pyaudio 里找名字含 Camera 或 USB 的
    如果 device_index=None 则使用系统默认麦克风
    """
    p = pyaudio.PyAudio()

    # 如果没指定设备，自动找第一个可用输入设备
    if device_index is None:
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                device_index = i
                print(f">>> 自动选择音频设备 [{i}]: {info['name']}")
                break

    try:
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=chunk_size,
        )
    except OSError as e:
        print(f"[错误] 无法打开音频设备 [{device_index}]: {e}")
        print("请运行 list_audio_devices() 查看可用设备，修改 MIC_DEVICE_INDEX")
        p.terminate()
        return

    print(f">>> 麦克风已打开，开始监听... (按 Ctrl+C 退出)")
    # try:
    while True:
        chunk = stream.read(chunk_size, exception_on_overflow=False)
        yield chunk

class RobotAssistant:
    def __init__(self):
        print("--- [系统初始化] 机器人启动 ---")
        self.vad = SileroVAD(min_silence_ms=Config.MIN_SILENCE_MS)
        self.pre_roll_buffer = deque(maxlen=15)
        self.recorded_audio = []
        self.is_recording = False
        self.is_busy = False

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
            x=1, y=4
        )

        self.tts_queue = TTSQueue(self.agent.tts)
        speaker.init(self.tts_queue)
        ctx = multiprocessing.get_context('spawn')
        main_mp_q = ctx.Queue()
        speaker.init_mp_queue(main_mp_q)

        def _bridge():
            while True:
                text = speaker._mp_q.get()
                self.tts_queue.speak(text)

        threading.Thread(target=_bridge, daemon=True).start()


    def _think_and_speak_thread(self, audio_data_list):
        """同步执行：ASR -> LLM -> TTS (返回是否识别到有效文本)"""
        is_valid_speech = False  # 增加一个标志位
        need_followup = False
        try:
            print("开始时间：", time.time())
            audio_bytes = b"".join(audio_data_list)
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            audio_tensor = torch.tensor(audio_np, dtype=torch.float32)

            hyp, timestamp = run_model(self.asr_model, audio_tensor, sample_rate=16000)
            text, _ = post_process(hyp, self.asr_vocab, timestamp)

            if text.strip():
                is_valid_speech = True
                print(f"\n[用户]: {text}")
                response = self.agent.run_workflow(text)
                need_followup = bool(getattr(self.agent, "expect_followup", False))
                print(f"[机器人]: {response}")
                if response:
                    self.tts_queue.speak(response)
                self.tts_queue.wait_until_done()
            else:
                print("[DEBUG] ASR 结果为空(可能是纯噪音)，忽略本次识别")

        except Exception as e:
            print(f"[系统错误]: {e}")
        finally:
            if is_valid_speech:
                print("\n>>> 回复完毕...")
            else:
                print("\n>>> 忽略噪音，重置状态...")
            
            # 无论成功失败，都重置录音状态
            self.recorded_audio = []
            self.is_recording = False
            self.pre_roll_buffer.clear()
            self.vad.reset()
            self.is_busy = False
            
        return is_valid_speech, need_followup  # 将结果返回给外层调用


    def run_once(self, stream_source):
        """只监听一次有效语音 → 识别 → 退出"""
        print("\n机器人已就绪（单次监听模式），请说话...")
        chunk_count = 0
        is_speaking = False
        for chunk in stream_source:
            chunk_count += 1

            self.pre_roll_buffer.append(chunk)
            is_speaking, is_final, prob = self.vad.process(chunk)
            print(is_speaking)

            if chunk_count % 50 == 0:
                pass
            if is_speaking and not self.is_recording:
                print(f"[DEBUG] 检测到声音波动 chunk#{chunk_count}, prob={prob:.3f}")
                self.is_recording = True
                self.recorded_audio = list(self.pre_roll_buffer)
                self.pre_roll_buffer.clear()

            elif self.is_recording:
                self.recorded_audio.append(chunk)
            if is_final:
                duration = (len(self.recorded_audio) * Config.CHUNK_SIZE) / Config.SAMPLE_RATE
                print(f"[DEBUG] 声音波动结束，duration={duration:.2f}s")

                if duration >= Config.MIN_SEGMENT_SEC:
                    print(f"\n[系统] 时长满足，开始识别...")
                    has_valid_text, need_followup = self._think_and_speak_thread(list(self.recorded_audio))
                    if has_valid_text:
                        if need_followup:
                            print("\n>>> 进入追问模式，继续监听姓名...")
                            continue
                        print("\n>>> 单次指令执行完成，程序退出")
                        break
                    else:
                        print("[DEBUG] 录音内容无意义(可能是咳嗽或敲击麦克风)，继续等待有效指令...")
                        continue
                else:
                    print(f"[DEBUG] 片段太短（{duration:.2f}s < {Config.MIN_SEGMENT_SEC}s），直接丢弃并继续...")
                    self.recorded_audio = []
                    self.is_recording = False
                    self.pre_roll_buffer.clear()
                    self.vad.reset()
                    continue
        print(f"\n[DEBUG] 结束，共处理 {chunk_count} 个 chunk")

if __name__ == "__main__":
    list_audio_devices()
    MIC_DEVICE_INDEX = 6
    assistant = RobotAssistant()
    mic_stream = get_mic_stream(
        device_index=MIC_DEVICE_INDEX,
        chunk_size=Config.CHUNK_SIZE,
        sample_rate=Config.SAMPLE_RATE
    )
    assistant.run_once(mic_stream)
