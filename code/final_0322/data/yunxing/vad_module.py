import os
import time
import numpy as np
import onnxruntime
from collections import deque

class SileroVAD:
    def __init__(self, model_name="silero_vad.onnx", 
                 threshold=0.5, 
                 threshold_low=0.2, 
                 min_silence_ms=1000):
        
        # 1. 加载模型
        model_path = os.path.join(os.path.dirname(__file__), model_name)
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = onnxruntime.InferenceSession(
            model_path, providers=["CPUExecutionProvider"], sess_options=opts
        )

        # 2. 逻辑参数
        self.threshold = threshold          # 高阈值：判定为“有声音”
        self.threshold_low = threshold_low  # 低阈值：判定为“彻底没声音”
        self.silence_limit = min_silence_ms / 1000  # 静默多久判定为结束
        
        # 3. 状态维护
        self.reset()

    def reset(self):
        """重置所有状态，准备听下一句话"""
        # 模型内部 RNN 状态
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, 64), dtype=np.float32)
        
        # 逻辑判断状态
        self.last_prob_is_voice = False     # 上一次的布尔结果（用于双阈值滞后判断）
        self.is_speaking = False            # 当前是否处于“正在说话”的状态
        self.has_spoken_anything = False    # 这一轮里是否曾经检测到过人声
        self.last_speech_time = 0           # 最后一次检测到人声的时间点
        
        # --- 核心修改：改为记录音频累计时长 ---
        self.audio_current_time = 0.0  # 累计处理了多少秒音频
        self.last_speech_time = 0.0   # 最后一次说话的音频时间点
        
        # 滑动窗口平滑处理 (窗口大小为3，只有2个以上为True才认为当前帧有效)
        self.window = deque(maxlen=4)
    
    def get_speech_prob(self, audio_chunk):
        """核心推理：只返回概率值"""
        if isinstance(audio_chunk, bytes):
            audio_int16 = np.frombuffer(audio_chunk, dtype=np.int16)
        else:
            audio_int16 = audio_chunk

        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        input_data = np.concatenate([self._context, audio_float32.reshape(1, -1)], axis=1).astype(np.float32)
        
        ort_inputs = {
            "input": input_data, "state": self._state, "sr": np.array(16000, dtype=np.int64)
        }
        out, state = self.session.run(None, ort_inputs)
        self._state = state
        self._context = input_data[:, -64:]
        return out.item()

    def process(self, audio_chunk):
        """
        处理 512 个采样的音频块
        返回: (status, is_final)
        - status: bool, 当前这一刻是否正在说话
        - is_final: bool, 是否检测到说话结束（用户讲完了）
        """
        # --- 1. 基础推理 ---
        if isinstance(audio_chunk, bytes):
            audio_int16 = np.frombuffer(audio_chunk, dtype=np.int16)
        else:
            audio_int16 = audio_chunk

        # 每次处理 512 个采样点，音频进度增加 0.032 秒 (16000Hz下)
        self.audio_current_time += (len(audio_int16) / 16000.0)
        
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        input_data = np.concatenate([self._context, audio_float32.reshape(1, -1)], axis=1).astype(np.float32)
        
        ort_inputs = {
            "input": input_data, "state": self._state, "sr": np.array(16000, dtype=np.int64)
        }
        out, state = self.session.run(None, ort_inputs)
        self._state = state
        self._context = input_data[:, -64:]
        
        prob = out.item()

        # --- 2. 双阈值逻辑 (防止概率在0.5附近抖动) ---
        if prob >= self.threshold:
            current_frame_voice = True
        elif prob <= self.threshold_low:
            current_frame_voice = False
        else:
            current_frame_voice = self.last_prob_is_voice # 维持原状
        
        self.last_prob_is_voice = current_frame_voice
        self.window.append(current_frame_voice)

        # --- 3. 滑动窗口平滑 ---
        # 窗口内超过 2 帧为 True，才认为这一刻真的有声音
        is_active_now = self.window.count(True) >= 3
        
        # --- 4. 说话结束（Endpointing）状态机 ---
        now = time.time()
        is_final = False

        if is_active_now:
            # 此时此刻正在说话
            if not self.is_speaking:
                self.is_speaking = True
                self.has_spoken_anything = True
            # self.last_speech_time = now  # 更新最后活跃时间
            # 更新最后说话的音频时刻
            self.last_speech_time = self.audio_current_time
        else:
            # 此时此刻是静音
            if self.is_speaking:
                # 之前在说，现在突然停了
                if self.audio_current_time - self.last_speech_time > self.silence_limit:
                    self.is_speaking = False
                    is_final = True  # 判定为讲完了！
        
        # 如果从没开过口，不能判定为结束
        if not self.has_spoken_anything:
            is_final = False

        return self.is_speaking, is_final, prob