import os
import multiprocessing

_tts_queue = None
_mp_q = None  # 初始为 None，等待被显式注入

def init(tts_queue):
    """主进程中调用，绑定真实的 TTS 播放器"""
    global _tts_queue
    _tts_queue = tts_queue
    print("[Speaker] TTS 队列已注入，语音播报已激活")

def init_mp_queue(mp_queue):
    """专门为子进程准备的注入接口，或者在主进程初始化通信管道"""
    global _mp_q
    _mp_q = mp_queue

def speak(text: str):
    if not text or not text.strip():
        return
        
    if _tts_queue is not None:
        # 主进程直接发声
        _tts_queue.speak(text)
    elif _mp_q is not None:
        # 子进程将文本推入跨进程队列
        _mp_q.put(text)
    else:
        print(f"[Speaker Warning] 既没有 TTS 引擎也没有进程队列，文字被丢弃: {text}")
