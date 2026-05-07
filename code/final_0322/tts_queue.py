import queue
import threading


class TTSQueue:
    """
    TTS 串行播放队列。

    所有 speak() 调用投入队列后立刻返回，
    后台工作线程按顺序逐条合成+播放，永远不会叠播。

    用法：
        tts_queue = TTSQueue(tts_engine)
        tts_queue.speak("你好")          # 立刻返回
        tts_queue.speak("世界")          # 排在"你好"后面
        tts_queue.wait_until_done()      # 阻塞，等两句都播完
    """

    def __init__(self, tts_engine):
        """
        :param tts_engine: 任意 TTS 实例（LocalTTS / LocalTTS2 / MatchaTTS），
                           必须有 speak(text, wait=True) 方法。
        """
        self.tts = tts_engine
        self._q = queue.Queue()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def speak(self, text: str):
        """投入一段文字，立刻返回，不阻塞调用方。"""
        if text and text.strip():
            self._q.put(text)

    def wait_until_done(self):
        """阻塞当前线程，直到队列中所有语音全部播放完毕。"""
        self._q.join()

    def _run(self):
        """工作线程：循环取出文字 → 合成 → 播放，播完再取下一条。"""
        while True:
            text = self._q.get()
            try:
                self.tts.speak(text, wait=True)
            except Exception as e:
                print(f"[TTSQueue] 播放出错: {e}")
            finally:
                self._q.task_done()

