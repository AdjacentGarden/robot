# ! 运动计数加一个语音提醒让人居于画面中心
import time
# import Radar
# import Vision
from final_0418.llm.main_robot import RobotAssistant, get_mic_stream, list_audio_devices
from final_0418.llm.speaker import speak
from ..test2 import Board

list_audio_devices()
MIC_DEVICE_INDEX = 6
robot = RobotAssistant()
# radar = Radar()
# vision = Vision()
board = Board()
class Config:
    ENCODER_PATH = "./model/encoder-epoch-99-avg-1.rknn"
    DECODER_PATH = "./model/decoder-epoch-99-avg-1.rknn"
    JOINER_PATH = "./model/joiner-epoch-99-avg-1.rknn"
    VOCAB_PATH = "./model/vocab.txt"
    LLM_MODE = "local"
    LLM_LOCAL = "http://localhost:8080/v1"
    API_KEY = "sk-ff528950477e421999763986692ce67e"
    # ! minimum sentence interval
    MIN_SILENCE_MS = 400
    # ! minimum voice duration
    MIN_SEGMENT_SEC = 0.4
    # ! voice chunk size
    CHUNK_SIZE = 512
    SAMPLE_RATE = 16000

if __name__ == "__main__":
    while True:
        try:
            wkup = board.get_wkup()
            if wkup is not None:
                print("唤醒状态:", wkup)
            time.sleep(0.01)
            mic_stream = get_mic_stream(
                device_index=MIC_DEVICE_INDEX,
                chunk_size=Config.CHUNK_SIZE,
                sample_rate=Config.SAMPLE_RATE
            )
            robot.run_once(mic_stream)
        except KeyboardInterrupt:
            break