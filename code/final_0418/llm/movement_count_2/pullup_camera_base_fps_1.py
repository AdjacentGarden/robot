import os
import sys
import cv2
import numpy as np
import time
import signal
import threading
import multiprocessing
import warnings

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, 'code')
POSE_SAMPLES_DIR = os.path.join(CODE_DIR, 'fitness_poses_csvs_out')
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from mediapipe.python.solutions import pose as mp_pose
import poseembedding as pe
import poseclassifier as pc
import resultsmooth as rs
import counter
import visualizer as vs

warnings.filterwarnings("ignore")


def _safe_remove_pid(pid_file: str) -> None:
    if not pid_file or not os.path.exists(pid_file):
        return
    try:
        os.remove(pid_file)
    except OSError:
        try:
            with open(pid_file, "w") as f:
                f.write("")
        except OSError:
            pass


def _section_to_chinese(section: int) -> str:
    digits = "零一二三四五六七八九"
    units = ["", "十", "百", "千"]
    result = ""
    zero_pending = False

    for idx in range(3, -1, -1):
        base = 10 ** idx
        digit = section // base
        section %= base

        if digit == 0:
            if result:
                zero_pending = True
            continue

        if zero_pending:
            result += "零"
            zero_pending = False

        if not (digit == 1 and idx == 1 and not result):
            result += digits[digit]
        result += units[idx]

    return result or "零"


def number_to_chinese(num: int) -> str:
    if num == 0:
        return "零"
    if num < 0:
        return f"负{number_to_chinese(-num)}"

    section_units = ["", "万", "亿", "兆"]
    sections = []
    while num > 0:
        sections.append(num % 10000)
        num //= 10000

    result = ""
    need_zero = False
    for idx in range(len(sections) - 1, -1, -1):
        section = sections[idx]
        if section == 0:
            need_zero = bool(result)
            continue

        if need_zero or (result and section < 1000):
            result += "零"

        result += _section_to_chinese(section) + section_units[idx]
        need_zero = False

    return result


class CameraReader:
    def __init__(self, src=0, width=320, height=240):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头流: {src}")
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.ret, self.frame = self.cap.read()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while not self._stop.is_set():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self._lock:
                    self.ret = ret
                    self.frame = frame
            else:
                time.sleep(0.005)
        if self.cap.isOpened():
            self.cap.release()

    def read(self):
        with self._lock:
            if self.frame is not None:
                return self.ret, self.frame.copy()
        return False, None

    def isOpened(self):
        return not self._stop.is_set() and self.cap.isOpened()

    def release(self):
        self._stop.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.cap.isOpened():
            self.cap.release()


def pullup_preproc(cv_img):
    """
    预处理图像为 RGB，用于 MediaPipe
    """
    return cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)


class PullupDet:
    def __init__(self, pose_samples_folder=POSE_SAMPLES_DIR):
        self.pose_tracker = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.pose_embedder = pe.FullBodyPoseEmbedder()
        self.pose_classifier = pc.PoseClassifier(
            pose_samples_folder=pose_samples_folder,
            class_name='pull_up',
            pose_embedder=self.pose_embedder,
            top_n_by_max_distance=30,
            top_n_by_mean_distance=10
        )
        self.pose_classification_filter = rs.EMADictSmoothing(
            window_size=10,
            alpha=0.2
        )
        self.last_result = None

    def infer(self, cv_img):
        image = pullup_preproc(cv_img)
        result = self.pose_tracker.process(image)
        self.last_result = result
        if result.pose_landmarks is None:
            return {}
        pose_landmarks = result.pose_landmarks.landmark
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in pose_landmarks])
        if landmarks.shape != (33, 3):
            return {}
        pose_classification = self.pose_classifier(landmarks)
        pose_classification_filtered = self.pose_classification_filter(pose_classification)
        return pose_classification_filtered

    def release(self):
        self.pose_tracker.close()


def background_counting_task(video_source, count_file_path, pid_file_path, tts_mp_q=None):
    is_running = True
    def handle_sigterm(signum, frame_obj):
        nonlocal is_running
        is_running = False

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)

    cap = None
    det = None

    SESSION_SECONDS = 30
    IDLE_SECONDS = 10

    try:
        import speaker
    except Exception:
        speaker = None

    def safe_speak(text: str):
        print(text)
        if speaker is not None:
            try:
                speaker.speak(text)
            except Exception:
                pass

    try:
        det = PullupDet()
        cap = CameraReader(video_source)

        repetition_counter = counter.RepetitionCounter(
            class_name='pull_up',
            enter_threshold=5,
            exit_threshold=4
        )

        with open(count_file_path, "w") as f:
            f.write("0")

        safe_speak("引体向上计数已开始，时长三十秒。")
        # speaker.speak("引体向上计数已开始，时长三十秒。")

        invalid_frame_count = 0
        last_count = 0

        session_start = time.monotonic()
        last_motion_time = session_start

        while cap.isOpened() and is_running:
            now = time.monotonic()
            # 30秒自动结束
            if now - session_start >= SESSION_SECONDS:
                safe_speak(f"三十秒已到，本次引体向上计数结束，共做了{number_to_chinese(last_count)}个。")
                # speaker.speak(f"本次引体向上计数结束，共做了{last_count}个")
                break
            # 10秒无动作自动结束
            if now - last_motion_time >= IDLE_SECONDS:
                safe_speak(f"连续十秒没有检测到新动作，本次引体向上计数自动结束，共做了{number_to_chinese(last_count)}个。")
                # speaker.speak(f"本次引体向上计数自动结束，共做了{last_count}个")
                break
            ret, frame = cap.read()
            if not ret or frame is None:
                invalid_frame_count += 1
                if invalid_frame_count % 50 == 0:
                    print(f"[警告] 收到无效帧")
                time.sleep(0.01)
                continue

            try:
                pose_classification = det.infer(frame)
            except Exception as e:
                print(f"[推理异常] {e}")
                continue

            # 没检测到人体时，不更新 last_motion_time
            if not pose_classification:
                continue

            pullup_count = repetition_counter(pose_classification)

            # 只要计数增加，就视为检测到一次动作
            if pullup_count > last_count:
                # 如果一次跳了多个，逐个播报
                for i in range(last_count + 1, pullup_count + 1):
                    chinese_number = number_to_chinese(i)
                    print(chinese_number)
                    safe_speak(f"第{chinese_number}个")
                last_count = pullup_count
                last_motion_time = now

                try:
                    with open(count_file_path, "w") as f:
                        f.write(str(pullup_count))
                except OSError:
                    pass

                print(f"[计数更新] Pull-ups: {pullup_count}")

    except Exception as e:
        print(f"异常: {e}")

    finally:
        if det is not None:
            det.release()

        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        _safe_remove_pid(pid_file_path)
        print("[引体向上进程] 已退出，资源已安全释放")


class PullupCountingSystem:
    def __init__(self):
        self.pid_file = "/tmp/pullup_pid.txt"
        self.count_file = "/tmp/pullup_count.txt"
        self._process: multiprocessing.Process = None

    def start_counting(self, video_source):
        if self._process is not None and self._process.is_alive():
            return

        _safe_remove_pid(self.pid_file)

        ctx = multiprocessing.get_context('spawn')
        p = ctx.Process(
            target=background_counting_task,
            args=(video_source, self.count_file, self.pid_file, None),
            daemon=True,
        )
        p.start()
        self._process = p

        try:
            with open(self.pid_file, "w") as f:
                f.write(str(p.pid))
        except OSError as e:
            print(f"写入 PID 文件失败: {e}")

        print(f"引体向上计数已启动，PID={p.pid}，将自动运行 30 秒；连续 10 秒无动作会提前结束。")

    def query_progress(self):
        count = 0
        if os.path.exists(self.count_file):
            try:
                txt = open(self.count_file).read().strip()
                if txt.isdigit():
                    count = int(txt)
            except Exception:
                pass
        print(f"您目前做了 {count} 个引体向上了")

    def stop_and_summarize(self):
        final_count = 0
        if os.path.exists(self.count_file):
            try:
                txt = open(self.count_file).read().strip()
                if txt.isdigit():
                    final_count = int(txt)
            except Exception:
                pass

        try:
            self._terminate_process()
        finally:
            _safe_remove_pid(self.pid_file)
            _safe_remove_pid(self.count_file)

        print(f"引体向上计数程序结束，您一共做了 {final_count} 个引体向上")

    def _terminate_process(self):
        if self._process is not None:
            try:
                if self._process.is_alive():
                    print("发送 SIGTERM，等待 3s...")
                    self._process.terminate()
                    self._process.join(timeout=3.0)
                    if self._process.is_alive():
                        self._process.kill()
                        self._process.join(timeout=2.0)
                    print("进程彻底终止" if not self._process.is_alive() else "进程仍然存活")
                else:
                    print("进程已自行退出")
            except Exception as e:
                print(f"终止进程时出错: {e}")
            finally:
                self._process = None
            return

        if not os.path.exists(self.pid_file):
            return

        try:
            pid_str = open(self.pid_file).read().strip()
            if not pid_str.isdigit():
                return
            pid = int(pid_str)
            try:
                os.kill(pid, signal.SIGTERM)
                time.sleep(3.0)
                try:
                    os.kill(pid, 0)
                    os.kill(pid, signal.SIGKILL)
                    time.sleep(0.5)
                except ProcessLookupError:
                    pass
            except ProcessLookupError:
                pass
            except PermissionError:
                print("无权限终止该进程")
        except Exception as e:
            print(f"通过 PID 文件终止时出错: {e}")


if __name__ == "__main__":
    CAMERA_ID = "/dev/video40"
    pullup_sys = PullupCountingSystem()

    print("\n" + "=" * 40)
    print(" [s]=开始  [q]=查询  [x]=停止汇总  [e]=退出")
    print("=" * 40)

    while True:
        cmd = input("\n请输入指令 (s/q/x/e): ").strip().lower()
        if cmd == "e":
            pullup_sys.stop_and_summarize()
            break
        elif cmd == "s":
            pullup_sys.start_counting(CAMERA_ID)
        elif cmd == "q":
            pullup_sys.query_progress()
        elif cmd == "x":
            pullup_sys.stop_and_summarize()
        else:
            print("无效指令")
