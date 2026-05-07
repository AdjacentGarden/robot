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

warnings.filterwarnings("ignore")

def _is_valid_bgr_frame(frame) -> bool:
    if frame is None or not isinstance(frame, np.ndarray):
        return False
    if frame.ndim not in (2, 3):
        return False
    if frame.shape[0] <= 0 or frame.shape[1] <= 0:
        return False
    if frame.ndim == 3 and frame.shape[2] not in (1, 3, 4):
        return False
    return True

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

class CameraReader:
    def __init__(self, src=0, width=320, height=240):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头流: {src}")
        self._lock  = threading.Lock()
        self._stop  = threading.Event()
        self.ret, self.frame = self.cap.read()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while not self._stop.is_set():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self._lock:
                    self.ret   = ret
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

def squat_preproc(cv_img):
    """
    预处理图像为 RGB，用于 MediaPipe
    """
    if not _is_valid_bgr_frame(cv_img):
        raise ValueError(f"非法图像输入，shape={getattr(cv_img, 'shape', None)}")
    if cv_img.ndim == 2:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
    elif cv_img.ndim == 3 and cv_img.shape[2] == 1:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
    elif cv_img.ndim == 3 and cv_img.shape[2] == 4:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

class SquatDet:
    def __init__(self, pose_samples_folder=POSE_SAMPLES_DIR):
        self.pose_tracker = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        self.pose_embedder = pe.FullBodyPoseEmbedder()
        self.pose_classifier = pc.PoseClassifier(
            pose_samples_folder=pose_samples_folder,
            class_name='squat_down',
            pose_embedder=self.pose_embedder,
            top_n_by_max_distance=30,
            top_n_by_mean_distance=10)
        self.pose_classification_filter = rs.EMADictSmoothing(
            window_size=10,
            alpha=0.2)
        self.last_result = None

    def infer(self, cv_img):
        image = squat_preproc(cv_img)
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
def background_counting_task(video_source, count_file_path, pid_file_path, tts_mp_q=None, det=None):
    is_running = True

    def handle_sigterm(signum, frame_obj):
        nonlocal is_running
        is_running = False

    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, handle_sigterm)
        signal.signal(signal.SIGINT, handle_sigterm)

    cap = None
    owns_detector = det is None

    SESSION_SECONDS = 30   # 总时长 30 秒
    IDLE_SECONDS = 10      # 连续 10 秒没有新动作就退出

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
        if det is None:
            det = SquatDet()
        cap = CameraReader(video_source)

        repetition_counter = counter.RepetitionCounter(
            class_name='squat_down',
            enter_threshold=5,
            exit_threshold=4
        )

        with open(count_file_path, "w") as f:
            f.write("0")

        safe_speak("深蹲计数已开始，时长三十秒。")

        invalid_frame_count = 0
        last_count = 0

        session_start = time.monotonic()
        last_motion_time = session_start

        while cap.isOpened() and is_running:
            now = time.monotonic()

            # 30 秒到达，自动结束
            if now - session_start >= SESSION_SECONDS:
                safe_speak(f"三十秒已到，本次深蹲计数结束，共做了 {last_count} 个。")
                break

            # 10 秒没有识别到新的动作，自动结束
            if now - last_motion_time >= IDLE_SECONDS:
                safe_speak(f"连续十秒没有检测到新动作，本次深蹲计数自动结束，共做了 {last_count} 个。")
                break

            ret, frame = cap.read()
            if not ret or not _is_valid_bgr_frame(frame):
                invalid_frame_count += 1
                if invalid_frame_count % 50 == 0:
                    print(f"[警告] 收到无效帧，shape={getattr(frame, 'shape', None)}")
                time.sleep(0.01)
                continue

            try:
                pose_classification = det.infer(frame)
            except Exception as e:
                print(f"[推理异常] {e}, shape={getattr(frame, 'shape', None)}")
                continue

            # 没检测到人体时，不更新 last_motion_time
            if not pose_classification:
                continue

            squat_count = repetition_counter(pose_classification)

            # 只要计数增加，就视为识别到一个新动作
            if squat_count > last_count:
                for i in range(last_count + 1, squat_count + 1):
                    chinese_number = number_to_chinese(i)
                    print(chinese_number)
                    safe_speak(f"第{chinese_number}个")

                last_count = squat_count
                last_motion_time = now

                try:
                    with open(count_file_path, "w") as f:
                        f.write(str(squat_count))
                except OSError:
                    pass

                print(f"-> 完成了一个深蹲！当前总数: {squat_count}")

    except Exception as e:
        print(f"异常: {e}")

    finally:
        if owns_detector and det is not None:
            det.release()

        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

        _safe_remove_pid(pid_file_path)
        print("\n[深蹲进程] 已退出，资源已安全释放")


class SquatCountingSystem:
    def __init__(self):
        self.pid_file   = "/tmp/squat_pid.txt"
        self.count_file = "/tmp/squat_count.txt"
        self._process: multiprocessing.Process = None
        self.detector = None

    def preload_detector(self):
        self._ensure_detector()

    def _ensure_detector(self):
        if self.detector is None:
            self.detector = SquatDet()
        return self.detector

    def start_counting(self, video_source):
        if self._process is not None and self._process.is_alive():
            try:
                import speaker
                speaker.speak("计数任务已经在后台运行中")
            except Exception:
                print("计数任务已经在后台运行中")
            return

        _safe_remove_pid(self.pid_file)

        background_counting_task(video_source, self.count_file, self.pid_file, None, self._ensure_detector())
        return

        try:
            with open(self.pid_file, "w") as f:
                f.write("")
        except OSError as e:
            print(f"写入 PID 文件失败: {e}")

        print("深蹲计数已在后台启动（最多 30 秒，连续 10 秒无动作会自动结束）")

    def query_progress(self):
        count = 0
        if os.path.exists(self.count_file):
            try:
                txt = open(self.count_file).read().strip()
                if txt.isdigit():
                    count = int(txt)
            except Exception:
                pass
        print(f"\n[查询结果] 您目前做了 {count} 个深蹲了")

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

        print(f"\n[汇总统计] 深蹲计数程序结束，您一共做了 {final_count} 个深蹲")

    def _terminate_process(self):
        if self._process is not None:
            try:
                if self._process.is_alive():
                    print("\n发送 SIGTERM，等待 3s...")
                    self._process.terminate()
                    self._process.join(timeout=3.0)
                    if self._process.is_alive():
                        self._process.kill()
                        self._process.join(timeout=2.0)
                    print("进程彻底终止" if not self._process.is_alive() else "进程仍然存活")
                else:
                    print("\n进程已自行退出")
            except Exception as e:
                print(f"终止进程时出错: {e}")
            finally:
                self._process = None
            return

        if not os.path.exists(self.pid_file):
            return
        try:
            pid_str = open(self.pid_file).read().strip()
            if not pid_str.isdigit(): return
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
    # 注意这里，请确保 /dev/video40 是真实有效的设备
    CAMERA_ID  = "/dev/video40"
    squat_sys = SquatCountingSystem()

    print("\n" + "="*40)
    print(" [s]=开始  [q]=查询  [x]=停止汇总  [e]=退出")
    print("="*40)

    while True:
        cmd = input("\n请输入指令 (s/q/x/e): ").strip().lower()
        if   cmd == "e": 
            squat_sys.stop_and_summarize()
            break
        elif cmd == "s": squat_sys.start_counting(CAMERA_ID)
        elif cmd == "q": squat_sys.query_progress()
        elif cmd == "x": squat_sys.stop_and_summarize()
        else:
            print(" 无效指令")
