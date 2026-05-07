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
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import drawing_styles as mp_drawing_styles

import poseembedding as pe
import poseclassifier as pc
import resultsmooth as rs
import counter

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


def pushup_preproc(cv_img):
    return cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)


class PushupDet:
    def __init__(self, pose_samples_folder=POSE_SAMPLES_DIR):
        self.pose_tracker = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.pose_embedder = pe.FullBodyPoseEmbedder()

        self.pose_classifier = pc.PoseClassifier(
            pose_samples_folder=pose_samples_folder,
            class_name='push_down',
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
        image = pushup_preproc(cv_img)
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


def draw_pose_and_info(frame, pose_result, count, remain_seconds):
    """
    在视频帧上画：
    1. MediaPipe 人体关键点和骨架
    2. 当前俯卧撑数量
    3. 剩余时间
    """

    annotated = frame.copy()

    if pose_result is not None and pose_result.pose_landmarks is not None:
        mp_drawing.draw_landmarks(
            annotated,
            pose_result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

    cv2.rectangle(annotated, (5, 5), (315, 85), (0, 0, 0), -1)

    cv2.putText(
        annotated,
        f"Push-ups: {count}",
        (15, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        annotated,
        f"Time left: {remain_seconds}s",
        (15, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return annotated


def create_video_writer(output_path, frame_width, frame_height, fps=20):
    """
    创建 mp4 视频写入器。
    优先使用 mp4v，适合 OpenCV 保存 mp4。
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (frame_width, frame_height)
    )

    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频文件: {output_path}")

    return writer


def background_counting_task(video_source, count_file_path, pid_file_path, video_output_path, det=None):
    is_running = True

    def handle_sigterm(signum, frame_obj):
        nonlocal is_running
        is_running = False

    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, handle_sigterm)
        signal.signal(signal.SIGINT, handle_sigterm)

    cap = None
    owns_detector = det is None
    video_writer = None

    SESSION_SECONDS = 30
    IDLE_SECONDS = 10
    SAVE_FPS = 20

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
            det = PushupDet()
        cap = CameraReader(video_source)

        repetition_counter = counter.RepetitionCounter(
            class_name='push_down',
            enter_threshold=5,
            exit_threshold=4
        )

        with open(count_file_path, "w") as f:
            f.write("0")

        safe_speak("俯卧撑计数已开始，时长三十秒。")

        invalid_frame_count = 0
        last_count = 0

        session_start = time.monotonic()
        last_motion_time = session_start

        while cap.isOpened() and is_running:
            now = time.monotonic()
            elapsed = now - session_start
            remain_seconds = max(0, int(SESSION_SECONDS - elapsed))

            if elapsed >= SESSION_SECONDS:
                safe_speak(f"三十秒已到，本次俯卧撑计数结束，共做了{number_to_chinese(last_count)}个。")
                break

            if now - last_motion_time >= IDLE_SECONDS:
                safe_speak(f"连续十秒没有检测到新动作，本次俯卧撑计数自动结束，共做了{number_to_chinese(last_count)}个。")
                break

            ret, frame = cap.read()

            if not ret or frame is None:
                invalid_frame_count += 1
                if invalid_frame_count % 50 == 0:
                    print("[警告] 收到无效帧")
                time.sleep(0.01)
                continue

            if video_writer is None:
                h, w = frame.shape[:2]
                video_writer = create_video_writer(
                    output_path=video_output_path,
                    frame_width=w,
                    frame_height=h,
                    fps=SAVE_FPS
                )
                print(f"[视频录制] 已开始保存到: {video_output_path}")

            try:
                pose_classification = det.infer(frame)
            except Exception as e:
                print(f"[推理异常] {e}")
                annotated_frame = draw_pose_and_info(
                    frame=frame,
                    pose_result=None,
                    count=last_count,
                    remain_seconds=remain_seconds
                )
                video_writer.write(annotated_frame)
                continue

            if pose_classification:
                pushup_count = repetition_counter(pose_classification)

                if pushup_count > last_count:
                    for i in range(last_count + 1, pushup_count + 1):
                        chinese_number = number_to_chinese(i)
                        print(chinese_number)
                        safe_speak(f"第{chinese_number}个")

                    last_count = pushup_count
                    last_motion_time = now

                    try:
                        with open(count_file_path, "w") as f:
                            f.write(str(pushup_count))
                    except OSError:
                        pass

                    print(f"[计数更新] Push-ups: {pushup_count}")

            annotated_frame = draw_pose_and_info(
                frame=frame,
                pose_result=det.last_result,
                count=last_count,
                remain_seconds=remain_seconds
            )

            video_writer.write(annotated_frame)

    except Exception as e:
        print(f"异常: {e}")

    finally:
        if video_writer is not None:
            try:
                video_writer.release()
                print(f"[视频录制] 已保存: {video_output_path}")
            except Exception as e:
                print(f"[视频保存异常] {e}")

        if owns_detector and det is not None:
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
        print("[俯卧撑进程] 已退出，资源已安全释放")


class PushupCountingSystem:
    def __init__(self):
        self.pid_file = "/tmp/pushup_pid.txt"
        self.count_file = "/tmp/pushup_count.txt"
        self.video_output_file = "/home/test/code/final_0418/llm/movement_count_2/pushup_record.mp4"
        self._process: multiprocessing.Process = None
        self.detector = None

    def preload_detector(self):
        self._ensure_detector()

    def _ensure_detector(self):
        if self.detector is None:
            self.detector = PushupDet()
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

        if os.path.exists(self.video_output_file):
            try:
                os.remove(self.video_output_file)
            except OSError:
                pass

        background_counting_task(
            video_source,
            self.count_file,
            self.pid_file,
            self.video_output_file,
            self._ensure_detector(),
        )
        return

        try:
            with open(self.pid_file, "w") as f:
                f.write("")
        except OSError as e:
            print(f"写入 PID 文件失败: {e}")

        print("俯卧撑计数已在后台启动（最多 30 秒，连续 10 秒无动作会自动结束）")
        print(f"录制视频将保存到: {self.video_output_file}")

    def query_progress(self):
        count = 0

        if os.path.exists(self.count_file):
            try:
                txt = open(self.count_file).read().strip()
                if txt.isdigit():
                    count = int(txt)
            except Exception:
                pass

        print(f"您目前做了 {count} 个俯卧撑了")
        print(f"当前录制视频路径: {self.video_output_file}")

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

        print(f"俯卧撑计数程序结束，您一共做了 {final_count} 个俯卧撑")
        print(f"带关键点标注的视频已保存到: {self.video_output_file}")

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

    pushup_sys = PushupCountingSystem()

    print("\n" + "=" * 40)
    print(" [s]=开始  [q]=查询  [x]=停止汇总  [e]=退出")
    print("=" * 40)

    while True:
        cmd = input("\n请输入指令 (s/q/x/e): ").strip().lower()

        if cmd == "e":
            pushup_sys.stop_and_summarize()
            break

        elif cmd == "s":
            pushup_sys.start_counting(CAMERA_ID)

        elif cmd == "q":
            pushup_sys.query_progress()

        elif cmd == "x":
            pushup_sys.stop_and_summarize()

        else:
            print("无效指令")
