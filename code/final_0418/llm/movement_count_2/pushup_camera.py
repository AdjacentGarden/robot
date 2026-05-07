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
# 添加本地模型路径
MODELS_DIR = os.path.join(SCRIPT_DIR, 'mediapipe_models')
POSE_MODEL_DIR = os.path.join(MODELS_DIR, 'pose')
# 支持三种模型选择: 'lite'(轻量级), 'full'(完整版), 'heavy'(高精度)
POSE_MODEL_TYPE = 'lite'  # 可修改为 'lite' 或 'heavy'
POSE_MODEL_PATHS = {
    'lite': os.path.join(POSE_MODEL_DIR, 'pose_landmark_lite.task'),
    'full': os.path.join(POSE_MODEL_DIR, 'pose_landmark_full.task'),
    'heavy': os.path.join(POSE_MODEL_DIR, 'pose_landmark_heavy.task')
}
POSE_MODEL_PATH = POSE_MODEL_PATHS.get(POSE_MODEL_TYPE, POSE_MODEL_PATHS['full'])

if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose
import poseembedding as pe
import poseclassifier as pc
import resultsmooth as rs
import counter
import visualizer as vs
import speaker

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


class CameraReader:
    def __init__(self, src=0, width=640, height=480):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
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
                    self.frame = cv2.flip(frame, -1)
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
    """
    预处理图像为 RGB，用于 MediaPipe
    """
    return cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)


class PushupDet:
    def __init__(self, pose_samples_folder=POSE_SAMPLES_DIR, use_local_model=True):
        # 检查是否使用本地模型
        if use_local_model and os.path.exists(POSE_MODEL_PATH):
            print("使用本地MediaPipe Pose模型...")
            self.pose_tracker = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                enable_segmentation=False,
                smooth_segmentation=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_asset_path=POSE_MODEL_PATH
            )
        else:
            if use_local_model:
                print("警告: 本地模型文件不存在，使用默认在线模型")
            self.pose_tracker = mp_pose.Pose()
        
        self.pose_embedder = pe.FullBodyPoseEmbedder()
        self.pose_classifier = pc.PoseClassifier(
            pose_samples_folder=pose_samples_folder,
            class_name='push_down',
            pose_embedder=self.pose_embedder,
            top_n_by_max_distance=30,
            top_n_by_mean_distance=10)
        self.pose_classification_filter = rs.EMADictSmoothing(
            window_size=10,
            alpha=0.2)

    def infer(self, cv_img):
        image = pushup_preproc(cv_img)
        result = self.pose_tracker.process(image)
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
    
    import speaker
    speaker.init_mp_queue(tts_mp_q)

    is_running = True

    def handle_sigterm(signum, frame_obj):
        nonlocal is_running
        is_running = False  

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT,  handle_sigterm)

    cap = None
    det = None
    try:
        det = PushupDet()
        cap = CameraReader(video_source)

        repetition_counter = counter.RepetitionCounter(
            class_name='push_down',
            enter_threshold=5,
            exit_threshold=4)

        with open(count_file_path, "w") as f:
            f.write("0")

        print("\n正在检测人物是否居中...")

        while cap.isOpened() and is_running:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            pose_classification = det.infer(frame)

            # 简单居中检测，假设基于姿势存在
            if not pose_classification:
                cv2.putText(frame, "Please move to center!",
                            (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("OpenBot Push-up Monitor", frame)
                cv2.waitKey(1)
                time.sleep(0.1)
                continue

            pushup_count = repetition_counter(pose_classification)

            try:
                with open(count_file_path, "w") as f:
                    f.write(str(pushup_count))
            except OSError:
                pass

            # 绘制姿势
            image = pushup_preproc(frame)
            result = det.pose_tracker.process(image)
            if result.pose_landmarks:
                mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.rectangle(frame, (10, 10), (320, 80), (0, 0, 0), -1)
            cv2.putText(frame, f"Push-ups: {pushup_count}",
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
            cv2.imshow("OpenBot Push-up Monitor", frame)
            cv2.waitKey(1)

    except Exception as e:
        print(f"异常: {e}")

    finally:
        if det is not None:
            det.release()
            
        if cap is not None:
            try:   cap.release()
            except Exception: pass
            
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except Exception: pass
        
        _safe_remove_pid(pid_file_path)
        print("[俯卧撑进程] 已退出，资源已安全释放")


class PushupCountingSystem:

    def __init__(self):
        self.pid_file   = "/tmp/pushup_pid.txt"
        self.count_file = "/tmp/pushup_count.txt"
        self._process: multiprocessing.Process = None

    def start_counting(self, video_source):
        if self._process is not None and self._process.is_alive():
            import speaker
            speaker.speak("计数任务已经在后台运行中")
            return

        _safe_remove_pid(self.pid_file)

        import speaker
        ctx = multiprocessing.get_context('spawn')
        if speaker._mp_q is None:
             speaker.init_mp_queue(ctx.Queue())

        p = ctx.Process(
            target=background_counting_task,
            args=(video_source, self.count_file, self.pid_file, speaker._mp_q),
            daemon=True,
        )
        p.start()
        self._process = p
        try:
            with open(self.pid_file, "w") as f:
                f.write(str(p.pid))
        except OSError as e:
            print(f"写入 PID 文件失败: {e}")
            
        speaker.speak("俯卧撑计数已在后台启动")

    def query_progress(self):
        import speaker
        count = 0
        if os.path.exists(self.count_file):
            try:
                txt = open(self.count_file).read().strip()
                if txt.isdigit():
                    count = int(txt)
            except Exception:
                pass
        speaker.speak(f"您目前做了{count}个俯卧撑了")

    def stop_and_summarize(self):
        import speaker
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

        speaker.speak(f"俯卧撑计数程序结束，您一共做了{final_count}个俯卧撑")

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
                    print(" 进程彻底终止" if not self._process.is_alive()
                          else "进程仍然存活")
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
    CAMERA_ID  = "/dev/video21"

    pushup_sys = PushupCountingSystem()

    print("\n" + "="*40)
    print(" [s]=开始  [q]=查询  [x]=停止汇总  [e]=退出")
    print("="*40)

    while True:
        cmd = input("\n请输入指令 (s/q/x/e): ").strip().lower()
        if   cmd == "e": 
            pushup_sys.stop_and_summarize()
            break
        elif cmd == "s": pushup_sys.start_counting(CAMERA_ID)
        elif cmd == "q": pushup_sys.query_progress()
        elif cmd == "x": pushup_sys.stop_and_summarize()
        else:
            print(" 无效指令")