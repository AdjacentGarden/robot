import os
import sys
import cv2
import numpy as np
import onnxruntime
import time
import signal
import threading
import multiprocessing
import warnings
from speaker import speak

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
                    self.frame = cv2.flip(frame, 1)
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

def situp_preproc(cv_img, resize=(224, 224),
                  mean=(103.53, 116.28, 123.675),
                  std=(57.375, 57.12, 58.395)):
    img = cv2.resize(cv_img, resize).astype(np.float32)
    img = (img - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    return np.expand_dims(img.transpose(2, 0, 1), axis=0)


class SitupDet:
    def __init__(self, onnx_path):
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"找不到模型文件: {onnx_path}")
        so = onnxruntime.SessionOptions()
        so.intra_op_num_threads = 1
        self.sess = onnxruntime.InferenceSession(onnx_path, sess_options=so)
        self.inp  = self.sess.get_inputs()[0]
        self.out  = self.sess.get_outputs()[0]

    def infer(self, cv_img):
        preds    = self.sess.run([self.out.name], {self.inp.name: situp_preproc(cv_img)})
        heatmaps = preds[0][0]          # shape [3, H, W]
        def peak(fm):
            idx = np.unravel_index(np.argmax(fm), fm.shape)
            return [(idx[0] + 0.5) / fm.shape[0], (idx[1] + 0.5) / fm.shape[1]]
        return peak(heatmaps[0]), peak(heatmaps[1]), peak(heatmaps[2])


def background_counting_task(video_source, model_path, count_file_path, pid_file_path):

    is_running = True

    def handle_sigterm(signum, frame_obj):
        nonlocal is_running
        is_running = False  

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT,  handle_sigterm)

    cap = None
    try:
        det = SitupDet(model_path)
        cap = CameraReader(video_source)

        situp_count  = 0
        last_pos     = "unknown"
        is_centered  = False

        with open(count_file_path, "w") as f:
            f.write("0")

        print("\n正在检测人物是否居中...")

        while cap.isOpened() and is_running:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            h, w = frame.shape[:2]
            p_head, p_knee, p_crotch = det.infer(frame)

            if not is_centered:
                if 0.2 < p_head[1] < 0.8 and 0.2 < p_crotch[1] < 0.8:
                    is_centered = True
                    print("\n人物已居中，正式开始为您计数！")
                else:
                    cv2.putText(frame, "Please move to center!",
                                (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow("OpenBot Sit-up Monitor", frame)
                    cv2.waitKey(1)
                    time.sleep(0.1)
                    continue

            if p_head[0] > p_knee[0] and p_crotch[0] > p_knee[0]:
                last_pos = "step1"
            if p_head[0] < p_knee[0] and p_crotch[0] > p_knee[0]:
                if last_pos == "step1":
                    situp_count += 1
                    try:
                        with open(count_file_path, "w") as f:
                            f.write(str(situp_count))
                    except OSError:
                        pass
                last_pos = "step2"

            head_px   = (int(p_head[1]   * w), int(p_head[0]   * h))
            knee_px   = (int(p_knee[1]   * w), int(p_knee[0]   * h))
            crotch_px = (int(p_crotch[1] * w), int(p_crotch[0] * h))

            cv2.line(frame, head_px, crotch_px, (255, 255, 0), 3)
            cv2.line(frame, crotch_px, knee_px, (0, 255, 255), 3)
            cv2.circle(frame, head_px,   8, (0, 0, 255),   -1)
            cv2.circle(frame, crotch_px, 8, (255, 0, 0),   -1)
            cv2.circle(frame, knee_px,   8, (0, 255, 0),   -1)
            cv2.rectangle(frame, (10, 10), (320, 80), (0, 0, 0), -1)
            cv2.putText(frame, f"Sit-ups: {situp_count}",
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
            state_color = (0, 255, 0) if last_pos == "step1" else (0, 165, 255)
            cv2.putText(frame, f"Pose: {last_pos}",
                        (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, state_color, 2)
            cv2.imshow("OpenBot Sit-up Monitor", frame)
            cv2.waitKey(1)

    except Exception as e:
        print(f"异常: {e}")

    finally:
        if cap is not None:
            try:   cap.release()
            except Exception: pass
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except Exception: pass
        _safe_remove_pid(pid_file_path)
        print("[仰卧起坐进程] 已退出，摄像头资源已释放")

class SitupCountingSystem:

    def __init__(self, model_path="model_24.onnx"):
        self.model_path = model_path
        self.pid_file   = "/tmp/situp_pid.txt"
        self.count_file = "/tmp/situp_count.txt"
        self._process: multiprocessing.Process = None

    def start_counting(self, video_source):
        if self._process is not None and self._process.is_alive():
            speak("计数任务已经在后台运行中")
            return

        _safe_remove_pid(self.pid_file)

        if not os.path.exists(self.model_path):
            print(f"找不到模型文件: {self.model_path}")
            return

        p = multiprocessing.Process(
            target=background_counting_task,
            args=(video_source, self.model_path, self.count_file, self.pid_file),
            daemon=True,
        )
        p.start()
        self._process = p
        try:
            with open(self.pid_file, "w") as f:
                f.write(str(p.pid))
        except OSError as e:
            print(f"写入 PID 文件失败: {e}")
        speak("仰卧起坐计数已在后台启动")

    def query_progress(self):
        count = 0
        if os.path.exists(self.count_file):
            try:
                txt = open(self.count_file).read().strip()
                if txt.isdigit():
                    count = int(txt)
            except Exception:
                pass
        speak("您目前做了{count}个仰卧起坐了")

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

        speak(f"仰卧起坐计数程序结束，您一共做了{final_count}个仰卧起坐")

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
    MODEL_PATH = "model_24.onnx"
    CAMERA_ID  = "/dev/video21"

    try:
        situp_sys = SitupCountingSystem(model_path=MODEL_PATH)
    except FileNotFoundError as e:
        print(e); exit(1)

    print("\n" + "="*40)
    print(" [s]=开始  [q]=查询  [x]=停止汇总  [e]=退出")
    print("="*40)

    while True:
        cmd = input("\n请输入指令 (s/q/x/e): ").strip().lower()
        if   cmd == "e": situp_sys.stop_and_summarize(); break
        elif cmd == "s": situp_sys.start_counting(CAMERA_ID)
        elif cmd == "q": situp_sys.query_progress()
        elif cmd == "x": situp_sys.stop_and_summarize()
        else:            print(" 无效指令")