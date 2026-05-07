import os
import sys
import cv2
import numpy as np
import time
import signal
import threading
import multiprocessing
import warnings

from rknnlite.api import RKNNLite

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


# 【修改2】重写预处理，对齐 RKNN 案例的输入要求
def situp_preproc(cv_img, resize=(224, 224)):
    """
    保持和 RKNN 转换流程一致的输入语义：
    1. cv2.imread -> BGR
    2. resize 到 224x224
    3. 不手工做 mean/std 归一化（交由 NPU 硬件处理）
    4. 转成 NCHW 4维输入
    """
    img = cv2.resize(cv_img, resize).astype(np.uint8)   # 保持 BGR
    img = np.transpose(img, (2, 0, 1))                  # HWC -> CHW
    img = np.expand_dims(img, axis=0)                   # CHW -> NCHW
    img = np.ascontiguousarray(img)
    return img


# 【修改3】彻底用 RKNNLite 替换 ONNXRuntime
class SitupDet:
    def __init__(self, rknn_path):
        if not os.path.exists(rknn_path):
            raise FileNotFoundError(f"找不到 RKNN 模型文件: {rknn_path}")
        
        self.rknn = RKNNLite()
        ret = self.rknn.load_rknn(rknn_path)
        if ret != 0:
            raise RuntimeError(f"load_rknn 失败，返回码: {ret}")

        try:
            ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
        except Exception:
            ret = self.rknn.init_runtime()

        if ret != 0:
            raise RuntimeError(f"init_runtime 失败，返回码: {ret}")

    def infer(self, cv_img):
        input_data = situp_preproc(cv_img)
        outputs = self.rknn.inference(inputs=[input_data], data_format=['nchw'])
        
        if outputs is None or len(outputs) == 0:
            raise RuntimeError("RKNN 推理失败，没有输出。")
            
        heatmaps = outputs[0][0]  # 取出 shape [3, H, W] 的热力图
        
        def peak(fm):
            idx = np.unravel_index(np.argmax(fm), fm.shape)
            return [(idx[0] + 0.5) / fm.shape[0], (idx[1] + 0.5) / fm.shape[1]]
            
        return peak(heatmaps[0]), peak(heatmaps[1]), peak(heatmaps[2])

    def release(self):
        if hasattr(self, "rknn"):
            self.rknn.release()


def background_counting_task(video_source, model_path, count_file_path, pid_file_path, tts_mp_q=None, det=None):
    
    # 在全新的 spawn 子进程中重新绑定跨进程队列
    import speaker
    speaker.init_mp_queue(tts_mp_q)

    is_running = True

    def handle_sigterm(signum, frame_obj):
        nonlocal is_running
        is_running = False  

    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, handle_sigterm)
        signal.signal(signal.SIGINT,  handle_sigterm)

    cap = None
    owns_detector = det is None
    try:
        if det is None:
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
                    # 让子进程直接通过队列呼叫发声
                    speaker.speak("人物已居中，正式开始为您计数！")
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
                    # speaker.speak(str(situp_count))
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
        # 【修改4】安全释放 NPU 资源，防止底板内存泄漏
        if owns_detector and det is not None:
            det.release()
            
        if cap is not None:
            try:   cap.release()
            except Exception: pass
            
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except Exception: pass
        
        _safe_remove_pid(pid_file_path)
        print("[仰卧起坐进程] 已退出，NPU 与摄像头资源已安全释放")


class SitupCountingSystem:

    def __init__(self, model_path="sitandup.rknn"): # 默认修改为 .rknn
        self.model_path = model_path
        self.pid_file   = "/tmp/situp_pid.txt"
        self.count_file = "/tmp/situp_count.txt"
        self._process: multiprocessing.Process = None
        self.detector = None

    def preload_detector(self):
        self._ensure_detector()

    def _ensure_detector(self):
        if self.detector is None:
            self.detector = SitupDet(self.model_path)
        return self.detector

    def start_counting(self, video_source):
        # 统一主进程发声调用
        if self._process is not None and self._process.is_alive():
            import speaker
            speaker.speak("计数任务已经在后台运行中")
            return

        _safe_remove_pid(self.pid_file)

        if not os.path.exists(self.model_path):
            print(f"找不到模型文件: {self.model_path}")
            return

        import speaker
        # 强制使用 spawn 上下文，并传递消息队列
        ctx = multiprocessing.get_context('spawn')
        if speaker._mp_q is None:
             speaker.init_mp_queue(ctx.Queue())

        background_counting_task(
            video_source,
            self.model_path,
            self.count_file,
            self.pid_file,
            speaker._mp_q,
            self._ensure_detector(),
        )
        return
        try:
            with open(self.pid_file, "w") as f:
                f.write("")
        except OSError as e:
            print(f"写入 PID 文件失败: {e}")
            
        speaker.speak("仰卧起坐计数已在后台启动")

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
        speaker.speak(f"您目前做了{number_to_chinese(count)}个仰卧起坐了")

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

        speaker.speak(f"仰卧起坐计数程序结束，您一共做了{number_to_chinese(final_count)}个仰卧起坐")

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
    MODEL_PATH = "model_24.rknn"
    CAMERA_ID  = "/dev/video40"

    try:
        situp_sys = SitupCountingSystem(model_path=MODEL_PATH)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    print("\n" + "="*40)
    print(" [s]=开始  [q]=查询  [x]=停止汇总  [e]=退出")
    print("="*40)

    while True:
        cmd = input("\n请输入指令 (s/q/x/e): ").strip().lower()
        if   cmd == "e": 
            situp_sys.stop_and_summarize()
            break
        elif cmd == "s": situp_sys.start_counting(CAMERA_ID)
        elif cmd == "q": situp_sys.query_progress()
        elif cmd == "x": situp_sys.stop_and_summarize()
        else:
            print(" 无效指令")
