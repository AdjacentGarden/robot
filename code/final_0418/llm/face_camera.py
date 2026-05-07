import os
import time
import queue
import pickle
import sqlite3
import threading
import multiprocessing

import cv2
import mediapipe as mp
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1

from speaker import speak

PENDING_FACE_KEY = "__pending_face__"
REGISTER_TIMEOUT_SEC = 3000.0
RECOGNIZE_TIMEOUT_SEC = 6.0


def _want_preview_window():
    v = str(os.getenv("FACE_CAMERA_SHOW_WINDOW", "0")).strip().lower()
    return v in {"1", "true", "yes", "on"}


def _safe_named_window(window_name):
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        return True
    except cv2.error as e:
        print(f"[WARN] 无法创建窗口，切换无窗口模式: {e}")
        return False


def _safe_imshow(enable_window, window_name, frame):
    if not enable_window:
        return
    try:
        cv2.imshow(window_name, frame)
    except cv2.error as e:
        print(f"[WARN] 显示画面失败，切换无窗口模式: {e}")


def _safe_wait_key(enable_window, delay=1):
    if not enable_window:
        return -1
    try:
        return cv2.waitKey(delay) & 0xFF
    except cv2.error:
        return -1


def _safe_destroy_windows(enable_window):
    if not enable_window:
        return
    try:
        cv2.destroyAllWindows()
        for _ in range(10):
            cv2.waitKey(1)
    except cv2.error:
        pass


class CameraReader:
    def __init__(self, src=0, width=640, height=480):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头流: {src}")

        self.ret, self.frame = self.cap.read()
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                if ret:
                    self.frame = frame
                else:
                    time.sleep(0.05)

    def read(self):
        with self.lock:
            if self.frame is not None and self.ret:
                return self.ret, self.frame.copy()
            return self.ret, None

    def isOpened(self):
        return self.running

    def release(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap.isOpened():
            self.cap.release()


def _get_padded_face_image(image, bbox):
    ih, iw, _ = image.shape
    raw_x, raw_y = int(bbox.xmin * iw), int(bbox.ymin * ih)
    raw_w, raw_h = int(bbox.width * iw), int(bbox.height * ih)
    pad_w, pad_h = int(raw_w * 0.2), int(raw_h * 0.2)
    x_start = max(0, raw_x - pad_w)
    y_start = max(0, raw_y - pad_h)
    x_end = min(iw, raw_x + raw_w + pad_w)
    y_end = min(ih, raw_y + raw_h + pad_h)
    return image[y_start:y_end, x_start:x_end]


def _extract_feature(face_image, model):
    resized = cv2.resize(face_image, (160, 160))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).float()
    tensor = (tensor - 127.5) / 128.0
    with torch.no_grad():
        vector = model(tensor).cpu().numpy().reshape(-1)
    return vector


def background_register_face_task(video_source, db_path, model_path, person_name, tts_mp_q=None, result_q=None):
    import speaker

    speaker.init_mp_queue(tts_mp_q)
    cap = None
    window_name = "Face Registration"
    target_name = str(person_name or "").strip()
    pending_mode = not bool(target_name)
    window_enabled = _want_preview_window()

    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS face_vectors (name TEXT PRIMARY KEY, vector BLOB)")
        conn.commit()

        model = InceptionResnetV1(pretrained=None)
        model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
        model.eval()

        cap = CameraReader(video_source)
        if window_enabled:
            window_enabled = _safe_named_window(window_name)

        window_ready = False
        wait_start = time.time()
        while cap.isOpened() and (time.time() - wait_start) < 5.0:
            success, image = cap.read()
            if success and image is not None:
                cv2.putText(image, "Camera Ready, Starting...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                _safe_imshow(window_enabled, window_name, image)
                _safe_wait_key(window_enabled, 10)
                window_ready = True
                break
            time.sleep(0.01)

        if not window_ready:
            speak("对不起我没识别到人脸")
            if result_q is not None:
                result_q.put({"status": "no_face"})
            return

        collected_vectors = []
        start_time = time.time()
        first_face_deadline = start_time + 5.0
        first_face_seen = False
        last_extract_ts = 0.0

        mp_face_detection = mp.solutions.face_detection
        with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6) as face_detection:
            while cap.isOpened() and (time.time() - start_time) < REGISTER_TIMEOUT_SEC:
                if len(collected_vectors) >= 5:
                    break

                success, image = cap.read()
                if not success or image is None:
                    time.sleep(0.01)
                    continue

                vis = image.copy()
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_detection.process(image_rgb)

                if results.detections:
                    first_face_seen = True
                    bbox = results.detections[0].location_data.relative_bounding_box
                    ih, iw, _ = vis.shape
                    bx, by, bw, bh = int(bbox.xmin * iw), int(bbox.ymin * ih), int(bbox.width * iw), int(bbox.height * ih)
                    cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (255, 165, 0), 2)

                    face_image = _get_padded_face_image(image, bbox)
                    print(face_image.size)
                    if face_image.size > 0:
                        gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                        focus_measure = cv2.Laplacian(gray_face, cv2.CV_64F).var()
                        if focus_measure > 10.0:
                            now_ts = time.time()
                            if now_ts - last_extract_ts >= 0.2:
                                vector = _extract_feature(face_image, model)
                                collected_vectors.append(vector)
                                last_extract_ts = now_ts
                                cv2.putText(vis, "Capturing...", (bx, max(20, by - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        else:
                            cv2.putText(vis, "Blurry", (bx, max(20, by - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                elif time.time() > first_face_deadline and not first_face_seen:
                    speak("对不起我没识别到人脸")
                    if result_q is not None:
                        result_q.put({"status": "no_face"})
                    return

                time_left = max(0, REGISTER_TIMEOUT_SEC - (time.time() - start_time))
                current_saved = len(collected_vectors)
                cv2.putText(vis, f"Time: {time_left:.1f}s | Saved: {current_saved}/5", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                _safe_imshow(window_enabled, window_name, vis)
                if _safe_wait_key(window_enabled, 1) == 27:
                    break

        final_count = len(collected_vectors)
        if final_count >= 5:
            mean_vector = np.mean(np.array(collected_vectors), axis=0)
            vector_blob = pickle.dumps(mean_vector)
            save_name = PENDING_FACE_KEY if pending_mode else target_name
            cursor.execute("INSERT OR REPLACE INTO face_vectors (name, vector) VALUES (?, ?)", (save_name, vector_blob))
            conn.commit()
            if result_q is not None:
                result_q.put({"status": "success", "pending": pending_mode})
        else:
            speak("对不起我没识别到人脸")
            if result_q is not None:
                result_q.put({"status": "no_face"})
    finally:
        if cap:
            cap.release()
        _safe_destroy_windows(window_enabled)
        if "conn" in locals():
            conn.close()


def background_recognize_face_task(video_source, db_path, model_path, tts_mp_q=None, result_q=None):
    import speaker

    speaker.init_mp_queue(tts_mp_q)
    cap = None
    window_name = "OpenBot Face Recognition"
    window_enabled = _want_preview_window()

    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS face_vectors (name TEXT PRIMARY KEY, vector BLOB)")

        cursor.execute("SELECT name, vector FROM face_vectors")
        rows = cursor.fetchall()
        all_known_faces = [(row[0], pickle.loads(row[1])) for row in rows]
        if not all_known_faces:
            msg = "当前没有已注册人脸，请先执行人脸录入。"
            print(msg)
            speak(msg)
            if result_q is not None:
                result_q.put({"status": "empty_db"})
            return

        model = InceptionResnetV1(pretrained=None)
        model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
        model.eval()

        print("正在调起摄像头...")
        cap = CameraReader(video_source)
        if window_enabled:
            window_enabled = _safe_named_window(window_name)

        window_ready = False
        wait_start = time.time()
        while cap.isOpened() and (time.time() - wait_start) < 5.0:
            success, image = cap.read()
            if success and image is not None:
                cv2.putText(image, "Camera Ready! Starting...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                _safe_imshow(window_enabled, window_name, image)
                _safe_wait_key(window_enabled, 50)
                window_ready = True
                break
            time.sleep(0.01)

        if not window_ready:
            speak("摄像头画面获取超时，我没有看到人")
            if result_q is not None:
                result_q.put({"status": "no_face"})
            return

        start_time = time.time()
        saw_face = False
        matched_name = None
        display_name = "Detecting..."
        display_color = (255, 255, 0)
        should_exit = False
        last_infer_ts = 0.0

        mp_face_detection = mp.solutions.face_detection
        with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.55) as face_detection:
            while cap.isOpened() and (time.time() - start_time) < RECOGNIZE_TIMEOUT_SEC:
                if should_exit:
                    break
                success, image = cap.read()
                if not success or image is None:
                    time.sleep(0.01)
                    continue

                vis = image.copy()
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_detection.process(image_rgb)

                if results.detections:
                    saw_face = True
                    bbox = results.detections[0].location_data.relative_bounding_box
                    ih, iw, _ = vis.shape
                    bx, by, bw, bh = int(bbox.xmin * iw), int(bbox.ymin * ih), int(bbox.width * iw), int(bbox.height * ih)
                    cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), display_color, 2)

                    face_image = _get_padded_face_image(image, bbox)
                    if face_image.size > 0:
                        gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                        focus_measure = cv2.Laplacian(gray_face, cv2.CV_64F).var()
                        if focus_measure > 10.0 and all_known_faces:
                            now_ts = time.time()
                            if now_ts - last_infer_ts >= 0.15:
                                current_vec = _extract_feature(face_image, model)
                                best_sim = -1.0
                                temp_matched_name = None
                                for f_name, f_vec in all_known_faces:
                                    sim = np.dot(current_vec, f_vec) / (np.linalg.norm(current_vec) * np.linalg.norm(f_vec))
                                    if sim > best_sim:
                                        best_sim = sim
                                        temp_matched_name = f_name
                                if best_sim > 0.7:
                                    matched_name = temp_matched_name
                                    display_name = f"{matched_name} ({best_sim:.2f})"
                                    display_color = (0, 255, 0)
                                    should_exit = True
                                else:
                                    display_name = "Unknown"
                                    display_color = (0, 0, 255)
                                last_infer_ts = now_ts
                    cv2.putText(vis, display_name, (bx, max(20, by - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, display_color, 2)

                time_left = max(0, RECOGNIZE_TIMEOUT_SEC - (time.time() - start_time))
                cv2.putText(vis, f"Time left: {time_left:.1f}s", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                _safe_imshow(window_enabled, window_name, vis)
                if _safe_wait_key(window_enabled, 1) == 27:
                    break

        if matched_name:
            speak(f"这是{matched_name}")
            if result_q is not None:
                result_q.put({"status": "matched", "name": matched_name})
        elif saw_face:
            speak("抱歉，我不认识这个人")
            if result_q is not None:
                result_q.put({"status": "unknown"})
        else:
            speak("对不起，我没有看到人")
            if result_q is not None:
                result_q.put({"status": "no_face"})
    except Exception as e:
        print(f"\n发生异常: {e}，正在退出...")
        if result_q is not None:
            result_q.put({"status": "error", "error": str(e)})
    finally:
        if cap:
            cap.release()
        _safe_destroy_windows(window_enabled)
        if "conn" in locals():
            conn.close()


class FaceRecognitionSystem:
    def __init__(self, db_path="face_vectors.db", model_path="20180402-114759-vggface2.pt"):
        self.db_path = db_path
        self.model_path = model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型权重文件: {model_path}")

    def _use_subprocess(self):
        v = str(os.getenv("FACE_CAMERA_USE_SUBPROCESS", "0")).strip().lower()
        return v in {"1", "true", "yes", "on"}

    def _run_register_process(self, video_source, person_name):
        import speaker
        print("启动人脸注册进程")

        ctx = multiprocessing.get_context("fork" if os.name == "posix" else "spawn")
        if speaker._mp_q is None:
            speaker.init_mp_queue(ctx.Queue())

        result_q = ctx.Queue()
        p = ctx.Process(
            target=background_register_face_task,
            args=(video_source, self.db_path, self.model_path, person_name, speaker._mp_q, result_q),
            daemon=False,
        )
        p.start()
        p.join()
        print("注册进程已退出")

        result = {"status": "failed"}
        if p.exitcode not in (0, None):
            print(f"[WARN] 人脸注册子进程异常退出, exitcode={p.exitcode}")
            return result
        while not result_q.empty():
            result = result_q.get()
        return result

    def _run_register_inline(self, video_source, person_name):
        result_q = queue.Queue()
        background_register_face_task(video_source, self.db_path, self.model_path, person_name, None, result_q)
        result = {"status": "failed"}
        while not result_q.empty():
            result = result_q.get()
        return result

    def _run_recognize_process(self, video_source):
        import speaker

        print("启动人脸识别进程")
        ctx = multiprocessing.get_context("fork" if os.name == "posix" else "spawn")
        if speaker._mp_q is None:
            speaker.init_mp_queue(ctx.Queue())

        result_q = ctx.Queue()
        p = ctx.Process(
            target=background_recognize_face_task,
            args=(video_source, self.db_path, self.model_path, speaker._mp_q, result_q),
            daemon=False,
        )
        p.start()
        p.join()

        if p.exitcode not in (0, None):
            print(f"[WARN] 人脸识别子进程异常退出, exitcode={p.exitcode}")
            return {"status": "error", "error": f"subprocess_exit_{p.exitcode}"}

        result = {"status": "failed"}
        while not result_q.empty():
            result = result_q.get()
        print("人脸识别进程已退出")
        return result

    def _run_recognize_inline(self, video_source):
        result_q = queue.Queue()
        background_recognize_face_task(video_source, self.db_path, self.model_path, None, result_q)
        result = {"status": "failed"}
        while not result_q.empty():
            result = result_q.get()
        return result

    def register_face(self, video_source, person_name):
        result = self._run_register_process(video_source, person_name) if self._use_subprocess() else self._run_register_inline(video_source, person_name)
        print(f"[FaceRegister] result={result}")
        return result.get("status") == "success"

    def start_face_enrollment(self, video_source):
        result = self._run_register_process(video_source, "") if self._use_subprocess() else self._run_register_inline(video_source, "")
        print(f"[FaceEnroll] result={result}")
        return result.get("status") == "success"

    def confirm_face_name(self, person_name):
        target_name = str(person_name or "").strip()
        if not target_name:
            return False

        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        try:
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE IF NOT EXISTS face_vectors (name TEXT PRIMARY KEY, vector BLOB)")
            cursor.execute("SELECT vector FROM face_vectors WHERE name = ?", (PENDING_FACE_KEY,))
            row = cursor.fetchone()
            if not row:
                return False
            cursor.execute("INSERT OR REPLACE INTO face_vectors (name, vector) VALUES (?, ?)", (target_name, row[0]))
            cursor.execute("DELETE FROM face_vectors WHERE name = ?", (PENDING_FACE_KEY,))
            conn.commit()
            return True
        finally:
            conn.close()

    def recognize_face(self, video_source):
        result = self._run_recognize_process(video_source) if self._use_subprocess() else self._run_recognize_inline(video_source)
        print(f"[FaceRecognize] result={result}")
        return result
