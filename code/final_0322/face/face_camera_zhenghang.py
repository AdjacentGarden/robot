import mediapipe as mp
import cv2
import time
from facenet_pytorch import InceptionResnetV1
import torch
import numpy as np
import sqlite3
import pickle
import os
import threading
import multiprocessing
from speaker import speak

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
                    self.frame = cv2.flip(frame, 1) 
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

def background_register_face_task(video_source, db_path, model_path, person_name):
    cap = None
    window_name = "Face Registration"
    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS face_vectors (name TEXT PRIMARY KEY, vector BLOB)''')
        conn.commit()

        model = InceptionResnetV1(pretrained=None)
        model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
        model.eval()

        cap = CameraReader(video_source)
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        window_ready = False
        wait_start = time.time()
        while cap.isOpened() and (time.time() - wait_start) < 5.0:
            success, image = cap.read()
            if success and image is not None:
                cv2.putText(image, "Camera Ready, Starting...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow(window_name, image)
                cv2.waitKey(10)
                window_ready = True
                break
            time.sleep(0.01)

        if not window_ready:
            speak('摄像头画面获取超时，请重新录入')
            return

        collected_vectors = []
        frame_count = 0
        start_time = time.time() 
        is_extracting = False
        vector_lock = threading.Lock()
        
        def do_extraction(face_img):
            nonlocal is_extracting
            try:
                vector = _extract_feature(face_img, model)
                with vector_lock:
                    collected_vectors.append(vector)
            finally:
                is_extracting = False

        mp_face_detection = mp.solutions.face_detection
        with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6) as face_detection:
            while cap.isOpened() and (time.time() - start_time) < 10.0:
                
                with vector_lock:
                    if len(collected_vectors) >= 5:
                        break 
                        
                success, image = cap.read()
                if not success or image is None:
                    time.sleep(0.01)
                    continue
                
                frame_count += 1
                vis = image.copy() 
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_detection.process(image_rgb)

                if results.detections:
                    bbox = results.detections[0].location_data.relative_bounding_box
                    ih, iw, _ = vis.shape
                    bx, by, bw, bh = int(bbox.xmin * iw), int(bbox.ymin * ih), int(bbox.width * iw), int(bbox.height * ih)
                    cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (255, 165, 0), 2)

                    if not is_extracting:
                        face_image = _get_padded_face_image(image, bbox)
                        if face_image.size > 0:
                            gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                            focus_measure = cv2.Laplacian(gray_face, cv2.CV_64F).var()
                            if focus_measure > 20.0:
                                is_extracting = True
                                threading.Thread(target=do_extraction, args=(face_image.copy(),), daemon=True).start()
                                cv2.putText(vis, "Capturing...", (bx, max(20, by - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            else:
                                cv2.putText(vis, "Blurry", (bx, max(20, by - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    else:
                        cv2.putText(vis, "Extracting...", (bx, max(20, by - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                time_left = max(0, 10.0 - (time.time() - start_time))
                with vector_lock:
                    current_saved = len(collected_vectors)
                cv2.putText(vis, f"Time: {time_left:.1f}s | Saved: {current_saved}/5", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.imshow(window_name, vis)
                if cv2.waitKey(1) & 0xFF == 27: 
                    break
        time.sleep(0.1)
        with vector_lock:
            final_count = len(collected_vectors)

        if final_count >= 5:
            mean_vector = np.mean(np.array(collected_vectors), axis=0)
            vector_blob = pickle.dumps(mean_vector)
            cursor.execute("INSERT OR REPLACE INTO face_vectors (name, vector) VALUES (?, ?)", (person_name, vector_blob))
            conn.commit()
            speak(f"这是{person_name}")
    finally:
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        for _ in range(10): cv2.waitKey(1) 
        if 'conn' in locals():
            conn.close()

def background_recognize_face_task(video_source, db_path, model_path):
    cap = None
    window_name = "OpenBot Face Recognition"
    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS face_vectors (name TEXT PRIMARY KEY, vector BLOB)''')
        
        cursor.execute("SELECT name, vector FROM face_vectors")
        rows = cursor.fetchall()
        all_known_faces = [(row[0], pickle.loads(row[1])) for row in rows]

        model = InceptionResnetV1(pretrained=None)
        model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
        model.eval()

        print("正在调起摄像头...")
        cap = CameraReader(video_source)
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        window_ready = False
        wait_start = time.time()
        while cap.isOpened() and (time.time() - wait_start) < 5.0:
            success, image = cap.read()
            if success and image is not None:
                cv2.putText(image, "Camera Ready! Starting...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow(window_name, image)
                cv2.waitKey(50) 
                window_ready = True
                break
            time.sleep(0.01)

        if not window_ready:
            speak(f"摄像头画面获取超时，我没有看到人")
            return

        start_time = time.time()
        saw_face = False
        matched_name = None
        
        is_inferencing = False 
        display_name = "Detecting..."
        display_color = (255, 255, 0)
        should_exit = False

        def do_inference(face_img):
            nonlocal is_inferencing, matched_name, display_name, display_color, should_exit
            try:
                gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                focus_measure = cv2.Laplacian(gray_face, cv2.CV_64F).var()

                if focus_measure > 60.0 and len(all_known_faces) > 0:
                    current_vec = _extract_feature(face_img, model)
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
                        time.sleep(0.5)
                        should_exit = True
                    else:
                        display_name = "Unknown"
                        display_color = (0, 0, 255)
            finally:
                is_inferencing = False 

        mp_face_detection = mp.solutions.face_detection
        with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.55) as face_detection:
            while cap.isOpened() and (time.time() - start_time) < 5.0:
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
                    cv2.putText(vis, display_name, (bx, max(20, by - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, display_color, 2)

                    if not is_inferencing:
                        face_image = _get_padded_face_image(image, bbox)
                        if face_image.size > 0:
                            is_inferencing = True
                            threading.Thread(target=do_inference, args=(face_image.copy(),), daemon=True).start()

                time_left = max(0, 5.0 - (time.time() - start_time))
                cv2.putText(vis, f"Time left: {time_left:.1f}s", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow(window_name, vis)
                if cv2.waitKey(1) & 0xFF == 27: 
                    break
                    
        if matched_name:
            speak(f"这是{matched_name}")
        elif saw_face:
            speak(f"抱歉，我不认识这个人")
        else:
            speak(f"对不起，我没有看到人")

    except Exception as e:
        print(f"\n发生异常: {e}，正在退出...")

    finally:
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        for _ in range(10): cv2.waitKey(1)
        if 'conn' in locals():
            conn.close()


class FaceRecognitionSystem:
    def __init__(self, db_path='face_vectors.db', model_path='20180402-114759-vggface2.pt'):
        self.db_path = db_path
        self.model_path = model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型权重文件: {model_path}")

    def register_face(self, video_source, person_name):
        print(f"启动人脸注册进程")
        p = multiprocessing.Process(target=background_register_face_task, args=(video_source, self.db_path, self.model_path, person_name), daemon=True)
        p.start()
        # waiting sub-process ends
        p.join()
        print(f"注册进程已退出")

    def recognize_face(self, video_source):
        print(f"启动人脸识别进程")
        p = multiprocessing.Process(target=background_recognize_face_task, args=(video_source, self.db_path, self.model_path), daemon=True)
        p.start()
        p.join()
        print(f"人脸识别进程已退出")
