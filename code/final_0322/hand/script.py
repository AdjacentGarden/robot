import cv2
import mediapipe as mp
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import threading
import os
from speaker import speak
import multiprocessing
import queue
import datetime 
import random  

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
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            # 只有在 running 为 True 时才读取，防止在释放期间读取
            ret, frame = self.cap.read()
            if ret:
                self.ret = ret
                self.frame = cv2.flip(frame, -1)
            else:
                time.sleep(0.01) # 防止死循环占满 CPU
                
        # 线程安全退出时，由线程自己负责释放 C++ 资源
        if self.cap.isOpened():
            self.cap.release()

    def read(self):
        if self.frame is not None: 
            return self.ret, self.frame.copy()
        return self.ret, None

    def isOpened(self): 
        return self.running

    def release(self):
        # 1. 切断循环标志
        self.running = False
        # 2. 等待后台线程自行安全结束，_update 函数最后会自动 release()
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)


def judge_gesture(hand_landmarks):
    direction = 'None'
    root = hand_landmarks[0]
    little_finger = hand_landmarks[13]
    vector = [little_finger[0] - root[0], little_finger[1] - root[1]]
    if vector[0] > vector[1]:
        if vector[0] > -vector[1]: direction = 'left'
        else: direction = 'up'
    else:
        if vector[0] > -vector[1]: direction = 'down'
        else: direction = 'right'
    
    thumb_straight = judge_straight(hand_landmarks[1:5])
    index_straight = judge_straight(hand_landmarks[5:9])
    middle_straight = judge_straight(hand_landmarks[9:13])
    ring_straight = judge_straight(hand_landmarks[13:17])
    little_straight = judge_straight(hand_landmarks[17:21])

    shape = 'None'
    if index_straight and middle_straight and ring_straight and little_straight: shape = 'palm'
    elif index_straight and middle_straight and not ring_straight and not little_straight: shape = 'two'
    elif not index_straight and not middle_straight and not ring_straight and not little_straight: shape = 'fist'
    else: shape = 'other'
    
    gesture = shape + '_' + direction
    if shape == 'fist': gesture = 'fist'
    if shape == 'other': gesture = 'other'
    return gesture

def judge_straight(points):
    vector1 = [points[1][0] - points[0][0], points[1][1] - points[0][1]]
    vector2 = [points[2][0] - points[1][0], points[2][1] - points[1][1]]
    vector3 = [points[3][0] - points[2][0], points[3][1] - points[2][1]]
    vector4 = [points[3][0] - points[0][0], points[3][1] - points[0][1]]
    vector5 = [points[2][0] - points[0][0], points[2][1] - points[0][1]]
    vector6 = [points[3][0] - points[1][0], points[3][1] - points[1][1]]
    
    n1 = np.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
    n2 = np.sqrt(vector2[0] ** 2 + vector2[1] ** 2)
    n3 = np.sqrt(vector3[0] ** 2 + vector3[1] ** 2)
    n4 = np.sqrt(vector4[0] ** 2 + vector4[1] ** 2)
    n5 = np.sqrt(vector5[0] ** 2 + vector5[1] ** 2)
    n6 = np.sqrt(vector6[0] ** 2 + vector6[1] ** 2)
    
    if n1*n2 == 0 or n1*n3 == 0 or n4*n5 == 0 or n4*n6 == 0: return False
        
    cos1 = (vector1[0] * vector2[0] + vector1[1] * vector2[1]) / (n1 * n2)
    cos2 = (vector1[0] * vector3[0] + vector1[1] * vector3[1]) / (n1 * n3)
    cos3 = (vector4[0] * vector5[0] + vector4[1] * vector5[1]) / (n4 * n5)
    cos4 = (vector4[0] * vector6[0] + vector4[1] * vector6[1]) / (n4 * n6)
    
    vector2_length = n2
    vector3_length = n3
    if vector3_length == 0 or vector2_length < 0.6 * vector3_length or vector2_length > 1.4 * vector3_length: return False
    if cos1 > 0.9 and cos2 > 0.9 and cos3 > 0.9 and cos4 > 0.9: return True
    return False

class StateMachine1():
    def __init__(self, max_count, other_count=3):
        self.other_to_any_time = 1
        self.any_to_any_time = 2
        self.state_time = time.time()
        self.current_state = 'other'
        self.no_lock_state = 'other'
        self.next_state = 'other'
        self.max_count = max_count
        self.next_count = 0
        self.action = None
        self.position_change = None
        self.other_count = other_count
        self.ok_time_stamp = time.time()
        self.max_ok_time = 0.7
        self.lock = False
        self.state_history = []
        self.state_history_times = []
        self.action_history = []
        self.action_history_times = []

    def _handle_state_history(self, state):
        self.state_history.append(state)
        self.state_history_times.append(time.time())
        while len(self.state_history_times) > 0 and time.time() - self.state_history_times[0] > 5:
            self.state_history.pop(0)
            self.state_history_times.pop(0)

    def _handle_action_history(self, action):
        self.action_history.append(action)
        self.action_history_times.append(time.time())
        while len(self.action_history_times) > 0 and time.time() - self.action_history_times[0] > 5:
            self.action_history.pop(0)
            self.action_history_times.pop(0)

    def change_state(self, state):
        self._handle_state_history(state)
        action = self._change_state(state)
        self._handle_action_history(action)
        if action in ['ok', 'sound_up', 'sound_down', 'light_up', 'light_down']:
            if time.time() - self.ok_time_stamp < self.max_ok_time: action = None
            else: self.ok_time_stamp = time.time()
        return action
            
    def _change_state(self, state):
        action = self.get_action()
        self.no_lock_state = self.next_state
        if self.current_state == self.next_state:
            self.state_time = time.time()
        elif self.current_state != self.next_state:
            if self.current_state == 'other' and time.time() - self.state_time > self.other_to_any_time:
                self.current_state = 'other' if len(self.state_history) < 1 else self.state_history[-1]
            elif self.current_state != 'other' and time.time() - self.state_time > self.any_to_any_time:
                self.current_state = 'other' if len(self.state_history) < 1 else self.state_history[-1]
        return action

    def get_action(self):
        if len(self.state_history) < 2: return None
        last_state = self.state_history[-1]
        if len(self.state_history) >= self.max_count and all([state == last_state for state in self.state_history[-self.max_count:]]):
            self.next_state = last_state
        
        if self.next_state == 'ok': return 'ok'
        if self.current_state == 'palm_up' and self.no_lock_state == 'palm_up':
            if self.next_state == 'fist': return 'screen_shot'
            if last_state == 'palm_down':
                self.next_state = 'palm_down'
                return 'down'
        if self.current_state == 'palm_down' and self.no_lock_state == 'palm_down':
            if last_state == 'palm_up':
                self.next_state = 'palm_up'
                return 'up'
        if self.current_state == 'palm_left' and self.no_lock_state == 'palm_left':
            if last_state == 'palm_right':
                self.next_state = 'palm_right'
                return 'right'
        if self.current_state == 'palm_right' and self.no_lock_state == 'palm_right':
            if last_state == 'palm_left':
                self.next_state = 'palm_left'
                return 'left'
        if self.next_state == 'two_up': return 'light_up'
        if self.next_state == 'two_down': return 'light_down'
        if self.next_state == 'two_left': return 'sound_down'
        if self.next_state == 'two_right': return 'sound_up'
        return None

class GestureModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.lstm1 = nn.LSTM(64, 32, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(64, 16, batch_first=True)
        self.classifier = nn.Linear(16, num_classes)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.permute(0, 2, 1)
        x, _ = self.lstm1(x)
        x, (h_n, c_n) = self.lstm2(x)
        x = x[:, -1, :]
        return self.classifier(x)

def background_gesture_task(camera_id, model_path, result_queue, tts_mp_q=None):
    import speaker
    speaker.init_mp_queue(tts_mp_q)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GestureModel(63, 10)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    index_2_gesture = {
        0: 'palm_up', 1: 'palm_down', 2: 'palm_left', 3: 'palm_right',
        4: 'fist', 5: 'two_up', 6: 'two_down', 7: 'two_left', 8: 'two_right', 9: 'ok'
    }
    gesture_chn_map = {
        'ok': 'OK', 'down': '向下', 'screen_shot': '截图', 'up': '向上',
        'right': '向右', 'left': '向左', 'light_up': '调亮', 'light_down': '调暗',
        'sound_up': '音量加', 'sound_down': '音量减'
    }

    window_name = "OpenBot Gesture Recognition"
    try:
        cap = CameraReader(src=camera_id)
    except Exception as e:
        print(f"无法打开摄像头设备: {e}")
        return

    state_machine = StateMachine1(3)
    gesture_times = {
        'left': 0, 'right': 0, 'up': 0, 'down': 0, 'screen_shot': 0,
        'sound_up': 0, 'sound_down': 0, 'light_up': 0, 'light_down': 0, 'ok': 0
    }

    record_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gesture_records")
    video_dir = os.path.join(record_dir, "videos")
    img_dir = os.path.join(record_dir, "screenshots")
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    
    session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(video_dir, f"gesture_{session_id}.mp4")
    video_writer = None

    final_action = None

    try:
        window_ready = False
        while cap.isOpened():
            success, image = cap.read()
            if success and image is not None:
                cv2.putText(image, "Camera Ready! Starting...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if video_writer is None:
                    h, w = image.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (w, h))
                if video_writer is not None:
                    video_writer.write(image)
            

                cv2.imshow(window_name, image)
                cv2.waitKey(200) 
                window_ready = True
                break
            time.sleep(0.01)

        if not window_ready:
            return

        start_time = time.time()

        with mp_hands.Hands(model_complexity=0, max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
            while cap.isOpened() and (time.time() - start_time) < 5.0:
                success, image = cap.read()
                if not success or image is None:
                    time.sleep(0.01)
                    continue

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )
                        single_frame = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                        direction = judge_gesture(single_frame)
                        data = np.array(single_frame).reshape(-1)
                        data = torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            output = model(data) 
                            output = torch.softmax(output, dim=1)
                            pred = torch.argmax(output, dim=1)
                            if output[0][pred.item()] > 0.95:
                                model_gesture = index_2_gesture[pred.item()]
                                if model_gesture == 'ok': direction = 'ok'
                                else: direction = model_gesture
                                action = state_machine.change_state(model_gesture)
                                if action is not None:
                                    gesture_times[action] += 1
                                    final_action = action
                                    break 

                time_left = max(0, 5.0 - (time.time() - start_time))
                cv2.putText(image, f"Detecting... {time_left:.1f}s", (400, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(image, f"State: {state_machine.current_state}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(image, f"L:{gesture_times['left']} R:{gesture_times['right']} U:{gesture_times['up']} D:{gesture_times['down']} Shot:{gesture_times['screen_shot']}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(image, f"Vol U:{gesture_times['sound_up']} D:{gesture_times['sound_down']} | Light U:{gesture_times['light_up']} D:{gesture_times['light_down']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(image, f"OK: {gesture_times['ok']}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                if video_writer is not None:
                    video_writer.write(image)
                
                if random.random() < 0.03:
                    img_name = f"shot_{session_id}_{int(time.time() * 1000)}.jpg"
                    cv2.imwrite(os.path.join(img_dir, img_name), image)

                cv2.imshow(window_name, image)
                cv2.waitKey(1)

                if final_action is not None: break
        
        print("[Debug]final_action:", final_action)

    finally:
        if video_writer is not None:
            video_writer.release()
        if cap is not None:
            cap.release()
        try:
            cv2.destroyAllWindows()
            for _ in range(5): 
                cv2.waitKey(1)
        except Exception:
            pass

    if final_action:
        action_chn = gesture_chn_map.get(final_action, final_action)
        result_queue.put(final_action)
        speak(f"这是{action_chn}手势")
        
    else:
        speak(f"对不起，我没有识别到手势")
        result_queue.put(None)
        
    time.sleep(0.2)

class GestureRecognitionSystem:
    def __init__(self, model_path='models/1-63-10/model_999.pth'):
        self.model_path = model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型权重: {model_path}")

    def identify_gesture(self, camera_id='/dev/video21'):
        import speaker 
        print("启动手势识别")
        speaker.speak("启动手势识别") 
        ctx = multiprocessing.get_context('spawn')
        result_q = ctx.Queue()
        if speaker._mp_q is None:
             speaker.init_mp_queue(ctx.Queue())
        
        p = ctx.Process(
            target=background_gesture_task,
            args=(camera_id, self.model_path, result_q, speaker._mp_q), 
            daemon=True
        )
        p.start()
        result = None
        try:
            result = result_q.get(timeout=10.0)
        except queue.Empty:
            print("手势识别超时 (等待了20秒没有结果)")
        except Exception as e:
            print(f"手势识别发生未知异常: {repr(e)}")
        p.join(timeout=5.0)
        if p.is_alive():
            p.terminate()
            p.join(timeout=2.0)
        print("手势识别进程已退出")
        return result