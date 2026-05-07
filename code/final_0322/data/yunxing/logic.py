#底层实现不要阻塞进程

def find_pet(target):
    print(f"[底层实现] 开始旋转寻找宠物: {target}")

def start_pet_tracking(target):
    print(f"[底层实现] 启动进程跟踪宠物: {target}")

def stop_pet_tracking():
    print("[底层实现] Kill 宠物跟踪进程")

def register_face(name):
    print(f"[底层实现] 录入人脸: {name}")

def recognize_face():
    print("[底层实现] 持续 5s 识别当前人脸")

def search_person(name):
    print(f"[底层实现] 开始搜寻人物: {name}")

def start_person_tracking(name):
    print(f"[底层实现] 启动进程跟踪人物: {name}")

def stop_person_tracking():
    print("[底层实现] Kill 人物跟踪进程")

def identify_gesture():
    print("[底层实现] 开启手势识别窗口")

def start_situp_counting():
    print("[底层实现] 启动仰卧起坐计数后台进程")

def query_situp_progress():
    print("[底层实现] 读取文件中的实时个数并播报")

def stop_and_summarize():
    print("[底层实现] 终止计数进程并汇总播报")