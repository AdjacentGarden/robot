import os
import fcntl
import time
import struct
import math
import threading
# 注意：确保 test3.py 中的 set_rotation 函数存在
from test3 import set_rotation

# I2C 系统调用常量
I2C_SLAVE_FORCE = 0x0706

# 配置参数
I2C_BUS = 4          # 对应 /dev/i2c-4
DEVICE_ADDR = 0x6A   # ASM330LHH I2C 从机地址

# 寄存器地址
CTRL1_XL = 0x10      # 加速度计控制寄存器
CTRL2_G = 0x11       # 陀螺仪控制寄存器
OUTX_L_G = 0x22      # 陀螺仪 X 轴低字节起始地址

# 灵敏度常量
ACCEL_SCALE = 0.061 / 1000.0         # ±2g 量程转换为 g
GYRO_SCALE_RAD = 0.000152716         # ±250dps 直接转换为 rad/s (用于Mahony滤波)

# 全局变量（线程共享）
roll, pitch, yaw = 0.0, 0.0, 0.0
# 增加锁，保证线程安全读取/写入全局变量
imu_lock = threading.Lock()

class MahonyAHRS:
    """ Mahony 姿态解算滤波器 (面向对象封装，避免全局变量污染) """
    def __init__(self, Kp=2.1, Ki=0.000):
        self.Kp = Kp
        self.Ki = Ki
        # 初始四元数 q0, q1, q2, q3 (1, 0, 0, 0 表示无旋转)
        self.q = [1.0, 0.0, 0.0, 0.0]
        # 积分误差累计
        self.eInt = [0.0, 0.0, 0.0]

    def update(self, gx, gy, gz, ax, ay, az, dt):
        q0, q1, q2, q3 = self.q
        halfT = dt / 2.0

        # 如果加速度计全为0，无法作为参考基准，跳过
        if ax == 0.0 and ay == 0.0 and az == 0.0:
            return

        # 1. 归一化加速度计数据
        norm = math.sqrt(ax*ax + ay*ay + az*az)
        ax /= norm
        ay /= norm
        az /= norm

        # 2. 从当前四元数推算出重力在机体坐标系下的期望方向
        vx = 2.0 * (q1*q3 - q0*q2)
        vy = 2.0 * (q0*q1 + q2*q3)
        vz = q0*q0 - q1*q1 - q2*q2 + q3*q3

        # 3. 计算期望重力方向与实际测得重力方向的叉乘，得到误差
        ex = (ay*vz - az*vy)
        ey = (az*vx - ax*vz)
        ez = (ax*vy - ay*vx)

        # 4. 误差积分累计
        self.eInt[0] += ex * self.Ki
        self.eInt[1] += ey * self.Ki
        self.eInt[2] += ez * self.Ki

        # 5. 用比例项和积分项调整陀螺仪的测量值
        gx += self.Kp * ex + self.eInt[0]
        gy += self.Kp * ey + self.eInt[1]
        gz += self.Kp * ez + self.eInt[2]

        # 6. 一阶龙格库塔法积分四元数导数
        temp0, temp1, temp2, temp3 = q0, q1, q2, q3
        q0 += (-temp1*gx - temp2*gy - temp3*gz) * halfT
        q1 += ( temp0*gx + temp2*gz - temp3*gy) * halfT
        q2 += ( temp0*gy - temp1*gz + temp3*gx) * halfT
        q3 += ( temp0*gz + temp1*gy - temp2*gx) * halfT

        # 7. 归一化四元数
        norm = math.sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3)
        self.q = [q0/norm, q1/norm, q2/norm, q3/norm]

    def get_euler_angles(self):
        """ 将四元数转换为欧拉角 (Roll, Pitch, Yaw) 单位：度 """
        q0, q1, q2, q3 = self.q
        
        # Roll
        roll = math.atan2(2*(q2*q3 + q0*q1), 1 - 2*(q1*q1 + q2*q2))
        
        # Pitch (加入边界保护防止浮点数精度超限导致 math.asin 报错)
        pitch_val = -2*(q1*q3) + 2*(q0*q2)
        pitch_val = max(-1.0, min(1.0, pitch_val))
        pitch = math.asin(pitch_val)
        
        # Yaw
        yaw = math.atan2(2*(q1*q2 + q0*q3), 1 - 2*(q2*q2 + q3*q3))
        
        return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def init_sensor(fd):
    """初始化传感器配置 (12.5Hz)"""
    try:
        # 陀螺仪配置：0x10 = ODR_G_12.5Hz (12.5Hz输出率)
        os.write(fd, bytes([CTRL2_G, 0x10]))
        time.sleep(0.01)
        # 加速度计配置：0x10 = ODR_XL_12.5Hz (12.5Hz输出率)
        os.write(fd, bytes([CTRL1_XL, 0x10]))
        time.sleep(0.01)
        print("传感器初始化成功")
    except Exception as e:
        raise RuntimeError(f"传感器初始化失败: {e}")

def parse_16bit_2s_complement(low_byte, high_byte):
    """解析16位补码数据"""
    return struct.unpack('<h', bytes([low_byte, high_byte]))[0]

def main_loop(fd):
    """IMU主循环：静态校正 + 姿态解算"""
    global roll, pitch, yaw
    # 实例化滤波器
    ahrs = MahonyAHRS(Kp=10.0, Ki=0.008)
    last_time = time.time()

    # 1. 陀螺仪静态校正
    print("开始陀螺仪静态校正...")
    gx_bias, gy_bias, gz_bias = 0.0, 0.0, 0.0
    gx_sum, gy_sum, gz_sum = 0.0, 0.0, 0.0
    calibrate_num = 50  # 校正次数
    calibrate_success = 0

    for i in range(calibrate_num):
        try:
            # 写入寄存器地址，读取12字节数据（陀螺仪6字节 + 加速度计6字节）
            os.write(fd, bytes([OUTX_L_G]))
            data = os.read(fd, 12)
            
            if len(data) == 12:
                # 读取陀螺仪原始数据
                gx = parse_16bit_2s_complement(data[0], data[1])
                gy = parse_16bit_2s_complement(data[2], data[3])
                gz = parse_16bit_2s_complement(data[4], data[5])
                gx_sum += gx
                gy_sum += gy
                gz_sum += gz
                calibrate_success += 1
            
            time.sleep(0.08)  # 12.5Hz采样间隔
        except Exception as e:
            print(f"校正第{i+1}次失败: {e}")

    if calibrate_success > 0:
        gx_bias = gx_sum / calibrate_success
        gy_bias = gy_sum / calibrate_success
        gz_bias = gz_sum / calibrate_success
        print(f"陀螺仪校正完成 | 偏移值：gx={gx_bias:.2f}, gy={gy_bias:.2f}, gz={gz_bias:.2f}")
    else:
        print("陀螺仪校正失败，使用默认偏移(0)")

    # 2. 姿态解算主循环
    alpha = 0.3  # 低通滤波系数
    gx, gy, gz = 0.0, 0.0, 0.0
    ax, ay, az = 0.0, 0.0, 0.0
    print("开始姿态解算...")

    while True:
        try:
            # 读取传感器数据
            os.write(fd, bytes([OUTX_L_G]))
            data = os.read(fd, 12)
            
            if len(data) == 12:
                # 计算时间增量（动态dt，避免写死0.08）
                current_time = time.time()
                dt = current_time - last_time
                last_time = current_time
                if dt <= 0:
                    dt = 0.08  # 保底值
                
                # 陀螺仪数据：去偏移 + 低通滤波 + 单位转换(rad/s)
                raw_gx = parse_16bit_2s_complement(data[0], data[1]) - gx_bias
                raw_gy = parse_16bit_2s_complement(data[2], data[3]) - gy_bias
                raw_gz = parse_16bit_2s_complement(data[4], data[5]) - gz_bias
                gx = alpha * (raw_gx * GYRO_SCALE_RAD) + (1 - alpha) * gx
                gy = alpha * (raw_gy * GYRO_SCALE_RAD) + (1 - alpha) * gy
                gz = alpha * (raw_gz * GYRO_SCALE_RAD) + (1 - alpha) * gz

                # 加速度计数据：低通滤波 + 单位转换(g)
                raw_ax = parse_16bit_2s_complement(data[6], data[7])
                raw_ay = parse_16bit_2s_complement(data[8], data[9])
                raw_az = parse_16bit_2s_complement(data[10], data[11])
                ax = alpha * (raw_ax * ACCEL_SCALE) + (1 - alpha) * ax
                ay = alpha * (raw_ay * ACCEL_SCALE) + (1 - alpha) * ay
                az = alpha * (raw_az * ACCEL_SCALE) + (1 - alpha) * az

                # 更新姿态解算
                ahrs.update(gx, gy, gz, ax, ay, az, dt)
                # 获取欧拉角并更新全局变量（加锁保证线程安全）
                with imu_lock:
                    roll, pitch, yaw = ahrs.get_euler_angles()
                    # 归一化roll到0-360度
                    if roll < 0:
                        roll += 360

            time.sleep(0.08)  # 控制采样频率
        except Exception as e:
            print(f"姿态解算出错: {e}")
            time.sleep(0.08)

def imu_thread_func(stop_event):
    """IMU线程函数（处理stop_event退出逻辑）"""
    fd = None
    try:
        # 打开I2C设备
        i2c_dev_path = f"/dev/i2c-{I2C_BUS}"
        fd = os.open(i2c_dev_path, os.O_RDWR)
        fcntl.ioctl(fd, I2C_SLAVE_FORCE, DEVICE_ADDR)
        print(f"成功打开I2C设备: {i2c_dev_path} (地址: 0x{DEVICE_ADDR:02X})")
        
        # 初始化传感器
        init_sensor(fd)
        
        # 运行主循环（新增：支持通过stop_event退出）
        while not stop_event.is_set():
            main_loop(fd)
    except Exception as e:
        print(f"IMU线程初始化/运行失败: {e}")
    finally:
        # 关闭文件描述符
        if fd is not None:
            os.close(fd)
            print("I2C设备已关闭")

def get_current_roll():
    """线程安全获取当前roll值"""
    with imu_lock:
        return roll

def rotation_roll(target_roll):
    """控制Roll角到目标值"""
    bias = 8  # 误差允许范围
    # 限制目标值范围
    target_roll = max(163, min(220, target_roll))
    print(f"开始调整Roll角到 {target_roll} 度（误差允许±{bias}度）")

    try:
        while True:
            current_roll = get_current_roll()
            # 计算误差（处理360度环绕）
            diff = target_roll - current_roll
            if abs(diff) <= bias:
                set_rotation(0)
                print(f"Roll角已到位: 当前={current_roll:.2f}, 目标={target_roll}")
                break
            
            # 控制旋转方向
            if diff > bias:
                set_rotation(-60)  # 抬头（角度变大）
            elif diff < -bias:
                set_rotation(60)   # 低头（角度变小）
            
            print(f"当前Roll: {current_roll:.2f} | 目标: {target_roll} | 差值: {diff:.2f}")
            time.sleep(0.05)  # 降低循环频率，避免占用过多资源
    except Exception as e:
        set_rotation(0)
        print(f"Roll调整出错: {e}")

# 转速为负，向上抬头
# 转速为正，向下低头
# 抬头，角度变大
# 低头，角度变小
def set_head(state):
    """设置抬头状态"""
    if state == 0:
        rotation_roll(163)
    elif state == 1:
        rotation_roll(185)
    elif state == 2:
        rotation_roll(220)
    else:
        print("无效状态！仅支持0/1/2")

if __name__ == "__main__":
    # 初始化旋转为0
    set_rotation(0)
    
    # 创建停止事件，启动IMU线程
    stop_imu_event = threading.Event()
    imu_thread = threading.Thread(target=imu_thread_func, args=(stop_imu_event,), daemon=True)
    print("启动IMU线程...")
    imu_thread.start()
    
    # 等待线程初始化完成
    time.sleep(3)
    
    # 打印初始姿态
    print(f"初始姿态 | Roll: {get_current_roll():.2f}°, Pitch: {pitch:.2f}°, Yaw: {yaw:.2f}°")

    # 主交互循环
    try:
        while True:
            print("\n请输入指令：0-163度，1-185度，2-220度（输入q退出）")
            word = input().strip()
            if word == "q":
                print("退出程序...")
                stop_imu_event.set()
                imu_thread.join(timeout=2)
                break
            elif word in ["0", "1", "2"]:
                set_head(int(word))
            else:
                print("无效指令！请输入0/1/2或q退出")
    except KeyboardInterrupt:
        stop_imu_event.set()
        imu_thread.join(timeout=2)
        set_rotation(0)
        print("程序被中断，已停止IMU线程")