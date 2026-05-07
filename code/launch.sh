#!/bin/bash

echo "🚀 等待系统和硬件初始化..."

# 等摄像头
while [ ! -e /dev/video40 ]; do
    echo "⏳ 等待摄像头..."
    sleep 1
done

# 等麦克风设备
while ! arecord -l | grep -q "rockchipi2sdmic"; do
    echo "⏳ 等待麦克风..."
    sleep 1
done

echo "✅ 硬件已就绪，启动主程序"

cd /home/test/code
exec /usr/bin/python3 pet_system_v2.py