#!/bin/bash

PORT=8080

echo "🔍 检查端口 $PORT 占用情况..."

PIDS=$(lsof -t -i:$PORT)

if [ -z "$PIDS" ]; then
    echo "✅ 端口 $PORT 未被占用"
else
    echo "⚠️ 端口 $PORT 被占用，进程: $PIDS"
    echo "🛑 正在杀掉进程..."

    kill -9 $PIDS

    sleep 1

    echo "✅ 已释放端口 $PORT"
fi