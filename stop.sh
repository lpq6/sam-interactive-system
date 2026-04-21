#!/bin/bash
# SAM Interactive System - 停止脚本
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "🛑 停止 SAM Interactive System..."

# 停止后端
if [ -f "$SCRIPT_DIR/backend.pid" ]; then
    PID=$(cat "$SCRIPT_DIR/backend.pid")
    kill $PID 2>/dev/null && echo "   ✅ 后端已停止 (PID: $PID)"
    rm "$SCRIPT_DIR/backend.pid"
fi

# 停止前端
if [ -f "$SCRIPT_DIR/frontend.pid" ]; then
    PID=$(cat "$SCRIPT_DIR/frontend.pid")
    kill $PID 2>/dev/null && echo "   ✅ 前端已停止 (PID: $PID)"
    rm "$SCRIPT_DIR/frontend.pid"
fi

# 备用: 通过端口停止
pkill -f "python.*app.py" 2>/dev/null
pkill -f "vite" 2>/dev/null

echo "✅ 所有服务已停止"
