#!/bin/bash
# SAM Interactive System - 一键启动脚本
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "╔══════════════════════════════════════════════╗"
echo "║  🎯 SAM Interactive System                  ║"
echo "║  基于 Segment Anything 的智能图像分割系统    ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# ── 检查依赖 ──
check_deps() {
    echo "📋 检查依赖..."
    
    if ! command -v python3 &>/dev/null; then
        echo "❌ 需要 Python 3.8+"
        exit 1
    fi
    
    if ! command -v node &>/dev/null; then
        echo "❌ 需要 Node.js 16+"
        exit 1
    fi
    
    echo "✅ 依赖检查通过"
}

# ── 设置后端 ──
setup_backend() {
    echo ""
    echo "🐍 设置后端..."
    
    cd "$SCRIPT_DIR/backend"
    
    # 直接使用Anaconda环境（GPU支持）
    echo "   使用 Anaconda GPU 环境: /mnt/d/Anaconda/envs/machine_learning"
    
    # 安装依赖
    echo "   安装 Python 依赖..."
    "/mnt/d/Anaconda/envs/machine_learning/python.exe" -m pip install -q fastapi uvicorn python-multipart pillow numpy
    
    # 检查 SAM 是否已安装
    if ! "/mnt/d/Anaconda/envs/machine_learning/python.exe" -c "import segment_anything" 2>/dev/null; then
        echo "   安装 segment_anything..."
        "/mnt/d/Anaconda/envs/machine_learning/python.exe" -m pip install -q segment-anything
    fi
    
    echo "✅ 后端设置完成"
}

# ── 设置前端 ──
setup_frontend() {
    echo ""
    echo "⚛️  设置前端..."
    
    cd "$SCRIPT_DIR/frontend"
    
    if [ ! -d "node_modules" ]; then
        echo "   安装 npm 依赖..."
        npm install --silent
    fi
    
    echo "✅ 前端设置完成"
}

# ── 检查GPU状态 ──
check_gpu_status() {
    echo ""
    echo "🖥️  检查GPU状态..."
    
    if python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null; then
        import_result=$(python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA devices:', torch.cuda.device_count())")
        echo "   ${import_result}"
        
        if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
            echo "   ✅ GPU可用 - 将使用GPU加速推理"
        else
            echo "   ⚠️  GPU不可用 (驱动过旧)，使用CPU模式"
        fi
    else
        echo "   ⚠️  无法导入torch模块"
    fi
}

# ── 下载模型 ──
download_models() {
    echo ""
    echo "📥 检查模型文件..."
    
    mkdir -p "$SCRIPT_DIR/backend/models"
    cd "$SCRIPT_DIR/backend/models"
    
    MODELS=(
        "sam_vit_b_01ec64.pth|https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth|375MB"
    )
    
    for entry in "${MODELS[@]}"; do
        IFS='|' read -r filename url size <<< "$entry"
        if [ ! -f "$filename" ]; then
            echo "   ⚠️  $filename 不存在 ($size)"
            echo "   请手动下载: $url"
            echo "   放到: $SCRIPT_DIR/backend/models/"
            echo ""
            read -p "   是否现在下载? (y/N) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                echo "   下载中... (可能需要几分钟)"
                curl -L -o "$filename" "$url" --progress-bar
                echo "   ✅ 下载完成"
            fi
        else
            echo "   ✅ $filename 已存在"
        fi
    done
}

# ── 启动服务 ──
start_services() {
    echo ""
    echo "🚀 启动服务..."
    
    # 启动后端（使用Anaconda GPU环境）
    echo "   启动后端 (端口 8000)..."
    nohup "/mnt/d/Anaconda/envs/machine_learning/python.exe" /mnt/d/OpenClaw_Workspace_full/sam-interactive-system/backend/app.py > "$SCRIPT_DIR/backend.log" 2>&1 &
    BACKEND_PID=$!
    echo $BACKEND_PID > "$SCRIPT_DIR/backend.pid"
    
    # 等待后端启动
    sleep 5
    
    # 启动前端
    cd "$SCRIPT_DIR/frontend"
    echo "   启动前端 (端口 5173)..."
    nohup npm run dev > "$SCRIPT_DIR/frontend.log" 2>&1 &
    FRONTEND_PID=$!
    echo $FRONTEND_PID > "$SCRIPT_DIR/frontend.pid"
    
    sleep 3
    
    echo ""
    echo "╔══════════════════════════════════════════════╗"
    echo "║  ✅ 系统启动成功!                           ║"
    echo "╠══════════════════════════════════════════════╣"
    echo "║  🎮 GPU模式已启用                          ║"
    echo "║  🌐 前端: http://localhost:5173             ║"
    echo "║  📡 后端: http://localhost:8000             ║"
    echo "║  📚 API文档: http://localhost:8000/docs     ║"
    echo "╠══════════════════════════════════════════════╣"
    echo "║  📋 日志:                                   ║"
    echo "║     后端: tail -f backend.log               ║"
    echo "║     前端: tail -f frontend.log              ║"
    echo "║  🛑 停止: ./stop.sh                         ║"
    echo "╚══════════════════════════════════════════════╝"
}

# ── 主流程 ──
main() {
    check_deps
    setup_backend
    setup_frontend
    check_gpu_status
    download_models
    start_services
}

main "$@"
