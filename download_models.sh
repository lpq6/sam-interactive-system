#!/bin/bash
# download_models.sh — 下载所需模型文件
# 使用方法: bash download_models.sh

set -e
cd "$(dirname "$0")"

echo "=== 下载模型文件 ==="
echo ""

# SAM ViT-B (358MB)
SAM_DIR="backend/models"
SAM_FILE="$SAM_DIR/sam_vit_b_01ec64.pth"
SAM_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

if [ -f "$SAM_FILE" ]; then
    echo "[SKIP] SAM ViT-B 已存在: $SAM_FILE"
else
    echo "[下载] SAM ViT-B (358MB)..."
    mkdir -p "$SAM_DIR"
    curl -L -o "$SAM_FILE" "$SAM_URL"
    echo "[OK] SAM ViT-B 下载完成"
fi

# YOLOv8n (6.3MB)
YOLO_FILE="backend/yolov8n.pt"

if [ -f "$YOLO_FILE" ]; then
    echo "[SKIP] YOLOv8n 已存在: $YOLO_FILE"
else
    echo "[下载] YOLOv8n (6.3MB)..."
    curl -L -o "$YOLO_FILE" "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"
    echo "[OK] YOLOv8n 下载完成"
fi

echo ""
echo "=== 模型文件 ==="
ls -lh "$SAM_FILE" "$YOLO_FILE" 2>/dev/null
echo ""
echo "完成！可以启动服务了: bash start.sh"
