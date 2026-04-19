# SAM Interactive System
# 基于 Segment Anything Model 的智能图像识别分割检测系统

## 快速启动

```bash
cd sam-interactive-system
./start.sh
```

## 项目结构

```
sam-interactive-system/
├── frontend/           # React 前端 (端口 5173)
├── backend/            # FastAPI 后端 (端口 8000)
├── docs/               # 设计文档
├── start.sh            # 一键启动
└── stop.sh             # 停止服务
```

## 功能特性

- 🎯 点击分割 (Point-based Segmentation)
- 📦 框选分割 (Box-based Segmentation)  
- 🖌️ 交互式标注 (Interactive Annotation)
- 🎨 实时预览 (Real-time Preview)
- 📊 检测识别 (Object Detection)

## 技术栈

- **前端**: React + Vite + Canvas API
- **后端**: FastAPI + Segment Anything
- **AI**: SAM ViT-B/L/H
