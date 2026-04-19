# SAM Interactive System v1.17.0

基于 Segment Anything Model 的智能图像识别分割检测系统

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
├── models/             # 预训练模型
├── docs/               # 设计文档
├── CHANGELOG.md        # 更新日志
├── start.sh            # 一键启动
└── stop.sh             # 停止服务
```

## 功能特性

### 核心功能
- 🎯 点击分割 (Point-based Segmentation)
- 📦 框选分割 (Box-based Segmentation)
- 🖌️ 交互式标注 (Interactive Annotation)
- 🎨 实时预览 (Real-time Preview)
- 📊 检测识别 (Object Detection)
- 🌈 彩色物体提取
- ⌨️ 键盘快捷键
- 📏 置信度过滤
- 🖼️ 边缘平滑
- 📁 批量处理
- 🌓 暗色/亮色主题
- ✏️ 掩码编辑（擦除/画笔）
- 📜 分割历史记录
- 🔄 模型切换（ViT-B/L/H）
- 📤 导出格式扩展（JSON/CSV）
- 📱 移动端适配

### 视频功能
- 🎬 视频帧分割
- 📷 摄像头实时流分割
- 🖥️ MJPEG 流预览

### 导出功能
- 📦 COCO 格式标注导出
- 📦 YOLO 格式标注导出

### 机器学习
- 🎯 自定义类别训练
- 📊 模型评估可视化
- 💾 模型导出/导入
- 📚 批量训练
- 🎨 数据增强

## 技术栈

- **前端**: React 18 + Vite + Canvas API
- **后端**: FastAPI + Uvicorn + PyTorch
- **AI 模型**: 
  - SAM: ViT-B (375MB), ViT-L (1.2GB), ViT-H (2.4GB)
  - ResNet: ResNet18/50 (分类器)
- **图像处理**: OpenCV + PIL + NumPy
- **深度学习**: PyTorch + torchvision

## 模型下载

- ViT-B: 375MB (默认)
- ViT-L: 1.2GB
- ViT-H: 2.4GB (最高精度)

模型文件位于 `backend/models/` 目录。

## API 端点

### 图像处理
- `POST /api/upload` - 上传图像
- `POST /api/segment` - 执行分割
- `POST /api/recognize` - 识别图像
- `POST /api/extract/all` - 提取彩色物体

### 视频处理
- `POST /api/video/upload` - 上传视频
- `GET /api/video/{id}/stream` - 视频流
- `GET /api/camera/list` - 摄像头列表
- `POST /api/camera/stop` - 停止摄像头

### 导出
- `GET /api/export/json/{id}` - JSON 导出
- `GET /api/export/csv/{id}` - CSV 导出
- `GET /api/export/coco/{id}` - COCO 导出
- `GET /api/export/yolo/{id}` - YOLO 导出

### 机器学习
- `POST /api/custom/train` - 训练分类器
- `POST /api/custom/batch-train` - 批量训练
- `GET /api/custom/classes` - 获取类别
- `POST /api/custom/predict` - 预测
- `GET /api/custom/evaluate` - 模型评估
- `GET /api/custom/export` - 导出模型
- `POST /api/custom/import` - 导入模型
- `GET /api/custom/augmentation` - 数据增强状态

## 使用说明

1. **启动系统**
   ```bash
   ./start.sh
   ```

2. **访问界面**
   - 前端: http://localhost:5173
   - API: http://localhost:8000

3. **上传图像**
   - 点击"上传图片"按钮
   - 或拖拽图片到上传区域

4. **执行分割**
   - 使用点击工具选择目标
   - 或使用框选工具选择区域

5. **导出结果**
   - 点击导出按钮选择格式
   - 支持 JSON/CSV/COCO/YOLO

6. **训练模型**
   - 添加自定义类别
   - 添加训练样本
   - 点击"开始训练"

## 系统要求

- Python 3.8+
- Node.js 16+
- CUDA (推荐，用于 GPU 加速)
- 内存: 8GB+ (推荐 16GB)
- 显存: 4GB+ (推荐 8GB)

## 许可证

MIT License
