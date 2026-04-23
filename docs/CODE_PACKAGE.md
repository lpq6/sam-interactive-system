# SAM 交互式分割系统 — 完整代码包

> **版本**: v1.20.0 | **日期**: 2026-04-23 | **总代码量**: ~6600 行

---

## 目录

1. [项目概览](#1-项目概览)
2. [系统架构](#2-系统架构)
3. [核心代码演示](#3-核心代码演示) **（50%）**
   - 3.1 模型加载 — SAM / ResNet / YOLO
   - 3.2 COCO→ImageNet 同义词映射引擎
   - 3.3 YOLO+SAM 联合检测流水线
   - 3.4 识别 API — YOLO→SAM→ResNet 三级流水线
   - 3.5 批量彩色物体提取
   - 3.6 图像工具函数集
   - 3.7 前端核心 — Canvas 交互 + API 调用
4. [API 接口一览](#4-api-接口一览)
5. [评估实验](#5-评估实验)
6. [部署与运行](#6-部署与运行)

---

## 1. 项目概览

### 1.1 功能矩阵

| 功能 | API | 按钮 | 说明 |
|------|-----|------|------|
| 交互式分割 | `/api/segment/point`, `/api/segment/box` | ✏️ 点击/框选 | SAM 精确 mask |
| 自动检测 | `/api/detect/auto` | 🔍 自动检测 | YOLO+SAM 联合，mask 叠加图 |
| 自动分割 | `/api/segment/auto` | ✨ 自动分割 | SAM 网格采样，纯 mask |
| 识别 | `/api/recognize` | 🏷️ 识别 | YOLO→SAM→ResNet 三级流水线 |
| 提取彩色 | `/api/extract/all` | 🎨 提取彩色 | YOLO→SAM+ResNet，透明 PNG |
| mask 编辑 | `/api/mask/edit` | 🖌️ 画笔/橡皮 | 历史记录 + 撤销/重做 |
| 导出 | `/api/export/*` | 📥 导出 | JSON/CSV/COCO/YOLO 格式 |
| 视频处理 | `/api/video/*` | 🎬 视频 | 帧级分割 + 摄像头流 |

### 1.2 技术栈

```
前端: React 18 + Vite + Canvas API
后端: FastAPI + Uvicorn + PyTorch
模型: SAM ViT-B (91M) + YOLOv8n (3.2M) + ResNet50 (25.6M)
GPU:  NVIDIA RTX 3060 Laptop, 6GB, CUDA 12.0
```

---

## 2. 系统架构

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    React Frontend (5173)                     │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐         │
│  │ 上传 │  │ 点击 │  │ 自动 │  │ 识别 │  │ 导出 │         │
│  └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘         │
│     │         │         │         │         │               │
│     └─────────┴─────────┴────┬────┴─────────┘               │
│                              │ Canvas + Overlay 渲染         │
└──────────────────────────────┼──────────────────────────────┘
                               │ HTTP REST
┌──────────────────────────────┼──────────────────────────────┐
│                    FastAPI Backend (8000)                    │
│                              │                               │
│  ┌───────────────────────────┼───────────────────────┐      │
│  │              API Gateway / Router                  │      │
│  └───────┬──────────┬────────┼────────┬──────────────┘      │
│          │          │        │        │                      │
│  ┌───────┴───┐ ┌────┴────┐ ┌┴──────┐ ┌┴───────┐            │
│  │ YOLOv8n   │ │ SAM     │ │ResNet │ │ Tools  │            │
│  │ 检测      │ │ 分割    │ │50 识别│ │ mask   │            │
│  │ 80类 bbox │ │ 精确mask│ │1000类 │ │ 编辑   │            │
│  └───────┬───┘ └────┬────┘ └┬──────┘ └────────┘            │
│          │          │       │                                │
│          └──────────┴───┬───┘                                │
│                         │                                    │
│              ┌──────────┴──────────┐                        │
│              │  COCO→ImageNet 映射  │                        │
│              │  80类 × 1700+ 同义词 │                        │
│              └─────────────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 核心流水线

```
用户上传图片
      │
      ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  YOLOv8n     │───▶│  SAM ViT-B   │───▶│  ResNet50    │
│  目标检测    │    │  精确分割    │    │  图像分类    │
│  bbox + 类名 │    │  mask        │    │  Top-50      │
└──────────────┘    └──────────────┘    └──────┬───────┘
                                               │
                                               ▼
                                     ┌──────────────────┐
                                     │ COCO→ImageNet    │
                                     │ 同义词匹配       │
                                     │ → 最终标签       │
                                     └──────────────────┘
```

---

## 3. 核心代码演示

> 以下展示系统关键模块的实现代码，包含详细注释。

<!-- PLACEHOLDER_PART2 -->

### 3.1 模型加载 — SAM / ResNet / YOLO

#### 3.1.1 SAM 预测器（支持 GPU/CPU，mock 兜底）

```python
class SAMPredictor:
    """SAM 模型管理器 — 支持 vit_b/vit_l/vit_h 三种规格"""

    def __init__(self):
        self.predictor = None
        self.device = "cpu"
        self.model_type = None

    def _init_device(self):
        if torch.cuda.is_available():
            self.device = "cuda"
            print(f"[GPU] Using GPU: {torch.cuda.get_device_name()}")
        else:
            print("[CPU] Using CPU (slower inference)")

    def load_model(self, model_type: str = "vit_b") -> bool:
        """加载 SAM 模型，检查点映射："""
        checkpoint_map = {
            "vit_b": "sam_vit_b_01ec64.pth",   # 375MB, 91M params
            "vit_l": "sam_vit_l_0b3195.pth",   # 1.2GB, 308M params
            "vit_h": "sam_vit_h_4b8939.pth",   # 2.6GB, 636M params
        }
        ckpt_path = MODELS_DIR / checkpoint_map.get(model_type, "")
        if not ckpt_path.exists():
            return False

        sam = sam_model_registry[model_type](checkpoint=str(ckpt_path))
        sam.to(self.device)
        self.predictor = SamPredictor(sam)
        self.model_type = model_type
        return True

    def predict_point(self, points, labels, multimask=True):
        """点提示分割 — 返回多候选 mask + 置信度"""
        if self.predictor:
            masks, scores, _ = self.predictor.predict(
                point_coords=points, point_labels=labels,
                multimask_output=multimask
            )
            return masks, scores
        return self._mock_point_predict(points)  # 无模型时 mock

    def predict_box(self, box):
        """框提示分割 — bbox → 精确 mask"""
        if self.predictor:
            masks, scores, _ = self.predictor.predict(
                box=box, multimask_output=True
            )
            return masks, scores
        return self._mock_box_predict(box)
```

#### 3.1.2 ResNet50 图像分类器

```python
class ImageClassifier:
    """ResNet50 ImageNet 1K 分类器 — 用于物体识别"""

    def load(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.model.to(self.device).eval()
        self.labels = ResNet50_Weights.IMAGENET1K_V1.meta["categories"]
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def classify(self, image: Image.Image, top_k: int = 50):
        """输入 PIL 图片，返回 Top-K 预测 [{label, prob}]"""
        input_tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.nn.functional.softmax(output[0], dim=0)
        top = torch.topk(probs, top_k)
        return [{"label": self.labels[idx], "prob": float(prob)}
                for idx, prob in zip(top.indices, top.values)]
```

#### 3.1.3 YOLOv8n 目标检测器

```python
class YOLODetector:
    """YOLOv8n COCO 80 类目标检测器"""

    def load(self) -> bool:
        from ultralytics import YOLO
        model_path = Path(__file__).parent / "yolov8n.pt"  # 6.2MB
        self.model = YOLO(str(model_path))
        self.names = self.model.names  # {0: 'person', 1: 'bicycle', ...}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[OK] YOLOv8n loaded, {len(self.names)} classes")
        return True

    def detect(self, img, conf_thresh=0.25, max_det=30) -> list:
        """返回 [{label, confidence, bbox: [x1,y1,x2,y2], class_id}]"""
        results = self.model.predict(
            source=img, conf=conf_thresh, max_det=max_det,
            verbose=False, device=self.device
        )
        detections = []
        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
                    detections.append({
                        "label": self.names.get(cls_id, f"class_{cls_id}"),
                        "confidence": round(conf, 3),
                        "bbox": [x1, y1, x2, y2],
                        "class_id": cls_id,
                    })
        return detections
```

### 3.2 COCO→ImageNet 同义词映射引擎

**核心问题**：ResNet50 预训练于 ImageNet (1000 类)，但 YOLO 使用 COCO (80 类)。两类标签体系不一致（如 YOLO: "car" vs ImageNet: "limousine"），需要映射桥接。

```python
COCO_TO_IMAGENET = {
    "person": [
        "person","man","woman","boy","girl","human","bridegroom","groom",
        "scuba diver","parachutist","backpacker","skier","swimmer","cowboy",
        "academic gown","apron","bib","poncho","raincoat","jersey","sweatshirt",
        # ... 共 35 个同义词
    ],
    "car": [
        "car","limousine","sports car","minivan","jeep","cab","convertible",
        "coupe","hatchback","sedan","taxi","police car","race car","beach wagon",
        # ... 共 18 个同义词
    ],
    "dog": [
        "dog","golden retriever","labrador","german shepherd","poodle","bulldog",
        "beagle","chihuahua","husky","dalmatian","collie","pug","corgi",
        "rottweiler","doberman","boxer","shih tzu","malamute","pomeranian",
        # ... 共 34 个同义词
    ],
    # ... 80 类全覆盖，总计 1700+ 个同义词
}

def match_coco_class(yolo_class: str, top_labels: list, top_probs: list):
    """
    判断 ResNet Top-K 是否命中 COCO 类别的同义词集合

    Args:
        yolo_class: YOLO 检测到的 COCO 类名 (如 "car")
        top_labels: ResNet Top-K 标签列表 (如 ["limousine", "sports car", ...])
        top_probs:  ResNet Top-K 概率列表

    Returns:
        (is_match: bool, match_prob: float, match_label: str)
    """
    if yolo_class not in COCO_TO_IMAGENET:
        return False, 0.0, ""
    synonyms = set(s.lower() for s in COCO_TO_IMAGENET[yolo_class])
    for label, prob in zip(top_labels, top_probs):
        if label.lower() in synonyms:
            return True, round(prob, 3), label
    return False, 0.0, ""
```

**消融实验结果**：

| 方案 | 匹配率 | 说明 |
|------|--------|------|
| 无映射 (直接字符串) | 6.7% | "car" vs "limousine" → 不匹配 |
| Top-5 + 基础映射 | 49.1% | 搜索范围小 |
| Top-20 + 完整映射 | 87.2% | 扩大搜索 + 补充同义词 |
| **Top-50 + 扩展映射** | **95.3%** | **当前版本，16 类同义词扩展** |

### 3.3 YOLO+SAM 联合检测流水线

```python
def yolo_sam_detect(img, conf_thresh=0.25, max_det=30,
                     min_area=100, overlay_alpha=0.4) -> dict:
    """
    YOLO 检测 + SAM 分割联合流水线:

    1. YOLO 检测 → bbox + label + confidence
    2. SAM 用 bbox 做精确分割 → mask + score
    3. 渲染彩色叠加图

    Returns:
        {
            "success": True,
            "count": 12,
            "detections": [{
                "id": 1,
                "label": "person",       # YOLO COCO 标签（可靠）
                "confidence": 0.92,
                "sam_score": 0.98,       # SAM mask 置信度
                "bbox": [100, 50, 300, 400],
                "area": 58000,
                "mask": "<base64>",      # 单独 mask
            }, ...],
            "overlay": "<base64>",       # 彩色叠加图
            "method": "yolo+sam"
        }
    """
    # Step 1: YOLO detection
    yolo_dets = yolo.detect(img, conf_thresh=conf_thresh, max_det=max_det)
    if not yolo_dets:
        return {"success": True, "count": 0, "detections": [], ...}

    # Step 2: SAM segmentation for each YOLO bbox
    sam.set_image(img)
    overlay = img.copy().astype(np.float32)
    detections = []

    for i, det in enumerate(yolo_dets):
        x1, y1, x2, y2 = det["bbox"]
        area = (x2 - x1) * (y2 - y1)
        if area < min_area:
            continue

        # SAM box segmentation → 精确 mask
        box_np = np.array([x1, y1, x2, y2])
        masks, scores = sam.predict_box(box_np)
        best_idx = int(np.argmax(scores))
        mask = masks[best_idx]
        sam_score = float(scores[best_idx])

        # 计算 mask 的实际 bbox
        ys, xs = np.where(mask)
        if len(ys) > 0:
            mx1, my1, mx2, my2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        else:
            mx1, my1, mx2, my2 = x1, y1, x2, y2

        # 生成颜色叠加
        color = colors[i % len(colors)]
        alpha = overlay_alpha
        for c in range(3):
            overlay[:, :, c] = np.where(mask,
                overlay[:, :, c] * (1 - alpha) + color[c] * alpha,
                overlay[:, :, c])

        # mask → base64
        mask_uint8 = (mask.astype(np.uint8) * 255)
        mask_b64 = mask_to_base64(mask_uint8)

        detections.append({
            "id": i + 1,
            "label": det["label"],
            "confidence": det["confidence"],
            "sam_score": round(sam_score, 3),
            "bbox": [mx1, my1, mx2, my2],
            "area": int(mask.sum()),
            "mask": mask_b64,
        })

    # Step 3: 生成叠加图
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    overlay_b64 = encode_image_base64(overlay)

    return {
        "success": True, "count": len(detections),
        "detections": detections, "overlay": overlay_b64,
        "method": "yolo+sam",
    }
```

<!-- PLACEHOLDER_PART3 -->

### 3.4 识别 API — YOLO→SAM→ResNet 三级流水线

```python
@app.post("/api/recognize")
async def recognize_object(image_id: str = None, file: UploadFile = None):
    """
    🏷️ 识别 — 逐物体识别 + 场景分析

    流水线:
      1. YOLO 检测 → bbox + label
      2. SAM 分割 → 精确 mask
      3. mask 裁剪 → 去除背景
      4. ResNet Top-50 分类
      5. COCO→ImageNet 同义词匹配
      6. 返回逐物体结果 + 全图场景分析
    """
    img = np.array(Image.open(path).convert("RGB"))

    # ── 场景分析（保留原有功能）──
    mean_color = img.mean(axis=(0, 1))
    brightness = float(mean_color.mean())
    dominant_colors = extract_dominant_colors(img)
    if brightness > 180:    scene = "明亮场景"
    elif brightness > 120:  scene = "正常光照"
    elif brightness > 80:   scene = "较暗场景"
    else:                   scene = "暗光场景"

    # ── YOLO→SAM→ResNet 流水线识别 ──
    object_results = []
    if yolo.model and classifier.model:
        yolo_dets = yolo.detect(img, conf_thresh=0.25, max_det=20)
        if yolo_dets:
            sam.set_image(img)
            for det in yolo_dets:
                x1, y1, x2, y2 = det["bbox"]
                box_np = np.array([x1, y1, x2, y2])

                # SAM 精确分割
                masks, scores = sam.predict_box(box_np)
                best_idx = int(np.argmax(scores))
                mask = masks[best_idx]

                # mask 裁剪 + ResNet 分类 (Top-50)
                crop_pil = crop_object_with_mask(img, mask, margin=10)
                resnet_result = classifier.classify(crop_pil, top_k=50)

                # COCO→ImageNet 同义词匹配
                is_match, match_prob, match_label = match_coco_class(
                    det["label"],
                    [r["label"] for r in resnet_result],
                    [r["prob"] for r in resnet_result]
                )

                object_results.append({
                    "yolo_class": det["label"],                    # 主标签 (COCO)
                    "yolo_confidence": det["confidence"],
                    "resnet_label": match_label if is_match else resnet_result[0]["label"],
                    "resnet_confidence": match_prob if is_match else resnet_result[0]["prob"],
                    "coco_matched": is_match,                     # 是否命中同义词
                    "top5": [{"label": r["label"], "prob": round(r["prob"], 3)}
                             for r in resnet_result[:5]],
                    "bbox": det["bbox"],
                })

    return {
        "success": True,
        "scene": scene,
        "brightness": round(brightness, 1),
        "dominant_colors": [{"rgb": list(c), "ratio": round(r, 3)}
                            for c, r in dominant_colors],
        "objects": object_results,          # 逐物体识别结果
        "method": "yolo+sam+resnet",
    }
```

**前端识别结果展示**：

```jsx
{/* 🏷️ 识别结果 */}
{recognition && (
  <div style={{ background: '#1e1e2e', borderRadius: 12, padding: 16 }}>
    <h3>🏷️ 识别结果 ({recognition.method})</h3>

    {/* 逐物体识别 */}
    {recognition.objects?.map((obj, i) => (
      <div key={i} style={{ borderBottom: '1px solid #333', padding: '8px 0' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ color: '#4fc3f7', fontWeight: 'bold' }}>
            #{i + 1} {obj.yolo_class}
          </span>
          {obj.coco_matched && (
            <span style={{ color: '#66bb6a', fontSize: 12 }}>
              ✓ ResNet 确认: {obj.resnet_label}
            </span>
          )}
        </div>
        <div style={{ fontSize: 12, color: '#aaa', marginTop: 4 }}>
          YOLO: {(obj.yolo_confidence * 100).toFixed(1)}%
          {' | '}
          ResNet: {(obj.resnet_confidence * 100).toFixed(1)}%
        </div>
        {/* Top-5 备选标签 */}
        <div style={{ fontSize: 11, color: '#888', marginTop: 2 }}>
          Top-5: {obj.top5?.map(t => `${t.label} ${(t.prob*100).toFixed(1)}%`).join(', ')}
        </div>
      </div>
    ))}
  </div>
)}
```

### 3.5 批量彩色物体提取

```python
@app.post("/api/extract/all")
async def extract_all_objects(image_id: str, min_area: int = 500):
    """
    🎨 批量提取彩色物体 — YOLO→SAM+ResNet 流水线

    返回每个物体的:
      - 透明背景 PNG (base64)
      - mask (base64)
      - YOLO 标签 + ResNet 识别结果
    """
    img = np.array(Image.open(path).convert("RGB"))

    # YOLO 检测
    yolo_dets = yolo.detect(img, conf_thresh=0.25, max_det=20)
    sam.set_image(img)
    objects = []

    for det in yolo_dets:
        x1, y1, x2, y2 = det["bbox"]
        box_np = np.array([x1, y1, x2, y2])

        # SAM 分割 → mask
        masks, scores = sam.predict_box(box_np)
        best_idx = int(np.argmax(scores)]
        mask = masks[best_idx]

        # ResNet 分类 + COCO 映射（主标签仍是 YOLO）
        crop_pil = crop_object_with_mask(img, mask, margin=10)
        result = classifier.classify(crop_pil, top_k=50)
        is_match, match_prob, match_label = match_coco_class(
            det["label"],
            [r["label"] for r in result],
            [r["prob"] for r in result]
        )

        # 提取彩色物体（精修掩码 + 平滑边缘 + 透明背景）
        refined_mask = refine_mask(mask, erode_px=3)
        rgba = create_rgba_from_mask(img, refined_mask, smooth=True)
        ys, xs = np.where(refined_mask)
        cropped = rgba[ys.min():ys.max()+1, xs.min():xs.max()+1]

        # PNG 编码
        result_img = Image.fromarray(cropped, 'RGBA')
        buffer = io.BytesIO()
        result_img.save(buffer, format='PNG')
        color_b64 = base64.b64encode(buffer.getvalue()).decode()

        objects.append({
            "id": len(objects) + 1,
            "label": det["label"],           # YOLO 标签
            "resnet_label": match_label if is_match else result[0]["label"],
            "resnet_matched": is_match,
            "confidence": det["confidence"],
            "color_image": color_b64,        # 透明 PNG
            "mask": mask_b64,
            "bbox": bbox,
            "area": mask_area,
            "source": "yolo+sam+resnet",
        })

    return {"success": True, "count": len(objects), "objects": objects}
```

### 3.6 图像工具函数集

#### 3.6.1 mask 裁剪（去除背景干扰）

```python
def crop_object_with_mask(img: np.ndarray, mask: np.ndarray, margin: int = 10) -> Image.Image:
    """
    用 mask 裁剪物体 — 将背景设为灰色，提升 ResNet 分类准确率

    之前: 直接 bbox 裁剪 → 包含大量背景 → ResNet 被干扰
    现在: mask 裁剪 → 背景变灰色 → ResNet 聚焦物体
    """
    # 计算 mask 边界
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return Image.fromarray(img)

    y1, y2 = max(0, int(ys.min()) - margin), min(img.shape[0], int(ys.max()) + margin)
    x1, x2 = max(0, int(xs.min()) - margin), min(img.shape[1], int(xs.max()) + margin)

    # 裁剪
    img_crop = img[y1:y2, x1:x2].copy()
    mask_crop = mask[y1:y2, x1:x2]

    # 背景设为灰色 (128, 128, 128) — ImageNet 均值附近
    bg = np.full_like(img_crop, 128)
    img_crop = np.where(mask_crop[:, :, np.newaxis], img_crop, bg)

    return Image.fromarray(img_crop)
```

#### 3.6.2 mask 精修（腐蚀去碎片）

```python
def refine_mask(mask: np.ndarray, erode_px: int = 3) -> np.ndarray:
    """精修 mask — 去除零散碎片，平滑边缘"""
    mask_uint8 = (mask.astype(np.uint8) * 255)

    # 去除小碎片
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        max_label = 1 + np.argmax(areas)
        mask_uint8 = ((labels == max_label).astype(np.uint8) * 255)

    # 腐蚀边缘（去除 SAM 过度扩展）
    if erode_px > 0:
        kernel = np.ones((erode_px, erode_px), np.uint8)
        mask_uint8 = cv2.erode(mask_uint8, kernel, iterations=1)

    return mask_uint8 > 128
```

#### 3.6.3 透明背景 RGBA 合成

```python
def create_rgba_from_mask(img, mask, smooth=True, blur_radius=3):
    """从原图 + mask 生成 RGBA（透明背景）"""
    h, w = img.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = img  # RGB

    if smooth:
        alpha = smooth_mask(mask, blur_radius=blur_radius, feather=True)
    else:
        alpha = (mask.astype(np.uint8) * 255)

    rgba[:, :, 3] = alpha
    return rgba

def smooth_mask(mask, blur_radius=3, feather=True):
    """平滑 mask 边缘 — 高斯模糊 + 羽化"""
    alpha = (mask.astype(np.uint8) * 255)
    if feather:
        alpha = cv2.GaussianBlur(alpha, (blur_radius * 2 + 1, blur_radius * 2 + 1), 0)
    return alpha
```

<!-- PLACEHOLDER_PART4 -->

### 3.7 前端核心 — Canvas 交互 + API 调用

#### 3.7.1 主组件结构

```jsx
// frontend/src/App.jsx — 核心状态管理
export default function App() {
  // ── 状态 ──
  const [image, setImage] = useState(null)        // 当前图片
  const [mask, setMask] = useState(null)           // 当前 mask (base64)
  const [mode, setMode] = useState('point')        // 模式: point/box/auto/detect/recognize/extract
  const [overlayData, setOverlayData] = useState(null)  // 叠加图数据
  const [recognizeData, setRecognizeData] = useState(null)  // 识别结果
  const [extractData, setExtractData] = useState(null)  // 提取结果
  const [autoDetectData, setAutoDetectData] = useState(null)  // 自动检测结果
  const [history, setHistory] = useState([])       // 分割历史
  const [brushSize, setBrushSize] = useState(15)   // 画笔大小
  const [isDrawing, setIsDrawing] = useState(false)
  const canvasRef = useRef(null)
  // ... 更多状态

  // ── 初始化 ──
  useEffect(() => {
    fetch(`${API}/api/health`).then(r => r.json()).then(setHealth)
    fetch(`${API}/api/models`).then(r => r.json()).then(data => {
      setModels(data.models)
      setCurrentModel(data.current)
    })
  }, [])
```

#### 3.7.2 四大功能按钮调用

```jsx
// 🔍 自动检测 — YOLO+SAM 联合
const runAutoDetect = async () => {
  setMode('detect')
  const res = await fetch(`${API}/api/detect/auto`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image_id: image.id, min_mask_region_area: 100 }),
  })
  const data = await res.json()
  setAutoDetectData(data)
  setOverlayData(data.overlay)
  // 显示检测结果卡片
}

// ✨ 自动分割 — SAM 网格采样
const runAutoSegment = async () => {
  setMode('auto')
  const res = await fetch(`${API}/api/segment/auto`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image_id: image.id, points_per_side: 20 }),
  })
  const data = await res.json()
  setOverlayData(data.overlay)
}

// 🏷️ 识别 — YOLO→SAM→ResNet 三级流水线
const runRecognize = async () => {
  setMode('recognize')
  const res = await fetch(`${API}/api/recognize?image_id=${image.id}`)
  const data = await res.json()
  setRecognizeData(data)
  // data.objects[] 包含逐物体识别结果
}

// 🎨 提取彩色 — YOLO→SAM+ResNet，透明 PNG
const runExtractColors = async () => {
  setMode('extract')
  const res = await fetch(`${API}/api/extract/all?image_id=${image.id}&min_area=500`)
  const data = await res.json()
  setExtractData(data)
}
```

#### 3.7.3 Canvas 点击分割

```jsx
// Canvas 点击事件 — 交互式 SAM 分割
const handleCanvasClick = async (e) => {
  if (mode !== 'point' || !image) return

  const canvas = canvasRef.current
  const rect = canvas.getBoundingClientRect()
  const scaleX = canvas.width / rect.width
  const scaleY = canvas.height / rect.height
  const x = Math.round((e.clientX - rect.left) * scaleX)
  const y = Math.round((e.clientY - rect.top) * scaleY)

  // 前景点 (label=1) / 背景点 (label=0)
  const label = e.shiftKey ? 0 : 1
  const newPoints = [...points, [x, y]]
  const newLabels = [...labels, label]
  setPoints(newPoints)
  setLabels(newLabels)

  // 调用 SAM 点分割 API
  const res = await fetch(`${API}/api/segment/point`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      image_id: image.id,
      points: newPoints,
      labels: newLabels,
    }),
  })
  const data = await res.json()
  setMask(data.mask)
  setScore(data.score)
}
```

#### 3.7.4 Canvas 叠加渲染

```jsx
// Canvas 渲染 — 图片 + mask 叠加 + 检测框
useEffect(() => {
  const canvas = canvasRef.current
  const ctx = canvas.getContext('2d')
  if (!image || !canvas) return

  const img = new Image()
  img.onload = () => {
    canvas.width = img.width
    canvas.height = img.height
    ctx.drawImage(img, 0, 0)

    // 叠加 mask
    if (mask) {
      const maskImg = new Image()
      maskImg.onload = () => {
        ctx.globalAlpha = 0.5
        ctx.drawImage(maskImg, 0, 0)
        ctx.globalAlpha = 1.0
      }
      maskImg.src = `data:image/png;base64,${mask}`
    }

    // 叠加自动检测结果
    if (overlayData && mode === 'detect') {
      const olImg = new Image()
      olImg.onload = () => {
        ctx.drawImage(olImg, 0, 0)
        // 绘制检测框 + 标签
        autoDetectData?.detections?.forEach(det => {
          const [x1, y1, x2, y2] = det.bbox
          ctx.strokeStyle = '#4fc3f7'
          ctx.lineWidth = 2
          ctx.strokeRect(x1, y1, x2-x1, y2-y1)
          ctx.fillStyle = 'rgba(0,0,0,0.7)'
          ctx.fillRect(x1, y1-20, 120, 20)
          ctx.fillStyle = '#fff'
          ctx.font = '12px monospace'
          ctx.fillText(`${det.label} ${(det.confidence*100).toFixed(0)}%`, x1+4, y1-6)
        })
      }
      olImg.src = `data:image/png;base64,${overlayData}`
    }

    // 绘制点击点
    points.forEach((pt, i) => {
      ctx.beginPath()
      ctx.arc(pt[0], pt[1], 6, 0, 2 * Math.PI)
      ctx.fillStyle = labels[i] === 1 ? '#4caf50' : '#f44336'
      ctx.fill()
      ctx.strokeStyle = '#fff'
      ctx.lineWidth = 2
      ctx.stroke()
    })
  }
  img.src = image.url
}, [image, mask, overlayData, points, labels, mode])
```

#### 3.7.5 导出功能

```jsx
{/* 导出菜单 */}
{['json', 'csv', 'coco', 'yolo'].map(fmt => (
  <button key={fmt} onClick={() => {
    const a = document.createElement('a')
    a.href = `${API}/api/export/${fmt}/${image.id}`
    a.download = `segmentation.${fmt === 'coco' ? 'json' : fmt}`
    a.click()
  }}>
    📥 {fmt.toUpperCase()}
  </button>
))}
```

<!-- PLACEHOLDER_PART5 -->

## 4. API 接口一览

### 4.1 核心 API

| 方法 | 路径 | 功能 | 说明 |
|------|------|------|------|
| `GET` | `/api/health` | 健康检查 | 返回模型状态 |
| `GET` | `/api/models` | 模型列表 | 当前加载的模型 |
| `POST` | `/api/models/switch` | 切换模型 | vit_b / vit_l / vit_h |
| `POST` | `/api/upload` | 上传图片 | 返回 image_id |
| `POST` | `/api/upload/batch` | 批量上传 | 返回多个 image_id |
| `GET` | `/api/image/{image_id}` | 获取图片 | 返回图片文件 |

### 4.2 分割 API

| 方法 | 路径 | 功能 | 参数 |
|------|------|------|------|
| `POST` | `/api/segment/point` | 点提示分割 | points, labels |
| `POST` | `/api/segment/box` | 框提示分割 | box [x1,y1,x2,y2] |
| `POST` | `/api/segment/multi` | 多候选分割 | points, labels, multimask |
| `POST` | `/api/detect/auto` | 🔍 自动检测 | min_mask_region_area |
| `POST` | `/api/segment/auto` | ✨ 自动分割 | points_per_side, max_objects |

### 4.3 识别 API

| 方法 | 路径 | 功能 | 返回 |
|------|------|------|------|
| `POST` | `/api/recognize` | 🏷️ 识别 | objects[], scene, dominant_colors |
| `POST` | `/api/extract/color` | 单物体提取 | color_image (PNG base64) |
| `POST` | `/api/extract/all` | 🎨 批量提取 | objects[] (透明 PNG) |

### 4.4 mask 编辑 API

| 方法 | 路径 | 功能 |
|------|------|------|
| `POST` | `/api/mask/edit` | 画笔/橡皮编辑 |
| `POST` | `/api/mask/undo` | 撤销 |
| `POST` | `/api/mask/redo` | 重做 |

### 4.5 导出 API

| 方法 | 路径 | 格式 |
|------|------|------|
| `GET` | `/api/export/json/{image_id}` | JSON |
| `GET` | `/api/export/csv/{image_id}` | CSV |
| `GET` | `/api/export/coco/{image_id}` | COCO JSON |
| `GET` | `/api/export/yolo/{image_id}` | YOLO TXT |

### 4.6 历史 & 视频 API

| 方法 | 路径 | 功能 |
|------|------|------|
| `GET` | `/api/history/{image_id}` | 获取分割历史 |
| `POST` | `/api/history/get` | 获取历史详情 |
| `DELETE` | `/api/history/{image_id}/{entry_id}` | 删除历史条目 |
| `POST` | `/api/video/upload` | 上传视频 |
| `GET` | `/api/video/{video_id}/frames` | 获取视频帧 |
| `POST` | `/api/video/segment/frame` | 分割视频帧 |

---

## 5. 评估实验

### 5.1 评估设置

| 项目 | 配置 |
|------|------|
| 数据集 | COCO val2017 (86 张子集, 10 目标类) |
| 目标类别 | person, car, dog, cat, bird, bottle, chair, couch, dining table, tv |
| 检测模型 | YOLOv8n (3.2M 参数) |
| 分割模型 | SAM ViT-B (91M 参数) |
| 识别模型 | ResNet50 (ImageNet 1K) + 1700+ 同义词 |
| GPU | NVIDIA RTX 3060 Laptop, 6GB, CUDA 12.0 |

### 5.2 端到端评估结果 (v4)

#### YOLOv8n 目标检测

| 指标 | 值 |
|------|-----|
| Precision | **0.7273** |
| Recall | **0.5540** |
| F1-Score | **0.6289** |
| mAP@50 | **0.4587** |
| mIoU (bbox) | **0.8615** |
| 速度 | 0.061s/image |

#### SAM ViT-B 图像分割

| 指标 | 值 |
|------|-----|
| mIoU (mask) | **0.5594** |
| 平均速度 | 0.21s/mask |
| 总 mask 数 | 275 |

#### ResNet50 物体识别

| 评估方式 | Top-1 | Top-5 | 匹配率 |
|----------|-------|-------|--------|
| 全图分类 (86 张) | 9.6% | 16.0% | — |
| YOLO+SAM+映射 (86 张) | — | — | **95.3%** (262/275) |
| 单张丰富场景 | — | — | **92.9%** (13/14) |

### 5.3 消融实验

| 方案 | 匹配率 | 提升 |
|------|--------|------|
| 无映射 (直接字符串匹配) | 6.7% | baseline |
| Top-5 + 基础映射 | 49.1% | +42.4pp |
| Top-20 + 基础映射 | 76.3% | +69.6pp |
| Top-20 + 完整映射 (80 类) | 87.2% | +80.5pp |
| **Top-50 + 扩展映射 (1700+ 同义词)** | **95.3%** | **+88.6pp** |

### 5.4 各类模型大小对比

| 模型 | 参数量 | 显存占用 | 用途 |
|------|--------|----------|------|
| YOLOv8n | 3.2M | ~50MB | 目标检测 |
| SAM ViT-B | 91M | ~1.5GB | 图像分割 |
| ResNet50 | 25.6M | ~400MB | 图像分类 |
| **总计** | **~120M** | **~2GB** | 三模型共存 |

---

## 6. 部署与运行

### 6.1 环境要求

```
Python: 3.10+
PyTorch: 2.0+ (CUDA 12.0)
Node.js: 18+
GPU: 推荐 6GB+ (RTX 3060 级别)
```

### 6.2 安装

```bash
# 后端
cd backend
pip install fastapi uvicorn torch torchvision segment_anything ultralytics pillow numpy opencv-python

# 下载 SAM 模型
mkdir -p models
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P models/

# 前端
cd frontend
npm install
```

### 6.3 启动

```bash
# 启动后端 (端口 8000)
cd backend && python app.py

# 启动前端 (端口 5173)
cd frontend && npm run dev

# 访问 http://localhost:5173
```

### 6.4 项目结构

```
sam-interactive-system/
├── backend/
│   ├── app.py              # 后端主文件 (3835 行)
│   ├── yolov8n.pt          # YOLO 模型 (6.2MB)
│   └── models/
│       └── sam_vit_b_01ec64.pth  # SAM 模型 (375MB)
├── frontend/
│   ├── src/
│   │   ├── App.jsx         # 前端主组件 (2777 行)
│   │   ├── main.jsx        # 入口
│   │   └── styles.css      # 样式
│   ├── index.html
│   ├── vite.config.js
│   └── package.json
├── docs/
│   ├── REPORT.md           # 技术报告
│   ├── DEMO.md             # 功能演示
│   ├── FEATURE_GUIDE.md    # 功能指南
│   └── CODE_PACKAGE.md     # 本文档
└── README.md
```

---

> **文档版本**: v1.20.0 (2026-04-23)
> **评估数据**: v4 (Top-50 + 扩展同义词, 匹配率 95.3%)
> **代码行数**: 后端 3835 行 + 前端 2777 行 = 6612 行




