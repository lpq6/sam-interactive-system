"""
SAM 交互式分割系统 - 后端服务
基于 Facebook Segment Anything Model
"""
import os, io, uuid, base64, json
from pathlib import Path
from typing import Optional, List
from datetime import datetime
from contextlib import asynccontextmanager

import numpy as np
from PIL import Image, ImageFilter
import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel

# ── 配置 ──
import tempfile
UPLOAD_DIR = Path(tempfile.gettempdir()) / "sam_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
MODELS_DIR = Path(__file__).parent / "models"

# ── 数据模型 ──
class PointPrompt(BaseModel):
    image_id: str
    points: List[List[float]]
    labels: List[int]  # 1=前景, 0=背景

class BoxPrompt(BaseModel):
    image_id: str
    box: List[float]  # [x1, y1, x2, y2]

class MultiMaskRequest(BaseModel):
    image_id: str
    points: List[List[float]]
    labels: List[int]
    multimask: bool = True

class AutoDetectRequest(BaseModel):
    image_id: str
    points_per_side: int = 32  # 采样点密度
    pred_iou_thresh: float = 0.88
    stability_score_thresh: float = 0.95
    min_mask_region_area: int = 100

class AutoSegmentRequest(BaseModel):
    image_id: str
    points_per_side: int = 20
    min_mask_region_area: int = 300
    max_objects: int = 15

# ── SAM 预测器管理 ──
class SAMPredictor:
    def __init__(self):
        self.predictor = None
        self.device = "cpu"
        self.model_type = None
        self._init_device()

    def _init_device(self):
        try:
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
                print(f"[GPU] Using GPU: {torch.cuda.get_device_name()}")
            else:
                print("[CPU] Using CPU (slower inference)")
        except ImportError:
            print("[WARN] PyTorch not installed")

    def load_model(self, model_type: str = "vit_b") -> bool:
        if self.model_type == model_type and self.predictor:
            return True

        try:
            from segment_anything import sam_model_registry, SamPredictor

            checkpoint_map = {
                "vit_b": "sam_vit_b_01ec64.pth",
                "vit_l": "sam_vit_l_0b3195.pth",
                "vit_h": "sam_vit_h_4b8939.pth",
            }

            ckpt_path = MODELS_DIR / checkpoint_map.get(model_type, "")
            if not ckpt_path.exists():
                print(f"[WARN] Model file not found: {ckpt_path}")
                print(f"   Download from https://github.com/facebookresearch/segment-anything")
                return False

            sam = sam_model_registry[model_type](checkpoint=str(ckpt_path))
            sam.to(self.device)
            self.predictor = SamPredictor(sam)
            self.model_type = model_type
            print(f"[OK] Model loaded: {model_type} on {self.device}")
            return True

        except ImportError:
            print("[WARN] segment_anything not installed, using mock mode")
            return False
        except Exception as e:
            print(f"[ERROR] Model load failed: {e}")
            return False

    def set_image(self, image: np.ndarray):
        if self.predictor:
            self.predictor.set_image(image.astype(np.uint8))
        self._current_image = image

    def predict_point(self, points: np.ndarray, labels: np.ndarray,
                      multimask: bool = True):
        if self.predictor:
            masks, scores, _ = self.predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=multimask
            )
            return masks, scores
        return self._mock_point_predict(points)

    def predict_box(self, box: np.ndarray):
        if self.predictor:
            masks, scores, _ = self.predictor.predict(
                box=box,
                multimask_output=True
            )
            return masks, scores
        return self._mock_box_predict(box)

    def _mock_point_predict(self, points):
        """模拟预测 - 无模型时使用"""
        h, w = self._current_image.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        for pt in points:
            x, y = int(pt[0]), int(pt[1])
            y1, y2 = max(0, y-40), min(h, y+40)
            x1, x2 = max(0, x-40), min(w, x+40)
            mask[y1:y2, x1:x2] = True
        return np.stack([mask]), np.array([0.85])

    def _mock_box_predict(self, box):
        h, w = self._current_image.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        x1, y1, x2, y2 = map(int, box)
        mask[max(0,y1):min(h,y2), max(0,x1):min(w,x2)] = True
        return np.stack([mask]), np.array([0.90])

sam = SAMPredictor()

# ── ResNet 物体识别模型 ──
class ImageClassifier:
    def __init__(self):
        self.model = None
        self.labels = None
        self.transform = None
        self.device = "cpu"

    def load(self):
        try:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            self.model.to(self.device)
            self.model.eval()
            self.labels = ResNet50_Weights.IMAGENET1K_V1.meta["categories"]
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            print(f"[OK] ResNet50 loaded on {self.device}")
            return True
        except Exception as e:
            print(f"[WARN] ResNet load failed: {e}")
            return False

    def classify(self, image: Image.Image, top_k: int = 5):
        if not self.model:
            return None
        input_tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.nn.functional.softmax(output[0], dim=0)
        top = torch.topk(probs, top_k)
        return [{"label": self.labels[idx], "prob": float(prob)}
                for idx, prob in zip(top.indices, top.values)]

classifier = ImageClassifier()

# ── 图像工具 ──
def mask_to_base64(mask: np.ndarray) -> str:
    img = Image.fromarray((mask * 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def smooth_mask(mask: np.ndarray, blur_radius: int = 3, feather: bool = True) -> np.ndarray:
    """
    平滑掩码边缘
    
    参数:
        mask: 二值掩码 (bool 或 0-255)
        blur_radius: 高斯模糊半径
        feather: 是否使用羽化效果
    
    返回:
        平滑后的掩码 (0-255 uint8)
    """
    # 转换为 PIL Image
    if mask.dtype == bool:
        mask_uint8 = (mask * 255).astype(np.uint8)
    else:
        mask_uint8 = mask.astype(np.uint8)
    
    mask_img = Image.fromarray(mask_uint8, 'L')
    
    if feather:
        # 羽化效果：先膨胀再模糊
        from PIL import ImageFilter
        # 轻微膨胀
        dilated = mask_img.filter(ImageFilter.MaxFilter(size=blur_radius * 2 + 1))
        # 高斯模糊
        smoothed = dilated.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    else:
        # 仅高斯模糊
        smoothed = mask_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    return np.array(smoothed)


def create_rgba_from_mask(img: np.ndarray, mask: np.ndarray, smooth: bool = True, blur_radius: int = 5) -> np.ndarray:
    """
    从掩码创建 RGBA 图像（透明背景）
    
    参数:
        img: RGB 图像
        mask: 分割掩码
        smooth: 是否平滑边缘
        blur_radius: 模糊半径（默认5，更大的值边缘更平滑）
    
    返回:
        RGBA 图像 (透明背景)
    """
    h, w = img.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = img
    
    if smooth:
        # 平滑掩码边缘
        alpha = smooth_mask(mask, blur_radius=blur_radius, feather=True)
        rgba[:, :, 3] = alpha
    else:
        # 原始二值掩码
        rgba[:, :, 3] = np.where(mask, 255, 0)
    
    return rgba

def create_overlay(image: np.ndarray, mask: np.ndarray,
                   color=(0, 200, 255), alpha=0.45) -> str:
    """创建原图+掩码叠加效果"""
    overlay = image.copy().astype(np.float32)
    mask_3d = np.stack([mask]*3, axis=-1)
    color_arr = np.array(color, dtype=np.float32)
    # 在掩码区域叠加颜色
    overlay = np.where(mask_3d,
                       overlay * (1 - alpha) + color_arr * alpha,
                       overlay)
    result = Image.fromarray(overlay.astype(np.uint8))
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def create_overlay(image: np.ndarray, mask: np.ndarray,
                   color=(255, 165, 0), alpha=0.45) -> str:
    overlay = image.copy().astype(np.float32)
    color_arr = np.array(color, dtype=np.float32)
    overlay[mask] = overlay[mask] * (1 - alpha) + color_arr * alpha
    return mask_to_base64(overlay.astype(np.uint8)[:,:,0])  # simplified

def find_image(image_id: str) -> Optional[Path]:
    for f in UPLOAD_DIR.glob(f"{image_id}.*"):
        if f.suffix.lower() in {'.jpg','.jpeg','.png','.webp','.bmp'}:
            return f
    return None


# ── 掩码编辑工具 ──
def apply_brush_stroke(mask: np.ndarray, x: int, y: int, radius: int = 10, 
                       mode: str = 'add') -> np.ndarray:
    """
    在掩码上应用画笔笔触
    
    参数:
        mask: 原始掩码
        x, y: 笔触中心坐标
        radius: 笔触半径
        mode: 'add' (添加) 或 'erase' (擦除)
    
    返回:
        修改后的掩码
    """
    h, w = mask.shape[:2]
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - x)**2 + (Y - y)**2)
    stroke = dist <= radius
    
    if mode == 'add':
        return mask | stroke
    else:  # erase
        return mask & ~stroke


def apply_brush_strokes(mask: np.ndarray, strokes: List[dict]) -> np.ndarray:
    """
    批量应用画笔笔触
    
    参数:
        mask: 原始掩码
        strokes: 笔触列表 [{"x": int, "y": int, "radius": int, "mode": "add"|"erase"}]
    
    返回:
        修改后的掩码
    """
    result = mask.copy()
    for stroke in strokes:
        result = apply_brush_stroke(
            result,
            stroke['x'],
            stroke['y'],
            stroke.get('radius', 10),
            stroke.get('mode', 'add')
        )
    return result


# ── 掩码历史记录 ──
class MaskHistory:
    def __init__(self, max_size: int = 20):
        self.history = []
        self.current = -1
        self.max_size = max_size
    
    def push(self, mask: np.ndarray):
        """添加新掩码到历史"""
        # 删除当前位置之后的历史
        self.history = self.history[:self.current + 1]
        # 添加新掩码
        self.history.append(mask.copy())
        # 限制历史大小
        if len(self.history) > self.max_size:
            self.history.pop(0)
        self.current = len(self.history) - 1
    
    def undo(self) -> Optional[np.ndarray]:
        """撤销到上一步"""
        if self.current > 0:
            self.current -= 1
            return self.history[self.current].copy()
        return None
    
    def redo(self) -> Optional[np.ndarray]:
        """重做到下一步"""
        if self.current < len(self.history) - 1:
            self.current += 1
            return self.history[self.current].copy()
        return None
    
    def get_current(self) -> Optional[np.ndarray]:
        """获取当前掩码"""
        if 0 <= self.current < len(self.history):
            return self.history[self.current].copy()
        return None
    
    def clear(self):
        """清空历史"""
        self.history = []
        self.current = -1


# ── 分割历史记录 ──
class SegmentationHistory:
    def __init__(self, max_size: int = 50):
        self.history = []  # [{"id", "timestamp", "tool", "points", "box", "mask", "overlay", "score", "area", "label"}]
        self.max_size = max_size
        self.next_id = 1
    
    def add(self, tool: str, mask_b64: str, overlay_b64: str, score: float, 
            area: int, label: str = None, points: List = None, box: List = None) -> dict:
        """添加分割结果到历史"""
        import time
        entry = {
            "id": self.next_id,
            "timestamp": int(time.time() * 1000),
            "tool": tool,
            "points": points or [],
            "box": box or [],
            "mask": mask_b64,
            "overlay": overlay_b64,
            "score": score,
            "area": area,
            "label": label or "分割区域"
        }
        self.next_id += 1
        self.history.append(entry)
        
        # 限制历史大小
        if len(self.history) > self.max_size:
            self.history.pop(0)
        
        return entry
    
    def get_all(self) -> List[dict]:
        """获取所有历史记录"""
        return self.history.copy()
    
    def get_by_id(self, entry_id: int) -> Optional[dict]:
        """根据 ID 获取历史记录"""
        for entry in self.history:
            if entry["id"] == entry_id:
                return entry
        return None
    
    def delete(self, entry_id: int) -> bool:
        """删除历史记录"""
        for i, entry in enumerate(self.history):
            if entry["id"] == entry_id:
                self.history.pop(i)
                return True
        return False
    
    def clear(self):
        """清空历史"""
        self.history = []
        self.next_id = 1

# 全局分割历史记录
segmentation_histories = {}  # image_id -> SegmentationHistory

# 全局掩码历史记录
mask_histories = {}  # image_id -> MaskHistory

# ── FastAPI App ──
app = FastAPI(title="SAM Interactive System", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API Routes ──

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "device": sam.device,
        "model_loaded": sam.predictor is not None,
        "model_type": sam.model_type
    }

@app.get("/api/models")
async def list_models():
    import glob
    available = []
    for name, file in [("vit_b","SAM ViT-B (375MB)"),("vit_l","SAM ViT-L (1.2GB)"),("vit_h","SAM ViT-H (2.4GB)")]:
        # 检查模型文件是否存在
        model_exists = False
        if MODELS_DIR.exists():
            pattern = str(MODELS_DIR / f"sam_{name}_*.pth")
            model_exists = len(glob.glob(pattern)) > 0
        
        available.append({
            "id": name,
            "name": file,
            "loaded": sam.model_type == name,
            "exists": model_exists
        })
    return {"models": available, "current": sam.model_type, "device": sam.device}

@app.post("/api/models/switch")
async def switch_model(model: dict):
    model_type = model.get("model_type")
    if model_type not in ["vit_b", "vit_l", "vit_h"]:
        raise HTTPException(400, "无效的模型类型")
    
    if sam.model_type == model_type:
        return {"success": True, "message": f"模型 {model_type} 已加载", "model_type": model_type}
    
    # 检查模型文件是否存在
    if MODELS_DIR.exists():
        import glob
        pattern = str(MODELS_DIR / f"sam_{model_type}_*.pth")
        matches = glob.glob(pattern)
        if not matches:
            raise HTTPException(404, f"模型文件不存在: {model_type}")
    
    # 加载新模型
    success = sam.load_model(model_type)
    if success:
        return {"success": True, "message": f"已切换到 {model_type}", "model_type": model_type, "device": sam.device}
    else:
        raise HTTPException(500, f"加载模型失败: {model_type}")

@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "请上传图片文件")

    image_id = uuid.uuid4().hex[:10]
    ext = Path(file.filename).suffix if file.filename else ".png"
    dest = UPLOAD_DIR / f"{image_id}{ext}"
    dest.write_bytes(await file.read())

    img = Image.open(dest).convert("RGB")
    return {
        "image_id": image_id,
        "width": img.width,
        "height": img.height,
        "filename": file.filename
    }

@app.post("/api/upload/batch")
async def upload_batch(files: List[UploadFile] = File(...)):
    """
    批量上传图片
    
    返回:
        所有上传图片的 ID 列表
    """
    results = []
    for file in files:
        if not file.content_type or not file.content_type.startswith("image/"):
            continue
        
        image_id = uuid.uuid4().hex[:10]
        ext = Path(file.filename).suffix if file.filename else ".png"
        dest = UPLOAD_DIR / f"{image_id}{ext}"
        dest.write_bytes(await file.read())
        
        img = Image.open(dest).convert("RGB")
        results.append({
            "image_id": image_id,
            "width": img.width,
            "height": img.height,
            "filename": file.filename
        })
    
    return {
        "success": True,
        "count": len(results),
        "images": results
    }

@app.post("/api/batch/process")
async def batch_process(image_ids: List[str], min_confidence: float = 0.3):
    """
    批量处理图片 - 自动检测 + 识别
    
    参数:
        image_ids: 图片 ID 列表
        min_confidence: 最低置信度阈值
    
    返回:
        每张图片的检测结果
    """
    results = []
    
    for image_id in image_ids:
        path = find_image(image_id)
        if not path:
            results.append({"image_id": image_id, "success": False, "error": "图片不存在"})
            continue
        
        try:
            img = np.array(Image.open(path).convert("RGB"))
            h, w = img.shape[:2]
            sam.set_image(img)
            
            # 网格采样点
            step = max(w, h) // 20
            step = max(step, 40)
            
            detections = []
            used_masks = []
            generic_labels = {"未知", "冷色物体", "暖色物体", "区域-light", "区域-dark"}
            
            for y in range(step//2, h, step):
                for x in range(step//2, w, step):
                    if len(detections) >= 10:
                        break
                    
                    try:
                        masks, scores, _ = sam.predictor.predict(
                            point_coords=np.array([[x, y]]),
                            point_labels=np.array([1]),
                            multimask_output=False
                        )
                        
                        mask = masks[0]
                        area = int(mask.sum())
                        
                        if area < 500:
                            continue
                        
                        # 检查重复
                        is_duplicate = False
                        for used in used_masks:
                            overlap = (mask & used).sum() / min(mask.sum(), used.sum())
                            if overlap > 0.5:
                                is_duplicate = True
                                break
                        if is_duplicate:
                            continue
                        
                        used_masks.append(mask)
                        
                        # 计算边界框
                        ys, xs = np.where(mask)
                        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                        
                        # 用 ResNet 识别
                        region_label = "未知"
                        region_prob = 0.0
                        if classifier.model:
                            try:
                                x1, y1, x2, y2 = bbox
                                margin = 5
                                crop = img[max(0,y1-margin):min(h,y2+margin), max(0,x1-margin):min(w,x2+margin)]
                                if crop.size > 0:
                                    crop_pil = Image.fromarray(crop)
                                    result = classifier.classify(crop_pil, top_k=1)
                                    if result:
                                        region_label = result[0]["label"]
                                        region_prob = result[0]["prob"]
                            except:
                                pass
                        
                        # 跳过低置信度
                        if region_prob < min_confidence or region_label in generic_labels:
                            continue
                        
                        detections.append({
                            "label": region_label,
                            "confidence": round(region_prob, 3),
                            "bbox": bbox,
                            "area": area
                        })
                        
                    except Exception:
                        continue
            
            results.append({
                "image_id": image_id,
                "success": True,
                "detections": detections,
                "count": len(detections)
            })
            
        except Exception as e:
            results.append({"image_id": image_id, "success": False, "error": str(e)})
    
    return {
        "success": True,
        "total": len(results),
        "results": results
    }

@app.get("/api/image/{image_id}")
async def get_image(image_id: str):
    path = find_image(image_id)
    if not path:
        raise HTTPException(404, "图片不存在")
    return FileResponse(path)

@app.post("/api/segment/point")
async def segment_by_point(req: PointPrompt):
    path = find_image(req.image_id)
    if not path:
        raise HTTPException(404, "请先上传图片")

    img = np.array(Image.open(path).convert("RGB"))
    sam.set_image(img)

    pts = np.array(req.points)
    lbs = np.array(req.labels)
    masks, scores = sam.predict_point(pts, lbs)

    best = int(np.argmax(scores))
    mask = masks[best]

    ys, xs = np.where(mask)
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) > 0 else None

    # 生成平滑掩码
    smooth_alpha = smooth_mask(mask, blur_radius=5, feather=True)
    smooth_mask_img = Image.fromarray(smooth_alpha, 'L')
    buf = io.BytesIO()
    smooth_mask_img.save(buf, format="PNG")
    smooth_mask_b64 = base64.b64encode(buf.getvalue()).decode()

    # 生成平滑 RGBA 图像
    rgba = create_rgba_from_mask(img, mask, smooth=True, blur_radius=5)
    rgba_img = Image.fromarray(rgba, 'RGBA')
    rgba_buf = io.BytesIO()
    rgba_img.save(rgba_buf, format="PNG")
    rgba_b64 = base64.b64encode(rgba_buf.getvalue()).decode()

    # 生成叠加图
    overlay_b64 = create_overlay(img, mask)
    
    # 保存到历史记录
    if req.image_id not in segmentation_histories:
        segmentation_histories[req.image_id] = SegmentationHistory()
    
    # 尝试识别物体标签
    label = "分割区域"
    if classifier.model:
        try:
            x1, y1, x2, y2 = bbox
            margin = 5
            crop = img[max(0,y1-margin):min(h,y2+margin), max(0,x1-margin):min(w,x2+margin)]
            if crop.size > 0:
                crop_pil = Image.fromarray(crop)
                result = classifier.classify(crop_pil, top_k=1)
                if result:
                    label = result[0]["label"]
        except:
            pass
    
    history_entry = segmentation_histories[req.image_id].add(
        tool="point",
        mask_b64=smooth_mask_b64,
        overlay_b64=overlay_b64,
        score=float(scores[best]),
        area=int(mask.sum()),
        label=label,
        points=req.points
    )

    return {
        "success": True,
        "mask": smooth_mask_b64,  # 平滑掩码
        "color_image": rgba_b64,  # 平滑彩色图像
        "overlay": overlay_b64,
        "score": float(scores[best]),
        "bbox": bbox,
        "area": int(mask.sum()),
        "history_id": history_entry["id"]
    }

@app.post("/api/segment/box")
async def segment_by_box(req: BoxPrompt):
    path = find_image(req.image_id)
    if not path:
        raise HTTPException(404, "请先上传图片")

    img = np.array(Image.open(path).convert("RGB"))
    h, w = img.shape[:2]
    sam.set_image(img)

    box = np.array(req.box)
    masks, scores = sam.predict_box(box)

    best = int(np.argmax(scores))
    mask = masks[best]

    ys, xs = np.where(mask)
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) > 0 else None

    # 生成平滑掩码
    smooth_alpha = smooth_mask(mask, blur_radius=5, feather=True)
    smooth_mask_img = Image.fromarray(smooth_alpha, 'L')
    buf = io.BytesIO()
    smooth_mask_img.save(buf, format="PNG")
    smooth_mask_b64 = base64.b64encode(buf.getvalue()).decode()

    # 生成平滑 RGBA 图像
    rgba = create_rgba_from_mask(img, mask, smooth=True, blur_radius=5)
    rgba_img = Image.fromarray(rgba, 'RGBA')
    rgba_buf = io.BytesIO()
    rgba_img.save(rgba_buf, format="PNG")
    rgba_b64 = base64.b64encode(rgba_buf.getvalue()).decode()

    # 生成叠加图
    overlay_b64 = create_overlay(img, mask)
    
    # 保存到历史记录
    if req.image_id not in segmentation_histories:
        segmentation_histories[req.image_id] = SegmentationHistory()
    
    # 尝试识别物体标签
    label = "分割区域"
    if classifier.model:
        try:
            x1, y1, x2, y2 = bbox
            margin = 5
            crop = img[max(0,y1-margin):min(h,y2+margin), max(0,x1-margin):min(w,x2+margin)]
            if crop.size > 0:
                crop_pil = Image.fromarray(crop)
                result = classifier.classify(crop_pil, top_k=1)
                if result:
                    label = result[0]["label"]
        except:
            pass
    
    history_entry = segmentation_histories[req.image_id].add(
        tool="box",
        mask_b64=smooth_mask_b64,
        overlay_b64=overlay_b64,
        score=float(scores[best]),
        area=int(mask.sum()),
        label=label,
        box=req.box
    )

    return {
        "success": True,
        "mask": smooth_mask_b64,  # 平滑掩码
        "color_image": rgba_b64,  # 平滑彩色图像
        "overlay": overlay_b64,
        "score": float(scores[best]),
        "bbox": bbox,
        "area": int(mask.sum()),
        "history_id": history_entry["id"]
    }

@app.post("/api/segment/multi")
async def segment_multi(req: MultiMaskRequest):
    path = find_image(req.image_id)
    if not path:
        raise HTTPException(404, "请先上传图片")

    img = np.array(Image.open(path).convert("RGB"))
    sam.set_image(img)

    pts = np.array(req.points)
    lbs = np.array(req.labels)
    masks, scores = sam.predict_point(pts, lbs, multimask=req.multimask)

    results = []
    for i in range(len(masks)):
        m = masks[i]
        ys, xs = np.where(m)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) > 0 else None
        results.append({
            "mask": mask_to_base64(m),
            "score": float(scores[i]),
            "bbox": bbox,
            "area": int(m.sum())
        })

    return {"success": True, "results": results}


@app.post("/api/detect/auto")
async def auto_detect(req: AutoDetectRequest):
    """自动检测 - 使用网格采样检测图中所有物体"""
    path = find_image(req.image_id)
    if not path:
        raise HTTPException(404, "请先上传图片")

    img = np.array(Image.open(path).convert("RGB"))
    h, w = img.shape[:2]

    try:
        sam.predictor.set_image(img)

        # 网格采样点
        step = max(w, h) // req.points_per_side
        step = max(step, 30)  # 最小间距

        detections = []
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
            (0, 128, 255), (128, 255, 0), (255, 0, 128), (0, 255, 128),
        ]

        overlay = img.copy().astype(np.float32)
        used_masks = []
        detection_id = 0

        # 在网格点上进行分割
        for y in range(step//2, h, step):
            for x in range(step//2, w, step):
                if detection_id >= 15:  # 最多15个
                    break

                try:
                    masks, scores, _ = sam.predictor.predict(
                        point_coords=np.array([[x, y]]),
                        point_labels=np.array([1]),
                        multimask_output=False
                    )

                    mask = masks[0]
                    score = float(scores[0])

                    # 过滤太小的区域
                    area = int(mask.sum())
                    if area < req.min_mask_region_area:
                        continue

                    # 检查是否与已有掩码重叠过多
                    is_duplicate = False
                    for used in used_masks:
                        overlap = (mask & used).sum() / min(mask.sum(), used.sum())
                        if overlap > 0.5:
                            is_duplicate = True
                            break
                    if is_duplicate:
                        continue

                    used_masks.append(mask)
                    detection_id += 1

                    # 计算边界框
                    ys, xs = np.where(mask)
                    bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

                    # 用 ResNet 识别该区域
                    region_label = "未知"
                    region_prob = 0.0
                    if classifier.model:
                        try:
                            x1, y1, x2, y2 = bbox
                            margin = 5
                            crop = img[max(0,y1-margin):min(h,y2+margin), max(0,x1-margin):min(w,x2+margin)]
                            if crop.size > 0:
                                crop_pil = Image.fromarray(crop)
                                result = classifier.classify(crop_pil, top_k=1)
                                if result:
                                    region_label = result[0]["label"]
                                    region_prob = result[0]["prob"]
                        except:
                            pass

                    # 跳过低置信度的泛化标签
                    generic_labels = {"未知", "冷色物体", "暖色物体", "区域-light", "区域-dark"}
                    if region_prob < 0.15 or region_label in generic_labels:
                        continue

                    # 叠加颜色
                    color = colors[detection_id % len(colors)]
                    alpha = 0.4
                    mask_3d = np.stack([mask]*3, axis=-1)
                    color_arr = np.array(color, dtype=np.float32)
                    overlay = np.where(mask_3d,
                                     overlay * (1 - alpha) + color_arr * alpha,
                                     overlay)

                    # 生成单个掩码的 base64
                    single_mask_b64 = mask_to_base64(mask)

                    detections.append({
                        "id": detection_id,
                        "label": region_label,
                        "confidence": round(region_prob, 3),
                        "mask": single_mask_b64,
                        "bbox": bbox,
                        "area": area,
                        "score": score,
                    })

                except Exception:
                    continue

        # 按面积排序
        detections.sort(key=lambda x: x['area'], reverse=True)
        for i, d in enumerate(detections):
            d['id'] = i + 1

        # 生成叠加图
        overlay_b64 = Image.fromarray(overlay.astype(np.uint8))
        buf = io.BytesIO()
        overlay_b64.save(buf, format="PNG")
        overlay_str = base64.b64encode(buf.getvalue()).decode()

        return {
            "success": True,
            "count": len(detections),
            "detections": detections,
            "overlay": overlay_str
        }

    except Exception as e:
        return {"success": False, "message": str(e)}


@app.post("/api/segment/auto")
async def auto_segment(req: AutoSegmentRequest):
    """自动分割 - 检测并分割图中所有物体，返回每个物体的掩码"""
    path = find_image(req.image_id)
    if not path:
        raise HTTPException(404, "请先上传图片")

    img = np.array(Image.open(path).convert("RGB"))
    h, w = img.shape[:2]

    try:
        sam.predictor.set_image(img)

        # 网格采样点
        step = max(w, h) // req.points_per_side
        step = max(step, 40)

        detections = []
        colors = [
            (255, 50, 50), (50, 255, 50), (50, 50, 255), (255, 255, 50),
            (255, 50, 255), (50, 255, 255), (255, 128, 0), (128, 0, 255),
            (0, 255, 128), (255, 0, 128), (128, 255, 0), (0, 128, 255),
        ]

        overlay = img.copy().astype(np.float32)
        used_masks = []
        detection_id = 0

        # 在网格点上进行分割
        for y in range(step//2, h, step):
            for x in range(step//2, w, step):
                if detection_id >= req.max_objects:
                    break

                try:
                    masks, scores, _ = sam.predictor.predict(
                        point_coords=np.array([[x, y]]),
                        point_labels=np.array([1]),
                        multimask_output=False
                    )

                    mask = masks[0]
                    score = float(scores[0])
                    area = int(mask.sum())

                    # 过滤太小的区域
                    if area < req.min_mask_region_area:
                        continue

                    # 检查是否与已有掩码重叠过多
                    is_duplicate = False
                    for used in used_masks:
                        overlap = (mask & used).sum() / min(mask.sum(), used.sum())
                        if overlap > 0.5:
                            is_duplicate = True
                            break
                    if is_duplicate:
                        continue

                    used_masks.append(mask)
                    detection_id += 1

                    # 计算边界框
                    ys, xs = np.where(mask)
                    bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

                    # 用 ResNet 识别该区域
                    region_label = "未知"
                    region_prob = 0.0
                    if classifier.model:
                        try:
                            x1, y1, x2, y2 = bbox
                            # 裁剪区域，保留一些边距
                            margin = 5
                            crop = img[max(0,y1-margin):min(h,y2+margin), max(0,x1-margin):min(w,x2+margin)]
                            if crop.size > 0:
                                crop_pil = Image.fromarray(crop)
                                result = classifier.classify(crop_pil, top_k=1)
                                if result:
                                    region_label = result[0]["label"]
                                    region_prob = result[0]["prob"]
                        except Exception as e:
                            pass

                    # 跳过低置信度的泛化标签
                    generic_labels = {"未知", "冷色物体", "暖色物体", "区域-light", "区域-dark"}
                    if region_prob < 0.15 or region_label in generic_labels:
                        continue

                    # 叠加颜色
                    color = colors[detection_id % len(colors)]
                    alpha = 0.5
                    mask_3d = np.stack([mask]*3, axis=-1)
                    color_arr = np.array(color, dtype=np.float32)
                    overlay = np.where(mask_3d,
                                     overlay * (1 - alpha) + color_arr * alpha,
                                     overlay)

                    # 绘制边界框
                    x1, y1, x2, y2 = bbox
                    overlay[y1:y1+3, x1:x2] = color_arr  # 上边
                    overlay[y2-3:y2, x1:x2] = color_arr  # 下边
                    overlay[y1:y2, x1:x1+3] = color_arr  # 左边
                    overlay[y1:y2, x2-3:x2] = color_arr  # 右边

                    # 生成单个掩码的 base64
                    single_mask_b64 = mask_to_base64(mask)

                    detections.append({
                        "id": detection_id,
                        "label": region_label,
                        "confidence": round(region_prob, 3),
                        "mask": single_mask_b64,
                        "bbox": bbox,
                        "area": area,
                        "score": score,
                    })

                except Exception:
                    continue

        # 按面积排序
        detections.sort(key=lambda x: x['area'], reverse=True)
        for i, d in enumerate(detections):
            d['id'] = i + 1

        # 生成叠加图
        overlay_b64 = Image.fromarray(overlay.astype(np.uint8))
        buf = io.BytesIO()
        overlay_b64.save(buf, format="PNG")
        overlay_str = base64.b64encode(buf.getvalue()).decode()

        return {
            "success": True,
            "count": len(detections),
            "detections": detections,
            "overlay": overlay_str
        }

    except Exception as e:
        return {"success": False, "message": str(e)}


def classify_region(mean_color, area, img_shape):
    """简单识别：基于颜色和面积判断区域类型"""
    h, w = img_shape[:2]
    total_pixels = h * w
    area_ratio = area / total_pixels

    r, g, b = mean_color

    # 基于颜色的简单分类
    if r > 150 and g > 150 and b > 150:
        color_hint = "light"
    elif r < 80 and g < 80 and b < 80:
        color_hint = "dark"
    elif r > g and r > b:
        color_hint = "warm"
    elif b > r and b > g:
        color_hint = "cool"
    elif g > r and g > b:
        color_hint = "green"
    else:
        color_hint = "neutral"

    # 基于面积的分类
    if area_ratio > 0.3:
        size = "background"
    elif area_ratio > 0.1:
        size = "large-object"
    elif area_ratio > 0.02:
        size = "medium-object"
    else:
        size = "small-object"

    labels = {
        ("light", "background"): "天空/墙面",
        ("dark", "background"): "阴影/背景",
        ("warm", "large-object"): "暖色物体",
        ("cool", "large-object"): "冷色物体",
        ("green", "large-object"): "植被",
        ("warm", "medium-object"): "中等物体",
        ("cool", "medium-object"): "中等物体",
        ("green", "medium-object"): "植物",
        ("neutral", "medium-object"): "物体",
        ("warm", "small-object"): "小物体",
        ("cool", "small-object"): "小物体",
        ("green", "small-object"): "叶片/小植物",
    }

    return labels.get((color_hint, size), f"区域-{color_hint}")


@app.post("/api/recognize")
async def recognize_object(image_id: str = None, file: UploadFile = None):
    """识别 - 对分割区域进行识别"""
    if image_id:
        path = find_image(image_id)
        if not path:
            raise HTTPException(404, "图片不存在")
        img = np.array(Image.open(path).convert("RGB"))
    elif file:
        content = await file.read()
        img = np.array(Image.open(io.BytesIO(content)).convert("RGB"))
    else:
        raise HTTPException(400, "需要提供 image_id 或上传文件")

    h, w = img.shape[:2]

    # 计算图像整体特征
    mean_color = img.mean(axis=(0, 1))
    std_color = img.std(axis=(0, 1))

    # 分析主要颜色
    dominant_colors = extract_dominant_colors(img)

    # 图像属性
    brightness = float(mean_color.mean())
    contrast = float(std_color.mean())

    # 简单场景判断
    if brightness > 180:
        scene = "明亮场景"
    elif brightness > 120:
        scene = "正常光照"
    elif brightness > 80:
        scene = "较暗场景"
    else:
        scene = "暗光场景"

    if dominant_colors[0][1] > 0.4:
        scene += " / 单一主色调"
    else:
        scene += " / 多彩场景"

    # 使用 ResNet 进行物体识别
    classifications = classifier.classify(Image.fromarray(img), top_k=5)

    return {
        "success": True,
        "image_size": [int(w), int(h)],
        "scene": scene,
        "brightness": round(brightness, 1),
        "contrast": round(contrast, 1),
        "dominant_colors": [{"rgb": list(c), "ratio": round(r, 3)} for c, r in dominant_colors],
        "classifications": classifications,  # 新增：ResNet 识别结果
    }


def extract_dominant_colors(img, n_colors=5):
    """提取主要颜色"""
    # 简化：将颜色量化
    small = Image.fromarray(img).resize((50, 50))
    pixels = np.array(small).reshape(-1, 3)

    # 简单的 K-means 近似
    from collections import Counter
    quantized = (pixels // 32) * 32  # 量化到8个级别
    color_counts = Counter(map(tuple, quantized))
    total = len(quantized)

    dominant = color_counts.most_common(n_colors)
    return [([int(x) for x in c], float(count/total)) for c, count in dominant]


@app.post("/api/detect")
async def detect_objects(file: UploadFile = File(...)):
    """目标检测接口"""
    content = await file.read()
    img = np.array(Image.open(io.BytesIO(content)).convert("RGB"))
    h, w = img.shape[:2]

    try:
        sam.set_image(img)

        # 网格采样
        step = max(w, h) // 20
        step = max(step, 40)

        detections = []
        used_masks = []

        for y in range(step//2, h, step):
            for x in range(step//2, w, step):
                if len(detections) >= 10:
                    break

                try:
                    masks, scores, _ = sam.predictor.predict(
                        point_coords=np.array([[x, y]]),
                        point_labels=np.array([1]),
                        multimask_output=False
                    )

                    mask = masks[0]
                    area = mask.sum()

                    if area < 500:
                        continue

                    # 检查重复
                    is_dup = False
                    for used in used_masks:
                        if (mask & used).sum() / min(mask.sum(), used.sum()) > 0.5:
                            is_dup = True
                            break
                    if is_dup:
                        continue

                    used_masks.append(mask)
                    ys, xs = np.where(mask)
                    bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

                    # 用 ResNet 识别该区域
                    region_label = "未知"
                    region_prob = 0.0
                    if classifier.model:
                        try:
                            x1, y1, x2, y2 = bbox
                            margin = 5
                            crop = img[max(0,y1-margin):min(h,y2+margin), max(0,x1-margin):min(w,x2+margin)]
                            if crop.size > 0:
                                crop_pil = Image.fromarray(crop)
                                result = classifier.classify(crop_pil, top_k=1)
                                if result:
                                    region_label = result[0]["label"]
                                    region_prob = result[0]["prob"]
                        except:
                            pass

                    if region_prob < 0.05:
                        region_mean = img[mask].mean(axis=0)
                        region_label = classify_region(region_mean, area, img.shape)

                    detections.append({
                        "label": region_label,
                        "confidence": float(scores[0]),
                        "box": bbox,
                        "area": int(area)
                    })
                except Exception:
                    continue

        return {"detections": detections, "count": len(detections)}

    except Exception as e:
        return {"detections": [], "error": str(e)}


@app.post("/api/extract/color")
async def extract_color_object(image_id: str, mask_base64: str = None, bbox: str = None):
    """
    彩色物体提取 - 从原图中提取彩色物体（透明背景）
    
    参数:
        image_id: 图片ID
        mask_base64: 掩码的base64编码（可选）
        bbox: 边界框 "x1,y1,x2,y2"（可选，如果没有掩码则用边界框裁剪）
    
    返回:
        彩色物体的base64编码PNG（透明背景）
    """
    path = find_image(image_id)
    if not path:
        raise HTTPException(404, "图片不存在")
    
    img = np.array(Image.open(path).convert("RGB"))
    h, w = img.shape[:2]
    
    # 创建RGBA图像（带透明通道）
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = img  # RGB通道
    
    if mask_base64:
        # 使用掩码提取
        mask_bytes = base64.b64decode(mask_base64)
        mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")
        mask = np.array(mask_img.resize((w, h))) > 128
        
        # 设置透明度：掩码区域不透明，其他区域透明
        rgba[:, :, 3] = np.where(mask, 255, 0)
        
        # 裁剪到边界框区域
        ys, xs = np.where(mask)
        if len(ys) > 0:
            y1, y2 = ys.min(), ys.max()
            x1, x2 = xs.min(), xs.max()
            cropped = rgba[y1:y2+1, x1:x2+1]
        else:
            cropped = rgba
    elif bbox:
        # 使用边界框裁剪
        coords = list(map(int, bbox.split(',')))
        x1, y1, x2, y2 = coords
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        cropped = rgba[y1:y2, x1:x2]
        cropped[:, :, 3] = 255  # 全不透明
    else:
        raise HTTPException(400, "需要提供 mask_base64 或 bbox")
    
    # 转为PNG base64
    result_img = Image.fromarray(cropped, 'RGBA')
    buffer = io.BytesIO()
    result_img.save(buffer, format='PNG')
    color_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    return {
        "success": True,
        "color_image": color_base64,
        "size": [cropped.shape[1], cropped.shape[0]],
        "format": "png"
    }


@app.post("/api/extract/all")
async def extract_all_objects(image_id: str, min_area: int = 500, min_confidence: float = 0.6):
    """
    批量提取高置信度彩色物体
    
    参数:
        image_id: 图片ID
        min_area: 最小区域面积（默认500）
        min_confidence: 最低置信度阈值（默认0.6，即60%）
    
    返回:
        置信度高于阈值的彩色物体列表
    """
    path = find_image(image_id)
    if not path:
        raise HTTPException(404, "图片不存在")
    
    img = np.array(Image.open(path).convert("RGB"))
    h, w = img.shape[:2]
    
    # 不要这些泛化标签
    generic_labels = {"未知", "冷色物体", "暖色物体", "区域-light", "区域-dark"}
    
    try:
        sam.set_image(img)
        
        # 网格采样点
        step = max(w, h) // 20
        step = max(step, 40)
        
        objects = []
        used_masks = []
        
        for y in range(step//2, h, step):
            for x in range(step//2, w, step):
                if len(objects) >= 20:
                    break
                
                try:
                    masks, scores, _ = sam.predictor.predict(
                        point_coords=np.array([[x, y]]),
                        point_labels=np.array([1]),
                        multimask_output=False
                    )
                    
                    mask = masks[0]
                    area = mask.sum()
                    
                    if area < min_area:
                        continue
                    
                    # 检查重复
                    is_dup = False
                    for used in used_masks:
                        if (mask & used).sum() / min(mask.sum(), used.sum()) > 0.5:
                            is_dup = True
                            break
                    if is_dup:
                        continue
                    
                    used_masks.append(mask)
                    
                    # 计算边界框
                    ys, xs = np.where(mask)
                    bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                    
                    # 用 ResNet 识别该区域
                    region_label = "未知"
                    region_prob = 0.0
                    if classifier.model:
                        try:
                            x1, y1, x2, y2 = bbox
                            margin = 5
                            crop = img[max(0,y1-margin):min(h,y2+margin), max(0,x1-margin):min(w,x2+margin)]
                            if crop.size > 0:
                                crop_pil = Image.fromarray(crop)
                                result = classifier.classify(crop_pil, top_k=1)
                                if result:
                                    region_label = result[0]["label"]
                                    region_prob = result[0]["prob"]
                        except:
                            pass
                    
                    # 跳过低置信度和泛化标签的物体
                    if region_prob < min_confidence or region_label in generic_labels:
                        continue
                    
                    # 提取彩色物体（平滑边缘，透明背景）
                    rgba = create_rgba_from_mask(img, mask, smooth=True, blur_radius=5)
                    
                    y1, y2 = ys.min(), ys.max()
                    x1, x2 = xs.min(), xs.max()
                    cropped = rgba[y1:y2+1, x1:x2+1]
                    
                    # 转为base64
                    result_img = Image.fromarray(cropped, 'RGBA')
                    buffer = io.BytesIO()
                    result_img.save(buffer, format='PNG')
                    color_b64 = base64.b64encode(buffer.getvalue()).decode()
                    
                    # 生成平滑掩码base64
                    smooth_alpha = smooth_mask(mask, blur_radius=5, feather=True)
                    mask_img = Image.fromarray(smooth_alpha, 'L')
                    mask_buffer = io.BytesIO()
                    mask_img.save(mask_buffer, format='PNG')
                    mask_b64 = base64.b64encode(mask_buffer.getvalue()).decode()
                    
                    objects.append({
                        "id": len(objects) + 1,
                        "label": region_label,
                        "confidence": round(float(region_prob), 3),
                        "score": round(float(scores[0]), 3),
                        "bbox": bbox,
                        "area": int(area),
                        "color_image": color_b64,
                        "mask": mask_b64,
                    })
                    
                except Exception:
                    continue
            
            if len(objects) >= 15:
                break
        
        return {
            "success": True,
            "image_id": image_id,
            "objects": objects,
            "count": len(objects)
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── 掩码编辑 API ──
class BrushStroke(BaseModel):
    x: int
    y: int
    radius: int = 10
    mode: str = 'add'  # 'add' 或 'erase'

class MaskEditRequest(BaseModel):
    image_id: str
    mask_base64: str
    strokes: List[BrushStroke]

class MaskUndoRequest(BaseModel):
    image_id: str

@app.post("/api/mask/edit")
async def edit_mask(req: MaskEditRequest):
    """
    编辑掩码 - 使用画笔添加或擦除掩码区域
    
    参数:
        image_id: 图片 ID
        mask_base64: 当前掩码的 base64 编码
        strokes: 笔触列表
    
    返回:
        编辑后的掩码
    """
    try:
        # 解码原始掩码
        mask_bytes = base64.b64decode(req.mask_base64)
        mask_img = Image.open(io.BytesIO(mask_bytes)).convert('L')
        mask = np.array(mask_img) > 128
        
        # 保存到历史
        if req.image_id not in mask_histories:
            mask_histories[req.image_id] = MaskHistory()
        mask_histories[req.image_id].push(mask)
        
        # 应用笔触
        strokes_data = [s.dict() for s in req.strokes]
        edited_mask = apply_brush_strokes(mask, strokes_data)
        
        # 保存编辑后到历史
        mask_histories[req.image_id].push(edited_mask)
        
        # 编码返回
        result_img = Image.fromarray((edited_mask * 255).astype(np.uint8))
        buf = io.BytesIO()
        result_img.save(buf, format="PNG")
        mask_b64 = base64.b64encode(buf.getvalue()).decode()
        
        return {
            "success": True,
            "mask": mask_b64,
            "area": int(edited_mask.sum()),
            "can_undo": mask_histories[req.image_id].current > 0,
            "can_redo": mask_histories[req.image_id].current < len(mask_histories[req.image_id].history) - 1
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/mask/undo")
async def undo_mask(req: MaskUndoRequest):
    """
    撤销掩码编辑
    
    参数:
        image_id: 图片 ID
    
    返回:
        上一步的掩码
    """
    if req.image_id not in mask_histories:
        return {"success": False, "error": "没有历史记录"}
    
    mask = mask_histories[req.image_id].undo()
    if mask is None:
        return {"success": False, "error": "无法撤销"}
    
    result_img = Image.fromarray((mask * 255).astype(np.uint8))
    buf = io.BytesIO()
    result_img.save(buf, format="PNG")
    mask_b64 = base64.b64encode(buf.getvalue()).decode()
    
    return {
        "success": True,
        "mask": mask_b64,
        "area": int(mask.sum()),
        "can_undo": mask_histories[req.image_id].current > 0,
        "can_redo": mask_histories[req.image_id].current < len(mask_histories[req.image_id].history) - 1
    }


@app.post("/api/mask/redo")
async def redo_mask(req: MaskUndoRequest):
    """
    重做掩码编辑
    
    参数:
        image_id: 图片 ID
    
    返回:
        下一步的掩码
    """
    if req.image_id not in mask_histories:
        return {"success": False, "error": "没有历史记录"}
    
    mask = mask_histories[req.image_id].redo()
    if mask is None:
        return {"success": False, "error": "无法重做"}
    
    result_img = Image.fromarray((mask * 255).astype(np.uint8))
    buf = io.BytesIO()
    result_img.save(buf, format="PNG")
    mask_b64 = base64.b64encode(buf.getvalue()).decode()
    
    return {
        "success": True,
        "mask": mask_b64,
        "area": int(mask.sum()),
        "can_undo": mask_histories[req.image_id].current > 0,
        "can_redo": mask_histories[req.image_id].current < len(mask_histories[req.image_id].history) - 1
    }


# ── 分割历史记录 API ──
class HistoryRequest(BaseModel):
    image_id: str

class HistoryGetRequest(BaseModel):
    image_id: str
    entry_id: int

@app.get("/api/history/{image_id}")
async def get_history(image_id: str):
    """
    获取图片的分割历史记录
    
    参数:
        image_id: 图片 ID
    
    返回:
        历史记录列表
    """
    if image_id not in segmentation_histories:
        return {"success": True, "history": [], "count": 0}
    
    history = segmentation_histories[image_id].get_all()
    # 返回简化的记录（不含完整的 mask/overlay 数据）
    simplified = []
    for entry in history:
        simplified.append({
            "id": entry["id"],
            "timestamp": entry["timestamp"],
            "tool": entry["tool"],
            "label": entry["label"],
            "score": entry["score"],
            "area": entry["area"],
            "bbox": entry.get("bbox")
        })
    
    return {"success": True, "history": simplified, "count": len(simplified)}

@app.post("/api/history/get")
async def get_history_entry(req: HistoryGetRequest):
    """
    获取单条历史记录详情
    
    参数:
        image_id: 图片 ID
        entry_id: 记录 ID
    
    返回:
        完整的历史记录（包含 mask 和 overlay）
    """
    if req.image_id not in segmentation_histories:
        return {"success": False, "error": "没有历史记录"}
    
    entry = segmentation_histories[req.image_id].get_by_id(req.entry_id)
    if entry is None:
        return {"success": False, "error": "记录不存在"}
    
    return {"success": True, "entry": entry}

@app.delete("/api/history/{image_id}/{entry_id}")
async def delete_history_entry(image_id: str, entry_id: int):
    """
    删除单条历史记录
    
    参数:
        image_id: 图片 ID
        entry_id: 记录 ID
    
    返回:
        删除结果
    """
    if image_id not in segmentation_histories:
        return {"success": False, "error": "没有历史记录"}
    
    deleted = segmentation_histories[image_id].delete(entry_id)
    if not deleted:
        return {"success": False, "error": "记录不存在"}
    
    return {"success": True}

@app.delete("/api/history/{image_id}")
async def clear_history(image_id: str):
    """
    清空图片的所有历史记录
    
    参数:
        image_id: 图片 ID
    
    返回:
        清空结果
    """
    if image_id not in segmentation_histories:
        return {"success": True}
    
    segmentation_histories[image_id].clear()
    return {"success": True}

# ── 导出格式扩展 ──
@app.get("/api/export/json/{image_id}")
async def export_json(image_id: str, entry_id: str = None):
    """
    导出 JSON 格式
    
    参数:
        image_id: 图片 ID
        entry_id: 历史记录 ID（可选，为空则导出所有记录）
    
    返回:
        JSON 文件
    """
    import tempfile
    import os
    
    if image_id not in segmentation_histories:
        raise HTTPException(404, "没有找到该图片的分割记录")
    
    history = segmentation_histories[image_id]
    
    if entry_id:
        # 导出单条记录
        entry = history.get_entry(entry_id)
        if not entry:
            raise HTTPException(404, "没有找到该记录")
        export_data = [entry]
    else:
        # 导出所有记录
        export_data = history.history
    
    # 添加元数据
    export_data_with_meta = {
        "image_id": image_id,
        "export_time": str(datetime.now()),
        "model_type": sam.model_type,
        "device": sam.device,
        "records": export_data
    }
    
    # 创建临时文件
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8')
    json.dump(export_data_with_meta, temp_file, ensure_ascii=False, indent=2)
    temp_file.close()
    
    return FileResponse(
        temp_file.name,
        media_type="application/json",
        filename=f"sam_export_{image_id}.json",
        background=lambda: os.unlink(temp_file.name)
    )

@app.get("/api/export/csv/{image_id}")
async def export_csv(image_id: str, entry_id: str = None):
    """
    导出 CSV 格式
    
    参数:
        image_id: 图片 ID
        entry_id: 历史记录 ID（可选，为空则导出所有记录）
    
    返回:
        CSV 文件
    """
    import tempfile
    import os
    import csv
    
    if image_id not in segmentation_histories:
        raise HTTPException(404, "没有找到该图片的分割记录")
    
    history = segmentation_histories[image_id]
    
    if entry_id:
        entry = history.get_entry(entry_id)
        if not entry:
            raise HTTPException(404, "没有找到该记录")
        entries = [entry]
    else:
        entries = history.history
    
    # 创建临时文件
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8-sig', newline='')
    
    writer = csv.writer(temp_file)
    
    # 写入表头
    writer.writerow([
        "ID", "标签", "置信度", "分数",
        "边界框_x1", "边界框_y1", "边界框_x2", "边界框_y2",
        "面积", "创建时间"
    ])
    
    # 写入数据
    for entry in entries:
        d = entry if isinstance(entry, dict) else entry.to_dict()
        bbox = d.get("box", [0, 0, 0, 0]) or [0, 0, 0, 0]
        writer.writerow([
            d.get("id", ""),
            d.get("label", ""),
            d.get("score", 0),
            d.get("score", 0),
            bbox[0] if len(bbox) > 0 else 0,
            bbox[1] if len(bbox) > 1 else 0,
            bbox[2] if len(bbox) > 2 else 0,
            bbox[3] if len(bbox) > 3 else 0,
            d.get("area", 0),
            d.get("timestamp", "")
        ])
    
    temp_file.close()
    
    return FileResponse(
        temp_file.name,
        media_type="text/csv",
        filename=f"sam_export_{image_id}.csv",
        background=lambda: os.unlink(temp_file.name)
    )

# ── COCO/YOLO 标注导出 ──

@app.get("/api/export/coco/{image_id}")
async def export_coco(image_id: str):
    """
    导出 COCO 格式标注
    
    参数:
        image_id: 图片 ID
    
    返回:
        COCO 格式 JSON 文件
    """
    import tempfile
    import os
    
    if image_id not in segmentation_histories:
        raise HTTPException(404, "没有找到该图片的分割记录")
    
    history = segmentation_histories[image_id]
    entries = history.history
    
    # 获取图片信息
    image_path = find_image(image_id)
    if not image_path:
        raise HTTPException(404, "图片不存在")
    
    img = Image.open(image_path)
    width, height = img.size
    
    # 构建 COCO 格式
    coco_data = {
        "images": [{
            "id": 1,
            "file_name": image_path.name,
            "width": width,
            "height": height
        }],
        "annotations": [],
        "categories": []
    }
    
    # 类别映射
    category_map = {}
    category_id = 1
    
    for idx, entry in enumerate(entries):
        label = entry.get("label", "object")
        
        # 添加类别
        if label not in category_map:
            category_map[label] = category_id
            coco_data["categories"].append({
                "id": category_id,
                "name": label,
                "supercategory": "object"
            })
            category_id += 1
        
        # 获取边界框
        bbox = entry.get("box", [0, 0, 0, 0])
        if not bbox or len(bbox) < 4:
            # 从掩码计算边界框
            mask_b64 = entry.get("mask")
            if mask_b64:
                mask_bytes = base64.b64decode(mask_b64)
                mask_img = Image.open(io.BytesIO(mask_bytes)).convert('L')
                mask_array = np.array(mask_img)
                ys, xs = np.where(mask_array > 128)
                if len(xs) > 0 and len(ys) > 0:
                    bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                else:
                    bbox = [0, 0, 0, 0]
            else:
                bbox = [0, 0, 0, 0]
        
        # COCO 格式: [x, y, width, height]
        x1, y1, x2, y2 = bbox
        coco_bbox = [x1, y1, x2 - x1, y2 - y1]
        
        annotation = {
            "id": idx + 1,
            "image_id": 1,
            "category_id": category_map.get(label, 1),
            "bbox": coco_bbox,
            "area": entry.get("area", 0),
            "iscrowd": 0,
            "score": entry.get("score", 0)
        }
        
        coco_data["annotations"].append(annotation)
    
    # 创建临时文件
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8')
    json.dump(coco_data, temp_file, ensure_ascii=False, indent=2)
    temp_file.close()
    
    return FileResponse(
        temp_file.name,
        media_type="application/json",
        filename=f"sam_coco_{image_id}.json",
        background=lambda: os.unlink(temp_file.name)
    )

@app.get("/api/export/yolo/{image_id}")
async def export_yolo(image_id: str):
    """
    导出 YOLO 格式标注
    
    参数:
        image_id: 图片 ID
    
    返回:
        YOLO 格式 TXT 文件（ZIP 包）
    """
    import tempfile
    import os
    import zipfile
    
    if image_id not in segmentation_histories:
        raise HTTPException(404, "没有找到该图片的分割记录")
    
    history = segmentation_histories[image_id]
    entries = history.history
    
    # 获取图片信息
    image_path = find_image(image_id)
    if not image_path:
        raise HTTPException(404, "图片不存在")
    
    img = Image.open(image_path)
    width, height = img.size
    
    # 类别映射
    category_map = {}
    category_id = 0
    
    # 生成 YOLO 格式标注
    yolo_lines = []
    
    for entry in entries:
        label = entry.get("label", "object")
        
        # 添加类别
        if label not in category_map:
            category_map[label] = category_id
            category_id += 1
        
        # 获取边界框
        bbox = entry.get("box", [0, 0, 0, 0])
        if not bbox or len(bbox) < 4:
            # 从掩码计算边界框
            mask_b64 = entry.get("mask")
            if mask_b64:
                mask_bytes = base64.b64decode(mask_b64)
                mask_img = Image.open(io.BytesIO(mask_bytes)).convert('L')
                mask_array = np.array(mask_img)
                ys, xs = np.where(mask_array > 128)
                if len(xs) > 0 and len(ys) > 0:
                    bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                else:
                    continue
            else:
                continue
        
        # YOLO 格式: class_id center_x center_y width height (归一化)
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2 / width
        center_y = (y1 + y2) / 2 / height
        bbox_width = (x2 - x1) / width
        bbox_height = (y2 - y1) / height
        
        yolo_lines.append(f"{category_map[label]} {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}")
    
    # 创建临时 ZIP 文件
    temp_zip = tempfile.NamedTemporaryFile(suffix='.zip', delete=False)
    temp_zip.close()
    
    with zipfile.ZipFile(temp_zip.name, 'w') as zf:
        # 写入标注文件
        label_content = '\n'.join(yolo_lines)
        zf.writestr(f"{image_id}.txt", label_content)
        
        # 写入类别文件
        classes_content = '\n'.join([f"{name}" for name, _ in sorted(category_map.items(), key=lambda x: x[1])])
        zf.writestr("classes.txt", classes_content)
        
        # 写入图片
        zf.write(image_path, image_path.name)
    
    return FileResponse(
        temp_zip.name,
        media_type="application/zip",
        filename=f"sam_yolo_{image_id}.zip",
        background=lambda: os.unlink(temp_zip.name)
    )

# ── 视频/摄像头流分割 ──

# 视频处理工具类
class VideoProcessor:
    def __init__(self):
        self.videos = {}  # video_id -> video info
    
    def extract_frames(self, video_path: str, max_frames: int = 100) -> List[dict]:
        """
        从视频中提取帧
        
        参数:
            video_path: 视频文件路径
            max_frames: 最大帧数
        
        返回:
            帧信息列表 [{"frame_id": int, "timestamp": float, "image_id": str}]
        """
        try:
            import cv2
        except ImportError:
            print("[WARN] OpenCV not installed, video processing disabled")
            return []
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 计算采样间隔
        if total_frames <= max_frames:
            interval = 1
        else:
            interval = total_frames // max_frames
        
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % interval == 0:
                # 转换为 RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                
                # 保存帧
                frame_id = uuid.uuid4().hex[:10]
                frame_path = UPLOAD_DIR / f"{frame_id}.jpg"
                img.save(frame_path, "JPEG", quality=85)
                
                timestamp = frame_count / fps if fps > 0 else 0
                
                frames.append({
                    "frame_id": frame_count,
                    "timestamp": round(timestamp, 2),
                    "image_id": frame_id,
                    "width": img.width,
                    "height": img.height
                })
            
            frame_count += 1
            
            if len(frames) >= max_frames:
                break
        
        cap.release()
        return frames
    
    def get_video_info(self, video_path: str) -> dict:
        """
        获取视频信息
        
        参数:
            video_path: 视频文件路径
        
        返回:
            视频信息 {"duration": float, "fps": float, "width": int, "height": int, "total_frames": int}
        """
        try:
            import cv2
        except ImportError:
            return {}
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {}
        
        info = {
            "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0,
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        }
        
        cap.release()
        return info

video_processor = VideoProcessor()

# 视频上传请求
class VideoUploadRequest(BaseModel):
    max_frames: int = 50

# 视频帧分割请求
class VideoFrameSegmentRequest(BaseModel):
    video_id: str
    frame_id: int
    points: List[List[float]] = []
    labels: List[int] = []
    box: List[float] = []

@app.post("/api/video/upload")
async def upload_video(file: UploadFile = File(...), max_frames: int = 50):
    """
    上传视频文件
    
    参数:
        file: 视频文件
        max_frames: 最大提取帧数
    
    返回:
        视频信息和帧列表
    """
    if not file.content_type or not file.content_type.startswith("video/"):
        # 检查文件扩展名
        ext = Path(file.filename).suffix.lower() if file.filename else ""
        if ext not in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            raise HTTPException(400, "请上传视频文件")
    
    video_id = uuid.uuid4().hex[:10]
    ext = Path(file.filename).suffix if file.filename else ".mp4"
    dest = UPLOAD_DIR / f"{video_id}{ext}"
    dest.write_bytes(await file.read())
    
    # 获取视频信息
    video_info = video_processor.get_video_info(str(dest))
    
    # 提取帧
    frames = video_processor.extract_frames(str(dest), max_frames)
    
    # 保存视频信息
    video_processor.videos[video_id] = {
        "path": str(dest),
        "filename": file.filename,
        "info": video_info,
        "frames": frames
    }
    
    return {
        "video_id": video_id,
        "filename": file.filename,
        "info": video_info,
        "frames": frames,
        "frame_count": len(frames)
    }

@app.get("/api/video/{video_id}/info")
async def get_video_info(video_id: str):
    """
    获取视频信息
    
    参数:
        video_id: 视频 ID
    
    返回:
        视频信息
    """
    if video_id not in video_processor.videos:
        raise HTTPException(404, "视频不存在")
    
    video_data = video_processor.videos[video_id]
    return {
        "video_id": video_id,
        "filename": video_data["filename"],
        "info": video_data["info"],
        "frame_count": len(video_data["frames"])
    }

@app.get("/api/video/{video_id}/frames")
async def get_video_frames(video_id: str):
    """
    获取视频帧列表
    
    参数:
        video_id: 视频 ID
    
    返回:
        帧列表
    """
    if video_id not in video_processor.videos:
        raise HTTPException(404, "视频不存在")
    
    return {
        "video_id": video_id,
        "frames": video_processor.videos[video_id]["frames"]
    }

@app.post("/api/video/segment/frame")
async def segment_video_frame(req: VideoFrameSegmentRequest):
    """
    对视频帧进行分割
    
    参数:
        video_id: 视频 ID
        frame_id: 帧 ID
        points: 点击坐标列表
        labels: 标签列表 (1=前景, 0=背景)
        box: 边界框 [x1, y1, x2, y2]
    
    返回:
        分割结果
    """
    if req.video_id not in video_processor.videos:
        raise HTTPException(404, "视频不存在")
    
    video_data = video_processor.videos[req.video_id]
    
    # 查找帧
    frame_info = None
    for frame in video_data["frames"]:
        if frame["frame_id"] == req.frame_id:
            frame_info = frame
            break
    
    if not frame_info:
        raise HTTPException(404, "帧不存在")
    
    # 加载帧图像
    image_path = find_image(frame_info["image_id"])
    if not image_path:
        raise HTTPException(404, "帧图像不存在")
    
    img = np.array(Image.open(image_path).convert("RGB"))
    sam.set_image(img)
    
    results = []
    
    # 点击分割
    if req.points:
        pts = np.array(req.points)
        lbls = np.array(req.labels) if req.labels else np.ones(len(pts))
        masks, scores = sam.predict_point(pts, lbls, multimask=True)
        
        for i, (mask, score) in enumerate(zip(masks, scores)):
            mask_b64 = mask_to_base64(mask)
            overlay_b64 = create_overlay(img, mask)
            area = int(mask.sum())
            
            results.append({
                "mask_id": i,
                "mask": mask_b64,
                "overlay": overlay_b64,
                "score": float(score),
                "area": area
            })
    
    # 框选分割
    elif req.box:
        box = np.array(req.box)
        masks, scores = sam.predict_box(box)
        
        for i, (mask, score) in enumerate(zip(masks, scores)):
            mask_b64 = mask_to_base64(mask)
            overlay_b64 = create_overlay(img, mask)
            area = int(mask.sum())
            
            results.append({
                "mask_id": i,
                "mask": mask_b64,
                "overlay": overlay_b64,
                "score": float(score),
                "area": area
            })
    
    return {
        "success": True,
        "video_id": req.video_id,
        "frame_id": req.frame_id,
        "results": results
    }

@app.get("/api/video/{video_id}/stream")
async def video_stream(video_id: str):
    """
    视频流（MJPEG）
    
    参数:
        video_id: 视频 ID
    
    返回:
        MJPEG 流
    """
    try:
        import cv2
    except ImportError:
        raise HTTPException(500, "OpenCV not installed")
    
    if video_id not in video_processor.videos:
        raise HTTPException(404, "视频不存在")
    
    video_path = video_processor.videos[video_id]["path"]
    
    def generate():
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 转换为 JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        cap.release()
    
    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# 摄像头流管理
class CameraStream:
    def __init__(self):
        self.active_streams = {}
    
    async def process_camera(self, camera_id: str, websocket: WebSocket):
        """
        处理摄像头 WebSocket 流
        
        参数:
            camera_id: 摄像头 ID
            websocket: WebSocket 连接
        """
        try:
            import cv2
        except ImportError:
            await websocket.send_json({"error": "OpenCV not installed"})
            return
        
        cap = cv2.VideoCapture(0)  # 默认摄像头
        if not cap.isOpened():
            await websocket.send_json({"error": "无法打开摄像头"})
            return
        
        self.active_streams[camera_id] = {
            "cap": cap,
            "websocket": websocket,
            "active": True
        }
        
        try:
            while self.active_streams[camera_id]["active"]:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 转换为 JPEG
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_bytes = buffer.tobytes()
                
                # 发送帧
                await websocket.send_bytes(frame_bytes)
                
                # 检查是否有控制命令
                try:
                    data = await websocket.receive_text()
                    command = json.loads(data)
                    
                    if command.get("action") == "stop":
                        break
                    elif command.get("action") == "segment":
                        # 处理分割请求
                        points = command.get("points", [])
                        labels = command.get("labels", [])
                        
                        if points:
                            # 转换为 RGB
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            img = np.array(frame_rgb)
                            sam.set_image(img)
                            
                            pts = np.array(points)
                            lbls = np.array(labels) if labels else np.ones(len(pts))
                            masks, scores = sam.predict_point(pts, lbls, multimask=True)
                            
                            # 发送分割结果
                            result = {
                                "type": "segmentation",
                                "masks": [mask_to_base64(masks[0])],
                                "scores": [float(scores[0])]
                            }
                            await websocket.send_json(result)
                
                except:
                    pass
        
        except WebSocketDisconnect:
            pass
        finally:
            cap.release()
            if camera_id in self.active_streams:
                del self.active_streams[camera_id]
    
    def stop_stream(self, camera_id: str):
        """停止流"""
        if camera_id in self.active_streams:
            self.active_streams[camera_id]["active"] = False

camera_manager = CameraStream()

@app.websocket("/ws/camera/{camera_id}")
async def camera_websocket(websocket: WebSocket, camera_id: str):
    """
    摄像头 WebSocket 端点
    
    参数:
        camera_id: 摄像头 ID
    """
    await websocket.accept()
    await camera_manager.process_camera(camera_id, websocket)

@app.get("/api/camera/list")
async def list_cameras():
    """
    列出可用摄像头
    
    返回:
        摄像头列表
    """
    try:
        import cv2
    except ImportError:
        return {"cameras": [], "error": "OpenCV not installed"}
    
    cameras = []
    for i in range(3):  # 只检查前 3 个摄像头，减少错误
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # 尝试读取一帧来验证摄像头是否真正可用
                ret, _ = cap.read()
                if ret:
                    cameras.append({
                        "id": i,
                        "name": f"Camera {i}",
                        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                })
            cap.release()
        except Exception:
            # 忽略摄像头错误（如 WSL 环境）
            continue
    
    return {"cameras": cameras}

@app.get("/api/camera/{camera_id}/stream")
async def camera_stream(camera_id: int):
    """
    摄像头流（MJPEG）
    
    参数:
        camera_id: 摄像头 ID
    
    返回:
        MJPEG 流
    """
    try:
        import cv2
    except ImportError:
        raise HTTPException(500, "OpenCV not installed")
    
    def generate():
        cap = None
        try:
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                # 发送错误帧
                error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(error_frame, f"Camera {camera_id} not available", (50, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                _, buffer = cv2.imencode('.jpg', error_frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                return
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 转换为 JPEG
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"[ERROR] Camera stream error: {e}")
        finally:
            if cap is not None:
                cap.release()
    
    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.post("/api/camera/stop")
async def stop_camera():
    """
    停止所有摄像头流
    
    返回:
        停止结果
    """
    try:
        import cv2
    except ImportError:
        return {"success": False, "error": "OpenCV not installed"}
    
    # 停止所有 WebSocket 流
    for camera_id in list(camera_manager.active_streams.keys()):
        camera_manager.stop_stream(camera_id)
    
    # 释放所有可能打开的摄像头
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.release()
    
    return {"success": True, "message": "已停止所有摄像头流"}

# ── 自定义类别训练 ──

import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

class CustomClassifier:
    """自定义类别分类器"""
    def __init__(self):
        self.model = None
        self.classes = []  # 类别列表
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.model_path = MODELS_DIR / "custom_classifier.pth"
        self.classes_path = MODELS_DIR / "custom_classes.json"
        self.load()
    
    def load(self):
        """加载已有模型"""
        if self.model_path.exists() and self.classes_path.exists():
            try:
                self.classes = json.loads(self.classes_path.read_text(encoding='utf-8'))
                num_classes = len(self.classes)
                
                # 使用预训练的 ResNet18 作为基础
                from torchvision.models import resnet18, ResNet18_Weights
                self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
                self.model.fc = nn.Linear(512, num_classes)
                
                state_dict = torch.load(str(self.model_path), map_location='cpu')
                self.model.load_state_dict(state_dict)
                self.model.eval()
                
                if torch.cuda.is_available():
                    self.model = self.model.cuda()
                
                print(f"[OK] 自定义分类器已加载: {self.classes}")
                return True
            except Exception as e:
                print(f"[WARN] 加载自定义分类器失败: {e}")
                self.model = None
                return False
        return False
    
    def save(self):
        """保存模型和类别"""
        if self.model is not None:
            torch.save(self.model.state_dict(), str(self.model_path))
            self.classes_path.write_text(json.dumps(self.classes, ensure_ascii=False), encoding='utf-8')
            print(f"[OK] 自定义分类器已保存: {self.classes}")
    
    def train(self, samples: List[dict], epochs: int = 10, lr: float = 0.001):
        """
        训练分类器
        
        参数:
            samples: [{"image": base64, "label": "类别名"}, ...]
            epochs: 训练轮数
            lr: 学习率
        
        返回:
            训练结果
        """
        # 收集所有类别
        self.classes = list(set(s["label"] for s in samples))
        num_classes = len(self.classes)
        
        if num_classes < 2:
            return {"success": False, "error": "至少需要2个不同类别"}
        
        # 准备数据
        images = []
        labels = []
        
        for sample in samples:
            try:
                img_bytes = base64.b64decode(sample["image"])
                img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                tensor = self.transform(img)
                images.append(tensor)
                labels.append(self.classes.index(sample["label"]))
            except Exception as e:
                print(f"[WARN] 处理样本失败: {e}")
                continue
        
        if len(images) < num_classes * 2:
            return {"success": False, "error": f"样本不足，每个类别至少需要2个样本"}
        
        # 创建数据加载器
        images_tensor = torch.stack(images)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        dataset = torch.utils.data.TensorDataset(images_tensor, labels_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=min(8, len(dataset)), shuffle=True)
        
        # 创建模型
        from torchvision.models import resnet18, ResNet18_Weights
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(512, num_classes)
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            images_tensor = images_tensor.cuda()
            labels_tensor = labels_tensor.cuda()
        
        # 训练
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_images, batch_labels in loader:
                if torch.cuda.is_available():
                    batch_images = batch_images.cuda()
                    batch_labels = batch_labels.cuda()
                
                optimizer.zero_grad()
                outputs = self.model(batch_images)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            losses.append(epoch_loss / len(loader))
        
        self.model.eval()
        self.save()
        
        return {
            "success": True,
            "classes": self.classes,
            "num_samples": len(images),
            "epochs": epochs,
            "final_loss": losses[-1],
            "losses": losses
        }
    
    def train_from_images(self, image_ids: List[str], labels: List[str], epochs: int = 10, lr: float = 0.001):
        """
        从图片批量训练分类器
        
        参数:
            image_ids: 图片ID列表
            labels: 对应的类别标签列表
            epochs: 训练轮数
            lr: 学习率
        
        返回:
            训练结果
        """
        if len(image_ids) != len(labels):
            return {"success": False, "error": "图片ID和标签数量不匹配"}
        
        # 收集所有类别
        self.classes = list(set(labels))
        num_classes = len(self.classes)
        
        if num_classes < 2:
            return {"success": False, "error": "至少需要2个不同类别"}
        
        # 准备数据
        images = []
        label_indices = []
        
        for img_id, label in zip(image_ids, labels):
            try:
                path = find_image(img_id)
                if not path:
                    print(f"[WARN] 图片不存在: {img_id}")
                    continue
                
                img = Image.open(path).convert('RGB')
                tensor = self.transform(img)
                images.append(tensor)
                label_indices.append(self.classes.index(label))
            except Exception as e:
                print(f"[WARN] 处理图片失败 {img_id}: {e}")
                continue
        
        if len(images) < num_classes * 2:
            return {"success": False, "error": f"样本不足，每个类别至少需要2个样本"}
        
        # 创建数据加载器
        images_tensor = torch.stack(images)
        labels_tensor = torch.tensor(label_indices, dtype=torch.long)
        
        dataset = torch.utils.data.TensorDataset(images_tensor, labels_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=min(8, len(dataset)), shuffle=True)
        
        # 创建模型
        from torchvision.models import resnet18, ResNet18_Weights
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(512, num_classes)
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        
        # 训练
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_images, batch_labels in loader:
                if torch.cuda.is_available():
                    batch_images = batch_images.cuda()
                    batch_labels = batch_labels.cuda()
                
                optimizer.zero_grad()
                outputs = self.model(batch_images)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            losses.append(epoch_loss / len(loader))
        
        self.model.eval()
        self.save()
        
        return {
            "success": True,
            "classes": self.classes,
            "num_samples": len(images),
            "epochs": epochs,
            "final_loss": losses[-1],
            "losses": losses
        }
    
    def predict(self, image: Image.Image) -> List[dict]:
        """预测图片类别"""
        if self.model is None or not self.classes:
            return []
        
        try:
            tensor = self.transform(image).unsqueeze(0)
            if torch.cuda.is_available():
                tensor = tensor.cuda()
            
            with torch.no_grad():
                output = self.model(tensor)
                probs = torch.softmax(output, dim=1)[0]
            
            results = []
            for i, cls in enumerate(self.classes):
                results.append({
                    "label": cls,
                    "prob": round(float(probs[i]), 4)
                })
            
            return sorted(results, key=lambda x: -x["prob"])
        except Exception as e:
            print(f"[ERROR] 预测失败: {e}")
            return []

# 全局自定义分类器
custom_classifier = CustomClassifier()

class TrainRequest(BaseModel):
    samples: List[dict]  # [{"image": base64, "label": "类别名"}, ...]
    epochs: int = 10
    lr: float = 0.001

@app.post("/api/custom/train")
async def custom_train(req: TrainRequest):
    """
    训练自定义分类器
    
    参数:
        samples: 训练样本列表 [{"image": base64, "label": "类别名"}, ...]
        epochs: 训练轮数（默认10）
        lr: 学习率（默认0.001）
    
    返回:
        训练结果
    """
    result = custom_classifier.train(req.samples, req.epochs, req.lr)
    return result

class BatchTrainRequest(BaseModel):
    images: List[dict]  # [{"image_id": "xxx", "label": "类别名"}, ...]
    epochs: int = 10
    lr: float = 0.001

@app.post("/api/custom/batch-train")
async def custom_batch_train(req: BatchTrainRequest):
    """
    批量训练自定义分类器
    
    参数:
        images: 图片列表 [{"image_id": "xxx", "label": "类别名"}, ...]
        epochs: 训练轮数（默认10）
        lr: 学习率（默认0.001）
    
    返回:
        训练结果
    """
    image_ids = [img["image_id"] for img in req.images]
    labels = [img["label"] for img in req.images]
    
    result = custom_classifier.train_from_images(image_ids, labels, req.epochs, req.lr)
    return result

@app.get("/api/custom/classes")
async def custom_classes():
    """获取自定义类别列表"""
    return {
        "success": True,
        "classes": custom_classifier.classes,
        "has_model": custom_classifier.model is not None
    }

@app.post("/api/custom/predict")
async def custom_predict(image_id: str = None):
    """
    使用自定义分类器预测
    
    参数:
        image_id: 图片ID（可选，不传则用当前图片）
    
    返回:
        预测结果
    """
    if custom_classifier.model is None:
        return {"success": False, "error": "模型未训练"}
    
    path = find_image(image_id) if image_id else None
    if not path:
        # 使用最后一个图片
        if image_registry:
            image_id = list(image_registry.keys())[-1]
            path = find_image(image_id)
    
    if not path:
        return {"success": False, "error": "图片不存在"}
    
    img = Image.open(path).convert("RGB")
    results = custom_classifier.predict(img)
    
    return {
        "success": True,
        "image_id": image_id,
        "predictions": results,
        "top_class": results[0]["label"] if results else None,
        "top_prob": results[0]["prob"] if results else 0
    }

@app.post("/api/custom/classify-object")
async def custom_classify_object(req: dict):
    """
    对单个物体进行自定义分类
    
    参数:
        image_base64: 物体图片的base64
    
    返回:
        分类结果
    """
    if custom_classifier.model is None:
        return {"success": False, "error": "模型未训练"}
    
    try:
        img_bytes = base64.b64decode(req.get("image_base64", ""))
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        results = custom_classifier.predict(img)
        
        return {
            "success": True,
            "predictions": results,
            "top_class": results[0]["label"] if results else None,
            "top_prob": results[0]["prob"] if results else 0
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# ── 模型评估可视化 ──

@app.get("/api/custom/evaluate")
async def custom_evaluate():
    """
    评估自定义分类器性能
    
    返回:
        评估指标（准确率、混淆矩阵、各类别指标）
    """
    if custom_classifier.model is None:
        return {"success": False, "error": "模型未训练"}
    
    try:
        # 加载训练样本进行评估
        if not training_samples_store:
            return {"success": False, "error": "没有训练样本数据"}
        
        # 准备数据
        all_preds = []
        all_labels = []
        all_probs = []
        
        for sample in training_samples_store:
            try:
                img_bytes = base64.b64decode(sample["image"])
                img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                predictions = custom_classifier.predict(img)
                
                if predictions:
                    pred_label = predictions[0]["label"]
                    pred_prob = predictions[0]["prob"]
                    true_label = sample["label"]
                    
                    all_preds.append(pred_label)
                    all_labels.append(true_label)
                    all_probs.append(pred_prob)
            except Exception as e:
                print(f"[WARN] 评估样本失败: {e}")
                continue
        
        if not all_labels:
            return {"success": False, "error": "没有有效的评估数据"}
        
        # 计算指标
        classes = custom_classifier.classes
        num_classes = len(classes)
        
        # 混淆矩阵
        confusion = [[0] * num_classes for _ in range(num_classes)]
        for true_label, pred_label in zip(all_labels, all_preds):
            if true_label in classes and pred_label in classes:
                true_idx = classes.index(true_label)
                pred_idx = classes.index(pred_label)
                confusion[true_idx][pred_idx] += 1
        
        # 各类别指标
        class_metrics = {}
        for i, cls in enumerate(classes):
            tp = confusion[i][i]
            fp = sum(confusion[j][i] for j in range(num_classes) if j != i)
            fn = sum(confusion[i][j] for j in range(num_classes) if j != i)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            class_metrics[cls] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": sum(1 for l in all_labels if l == cls)
            }
        
        # 总体准确率
        correct = sum(1 for p, t in zip(all_preds, all_labels) if p == t)
        accuracy = correct / len(all_labels) if all_labels else 0
        
        # 平均置信度
        avg_confidence = sum(all_probs) / len(all_probs) if all_probs else 0
        
        return {
            "success": True,
            "accuracy": round(accuracy, 4),
            "avg_confidence": round(avg_confidence, 4),
            "total_samples": len(all_labels),
            "confusion_matrix": {
                "classes": classes,
                "matrix": confusion
            },
            "class_metrics": class_metrics,
            "predictions": [
                {"true": t, "pred": p, "correct": t == p}
                for t, p in zip(all_labels, all_preds)
            ]
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# 训练样本存储（用于评估）
training_samples_store = []

@app.post("/api/custom/store-samples")
async def store_training_samples(req: dict):
    """
    存储训练样本（用于评估）
    
    参数:
        samples: 训练样本列表
    """
    global training_samples_store
    training_samples_store = req.get("samples", [])
    return {"success": True, "stored": len(training_samples_store)}

@app.get("/api/custom/training-history")
async def get_training_history():
    """
    获取训练历史记录
    
    返回:
        训练历史（损失曲线等）
    """
    if custom_classifier.model is None:
        return {"success": False, "error": "模型未训练"}
    
    # 这里可以从文件加载训练历史
    # 暂时返回基本信息
    return {
        "success": True,
        "classes": custom_classifier.classes,
        "has_model": True,
        "model_path": str(custom_classifier.model_path)
    }

@app.get("/api/custom/export")
async def export_custom_model():
    """
    导出自定义分类器模型
    
    返回:
        模型文件 (ZIP 包含模型和类别)
    """
    if custom_classifier.model is None:
        raise HTTPException(400, "模型未训练")
    
    import tempfile
    import zipfile
    
    # 创建临时 ZIP 文件
    temp_zip = tempfile.NamedTemporaryFile(suffix='.zip', delete=False)
    temp_zip.close()
    
    with zipfile.ZipFile(temp_zip.name, 'w') as zf:
        # 写入模型文件
        if custom_classifier.model_path.exists():
            zf.write(custom_classifier.model_path, "model.pth")
        
        # 写入类别文件
        if custom_classifier.classes_path.exists():
            zf.write(custom_classifier.classes_path, "classes.json")
        
        # 写入元数据
        metadata = {
            "model_type": "resnet18",
            "classes": custom_classifier.classes,
            "num_classes": len(custom_classifier.classes),
            "export_time": str(datetime.now())
        }
        zf.writestr("metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2))
    
    return FileResponse(
        temp_zip.name,
        media_type="application/zip",
        filename=f"custom_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
        background=lambda: os.unlink(temp_zip.name)
    )

@app.post("/api/custom/import")
async def import_custom_model(file: UploadFile = File(...)):
    """
    导入自定义分类器模型
    
    参数:
        file: 模型 ZIP 文件
    
    返回:
        导入结果
    """
    import zipfile
    import tempfile
    
    if not file.filename.endswith('.zip'):
        raise HTTPException(400, "请上传 ZIP 文件")
    
    try:
        # 保存上传文件
        content = await file.read()
        temp_zip = tempfile.NamedTemporaryFile(suffix='.zip', delete=False)
        temp_zip.write(content)
        temp_zip.close()
        
        # 解压文件
        with zipfile.ZipFile(temp_zip.name, 'r') as zf:
            # 检查必要文件
            required_files = ['model.pth', 'classes.json']
            for f in required_files:
                if f not in zf.namelist():
                    raise HTTPException(400, f"ZIP 文件缺少 {f}")
            
            # 提取文件
            zf.extract('model.pth', str(MODELS_DIR))
            zf.extract('classes.json', str(MODELS_DIR))
            
            # 读取元数据
            metadata = {}
            if 'metadata.json' in zf.namelist():
                metadata = json.loads(zf.read('metadata.json'))
        
        # 清理临时文件
        os.unlink(temp_zip.name)
        
        # 重新加载模型
        custom_classifier.load()
        
        return {
            "success": True,
            "classes": custom_classifier.classes,
            "metadata": metadata,
            "message": f"成功导入模型，类别: {', '.join(custom_classifier.classes)}"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/custom/model-info")
async def get_custom_model_info():
    """
    获取自定义模型信息
    
    返回:
        模型详细信息
    """
    if custom_classifier.model is None:
        return {
            "success": True,
            "has_model": False,
            "classes": [],
            "message": "模型未训练"
        }
    
    # 获取模型文件信息
    model_size = custom_classifier.model_path.stat().st_size if custom_classifier.model_path.exists() else 0
    
    return {
        "success": True,
        "has_model": True,
        "classes": custom_classifier.classes,
        "num_classes": len(custom_classifier.classes),
        "model_path": str(custom_classifier.model_path),
        "model_size_mb": round(model_size / 1024 / 1024, 2),
        "classes_path": str(custom_classifier.classes_path)
    }

# ── 启动 ──
# 尝试加载默认模型
MODELS_DIR.mkdir(exist_ok=True)
for mt in ["vit_b", "vit_l", "vit_h"]:
    if sam.load_model(mt):
        break

# 加载 ResNet 识别模型
classifier.load()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
