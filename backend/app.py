"""
SAM 交互式分割系统 - 后端服务
基于 Facebook Segment Anything Model
"""
import os, io, uuid, base64, json
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

import numpy as np
from PIL import Image, ImageFilter
import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
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


def create_rgba_from_mask(img: np.ndarray, mask: np.ndarray, smooth: bool = True) -> np.ndarray:
    """
    从掩码创建 RGBA 图像（透明背景）
    
    参数:
        img: RGB 图像
        mask: 分割掩码
        smooth: 是否平滑边缘
    
    返回:
        RGBA 图像 (透明背景)
    """
    h, w = img.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = img
    
    if smooth:
        # 平滑掩码边缘
        alpha = smooth_mask(mask, blur_radius=3, feather=True)
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
    available = []
    for name, file in [("vit_b","SAM ViT-B (375MB)"),("vit_l","SAM ViT-L (1.2GB)"),("vit_h","SAM ViT-H (2.4GB)")]:
        available.append({
            "id": name,
            "name": file,
            "loaded": sam.model_type == name,
            "exists": (MODELS_DIR / f"sam_{name}_*.pth").exists() if MODELS_DIR.exists() else False
        })
    return {"models": available, "current": sam.model_type, "device": sam.device}

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
    smooth_alpha = smooth_mask(mask, blur_radius=3, feather=True)
    smooth_mask_img = Image.fromarray(smooth_alpha, 'L')
    buf = io.BytesIO()
    smooth_mask_img.save(buf, format="PNG")
    smooth_mask_b64 = base64.b64encode(buf.getvalue()).decode()

    # 生成平滑 RGBA 图像
    rgba = create_rgba_from_mask(img, mask, smooth=True)
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
    smooth_alpha = smooth_mask(mask, blur_radius=3, feather=True)
    smooth_mask_img = Image.fromarray(smooth_alpha, 'L')
    buf = io.BytesIO()
    smooth_mask_img.save(buf, format="PNG")
    smooth_mask_b64 = base64.b64encode(buf.getvalue()).decode()

    # 生成平滑 RGBA 图像
    rgba = create_rgba_from_mask(img, mask, smooth=True)
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
async def extract_all_objects(image_id: str, min_area: int = 500, min_confidence: float = 0.3):
    """
    批量提取高置信度彩色物体
    
    参数:
        image_id: 图片ID
        min_area: 最小区域面积（默认500）
        min_confidence: 最低置信度阈值（默认0.3，即30%）
    
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
                    rgba = create_rgba_from_mask(img, mask, smooth=True)
                    
                    y1, y2 = ys.min(), ys.max()
                    x1, x2 = xs.min(), xs.max()
                    cropped = rgba[y1:y2+1, x1:x2+1]
                    
                    # 转为base64
                    result_img = Image.fromarray(cropped, 'RGBA')
                    buffer = io.BytesIO()
                    result_img.save(buffer, format='PNG')
                    color_b64 = base64.b64encode(buffer.getvalue()).decode()
                    
                    # 生成平滑掩码base64
                    smooth_alpha = smooth_mask(mask, blur_radius=3, feather=True)
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
