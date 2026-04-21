"""生成演示对比图"""
import requests
import base64
import json
from PIL import Image
import io
import os

BASE = "http://localhost:8000"
DEMO_DIR = r"D:\OpenClaw_Workspace_full\sam-interactive-system\demo"
os.makedirs(DEMO_DIR, exist_ok=True)

TEST_IMG = r"D:\OpenClaw_Workspace_full\sam-interactive-system\test_images\truck.jpg"

# 1. 上传卡车图片
print("1. 上传图片...")
with open(TEST_IMG, "rb") as f:
    resp = requests.post(f"{BASE}/api/upload", files={"file": ("truck.jpg", f, "image/jpeg")})
    data = resp.json()
    image_id = data["image_id"]
    print(f"   Image ID: {image_id}, Size: {data['width']}x{data['height']}")

# 2. 保存原图
print("2. 保存原图...")
resp = requests.get(f"{BASE}/api/image/{image_id}")
original = Image.open(io.BytesIO(resp.content))
original.save(os.path.join(DEMO_DIR, "01_original.png"))
print(f"   原图已保存: {original.size}")

# 3. 点击分割 - 点击卡车
print("3. 点击分割 (点击卡车中心 500,375)...")
resp = requests.post(f"{BASE}/api/segment/point", json={
    "image_id": image_id,
    "points": [[500, 375]],
    "labels": [1]
})
seg_result = resp.json()
print(f"   成功: {seg_result['success']}")
print(f"   置信度: {seg_result['score']:.2%}")
print(f"   面积: {seg_result['area']} px")

# 保存分割效果图
if seg_result.get("overlay"):
    overlay_bytes = base64.b64decode(seg_result["overlay"])
    Image.open(io.BytesIO(overlay_bytes)).save(os.path.join(DEMO_DIR, "02_segmentation.png"))
    print("   分割效果图已保存")

# 保存掩码
if seg_result.get("mask"):
    mask_bytes = base64.b64decode(seg_result["mask"])
    Image.open(io.BytesIO(mask_bytes)).save(os.path.join(DEMO_DIR, "03_mask.png"))
    print("   掩码已保存")

# 4. 自动检测
print("4. 自动检测所有物体...")
resp = requests.post(f"{BASE}/api/detect/auto", json={
    "image_id": image_id,
    "points_per_side": 20,
    "min_mask_region_area": 500
})
detect_result = resp.json()
print(f"   成功: {detect_result['success']}")
print(f"   检测到 {detect_result.get('count', 0)} 个区域")
for d in detect_result.get("detections", [])[:5]:
    print(f"     #{d['id']} {d['label']} - {d['score']:.2%}")

if detect_result.get("overlay"):
    detect_bytes = base64.b64decode(detect_result["overlay"])
    Image.open(io.BytesIO(detect_bytes)).save(os.path.join(DEMO_DIR, "04_detection.png"))
    print("   检测效果图已保存")

# 5. 图像识别
print("5. 图像识别...")
resp = requests.post(f"{BASE}/api/recognize?image_id={image_id}")
recognize_result = resp.json()
print(f"   成功: {recognize_result['success']}")
print(f"   场景: {recognize_result.get('scene', 'N/A')}")
print(f"   亮度: {recognize_result.get('brightness', 'N/A')}")
print(f"   对比度: {recognize_result.get('contrast', 'N/A')}")

# 6. 生成对比拼图
print("6. 生成对比拼图...")
imgs = []
labels = ["原图", "点击分割", "掩码", "自动检测"]
files = ["01_original.png", "02_segmentation.png", "03_mask.png", "04_detection.png"]

for f in files:
    path = os.path.join(DEMO_DIR, f)
    if os.path.exists(path):
        imgs.append(Image.open(path))

if len(imgs) >= 2:
    # 统一高度
    h = 400
    imgs_resized = []
    for img in imgs:
        ratio = h / img.height
        w = int(img.width * ratio)
        imgs_resized.append(img.resize((w, h), Image.LANCZOS))

    # 拼接
    gap = 20
    total_w = sum(img.width for img in imgs_resized) + gap * (len(imgs_resized) - 1)
    comparison = Image.new("RGB", (total_w, h), (15, 23, 42))
    x = 0
    for img in imgs_resized:
        comparison.paste(img, (x, 0))
        x += img.width + gap

    comparison.save(os.path.join(DEMO_DIR, "comparison.png"))
    print("   对比拼图已保存")

print(f"\n完成! 图片保存在: {DEMO_DIR}")
