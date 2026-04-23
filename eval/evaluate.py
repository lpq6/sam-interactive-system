#!/usr/bin/env python3
"""
COCO val 评估脚本 — 通过 SAM 后端 API 进行评估
指标: mIoU, mAP@50, mAP@75, 分类 Top-1/Top-5 准确率
使用后端已加载的 SAM ViT-L + ResNet50
"""
import os, json, time, base64, io
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
import requests

API = 'http://localhost:8000'

# ── COCO paths ──
COCO_DIR = '/mnt/d/OpenClaw_Workspace_full/coco_eval'
ANN_FILE = f'{COCO_DIR}/annotations/instances_val2017.json'
IMG_DIR  = f'{COCO_DIR}/images'
RESULTS  = f'{COCO_DIR}/results.json'

# ── IoU calculation ──
def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0

# ── COCO ──
coco = COCO(ANN_FILE)
img_ids = sorted([int(f.replace('.jpg','')) for f in os.listdir(IMG_DIR) if f.endswith('.jpg')])
img_ids = [iid for iid in img_ids if iid in coco.getImgIds()]
print(f'Evaluating {len(img_ids)} images')

# ── Evaluate ──
all_ious = []
cls_correct_top1 = 0
cls_correct_top5 = 0
cls_total = 0
per_image_results = []

start = time.time()
for idx, img_id in enumerate(img_ids):
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(IMG_DIR, img_info['file_name'])
    if not os.path.exists(img_path):
        continue

    img = Image.open(img_path).convert('RGB')
    img_np = np.array(img)
    H, W = img_np.shape[:2]

    # ── Upload image ──
    with open(img_path, 'rb') as f:
        resp = requests.post(f'{API}/api/upload', files={'file': (img_info['file_name'], f, 'image/jpeg')})
    image_id = resp.json()['image_id']

    # ── Auto detect (SAM) ──
    resp = requests.post(f'{API}/api/detect/auto', json={
        'image_id': image_id,
        'points_per_side': 16,
        'min_mask_region_area': 50
    })
    detect_data = resp.json()
    masks_data = detect_data.get('detections', [])
    # Sort by confidence
    masks_data.sort(key=lambda x: x.get('confidence', 0), reverse=True)

    # ── GT annotations ──
    ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
    anns = coco.loadAnns(ann_ids)

    # For each GT annotation, find best matching predicted mask
    img_ious = []
    for ann in anns:
        gt_mask = coco.annToMask(ann)
        if gt_mask.shape != (H, W):
            gt_mask = np.array(Image.fromarray(gt_mask).resize((W, H), Image.NEAREST))

        best_iou = 0
        for pred in masks_data:
            # Decode predicted mask from base64
            mask_bytes = base64.b64decode(pred['mask'])
            pred_mask = Image.open(io.BytesIO(mask_bytes)).convert('L')
            pred_mask = np.array(pred_mask.resize((W, H), Image.NEAREST))
            if pred_mask.max() > 1:
                pred_mask = (pred_mask > 127).astype(np.uint8)

            iou = compute_iou(gt_mask, pred_mask)
            if iou > best_iou:
                best_iou = iou

        img_ious.append(best_iou)
        all_ious.append(best_iou)

    # ── Classification via API on largest mask ──
    if masks_data:
        largest = max(masks_data, key=lambda x: x.get('area', 0))
        # Decode mask
        mask_bytes = base64.b64decode(largest['mask'])
        mask_img = Image.open(io.BytesIO(mask_bytes)).convert('L')
        mask_np = np.array(mask_img.resize((W, H), Image.NEAREST))
        if mask_np.max() > 1:
            mask_np = (mask_np > 127).astype(np.uint8)

        # Crop with mask
        masked = img_np.copy()
        masked[~mask_np.astype(bool)] = 255
        x, y, w, h = largest['bbox']
        x, y, w, h = int(x), int(y), int(w), int(h)
        margin = 5
        x1, y1 = max(0, x-margin), max(0, y-margin)
        x2, y2 = min(W, x+w+margin), min(H, y+h+margin)
        crop = Image.fromarray(masked[y1:y2, x1:x2])

        # Upload cropped image for recognition
        buf = io.BytesIO()
        crop.save(buf, format='JPEG')
        buf.seek(0)
        resp = requests.post(f'{API}/api/upload', files={'file': ('crop.jpg', buf, 'image/jpeg')})
        crop_image_id = resp.json()['image_id']

        # Recognize
        resp = requests.post(f'{API}/api/recognize', json={'image_id': crop_image_id})
        rec = resp.json()
        predictions = rec.get('top_classifications', rec.get('predictions', []))

        # Get GT category name
        gt_cat_ids = [ann['category_id'] for ann in anns]
        gt_cat_names = set()
        for cid in gt_cat_ids:
            cat = coco.loadCats(cid)
            if cat:
                gt_cat_names.add(cat[0]['name'])

        if predictions:
            cls_total += 1
            pred_names = [p['label'] for p in predictions[:5]]
            # Top-1
            if pred_names[0] in gt_cat_names:
                cls_correct_top1 += 1
            # Top-5
            if any(l in gt_cat_names for l in pred_names):
                cls_correct_top5 += 1

    per_image_results.append({
        'img_id': img_id,
        'num_gt': len(anns),
        'num_pred': len(masks_data),
        'miou': np.mean(img_ious) if img_ious else 0,
    })

    if (idx + 1) % 5 == 0:
        elapsed = time.time() - start
        miu = np.mean(all_ious) if all_ious else 0
        print(f'  [{idx+1}/{len(img_ids)}] mIoU={miu:.3f} | top1={cls_correct_top1}/{cls_total} | {elapsed:.0f}s')

# ── Results ──
elapsed = time.time() - start

miou = np.mean(all_ious) if all_ious else 0
ap50 = np.mean([1.0 if iou >= 0.5 else 0.0 for iou in all_ious])
ap75 = np.mean([1.0 if iou >= 0.75 else 0.0 for iou in all_ious])
thresholds = np.arange(0.50, 1.00, 0.05)
ap_per_t = [np.mean([1.0 if iou >= t else 0.0 for iou in all_ious]) for t in thresholds]
map_avg = np.mean(ap_per_t)

top1_acc = cls_correct_top1 / cls_total if cls_total > 0 else 0
top5_acc = cls_correct_top5 / cls_total if cls_total > 0 else 0

results = {
    'dataset': 'COCO val2017 (subset)',
    'num_images': len(img_ids),
    'num_annotations_evaluated': len(all_ious),
    'model': 'SAM ViT-L (auto detect) + ResNet50 (classification)',
    'device': 'cuda (RTX 3060 Laptop)',
    'elapsed_seconds': round(elapsed, 1),
    'segmentation': {
        'mIoU': round(float(miou), 4),
        'AP@50': round(float(ap50), 4),
        'AP@75': round(float(ap75), 4),
        'mAP@[.50:.95]': round(float(map_avg), 4),
        'AP_per_threshold': {f'{t:.2f}': round(float(ap), 4) for t, ap in zip(thresholds, ap_per_t)},
    },
    'classification': {
        'top1_accuracy': round(float(top1_acc), 4),
        'top5_accuracy': round(float(top5_acc), 4),
        'samples': cls_total,
    },
    'per_image': per_image_results
}

with open(RESULTS, 'w') as f:
    json.dump(results, f, indent=2)

print(f'\n{"="*50}')
print(f'COCO val2017 Evaluation Results')
print(f'{"="*50}')
print(f'Images:        {len(img_ids)}')
print(f'Annotations:   {len(all_ious)}')
print(f'Time:          {elapsed:.1f}s ({elapsed/60:.1f} min)')
print(f'')
print(f'Segmentation (SAM ViT-L Auto Detect):')
print(f'  mIoU:          {miou:.4f}')
print(f'  AP@50:         {ap50:.4f}')
print(f'  AP@75:         {ap75:.4f}')
print(f'  mAP@[.50:.95]: {map_avg:.4f}')
print(f'')
print(f'Classification (ResNet50 on largest mask):')
print(f'  Top-1 Acc:     {top1_acc:.4f} ({cls_correct_top1}/{cls_total})')
print(f'  Top-5 Acc:     {top5_acc:.4f} ({cls_correct_top5}/{cls_total})')
print(f'')
print(f'Results saved to: {RESULTS}')
