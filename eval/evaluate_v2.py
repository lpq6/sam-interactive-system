#!/usr/bin/env python3
"""
COCO val 评估脚本 v2 — SAM 分割 + ResNet50 识别
通过后端 API 调用，使用 bbox IoU 评估检测质量
指标: mIoU (mask), bbox mAP, 分类准确率
"""
import os, json, time, base64, io
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
import requests

API = 'http://localhost:8000'

# ── paths ──
COCO_DIR = '/mnt/d/OpenClaw_Workspace_full/coco_eval'
ANN_FILE = f'{COCO_DIR}/annotations/instances_val2017.json'
IMG_DIR  = f'{COCO_DIR}/images'
RESULTS  = f'{COCO_DIR}/results_v2.json'

# ── helpers ──
def compute_iou(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return inter / union if union > 0 else 0

def bbox_iou(box1, box2):
    # box = [x, y, w, h]
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    x1b, y1b = x1+w1, y1+h1
    x2b, y2b = x2+w2, y2+h2
    xi = max(0, min(x1b, x2b) - max(x1, x2))
    yi = max(0, min(y1b, y2b) - max(y1, y2))
    inter = xi * yi
    union = w1*h1 + w2*h2 - inter
    return inter / union if union > 0 else 0

# ── COCO ──
coco = COCO(ANN_FILE)
img_ids = sorted([int(f.replace('.jpg','')) for f in os.listdir(IMG_DIR) if f.endswith('.jpg')])
img_ids = [iid for iid in img_ids if iid in coco.getImgIds()]
print(f'Evaluating {len(img_ids)} images')

# ── Evaluate ──
mask_ious = []        # best mask IoU for each GT
bbox_ious = []        # best bbox IoU for each GT
all_gt = 0
all_pred = 0
cls_correct_top1 = 0
cls_correct_top5 = 0
cls_total = 0
per_image = []

start = time.time()
for idx, img_id in enumerate(img_ids):
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(IMG_DIR, img_info['file_name'])
    if not os.path.exists(img_path):
        continue

    img = Image.open(img_path).convert('RGB')
    img_np = np.array(img)
    H, W = img_np.shape[:2]

    # Upload & detect
    with open(img_path, 'rb') as f:
        resp = requests.post(f'{API}/api/upload', files={'file': (img_info['file_name'], f, 'image/jpeg')})
    image_id = resp.json()['image_id']

    resp = requests.post(f'{API}/api/detect/auto', json={
        'image_id': image_id, 'points_per_side': 16, 'min_mask_region_area': 50
    })
    detections = resp.json().get('detections', [])
    detections.sort(key=lambda x: x.get('confidence', 0), reverse=True)
    all_pred += len(detections)

    # GT
    ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
    anns = coco.loadAnns(ann_ids)
    all_gt += len(anns)

    # Match each GT to best predicted mask/bbox
    matched_preds = set()
    for ann in anns:
        gt_bbox = ann['bbox']  # [x, y, w, h] COCO format
        gt_mask = coco.annToMask(ann)
        if gt_mask.shape != (H, W):
            gt_mask = np.array(Image.fromarray(gt_mask).resize((W, H), Image.NEAREST))

        best_mask_iou = 0
        best_bbox_iou = 0
        for p_idx, pred in enumerate(detections):
            # Bbox IoU
            bi = bbox_iou(gt_bbox, pred['bbox'])
            if bi > best_bbox_iou:
                best_bbox_iou = bi

            # Mask IoU
            try:
                mask_bytes = base64.b64decode(pred['mask'])
                pred_mask = np.array(Image.open(io.BytesIO(mask_bytes)).convert('L').resize((W, H), Image.NEAREST))
                if pred_mask.max() > 1:
                    pred_mask = (pred_mask > 127).astype(np.uint8)
                mi = compute_iou(gt_mask, pred_mask)
                if mi > best_mask_iou:
                    best_mask_iou = mi
                    if mi >= 0.5:
                        matched_preds.add(p_idx)
            except:
                pass

        mask_ious.append(best_mask_iou)
        bbox_ious.append(best_bbox_iou)

    # Classification: use recognize API on the full image
    if anns:
        resp = requests.post(f'{API}/api/recognize', json={'image_id': image_id})
        rec = resp.json()
        predictions = rec.get('top_classifications', rec.get('predictions', []))

        gt_cat_names = set()
        for ann in anns:
            cat = coco.loadCats(ann['category_id'])
            if cat:
                gt_cat_names.add(cat[0]['name'])

        if predictions:
            cls_total += 1
            pred_names = [p['label'] for p in predictions[:5]]
            if pred_names[0] in gt_cat_names:
                cls_correct_top1 += 1
            if any(l in gt_cat_names for l in pred_names):
                cls_correct_top5 += 1

    per_image.append({
        'img_id': img_id, 'gt': len(anns), 'pred': len(detections),
        'miou': float(np.mean(mask_ious[-len(anns):])) if anns else 0
    })

    if (idx + 1) % 10 == 0:
        elapsed = time.time() - start
        mmiou = np.mean(mask_ious) if mask_ious else 0
        mbiou = np.mean(bbox_ious) if bbox_ious else 0
        print(f'  [{idx+1}/{len(img_ids)}] mask_mIoU={mmiou:.3f} bbox_mIoU={mbiou:.3f} top1={cls_correct_top1}/{cls_total} {elapsed:.0f}s')

# ── Results ──
elapsed = time.time() - start

def calc_ap(ious, thresh):
    return np.mean([1.0 if i >= thresh else 0.0 for i in ious])

thresholds = np.arange(0.50, 1.00, 0.05)

results = {
    'dataset': 'COCO val2017 (94 images subset)',
    'num_images': len(img_ids),
    'gt_annotations': all_gt,
    'predicted_masks': all_pred,
    'model': 'SAM ViT-L + ResNet50 (via API)',
    'device': 'cuda',
    'time_seconds': round(elapsed, 1),
    'segmentation_mask': {
        'mIoU': round(float(np.mean(mask_ious)), 4),
        'AP@50': round(float(calc_ap(mask_ious, 0.5)), 4),
        'AP@75': round(float(calc_ap(mask_ious, 0.75)), 4),
        'mAP': round(float(np.mean([calc_ap(mask_ious, t) for t in thresholds])), 4),
    },
    'detection_bbox': {
        'mIoU': round(float(np.mean(bbox_ious)), 4),
        'AP@50': round(float(calc_ap(bbox_ious, 0.5)), 4),
        'AP@75': round(float(calc_ap(bbox_ious, 0.75)), 4),
        'mAP': round(float(np.mean([calc_ap(bbox_ious, t) for t in thresholds])), 4),
    },
    'classification': {
        'top1_accuracy': round(cls_correct_top1 / cls_total, 4) if cls_total else 0,
        'top5_accuracy': round(cls_correct_top5 / cls_total, 4) if cls_total else 0,
        'samples': cls_total,
    }
}

with open(RESULTS, 'w') as f:
    json.dump(results, f, indent=2)

print(f'\n{"="*55}')
print(f'  COCO val2017 Evaluation — SAM Interactive System')
print(f'{"="*55}')
print(f'Images:          {len(img_ids)}')
print(f'GT Annotations:  {all_gt}')
print(f'Predicted Masks:  {all_pred}')
print(f'Time:            {elapsed:.0f}s ({elapsed/60:.1f} min)')
print()
print(f'  Mask Segmentation:')
print(f'    mIoU:          {np.mean(mask_ious):.4f}')
print(f'    AP@50:         {calc_ap(mask_ious, 0.5):.4f}')
print(f'    AP@75:         {calc_ap(mask_ious, 0.75):.4f}')
print(f'    mAP@[.50:.95]: {np.mean([calc_ap(mask_ious, t) for t in thresholds]):.4f}')
print()
print(f'  Bbox Detection:')
print(f'    mIoU:          {np.mean(bbox_ious):.4f}')
print(f'    AP@50:         {calc_ap(bbox_ious, 0.5):.4f}')
print(f'    AP@75:         {calc_ap(bbox_ious, 0.75):.4f}')
print(f'    mAP@[.50:.95]: {np.mean([calc_ap(bbox_ious, t) for t in thresholds]):.4f}')
print()
if cls_total:
    print(f'  Classification (full image):')
    print(f'    Top-1:         {cls_correct_top1}/{cls_total} = {cls_correct_top1/cls_total:.4f}')
    print(f'    Top-5:         {cls_correct_top5}/{cls_total} = {cls_correct_top5/cls_total:.4f}')
print(f'\nSaved: {RESULTS}')
