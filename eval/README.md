# 评估代码

## 文件说明

| 文件 | 说明 |
|------|------|
| `yolo_sam_eval.py` | 主评估 — YOLOv8n + SAM ViT-B + ResNet50 端到端 (v4) |
| `evaluate.py` | 早期评估 (全图 ResNet 分类) |
| `evaluate_v2.py` | v2 评估 |
| `download_subset.py` | COCO val2017 子集下载 |
| `yolo_sam_results_v4.json` | v4 结果 (匹配率 95.3%) |
| `yolo_sam_results_v3.json` | v3 结果 (匹配率 87.2%) |

## 使用方法

```bash
# 1. 下载 COCO 数据集子集
python download_subset.py

# 2. 运行评估 (需要 GPU + 已下载模型)
python yolo_sam_eval.py

# 3. 输出: yolo_sam_results_v4.json
# 预计耗时: ~15 分钟 (RTX 3060)
```

## 评估结果 (v4)

| 指标 | 值 |
|------|-----|
| ResNet 匹配率 | **95.3%** (262/275) |
| YOLO mAP@50 | 0.4587 |
| SAM mIoU (mask) | 0.5594 |
| YOLO Precision | 0.7273 |
| YOLO Recall | 0.5540 |

## 消融实验

| 方案 | 匹配率 |
|------|--------|
| 无映射 | 6.7% |
| Top-5 + 基础映射 | 49.1% |
| Top-20 + 基础映射 | 76.3% |
| Top-20 + 完整映射 | 87.2% |
| **Top-50 + 扩展映射** | **95.3%** |
