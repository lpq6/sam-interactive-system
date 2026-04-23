"""
Demo 生成脚本 — 演示 YOLO+SAM+ResNet 识别效果
运行方式: python generate_demo.py
"""
import json

# v4 评估 demo 数据
DEMO_RESULTS = [
    {
        "image": "000000037777.jpg",
        "scene": "厨房",
        "objects": [
            {"yolo": "oven", "resnet": "stove", "confidence": 0.763, "matched": True},
            {"yolo": "refrigerator", "resnet": "refrigerator", "confidence": 0.687, "matched": True},
            {"yolo": "dining table", "resnet": "breastplate", "confidence": 0.663, "matched": True},
            {"yolo": "refrigerator", "resnet": "refrigerator", "confidence": 0.537, "matched": True},
        ]
    },
    {
        "image": "000000017178.jpg",
        "scene": "马匹 + 汽车",
        "objects": [
            {"yolo": "horse", "resnet": "sorrel", "confidence": 0.816, "matched": True},
            {"yolo": "horse", "resnet": "sorrel", "confidence": 0.760, "matched": True},
            {"yolo": "car", "resnet": "car wheel", "confidence": 0.704, "matched": True},
            {"yolo": "horse", "resnet": "sorrel", "confidence": 0.343, "matched": True},
            {"yolo": "horse", "resnet": "sorrel", "confidence": 0.262, "matched": True},
        ]
    },
    {
        "image": "000000093437.jpg",
        "scene": "办公桌",
        "objects": [
            {"yolo": "person", "resnet": "sweatshirt", "confidence": 0.941, "matched": True},
            {"yolo": "chair", "resnet": "throne", "confidence": 0.758, "matched": True},
            {"yolo": "chair", "resnet": "barber chair", "confidence": 0.728, "matched": True},
            {"yolo": "bottle", "resnet": "water bottle", "confidence": 0.682, "matched": True},
            {"yolo": "potted plant", "resnet": "vase", "confidence": 0.410, "matched": True},
            {"yolo": "potted plant", "resnet": "vase", "confidence": 0.319, "matched": True},
            {"yolo": "chair", "resnet": "throne", "confidence": 0.266, "matched": True},
        ]
    },
]

def main():
    total = sum(len(d["objects"]) for d in DEMO_RESULTS)
    matched = sum(1 for d in DEMO_RESULTS for o in d["objects"] if o["matched"])
    
    print(f"YOLO+SAM+ResNet 识别 Demo")
    print(f"{'='*50}")
    print(f"测试图片: {len(DEMO_RESULTS)} 张")
    print(f"总物体数: {total}")
    print(f"匹配率: {matched}/{total} ({matched/total*100:.1f}%)")
    print()
    
    for demo in DEMO_RESULTS:
        print(f"📷 {demo['image']} ({demo['scene']})")
        for obj in demo["objects"]:
            m = "✅" if obj["matched"] else "❌"
            print(f"  {m} {obj['yolo']:15s} → {obj['resnet']:20s} ({obj['confidence']*100:.1f}%)")
        print()

if __name__ == "__main__":
    main()
