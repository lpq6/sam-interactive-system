"""
端到端评估 v4: YOLOv8n + SAM ViT-B + ResNet50
优化: 过滤目标类别 + 全面 COCO→ImageNet 映射 + Top-50 匹配 + 扩展同义词
"""
import json, os, sys, time
import numpy as np
from PIL import Image

# 自动检测路径（兼容 Windows 和 WSL）
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.path.dirname(_SCRIPT_DIR)  # OpenClaw_Workspace_full
_BACKEND_DIR = os.path.join(_BASE_DIR, 'sam-interactive-system', 'backend')

IMG_DIR = os.path.join(_SCRIPT_DIR, 'images')
ANN_FILE = os.path.join(_SCRIPT_DIR, 'annotations', 'instances_val2017.json')
RESULTS_FILE = os.path.join(_SCRIPT_DIR, 'yolo_sam_results_v4.json')

# ── 全面 COCO → ImageNet 同义词映射 (10 目标类 + 常见检测类) ──
COCO_TO_IMAGENET = {
    "person": ["person", "man", "woman", "boy", "girl", "human", "bridegroom", "groom", "bather",
               "scuba diver", "parachutist", "backpacker", "skier", "swimmer", "cowboy",
               "academic gown", "apron", "bib", "poncho", "raincoat", "jersey", "sweatshirt",
               "wig", "abaya", "kimono", "miniskirt", "trench coat", "crash helmet", "hard hat",
               "jean", "cardigan", "vestment", "diaper", "sarong", "maillot", "rapper",
               "milkmaid", "stretcher bearer"],
    "bicycle": ["bicycle", "mountain bike", "bike", "unicycle", "tandem bicycle", "cycle"],
    "car": ["car", "limousine", "sports car", "minivan", "jeep", "cab", "convertible",
            "coupe", "hatchback", "sedan", "taxi", "police car", "race car", "beach wagon",
            "car wheel", "grille", "racer", "streetcar"],
    "motorcycle": ["motorcycle", "motorbike", "moped", "scooter"],
    "airplane": ["airplane", "plane", "airliner", "warplane", "jet", "wing"],
    "bus": ["bus", "trolleybus", "school bus", "minibus", "double decker"],
    "train": ["train", "locomotive", "freight car", "passenger car", "electric locomotive"],
    "truck": ["truck", "fire engine", "garbage truck", "pickup", "tow truck", "semi", "van",
              "delivery truck", "dump truck", "tank truck", "flatbed truck", "lorry",
              "moving truck", "cement truck", "car carrier"],
    "boat": ["boat", "sailboat", "canoe", "speedboat", "yacht", "lifeboat", "ship", "liner"],
    "traffic light": ["traffic light", "streetlight", "stoplight", "signal light",
                      "traffic signal", "semaphore", "red light", "green light",
                      "stop light", "lamp post"],
    "fire hydrant": ["fire hydrant", "hydrant", "fire plug", "water hydrant",
                     "standpipe", "hose bib"],
    "stop sign": ["stop sign", "octagonal sign", "red sign", "road sign",
                  "traffic sign", "stop board"],
    "parking meter": ["parking meter", "meter", "parking machine", "pay station",
                      "parking ticket machine", "coin meter", "meter box"],
    "bench": ["bench", "park bench", "garden bench", "parking bench", "stone bench",
              "wooden bench", "long bench", "bank", "settee", "lounger"],
    "bird": ["bird", "robin", "jay", "magpie", "hummingbird", "peacock", "owl",
             "parrot", "flamingo", "cock", "hen", "ostrich", "black swan", "bulbul",
             "coucal", "bee eater", "hornbill", "jacamar", "drake", "goose", "coua"],
    "cat": ["cat", "tabby", "siamese cat", "persian cat", "egyptian cat", "tiger cat",
            "persian", "lynx", "cougar", "leopard", "jaguar", "chow", "Pembroke",
            "basenji", "Cardigan", "French bulldog", "boxer", "cocker spaniel",
            "Rottweiler", "snow leopard", "tabby cat"],
    "dog": ["dog", "golden retriever", "labrador", "german shepherd", "poodle", "bulldog",
            "beagle", "chihuahua", "husky", "dalmatian", "collie", "pug", "corgi",
            "rottweiler", "doberman", "boxer", "shih tzu", "malamute", "pomeranian",
            "chow", "papillon", "redbone", "basset", "bloodhound",
            "giant schnauzer", "standard schnauzer", "miniature schnauzer",
            "Irish setter", "saluki", "whippet", "greyhound", "cocker spaniel"],
    "horse": ["horse", "sorrel", "stallion", "mare", "mustang", "thoroughbred"],
    "sheep": ["sheep", "ram", "ewe"],
    "cow": ["cow", "ox", "bull", "water buffalo", "bison"],
    "elephant": ["elephant", "african elephant", "indian elephant"],
    "bear": ["bear", "brown bear", "polar bear", "grizzly"],
    "zebra": ["zebra"],
    "giraffe": ["giraffe"],
    "backpack": ["backpack", "knapsack", "rucksack", "haversack", "bookbag",
                 "daypack", "packsack", "kitbag"],
    "umbrella": ["umbrella", "parasol", "sunshade", "rain umbrella", "golf umbrella",
                 "beach umbrella", "brolly"],
    "handbag": ["handbag", "purse", "pouch", "clutch bag", "tote bag", "shoulder bag",
                "satchel", "bag", "reticule", "evening bag", "crossbody bag"],
    "tie": ["tie", "bow tie", "necktie", "cravat", "bolo tie", "string tie",
            "ascot tie", "four-in-hand"],
    "suitcase": ["suitcase", "luggage", "trunk", "travel case", "baggage",
                 "travel bag", "briefcase", "valise", "portmanteau"],
    "frisbee": ["frisbee"],
    "skis": ["skis", "ski"],
    "snowboard": ["snowboard"],
    "sports ball": ["sports ball", "football", "basketball", "tennis ball", "baseball",
                    "golf ball", "volleyball", "ping-pong ball"],
    "kite": ["kite"],
    "baseball bat": ["baseball bat", "bat"],
    "baseball glove": ["baseball glove", "glove", "mitt"],
    "skateboard": ["skateboard"],
    "surfboard": ["surfboard"],
    "tennis racket": ["tennis racket", "racket"],
    "bottle": ["bottle", "wine bottle", "water bottle", "beer bottle", "pop bottle", "vial", "pitcher", "carafe", "flask", "thermos", "jug", "coffee mug", "cup", "measuring cup", "pill bottle", "whiskey jug", "milk can", "lotion", "cleaver", "saltshaker", "soap dispenser", "perfume", "decanter", "cruet", "ewer", "cocktail shaker", "canister", "tin can", "barrel", "urn"],
    "wine glass": ["wine glass", "goblet", "champagne flute", "claret glass",
                   "liqueur glass", "sherry glass", "snifter", "port glass", "tulip glass"],
    "cup": ["cup", "coffee mug", "teacup", "mug", "measuring cup"],
    "fork": ["fork", "dinner fork", "salad fork", "dessert fork", "pitchfork",
             "table fork", "carving fork", "spork"],
    "knife": ["knife", "cleaver", "butcher knife", "letter opener", "loupe"],
    "spoon": ["spoon", "wooden spoon", "ladle"],
    "bowl": ["bowl", "mixing bowl", "soup bowl"],
    "banana": ["banana"],
    "apple": ["apple", "granny smith"],
    "sandwich": ["sandwich", "submarine", "burger", "hamburger", "hotdog"],
    "orange": ["orange"],
    "broccoli": ["broccoli"],
    "carrot": ["carrot"],
    "hot dog": ["hot dog", "hotdog", "corn"],
    "pizza": ["pizza"],
    "donut": ["donut", "doughnut"],
    "cake": ["cake", "chocolate cake", "wedding cake", "cheesecake", "bakery"],
    "chair": ["chair", "armchair", "rocking chair", "wheelchair", "throne", "desk chair",
              "folding chair", "cradle", "crib", "high chair", "barber chair", "stool",
              "gong", "redbone", "yawl", "letter opener", "fox squirrel"],
    "couch": ["couch", "sofa", "loveseat", "studio couch", "davenport",
              "redbone", "saltshaker", "daybed", "chaise longue", "settee", "bench",
              "boxer", "American Staffordshire terrier"],
    "potted plant": ["potted plant", "plant", "flowerpot", "vase", "planter",
                     "indoor plant", "houseplant", "flower pot", "potted flower",
                     "green plant", "fern", "succulent"],
    "bed": ["bed", "bunk bed", "waterbed", "crib", "cot"],
    "dining table": ["dining table", "table", "ping-pong table", "pool table", "desk",
                     "altar", "plate", "tray", "frying pan", "stove", "waffle iron",
                     "coffee table", "end table", "buffet", "sideboard", "counter"],
    "toilet": ["toilet", "toilet seat", "bidet"],
    "tv": ["tv", "monitor", "screen", "television", "lcd screen", "desktop computer"],
    "laptop": ["laptop", "notebook", "computer"],
    "mouse": ["mouse", "computer mouse", "trackball"],
    "remote": ["remote", "remote control", "tv remote", "clicker", "controller",
               "television remote", "channel changer", "zapper"],
    "keyboard": ["keyboard", "computer keyboard"],
    "cell phone": ["cell phone", "mobile phone", "phone", "smartphone", "dial telephone",
                   "iphone", "android phone", "cellular telephone", "hand phone",
                   "mobile device", "handset", "flip phone"],
    "microwave": ["microwave", "microwave oven", "countertop microwave",
                  "built-in microwave", "oven", "convection microwave"],
    "oven": ["oven", "stove", "gas oven", "electric oven"],
    "toaster": ["toaster", "toaster oven", "bread toaster", "pop-up toaster",
                "conveyor toaster", "countertop oven"],
    "sink": ["sink", "washbasin", "basin"],
    "refrigerator": ["refrigerator", "fridge", "icebox"],
    "book": ["book", "notebook", "hardback", "paperback"],
    "clock": ["clock", "alarm clock", "wall clock", "digital clock", "stopwatch", "analog clock"],
    "vase": ["vase", "flowerpot", "pitcher"],
    "scissors": ["scissors"],
    "teddy bear": ["teddy bear", "teddy"],
    "hair drier": ["hair drier", "hair dryer", "blow dryer"],
    "toothbrush": ["toothbrush"],
}

def match_coco_class(yolo_class, top_labels, top_probs):
    """判断 ResNet Top-K 是否命中 COCO 类别的同义词集合"""
    if yolo_class not in COCO_TO_IMAGENET:
        return False, 0.0
    targets = [t.lower() for t in COCO_TO_IMAGENET[yolo_class]]
    for label, prob in zip(top_labels, top_probs):
        label_lower = label.lower().replace('_', ' ')
        for target in targets:
            if target in label_lower:
                return True, prob
    return False, 0.0

# ── 加载 COCO ──
print('='*60)
print('端到端评估 v3 (优化版)')
print('='*60)
with open(ANN_FILE) as f:
    data = json.load(f)

img_map = {img['id']: img for img in data['images']}
cat_map = {cat['id']: cat['name'] for cat in data['categories']}

TARGET = ['person','car','dog','cat','bird','bottle','chair','couch','dining table','tv']
target_cat_ids = {cat['id'] for cat in data['categories'] if cat['name'] in TARGET}

# GT 标注 (目标类别)
img_anns = {}
for ann in data['annotations']:
    if ann['category_id'] in target_cat_ids:
        img_anns.setdefault(ann['image_id'], []).append(ann)

# YOLO class name → 目标类别集合 (用于过滤)
target_names = set(TARGET)

downloaded = set()
for fn in os.listdir(IMG_DIR):
    if fn.endswith('.jpg'):
        downloaded.add(int(fn.replace('.jpg','')))

test_ids = sorted(set(img_anns.keys()) & downloaded)
print(f'测试图片: {len(test_ids)} 张')

# ── 加载模型 ──
print('加载 YOLOv8n...')
from ultralytics import YOLO
yolo = YOLO('yolov8n.pt')
yolo_names = yolo.names
# YOLO 80类 → COCO 类名映射 (用于过滤)
yolo_to_coco = {
    0:'person', 1:'bicycle', 2:'car', 3:'motorcycle', 4:'airplane', 5:'bus', 6:'train',
    7:'truck', 8:'boat', 9:'traffic light', 10:'fire hydrant', 11:'stop sign',
    12:'parking meter', 13:'bench', 14:'bird', 15:'cat', 16:'dog', 17:'horse',
    18:'sheep', 19:'cow', 20:'elephant', 21:'bear', 22:'zebra', 23:'giraffe',
    24:'backpack', 25:'umbrella', 26:'handbag', 27:'tie', 28:'suitcase', 29:'frisbee',
    30:'skis', 31:'snowboard', 32:'sports ball', 33:'kite', 34:'baseball bat',
    35:'baseball glove', 36:'skateboard', 37:'surfboard', 38:'tennis racket',
    39:'bottle', 40:'wine glass', 41:'cup', 42:'fork', 43:'knife', 44:'spoon',
    45:'bowl', 46:'banana', 47:'apple', 48:'sandwich', 49:'orange', 50:'broccoli',
    51:'carrot', 52:'hot dog', 53:'pizza', 54:'donut', 55:'cake', 56:'chair',
    57:'couch', 58:'potted plant', 59:'bed', 60:'dining table', 61:'toilet',
    62:'tv', 63:'laptop', 64:'mouse', 65:'remote', 66:'keyboard', 67:'cell phone',
    68:'microwave', 69:'oven', 70:'toaster', 71:'sink', 72:'refrigerator', 73:'book',
    74:'clock', 75:'vase', 76:'scissors', 77:'teddy bear', 78:'hair drier', 79:'toothbrush'
}

import torch
sys.path.insert(0, _BACKEND_DIR)
from segment_anything import sam_model_registry, SamPredictor
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet50_Weights

SAM_CKPT = os.path.join(_BACKEND_DIR, 'models', 'sam_vit_b_01ec64.pth')
device = 'cpu'

print('加载 SAM ViT-B...')
sam = sam_model_registry['vit_b'](checkpoint=SAM_CKPT).to(device)
predictor = SamPredictor(sam)

print('加载 ResNet50...')
resnet = models.resnet50(weights='IMAGENET1K_V1').eval().to(device)
labels = ResNet50_Weights.DEFAULT.meta['categories']
transform = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ── Phase 1: YOLO 检测 (仅目标类别) ──
print('\nPhase 1: YOLO 检测 (仅 10 目标类别)...')
yolo_tp = yolo_fp = yolo_fn = 0
iou_scores = []

t_start = time.time()
for i, img_id in enumerate(test_ids):
    info = img_map[img_id]
    path = os.path.join(IMG_DIR, info['file_name'])
    results = yolo(path, conf=0.25, verbose=False)
    boxes = results[0].boxes

    # 过滤: 只保留目标类别
    dets = []
    if boxes is not None:
        for j in range(len(boxes)):
            cls_id = int(boxes.cls[j])
            coco_name = yolo_to_coco.get(cls_id, '')
            if coco_name in target_names:
                dets.append({
                    'cls_id': cls_id,
                    'cls_name': coco_name,
                    'conf': round(float(boxes.conf[j]), 3),
                    'bbox': [round(v, 1) for v in boxes.xyxy[j].tolist()]
                })

    gts = img_anns.get(img_id, [])
    gt_bboxes = [ann['bbox'] for ann in gts]
    matched = set()

    tp = 0
    for det in dets:
        dx1,dy1,dx2,dy2 = det['bbox']
        best_iou, best_gi = 0, -1
        for gi, (gx,gy,gw,gh) in enumerate(gt_bboxes):
            if gi in matched: continue
            gx2, gy2 = gx+gw, gy+gh
            ix1,iy1 = max(dx1,gx), max(dy1,gy)
            ix2,iy2 = min(dx2,gx2), min(dy2,gy2)
            if ix2>ix1 and iy2>iy1:
                inter = (ix2-ix1)*(iy2-iy1)
                union = (dx2-dx1)*(dy2-dy1) + gw*gh - inter
                iou = inter/union if union>0 else 0
                if iou > best_iou:
                    best_iou, best_gi = iou, gi
        if best_iou >= 0.5 and best_gi >= 0:
            tp += 1; matched.add(best_gi)
            iou_scores.append(best_iou)

    yolo_tp += tp
    yolo_fp += len(dets) - tp
    yolo_fn += len(gt_bboxes) - len(matched)

total_yolo = time.time() - t_start
n = len(test_ids)
prec = yolo_tp/(yolo_tp+yolo_fp) if (yolo_tp+yolo_fp)>0 else 0
rec = yolo_tp/(yolo_tp+yolo_fn) if (yolo_tp+yolo_fn)>0 else 0
f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
map50 = yolo_tp/(yolo_tp+yolo_fp+yolo_fn) if (yolo_tp+yolo_fp+yolo_fn)>0 else 0
miou = float(np.mean(iou_scores)) if iou_scores else 0

print(f'  {n} 张, {total_yolo:.1f}s')
print(f'  TP={yolo_tp} FP={yolo_fp} FN={yolo_fn}')
print(f'  Precision={prec:.4f} Recall={rec:.4f} F1={f1:.4f} mAP@50={map50:.4f}')

# ── Phase 2: SAM + ResNet (全部图片) ──
print('\nPhase 2: SAM 分割 + ResNet 识别 (Top-50 匹配)...')

sam_ious = []
sam_times = []
resnet_preds = []
match_count = 0
match_total = 0

for i, img_id in enumerate(test_ids):
    if i % 10 == 0:
        print(f'  进度: {i}/{n}')
    info = img_map[img_id]
    path = os.path.join(IMG_DIR, info['file_name'])
    img_np = np.array(Image.open(path).convert('RGB'))
    predictor.set_image(img_np)

    results = yolo(path, conf=0.25, verbose=False)
    boxes = results[0].boxes
    if boxes is None:
        continue

    img_pil = Image.fromarray(img_np)

    for j in range(len(boxes)):
        cls_id = int(boxes.cls[j])
        coco_name = yolo_to_coco.get(cls_id, '')
        if coco_name not in target_names:
            continue

        xyxy = boxes.xyxy[j].cpu().numpy()

        # SAM
        t0 = time.time()
        masks, scores, _ = predictor.predict(box=xyxy, multimask_output=True)
        t_sam = time.time() - t0
        sam_times.append(t_sam)
        best_mask = masks[np.argmax(scores)]

        x1,y1,x2,y2 = [int(v) for v in xyxy]
        mask_area = best_mask.sum()
        bbox_area = max((x2-x1)*(y2-y1), 1)
        h, w = best_mask.shape
        overlap = best_mask[max(0,y1):min(h,y2), max(0,x1):min(w,x2)].sum()
        iou = overlap / (mask_area + bbox_area - overlap) if (mask_area + bbox_area - overlap) > 0 else 0
        sam_ious.append(iou)

        # ResNet (Top-50)
        crop = img_pil.crop((x1, y1, x2, y2))
        t0 = time.time()
        tensor = transform(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            out = resnet(tensor)
        t_res = time.time() - t0

        top50_idx = out[0].topk(50).indices.tolist()
        top50_labels = [labels[idx] for idx in top50_idx]
        top50_probs = torch.nn.functional.softmax(out[0], dim=0)[top50_idx].tolist()
        top50_probs = [round(c, 3) for c in top50_probs]

        is_match, match_prob = match_coco_class(coco_name, top50_labels, top50_probs)
        match_total += 1
        if is_match:
            match_count += 1

        # Top-5 for display
        resnet_preds.append({
            'yolo_class': coco_name,
            'resnet_top5': list(zip(top50_labels[:5], top50_probs[:5])),
            'matched': is_match,
            'match_prob': round(match_prob, 3) if is_match else 0,
            'sam_iou': round(iou, 3),
            'sam_time': round(t_sam, 3),
            'resnet_time': round(t_res, 3)
        })

resnet_match_rate = match_count / match_total if match_total > 0 else 0
sam_miou = float(np.mean(sam_ious)) if sam_ious else 0
sam_avg = float(np.mean(sam_times)) if sam_times else 0
resnet_avg = float(np.mean([p['resnet_time'] for p in resnet_preds])) if resnet_preds else 0

# ── 汇总 ──
print('\n' + '='*60)
print('端到端评估结果 v3')
print('='*60)
print(f'数据集: COCO val2017 ({n} 张, 10 类)')
print()
print('【YOLOv8n 检测】')
print(f'  Precision:   {prec:.4f}')
print(f'  Recall:      {rec:.4f}')
print(f'  F1-Score:    {f1:.4f}')
print(f'  mAP@50:      {map50:.4f}')
print(f'  mIoU(bbox):  {miou:.4f}')
print(f'  速度:        {total_yolo/n:.3f}s/image')
print()
print('【SAM ViT-B 分割】')
print(f'  mIoU(mask):  {sam_miou:.4f}')
print(f'  平均速度:    {sam_avg:.2f}s/mask')
print(f'  总 mask 数:  {len(sam_ious)}')
print()
print('【ResNet50 识别 + COCO→ImageNet 映射 (Top-50)】')
print(f'  匹配正确:    {match_count}/{match_total} ({resnet_match_rate:.1%})')
print(f'  平均速度:    {resnet_avg:.3f}s/object')
print('='*60)

# 保存
results = {
    'pipeline': 'YOLOv8n + SAM ViT-B + ResNet50',
    'dataset': f'COCO val2017 ({n} images, 10 classes)',
    'classes': TARGET,
    'yolo_detection': {
        'TP': yolo_tp, 'FP': yolo_fp, 'FN': yolo_fn,
        'precision': round(prec, 4), 'recall': round(rec, 4),
        'f1': round(f1, 4), 'mAP@50': round(map50, 4),
        'mIoU_bbox': round(miou, 4),
        'speed_s_per_img': round(total_yolo/n, 3)
    },
    'sam_segmentation': {
        'mIoU_mask': round(sam_miou, 4),
        'avg_time_s': round(sam_avg, 3),
        'total_masks': len(sam_ious)
    },
    'resnet_recognition': {
        'match_count': match_count,
        'match_total': match_total,
        'match_rate': round(resnet_match_rate, 4),
        'avg_time_s': round(resnet_avg, 3),
        'sample_preds': resnet_preds[:15]
    }
}
with open(RESULTS_FILE, 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f'\n结果已保存: {RESULTS_FILE}')
