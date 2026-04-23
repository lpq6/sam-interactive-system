import json, os, urllib.request, ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

with open('annotations/instances_val2017.json') as f:
    data = json.load(f)

# Get image IDs that have annotations from diverse categories
cat_counts = {}
selected_images = set()
target_per_cat = 10  # ~10 images per category → ~100 total

for ann in data['annotations']:
    img_id = ann['image_id']
    if img_id in selected_images:
        continue
    cats = [c['name'] for c in data['categories'] if c['id'] == ann['category_id']]
    cat_name = cats[0] if cats else 'unknown'
    if cat_counts.get(cat_name, 0) < target_per_cat:
        selected_images.add(img_id)
        cat_counts[cat_name] = cat_counts.get(cat_name, 0) + 1
    if len(selected_images) >= 100:
        break

img_map = {img['id']: img['file_name'] for img in data['images']}
download_ids = list(selected_images)[:100]

print(f'Selected {len(download_ids)} images from {len(cat_counts)} categories')
print(f'Categories: {sorted(cat_counts.keys())}')

# Download
os.makedirs('images', exist_ok=True)
for i, img_id in enumerate(download_ids):
    fname = img_map[img_id]
    url = f'https://images.cocodataset.org/val2017/{fname}'
    out_path = f'images/{fname}'
    if os.path.exists(out_path):
        continue
    try:
        urllib.request.urlretrieve(url, out_path)
        if (i+1) % 20 == 0:
            print(f'  Downloaded {i+1}/{len(download_ids)}')
    except Exception as e:
        print(f'  Failed {fname}: {e}')

print(f'Done. Total images: {len(os.listdir("images"))}')
