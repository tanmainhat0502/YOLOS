# import os
# import shutil
# import json
# from PIL import Image

# def read_classes(classes_txt):
#     with open(classes_txt) as f:
#         return [line.strip() for line in f.readlines() if line.strip()]

# def yolo2coco(images_dir, labels_dir, output_json, class_names):
#     coco = {"images": [], "annotations": [], "categories": []}
    
#     for i, class_name in enumerate(class_names):
#         coco["categories"].append({
#             "id": i,
#             "name": class_name,
#             "supercategory": "none"
#         })

#     ann_id = 1
#     image_id = 1

#     for img_name in sorted(os.listdir(images_dir)):
#         if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
#             continue

#         img_path = os.path.join(images_dir, img_name)
#         label_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + ".txt")

#         if not os.path.exists(label_path):
#             continue

#         width, height = Image.open(img_path).size

#         coco["images"].append({
#             "file_name": img_name,
#             "height": height,
#             "width": width,
#             "id": image_id
#         })

#         with open(label_path) as f:
#             for line in f:
#                 parts = line.strip().split()
#                 if len(parts) != 5:
#                     continue  # Skip malformed lines
#                 cls_id, x_center, y_center, w, h = map(float, parts)
#                 bbox_width = w * width
#                 bbox_height = h * height
#                 x = (x_center * width) - bbox_width / 2
#                 y = (y_center * height) - bbox_height / 2

#                 coco["annotations"].append({
#                     "id": ann_id,
#                     "image_id": image_id,
#                     "category_id": int(cls_id),
#                     "bbox": [x, y, bbox_width, bbox_height],
#                     "area": bbox_width * bbox_height,
#                     "iscrowd": 0
#                 })
#                 ann_id += 1

#         image_id += 1

#     with open(output_json, "w") as f:
#         json.dump(coco, f, indent=2)

# def prepare_dataset(yolo_root, output_root):
#     class_names = read_classes(os.path.join(yolo_root, "classes.txt"))
#     os.makedirs(os.path.join(output_root, "annotations"), exist_ok=True)

#     for split in ["train", "val", "test"]:
#         img_src = os.path.join(yolo_root, "images", split)
#         label_src = os.path.join(yolo_root, "labels", split)
#         img_dst = os.path.join(output_root, f"{split}2017")
#         os.makedirs(img_dst, exist_ok=True)

#         # Copy all images
#         for fname in os.listdir(img_src):
#             if fname.lower().endswith((".jpg", ".jpeg", ".png")):
#                 shutil.copy(os.path.join(img_src, fname), os.path.join(img_dst, fname))

#         # Convert YOLO → COCO
#         yolo2coco(
#             images_dir=img_dst,
#             labels_dir=label_src,
#             output_json=os.path.join(output_root, "annotations", f"instances_{split}2017.json"),
#             class_names=class_names
#         )

# # ---- Run ----
# prepare_dataset(
#     yolo_root="/home/tiennv/nnthanh/datasets/EmoticGender",
#     output_root="/home/tiennv/nnthanh/datasets/EmoticGender_COCO"
# )


import json
import sys
import os

def add_info_to_coco(json_path):
    if not os.path.exists(json_path):
        print(f"❌ File không tồn tại: {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    if "info" not in data:
        data["info"] = {
            "description": "Converted COCO Dataset",
            "version": "1.0",
            "year": 2025,
            "contributor": "YOLO2COCO Converter",
            "date_created": "2025-06-12"
        }
        print("✅ Đã thêm trường 'info'.")
    else:
        print("ℹ️ Trường 'info' đã tồn tại, không cần thêm.")

    # (Tùy chọn) Thêm 'licenses' nếu bạn cũng muốn đảm bảo có:
    if "licenses" not in data:
        data["licenses"] = [{
            "id": 1,
            "name": "Unknown",
            "url": ""
        }]
        print("✅ Đã thêm trường 'licenses'.")

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
        print(f"✅ Đã lưu file: {json_path}")

# ------- Cách dùng -------
if __name__ == "__main__":
    path = "/mnt/Userdrive/tiennv/nnthanh/datasets/EmoticGender_COCO/annotations/instances_val2017.json"
    add_info_to_coco(path)
