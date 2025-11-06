import os 
import cv2
import subprocess
import sys
import matplotlib.pyplot as plt
import random
import matplotlib.patches as patches
import numpy as np

class Preprocessor:
    def __init__(self):
        """Handles dataset visualization and conversion between YOLO and COCO formats."""
        print("🧩 Preprocessor initialized.")

    def _install(self, package):
        print(f"📦 Installing: {package}")
        subprocess.run([sys.executable, "-m", "pip", "install", *package.split(), "-q"], check=True)   
    
    def _visualize(self, dataset_path, model, num_images = 5):
        from pycocotools.coco import COCO
        import pycocotools.mask as maskUtils

        if model == "yolov11":
            images_dir = os.path.join(dataset_path, "train", "images")
            labels_dir = os.path.join(dataset_path, "train", "labels")

            if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
                print("❌ Could not find dataset structure like /images/train and /labels/train.")
                return

            image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            if len(image_files) == 0:
                print("❌ No images found in the dataset folder.")
                return

            sample_images = random.sample(image_files, min(num_images, len(image_files)))

            for img_name in sample_images:
                img_path = os.path.join(images_dir, img_name)
                label_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + ".txt")

                img = cv2.imread(img_path)
                if img is None:
                    print(f"⚠️ Could not read image {img_name}")
                    continue

                h, w = img.shape[:2]

                if os.path.exists(label_path):
                    with open(label_path, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) < 5:
                                continue
                            try:
                                cls, x, y, bw, bh = map(float, parts[:5])
                            except ValueError:
                                print(f"⚠️ Invalid label format in {label_path}")
                                continue

                            # Convert YOLO normalized coords to pixel coords
                            x1 = int((x - bw / 2) * w)
                            y1 = int((y - bh / 2) * h)
                            x2 = int((x + bw / 2) * w)
                            y2 = int((y + bh / 2) * h)

                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(img, f"{int(cls)}", (x1, max(y1 - 5, 15)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    print(f"⚠️ No label file found for {img_name}")

                # Convert BGR to RGB for matplotlib
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.figure(figsize=(6, 6))
                plt.imshow(img_rgb)
                plt.axis("off")
                plt.title(f"📷 {img_name}")
                plt.show()

        elif model == "rfdetr":
            coco_json_path = os.path.join(dataset_path, "train", "_annotations.coco.json")
            images_dir = os.path.join(dataset_path, "train")

            if not os.path.exists(coco_json_path):
                print("❌ Could not find COCO annotation file at:", coco_json_path)
                return

            coco = COCO(coco_json_path)
            img_ids = coco.getImgIds()
            sample_ids = random.sample(img_ids, min(num_images, len(img_ids)))

            for img_id in sample_ids:
                img_info = coco.loadImgs(img_id)[0]
                img_path = os.path.join(images_dir, img_info["file_name"])

                if not os.path.exists(img_path):
                    print(f"⚠️ Missing image: {img_path}")
                    continue

                img = cv2.imread(img_path)
                if img is None:
                    print(f"⚠️ Could not load image {img_info['file_name']}")
                    continue

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Load annotations
                ann_ids = coco.getAnnIds(imgIds=img_id)
                anns = coco.loadAnns(ann_ids)

                for ann in anns:
                    # Draw bounding boxes
                    if "bbox" in ann:
                        x, y, w, h = map(int, ann["bbox"])
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cat_name = coco.loadCats(ann["category_id"])[0]["name"]
                        cv2.putText(img, cat_name, (x, max(y - 5, 15)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                plt.figure(figsize=(7, 7))
                plt.imshow(img)
                plt.axis("off")
                plt.title(f"📸 {img_info['file_name']}")
                plt.show()  
        else:
            print("Model not supported, currently only the YOLO annotation format(yolo) and the COCO annotation format(rfdetr) are available")

    def _convert_yolo_to_coco(self, dataset_path, output_json="dataset_coco.json"):
        """Converts YOLO-format dataset to COCO-format JSON locally."""
        self._install("pylabel")
        from pylabel import importer

        print("⚙️ Converting YOLO to COCO locally...")

        yolo_dataset = importer.ImportYoloV5(path=dataset_path, path_to_annotations="labels", path_to_images="images")
        yolo_dataset.export.ExportToCoco(output_path=output_json)

        print(f"✅ COCO JSON saved to {output_json}")


    def _convert_coco_to_yolo(self, dataset_path, output_dir="dataset_yolo", move=True):
        """Converts COCO-format dataset to YOLO-format locally and creates data.yaml."""
        import shutil
        import json
        from os.path import join

        # Install pylabel if needed
        self._install("pylabel")
        from pylabel import importer

        splits = ["train", "test", "valid"]
        all_classes = {}

        for split in splits:
            coco_json = join(dataset_path, split, "_annotations.coco.json")
            if not os.path.exists(coco_json):
                print(f"⚠️ Skipping {split}: No annotation file found at {coco_json}")
                continue

            # Import COCO dataset
            coco_dataset = importer.ImportCoco(path=coco_json)

            # Export to YOLO format
            export_path = join(output_dir, "labels", split)
            os.makedirs(export_path, exist_ok=True)
            coco_dataset.export.ExportToYoloV5(output_path=export_path)
            print(f"✅ Exported labels to: {export_path}")

            # Copy or move images
            src_img_dir = join(dataset_path, split)
            dst_img_dir = join(output_dir, "images", split)
            os.makedirs(dst_img_dir, exist_ok=True)

            for file in os.listdir(src_img_dir):
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    src_file = join(src_img_dir, file)
                    dst_file = join(dst_img_dir, file)
                    if move:
                        shutil.move(src_file, dst_file)
                    else:
                        shutil.copy2(src_file, dst_file)

            print(f"📸 {'Moved' if move else 'Copied'} images for {split} split.\n")

            # Collect category names from COCO JSON
            with open(coco_json, "r") as f:
                data = json.load(f)
                for cat in data.get("categories", []):
                    all_classes[cat["id"]] = cat["name"]

        # 🔹 Generate data.yaml
        yaml_path = join(output_dir, "data.yaml")
        with open(yaml_path, "w") as f:
            f.write(f"path: {os.path.abspath(output_dir)}\n")
            f.write("train: images/train\n")
            f.write("val: images/valid\n")
            f.write("test: images/test\n\n")
            f.write("names:\n")
            for cid, cname in sorted(all_classes.items()):
                f.write(f"  {cid - min(all_classes.keys())}: {cname}\n")  # normalize IDs to start from 0

        print(f"🧾 data.yaml created at: {yaml_path}")
        print(f"✅ YOLO dataset ready at: {output_dir}")



            