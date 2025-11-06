import os 
import cv2
import subprocess
import sys
import matplotlib.pyplot as plt
import random
import matplotlib.patches as patches
import numpy as np
import shutil
from os.path import join
import json

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
            images_dir = os.path.join(dataset_path, "images", "train")
            labels_dir = os.path.join(dataset_path, "labels", "train")

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
                plt.show()  
        else:
            print("Model not supported, currently only the YOLO annotation format(yolo) and the COCO annotation format(rfdetr) are available")

    def _convert_yolo_to_coco(self, dataset_path, output_dir="dataset_coco", move=True, test_split=0.5):
        """Converts YOLO-format dataset to COCO-format locally.
        If YOLO dataset has no test split, uses 50% of val as COCO test.
        """
        self._install("pylabel")
        from pylabel import importer

        def export_split(yolo_split, coco_split, labels_dir, images_dir):
            """Helper to import YOLO and export COCO for one split."""
            print(f"⚙️ Processing split: {yolo_split} → {coco_split}")

            yolo_dataset = importer.ImportYoloV5(labels_dir)
            split_out_dir = join(output_dir, coco_split)
            os.makedirs(split_out_dir, exist_ok=True)

            output_json = join(split_out_dir, "_annotations.coco.json")
            yolo_dataset.export.ExportToCoco(output_path=output_json)
            print(f"✅ Exported COCO JSON to: {output_json}")

            for file in os.listdir(images_dir):
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    src_file = join(images_dir, file)
                    dst_file = join(split_out_dir, file)
                    if move:
                        shutil.move(src_file, dst_file)
                    else:
                        shutil.copy2(src_file, dst_file)
            print(f"📸 {'Moved' if move else 'Copied'} images for {coco_split}.\n")

        print("⚙️ Converting YOLO to COCO...")

        yolo_splits = ["train", "val", "test"]
        available_splits = [s for s in yolo_splits if os.path.exists(join(dataset_path, "labels", s))]

        has_test = "test" in available_splits
        print(f"📊 Found splits: {available_splits}")
        print(f"🧠 Test split detected: {has_test}")

        # --- Train always exists ---
        if "train" in available_splits:
            export_split(
                "train",
                "train",
                join(dataset_path, "labels/train"),
                join(dataset_path, "images/train"),
            )

        # --- Handle validation and test ---
        if has_test:

            if "val" in available_splits:
                export_split(
                    "val",
                    "valid",
                    join(dataset_path, "labels/val"),
                    join(dataset_path, "images/val"),
                )

            export_split(
                "test",
                "test",
                join(dataset_path, "labels/test"),
                join(dataset_path, "images/test"),
            )
        else:
            # Split val accordinf to the ratio into valid/test
            val_labels = join(dataset_path, "labels/val")
            val_images = join(dataset_path, "images/val")
            if not os.path.exists(val_labels) or not os.path.exists(val_images):
                print("❌ No val split found. Cannot create COCO valid/test splits.")
                return

            print("⚠️ No YOLO test split found — splitting val into valid/test according to the ratio: ", test_ratio)

            # Get all images and shuffle
            val_files = [f for f in os.listdir(val_images) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            random.shuffle(val_files)
            test_count = int(len(val_files) * test_split)
            test_files = val_files[:test_count]
            valid_files = val_files[test_count:]

            # Create temp split directories
            temp_base = join(dataset_path, "_temp_split")
            os.makedirs(temp_base, exist_ok=True)
            valid_img_dir = join(temp_base, "images/valid")
            test_img_dir = join(temp_base, "images/test")
            valid_lbl_dir = join(temp_base, "labels/valid")
            test_lbl_dir = join(temp_base, "labels/test")
            os.makedirs(valid_img_dir, exist_ok=True)
            os.makedirs(test_img_dir, exist_ok=True)
            os.makedirs(valid_lbl_dir, exist_ok=True)
            os.makedirs(test_lbl_dir, exist_ok=True)

            # Copy images and labels
            for f in valid_files:
                lbl = os.path.splitext(f)[0] + ".txt"
                if os.path.exists(join(val_images, f)):
                    shutil.copy2(join(val_images, f), join(valid_img_dir, f))
                if os.path.exists(join(val_labels, lbl)):
                    shutil.copy2(join(val_labels, lbl), join(valid_lbl_dir, lbl))

            for f in test_files:
                lbl = os.path.splitext(f)[0] + ".txt"
                if os.path.exists(join(val_images, f)):
                    shutil.copy2(join(val_images, f), join(test_img_dir, f))
                if os.path.exists(join(val_labels, lbl)):
                    shutil.copy2(join(val_labels, lbl), join(test_lbl_dir, lbl))

            # Export both splits
            export_split("valid", "valid", valid_lbl_dir, valid_img_dir)
            export_split("test", "test", test_lbl_dir, test_img_dir)

            # Cleanup temp dirs
            shutil.rmtree(temp_base, ignore_errors=True)

        print(f"🎯 COCO dataset ready at: {output_dir}")


    def _convert_coco_to_yolo(self, dataset_path, output_dir="dataset_yolo", move=True):
        """Converts COCO-format dataset to YOLO-format locally and creates data.yaml with 'val' instead of 'valid'."""
        self._install("pylabel")
        from pylabel import importer

        # COCO uses "valid", YOLO will use "val"
        coco_splits = ["train", "test", "valid"]
        yolo_splits = {"train": "train", "test": "test", "valid": "val"}
        all_classes = {}

        for split in coco_splits:
            coco_json = join(dataset_path, split, "_annotations.coco.json")
            if not os.path.exists(coco_json):
                print(f"⚠️ Skipping {split}: No annotation file found at {coco_json}")
                continue

            # Import COCO dataset
            coco_dataset = importer.ImportCoco(path=coco_json)

            # Export to YOLO format — use remapped split name
            yolo_split = yolo_splits[split]
            export_path = join(output_dir, "labels", yolo_split)
            os.makedirs(export_path, exist_ok=True)
            coco_dataset.export.ExportToYoloV5(output_path=export_path)
            print(f"✅ Exported labels to: {export_path}")

            # Copy or move images
            src_img_dir = join(dataset_path, split)
            dst_img_dir = join(output_dir, "images", yolo_split)
            os.makedirs(dst_img_dir, exist_ok=True)

            for file in os.listdir(src_img_dir):
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    src_file = join(src_img_dir, file)
                    dst_file = join(dst_img_dir, file)
                    if move:
                        shutil.move(src_file, dst_file)
                    else:
                        shutil.copy2(src_file, dst_file)

            print(f"📸 {'Moved' if move else 'Copied'} images for {yolo_split} split.\n")

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
            f.write("val: images/val\n")
            f.write("test: images/test\n\n")
            f.write("names:\n")
            for cid, cname in sorted(all_classes.items()):
                f.write(f"  {cid - min(all_classes.keys())}: {cname}\n")  # normalize IDs to start from 0

        print(f"🧾 data.yaml created at: {yaml_path}")
        print(f"✅ YOLO dataset ready at: {output_dir}")
