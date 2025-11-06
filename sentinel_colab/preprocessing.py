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
    
    def _visualize(self, dataset_path, model):
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

                    # Draw masks (if available)
                    if show_masks and "segmentation" in ann:
                        seg = ann["segmentation"]
                        if isinstance(seg, list):  # polygon
                            for poly in seg:
                                poly = np.array(poly).reshape((-1, 2)).astype(np.int32)
                                cv2.polylines(img, [poly], True, (255, 0, 0), 2)
                        elif isinstance(seg, dict):  # RLE mask
                            mask = maskUtils.decode(seg)
                            img[mask == 1] = img[mask == 1] * 0.5 + np.array([255, 0, 0]) * 0.5

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


    def _convert_coco_to_yolo(self, dataset_path, output_dir="dataset_yolo"):
        """Converts COCO-format dataset to YOLO-format locally."""
        self._install("pylabel")
        from pylabel import importer

        print("⚙️ Converting COCO to YOLO locally...")

        coco_dataset = importer.ImportCoco(path=dataset_path)
        coco_dataset.export.ExportToYoloV5(output_path=output_dir)

        print(f"✅ YOLO dataset saved to {output_dir}")

    