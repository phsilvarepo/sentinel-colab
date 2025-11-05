import subprocess
import sys
import torch

class Trainer:
    def __init__(self, model_name: str, dataset_path: str, model_type: str, model_size: str = "medium", resolution: str = "384", output_dir: str = "/content/output"):
        self.model_name = model_name.lower()
        self.dataset_path = dataset_path
        self.model_type = model_type
        self.model_size = model_size
        self.resolution = resolution
        self.output_dir = output_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"🚀 Initializing trainer for model: {self.model_name}")
        print(f"📂 Dataset path: {self.dataset_path}")
        print(f"💾 Output directory: {self.output_dir}")
        print(f"🖥️ Using device: {self.device}")

        self._validate_labels()
        self.model = self._load_model()

    def _validate_labels(self, num_images=5, show_masks=True):
        import os
        import random
        import cv2
        import numpy as np
        import matplotlib.pyplot as plt
        from pycocotools.coco import COCO
        import pycocotools.mask as maskUtils

        if self.model_name == "yolov11":
            images_dir = os.path.join(self.dataset_path, "train", "images")
            labels_dir = os.path.join(self.dataset_path, "train", "labels")
            print(images_dir)

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

        elif self.model_name == "rfdetr":
            coco_json_path = os.path.join(self.dataset_path, "train", "_annotations.coco.json")
            images_dir = os.path.join(self.dataset_path, "train")

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

    def _install(self, package):
        print(f"📦 Installing: {package}")
        subprocess.run([sys.executable, "-m", "pip", "install", *package.split(), "-q"], check=True)

    def _load_model(self):
        if self.model_name == "rfdetr":
            self._install("rfdetr supervision albumentations")
            if self.model_type == "detection":
                if self.model_size == "nano":
                    from rfdetr import RFDETRNano
                    model = RFDETRNano()
                elif self.model_size == "small":
                    from rfdetr import RFDETRSmall
                    model = RFDETRSmall()
                elif self.model_size == "medium":
                    from rfdetr import RFDETRMedium
                    model = RFDETRMedium()
                elif self.model_size == "large":
                    from rfdetr import RFDETRLarge
                    model = RFDETRLarge()
            elif self.model_type == "segmentation":
                from rfdetr import RFDETRSegPreview
                if self.resolution == "312":
                    model = RFDETRSegPreview(resolution=312)
                elif self.resolution == "384":
                    model = RFDETRSegPreview(resolution=384)
                elif self.resolution == "432":
                    model = RFDETRSegPreview(resolution=432)

            return model

        elif self.model_name == "yolov11":
            self._install("ultralytics")
            from ultralytics import YOLO

            print(f"📘 Loading YOLOv11 ({self.model_type}, {self.model_size}) model...")

            size_map = {
                "nano": "n",
                "small": "s",
                "medium": "m",
                "large": "l",
                "largest": "x",
            }

            suffix = size_map.get(self.model_size, "m")

            if self.model_type == "detection":
                model_path = f"yolo11{suffix}.pt"
            elif self.model_type == "segmentation":
                model_path = f"yolo11{suffix}-seg.pt"

            model = YOLO(model_path)
            return model

        else:
            raise ValueError(f"❌ Unsupported model: {self.model_name}")

    def train(self, **kwargs):
        if self.model_name == "rfdetr":
            print("🏋️ Training RF-DETR...")
            self.model.train(
                dataset_dir=self.dataset_path,
                output_dir=self.output_dir,
                **kwargs
            )

        elif self.model_name == "yolov11":
            print("🏋️ Training YOLOv11...")
            results = self.model.train(data=self.dataset_path, **kwargs)
            print("✅ Training complete.")
            return results
