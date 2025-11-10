import os 
import cv2
import subprocess
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Predictor:
    def __init__(self, model_name: str, model_type: str, data_path: str, weights: str = None, model_size: str = "medium", save: bool = False, conf: float = 0.4, resolution: int = 384):
        self.model_name = model_name
        self.model_type = model_type
        self.model_size = model_size
        self.data_path = data_path
        self.weights = weights
        self.save = save
        self.conf = conf
        self.resolution = resolution

        print(f"🚀 Performing inference for model: {self.model_name}")
        print(f"📂 Data path: {self.data_path}")

        if self.model_name == "yolov11" or self.model_name == "yolov8":
            model = self._load_model()
            results = model(data_path, conf=conf, save=save)

            img_bgr = results[0].plot()     
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            plt.imshow(img_rgb)
            plt.axis("off")
            plt.show()

        elif self.model_name == "rfdetr":
            model = self._load_model()
            detections = model.predict(data_path, threshold=conf)
        
            image = cv2.cvtColor(cv2.imread(data_path), cv2.COLOR_BGR2RGB)

            boxes = detections.xyxy
            scores = detections.confidence
            labels = detections.class_id

            plt.figure(figsize=(10, 10))
            plt.imshow(image)

            for (x1, y1, x2, y2), conf in zip(boxes, scores):
                rect = patches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=2,
                    edgecolor='lime',
                    facecolor='none'
                )
                plt.gca().add_patch(rect)
                plt.text(
                    x1,
                    max(y1 - 5, 15),
                    f"{conf:.2f}",
                    color="white",
                    fontsize=10,
                    bbox=dict(facecolor="black", alpha=0.5)
                )

            plt.axis("off")
            plt.show()

            if self.save == True:
                save_dir = "/rfdetr_results"
                os.makedirs(save_dir, exist_ok=True)
                output_path = os.path.join(save_dir, os.path.basename(self.data_path))
                plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
                print(f"💾 Saved annotated image at: {output_path}")
                plt.close()

    def _install(self, package):
        print(f"📦 Installing: {package}")
        subprocess.run([sys.executable, "-m", "pip", "install", *package.split(), "-q"], check=True)

    def _load_model(self):
        if self.model_name == "rfdetr":
            self._install("rfdetr supervision albumentations")

            if self.weights != None:
                if self.model_type == "detection":
                    if self.model_size == "nano":
                        from rfdetr import RFDETRNano
                        model = RFDETRNano(pretrain_weights=self.weights)
                    elif self.model_size == "small":
                        from rfdetr import RFDETRSmall
                        model = RFDETRSmall(pretrain_weights=self.weights)
                    elif self.model_size == "medium":
                        from rfdetr import RFDETRMedium
                        model = RFDETRMedium(pretrain_weights=self.weights)
                    elif self.model_size == "large":
                        from rfdetr import RFDETRLarge
                        model = RFDETRLarge(pretrain_weights=self.weights)
                elif self.model_type == "segmentation":
                    from rfdetr import RFDETRSegPreview
                    if self.resolution == 312:
                        model = RFDETRSegPreview(resolution=312, pretrain_weights=self.weights)
                    elif self.resolution == 384:
                        model = RFDETRSegPreview(resolution=384, pretrain_weights=self.weights)
                    elif self.resolution == 432:
                        model = RFDETRSegPreview(resolution=432, pretrain_weights=self.weights)
            else:
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
                    if self.resolution == 312:
                        model = RFDETRSegPreview(resolution=312)
                    elif self.resolution == 384:
                        model = RFDETRSegPreview(resolution=384)
                    elif self.resolution == 432:
                        model = RFDETRSegPreview(resolution=432)

            return model

        elif self.model_name == "yolov11":
            self._install("ultralytics")
            from ultralytics import YOLO

            if self.weights is None:
                self.weights = "yolo11m.pt"

            model = YOLO(self.weights)
            return model

        elif self.model_name == "yolov8":
            self._install("ultralytics")
            from ultralytics import YOLO

            if self.weights is None:
                self.weights = "yolov8m.pt"

            model = YOLO(self.weights)
            return model