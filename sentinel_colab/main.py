import subprocess
import sys
import torch

class Trainer:
    def __init__(self, model_name: str, dataset_path: str, type: str, model_size: str, resolution: str, output_dir: str = "/content/output"):
        self.model_name = model_name.lower()
        self.dataset_path = dataset_path
        self.type = model_type
        self.model_size = model_size
        self.resolution = resolution
        self.output_dir = output_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"🚀 Initializing trainer for model: {self.model_name}")
        print(f"📂 Dataset path: {self.dataset_path}")
        print(f"💾 Output directory: {self.output_dir}")
        print(f"🖥️ Using device: {self.device}")

        self.model = self._load_model()

    def _install(self, package):
        print(f"📦 Installing: {package}")
        subprocess.run([sys.executable, "-m", "pip", "install", *package.split(), "-q"], check=True)

    def _load_model(self):
        if self.model_name == "rfdetr":
            self._install("rfdetr supervision albumentations")
            if model_type == "detection":
                if model_size == "nano":
                    from rfdetr import RFDETRNano
                    model = RFDETRNano()
                elif model_size == "small":
                    from rfdetr import RFDETRSmall
                    model = RFDETRSmall()
                elif model_size == "medium":
                    from rfdetr import RFDETRMedium
                    model = RFDETRMedium()
                elif model_size == "large":
                    from rfdetr import RFDETRLarge
                    model = RFDETRLarge()
            elif model_type == "segmentation":
                from rfdetr import RFDETRSegPreview
                if resolution == "312":
                    model = RFDETRSegPreview(resolution=312)
                elif resolution == "384":
                    model = RFDETRSegPreview(resolution=384)
                elif resolution == "432":
                    model = RFDETRSegPreview(resolution=432)

            return model

        elif self.model_name == "yolov11":
            self._install("ultralytics")
            from ultralytics import YOLO

            print(f"📘 Loading YOLOv11 ({self.task_type}, {self.model_size}) model...")

            size_map = {
                "nano": "n",
                "small": "s",
                "medium": "m",
                "large": "l",
                "largest": "x",
            }

            suffix = size_map.get(self.model_size, "m")

            if self.task_type == "detection":
                model_path = f"yolo11{suffix}.pt"
            elif self.task_type == "segmentation":
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
