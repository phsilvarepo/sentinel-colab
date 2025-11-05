import subprocess
import sys
import torch

class Trainer:
    def __init__(self, model_name: str, dataset_path: str, output_dir: str = "/content/output"):
        self.model_name = model_name.lower()
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"🚀 Initializing trainer for model: {self.model_name}")
        print(f"📂 Dataset path: {self.dataset_path}")
        print(f"💾 Output directory: {self.output_dir}")
        print(f"🖥️ Using device: {self.device}")

        self.model = self._load_model()

    def _install(self, package):
        print("Entrou")
        subprocess.run([sys.executable, "-m", "pip", "install", *package.split(), "-q"], check=True)

    def _load_model(self):
        if self.model_name == "rfdetr":
            self._install("rfdetr supervision albumentations")
            from rfdetr import RFDETRMedium
            model = RFDETRMedium()
            return model

        elif self.model_name == "yolov11":
            self._install("ultralytics")
            from ultralytics import YOLO
            model = YOLO("yolo11m-seg.pt")
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
