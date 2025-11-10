import subprocess
import sys
import torch
from .preprocessing import Preprocessor

class Trainer:
    def __init__(self, model_name: str, dataset_path: str, model_type: str,
                 model_size: str = "medium", resolution: str = "384",
                 output_dir: str = "/content/output"):

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

        self.preprocessor = Preprocessor()
        self.preprocessor._visualize(self.dataset_path, self.model_name)

        self.model = self._load_model()

        self.default_hyperparams = self._get_default_hyperparams()

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

        elif self.model_name == "yolov8":
            self._install("ultralytics")
            from ultralytics import YOLO

            print(f"📘 Loading YOLOv8 ({self.model_type}, {self.model_size}) model...")

            size_map = {
                "nano": "n",
                "small": "s",
                "medium": "m",
                "large": "l",
                "largest": "x",
            }

            suffix = size_map.get(self.model_size, "m")

            if self.model_type == "detection":
                model_path = f"yolov8{suffix}.pt"
            elif self.model_type == "segmentation":
                model_path = f"yolov8{suffix}-seg.pt"

            model = YOLO(model_path)
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

    def _get_default_hyperparams(self):
        """Define default hyperparameters for each model type."""
        if self.model_name == "yolov11" or self.model_name == "yolov8":
            return {
                "epochs": 50,
                "batch_size": 16,
                "imgsz": 640,
                "lr0": 0.01,
                "momentum": 0.937,
                "weight_decay": 0.0005,
                "patience": 20,
            }

        elif self.model_name == "rfdetr":
            return {
                "epochs": 50,
                "batch_size": 8,
                "lr": 1e-4,
                "weight_decay": 1e-4,
                "warmup_steps": 1000,
                "gradient_clip_norm": 0.1,
            }
        else:
            raise ValueError(f"❌ Unsupported model: {self.model_name}")

    def train(self, **kwargs):
        """Train with defaults that can be overridden by user-provided kwargs."""
        train_args = {**self.default_hyperparams, **kwargs}

        print(f"⚙️ Training config: {train_args}")

        if self.model_name == "rfdetr":
            print("🏋️ Training RF-DETR...")
            self.model.train(
                dataset_dir=self.dataset_path,
                output_dir=self.output_dir,
                batch_size=train_args["batch_size"],
                num_epochs=train_args["epochs"],
                learning_rate=train_args["lr"],
                weight_decay=train_args["weight_decay"],
            )

        elif self.model_name == "yolov11" or self.model_name == "yolov8":
            print("🏋️ Training YOLOv11/YOLOv8...")
            from os.path import join
            yaml_path = join(self.dataset_path, "data.yaml")

            results = self.model.train(
                data=yaml_path,
                imgsz=train_args["imgsz"],
                epochs=train_args["epochs"],
                batch=train_args["batch_size"],
                lr0=train_args["lr0"],
                momentum=train_args["momentum"],
                weight_decay=train_args["weight_decay"],
                patience=train_args["patience"],
                project=self.output_dir
            )

            print("✅ Training complete.")
            return results

    '''
    def resume_training(self, checkpoint_path, **kwargs):
    print("⚙️ Resuming training")

    train_args = {**self.default_hyperparams, **kwargs}

    if self.model_name == "rfdetr":
        print("🏋️ Resuming training of RF-DETR...")
        self.model.train(
            dataset_dir=self.dataset_path,
            output_dir=self.output_dir,
            batch_size=train_args["batch_size"],
            num_epochs=train_args["epochs"],
            learning_rate=train_args["lr"],
            weight_decay=train_args["weight_decay"],
            resume_from=checkpoint_path
        )

    elif self.model_name == "yolov11":
        print("🏋️ Resuming training of YOLOv11...")

        model = YOLO(checkpoint_path)
        yaml_path = f"{self.dataset_path}/data.yaml"

        model.train(
            data=yaml_path,
            epochs=train_args["epochs"],
            resume=True,
            project=self.output_dir
        )
    '''