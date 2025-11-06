import os 
import cv2
import subprocess
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Exporter:
    def __init__(self, model_name: str, model_path: str, output_dir: str):
        self.model_name = model_name

        if self.model_name == "yolo":
            print(f"🚀 Exporting model: {self.model_name}")
            self._install("ultralytics")
            import YOLO
            model = YOLO(model_path)
            model.export(format="tfjs")
        else:
            print("Currently only YOLO models in TFJS format are supported for exporting...")

        def _install(self, package):
        print(f"📦 Installing: {package}")
        subprocess.run([sys.executable, "-m", "pip", "install", *package.split(), "-q"], check=True) 