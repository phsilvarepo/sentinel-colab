from ..utils.installer import install_package

class Exporter:
    def __init__(self, model_name: str, model_path: str, output_dir: str):
        self.model_name = model_name

        if self.model_name == "yolov8" or self.model_name == "yolov11"::
            print(f"🚀 Exporting model: {self.model_name}")
            install_package("ultralytics")
            from ultralytics import YOLO
            model = YOLO(model_path)
            model.export(format="tfjs")
        else:
            print("Currently only YOLO models in TFJS format are supported for exporting...")