## Sentinel Colab
Package to streamline the training of custom YOLOv8/v11 and RFDETR models in the Google Colab env.

# Features

Dataset Preprocessing:
Convert datasets to YOLO or COCO format
Merge datasets

Training:
Train models

Export:
Export YOLO models to TFJS format

# Instructions

!pip install git+https://github.com/phsilvarepo/sentinel-colab.git

from sentinel_colab.inference import Predictor

Predictor(model_name="rfdetr", data_path="/content/dog.jpg", model_type="detection")
Predictor(model_name="yolov11", data_path="/content/dog.jpg", model_type="detection")

from sentinel_colab.preprocessing import Preprocessor

preprocessor = Preprocessor()
preprocessor._convert_coco_to_yolo("/content/UAV-2")

from sentinel_colab import Trainer

trainer = Trainer("yolov11", "/content/dataset_yolo", "detection")
trainer.train(epochs=5)

from sentinel_colab.export import Exporter

exporter = Exporter("yolo", "yolo11m.pt", "asdasd")
