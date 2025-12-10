# Sentinel Colab

A package to streamline the training of custom YOLOv8/v11 and RFDETR models in the Google Colab environment.

## Features

### Dataset Preprocessing
- Convert datasets to YOLO or COCO format
- Merge datasets

### Training
- Train custom YOLOv8/v11 or RFDETR models

### Export
- Export YOLO models to TensorFlow.js (TFJS) format

## Installation

To install the package, run the following command in your Google Colab environment:

```
!pip install git+https://github.com/phsilvarepo/sentinel-colab.git
```

Validate labels:
```
from sentinel_colab.inference import Predictor

Predictor(model_name="rfdetr", data_path="/content/dog.jpg", model_type="detection")
Predictor(model_name="yolov11", data_path="/content/dog.jpg", model_type="detection")
```
Convert labels:
```
from sentinel_colab.preprocessing import Preprocessor

preprocessor = Preprocessor()
preprocessor._convert_coco_to_yolo("/content/UAV-2")
```
Train model:
```
from sentinel_colab import Trainer

trainer = Trainer("yolov11", "/content/dataset_yolo", "detection")
trainer.train(epochs=5)
```
Export model:
```
from sentinel_colab.export import Exporter

exporter = Exporter("yolo", "yolo11m.pt", "asdasd")
```
