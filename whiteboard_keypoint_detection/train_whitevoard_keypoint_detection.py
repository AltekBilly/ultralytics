import os
import cv2
import numpy

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

import ultralytics
from ultralytics import YOLO
from ultralytics.yolo.engine.results import Boxes
from ultralytics.yolo.engine.results import Keypoints


def main(model: str, data: str, epochs: int = 600, imgsz = 640):
    # Load a model
    model = YOLO(model)  # build a new model from YAML

    # Train the model
    model.train(data=data, epochs=epochs, imgsz=imgsz)

if __name__ == '__main__':
    model = 'yolov8-whiteboard-keypoints.yaml'
    data = 'whiteboard-keypoints.yaml'
    main(model, data, 600, [640, 640])