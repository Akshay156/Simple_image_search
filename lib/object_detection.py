import cv2
import numpy as np
from lib.utils import xywh2xyxy, nms, draw_detections
from pathlib import Path
from argparse import ArgumentParser
from urllib.request import urlopen
from ultralytics import YOLO

class YOLOv8:
    """
    YOLOv8 object detector.

    Args:
        model_path (str): Path to the YOLOv8 model file.
        conf_thres (float, optional): Confidence threshold for filtering detections. Defaults to 0.7.
        iou_thres (float, optional): IoU threshold for non-maximum suppression. Defaults to 0.5.

    Attributes:
        conf_threshold (float): Confidence threshold for filtering detections.
        iou_threshold (float): IoU threshold for non-maximum suppression.
        model (YOLO): YOLOv8 model object.

    Methods:
        __call__(self, image): Perform object detection on the input image.
        detect_objects(self, image): Detect objects in the input image.
        process_output(self, predictions): Process YOLOv8 model predictions.
    """

    def __init__(self, model_path, conf_thres=0.7, iou_thres=0.5):
        """
        Initialize the YOLOv8 object detector.

        Args:
            model_path (str): Path to the YOLOv8 model file.
            conf_thres (float, optional): Confidence threshold for filtering detections. Defaults to 0.7.
            iou_thres (float, optional): IoU threshold for non-maximum suppression. Defaults to 0.5.
        """
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres

        # Initialize model
        self.model = YOLO(model_path)

    def __call__(self, image):
        """
        Perform object detection on the input image.

        Args:
            image (ndarray): Input image for object detection.

        Returns:
            tuple: Detected boxes, scores, and class IDs.
        """
        return self.detect_objects(image)

    def detect_objects(self, image):
        """
        Detect objects in the input image.

        Args:
            image (ndarray): Input image for object detection.

        Returns:
            tuple: Detected boxes, scores, and class IDs.
        """
        # Predict using YOLOv8
        predictions = self.model(image)

        self.boxes, self.scores, self.class_ids = self.process_output(predictions)

        return self.boxes, self.scores, self.class_ids

    def process_output(self, predictions):
        """
        Process YOLOv8 model predictions.

        Args:
            predictions (list): List containing YOLOv8 model predictions.

        Returns:
            tuple: Processed boxes, scores, and class IDs.
        """
        # Extract predictions from the first element of the list
        boxes = predictions[0].boxes.xyxy.numpy()
        scores = predictions[0].boxes.conf.numpy()
        class_ids = predictions[0].boxes.cls.numpy()

        # Filter out low confidence predictions
        mask = scores > self.conf_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        # Apply non-maximum suppression
        indices = nms(boxes, scores, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices]  

if __name__ == '__main__':
    # Define the YOLOv5 model path
    MODEL_PATH = 'models/best.pt'

    # Initialize YOLOv8 object detectors
    yolov8_detector = YOLOv8(MODEL_PATH, conf_thres=0.3, iou_thres=0.5)

    img = cv2.imread("/mnt/Data/PD/InfiViz/Problem_1/Dataset/product_detection_from_packshots/shelf_images/db15.jpg")

    # Detect objects
    yolov8_detector(img)

    # Draw detections
    combined_img = draw_detections(img, yolov8_detector.boxes, yolov8_detector.scores, yolov8_detector.class_ids)
    cv2.imwrite("output.jpg", combined_img)
