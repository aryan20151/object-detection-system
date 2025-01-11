from ultralytics import YOLO
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import time

class ObjectDetector:
    """Main object detector class using YOLOv8."""
    
    def __init__(self, config: Dict):
        """
        Initialize the object detector.
        
        Args:
            config (Dict): Configuration dictionary containing model parameters
        """
        self.config = config
        self.model = YOLO(config['model']['object_detector']['model_path'])
        self.conf_threshold = config['model']['object_detector']['confidence_threshold']
        self.iou_threshold = config['model']['object_detector']['iou_threshold']
        self.device = config['processing']['device']
        self.object_count = {}
        
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect objects in a frame.
        
        Args:
            frame (np.ndarray): Input frame
            
        Returns:
            List[Dict]: List of detections in the specified JSON format
        """
        results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold)[0]
        detections = []
        
        for box in results.boxes:
            class_id = int(box.cls)
            conf = float(box.conf)
            bbox = box.xyxy[0].cpu().numpy()
            
            obj_name = results.names[class_id]
            
            # Generate unique ID for the object
            if obj_name not in self.object_count:
                self.object_count[obj_name] = 0
            self.object_count[obj_name] += 1
            
            detection = {
                "object": obj_name,
                "id": self.object_count[obj_name],
                "bbox": bbox.tolist(),
                "confidence": conf,
                "subobject": None  # Will be filled by SubObjectDetector
            }
            
            detections.append(detection)
            
        return detections
    
    def process_video(self, video_path: str, callback=None) -> List[Dict]:
        """
        Process video file and detect objects.
        
        Args:
            video_path (str): Path to input video
            callback (callable, optional): Callback function for real-time visualization
            
        Returns:
            List[Dict]: List of detections for each frame
        """
        cap = cv2.VideoCapture(video_path)
        fps = []
        all_detections = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            start_time = time.time()
            
            # Perform detection
            detections = self.detect(frame)
            all_detections.append(detections)
            
            # Calculate FPS
            fps.append(1 / (time.time() - start_time))
            
            if callback:
                callback(frame, detections)
                
        cap.release()
        
        avg_fps = sum(fps) / len(fps)
        print(f"Average FPS: {avg_fps:.2f}")
        
        return all_detections
    
    def extract_subobject(self, frame: np.ndarray, detection: Dict, subobject_name: str) -> Optional[np.ndarray]:
        """
        Extract subobject image from frame based on detection.
        
        Args:
            frame (np.ndarray): Input frame
            detection (Dict): Detection dictionary
            subobject_name (str): Name of subobject to extract
            
        Returns:
            Optional[np.ndarray]: Cropped image of subobject if found
        """
        if detection['subobject'] is None or detection['subobject']['object'] != subobject_name:
            return None
            
        bbox = detection['subobject']['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        return frame[y1:y2, x1:x2]
