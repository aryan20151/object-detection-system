from ultralytics import YOLO
import cv2
import numpy as np
from typing import Dict, List

class SubObjectDetector:
    """Sub-object detector class for detecting hierarchical objects."""
    
    def __init__(self, config: Dict):
        """
        Initialize the sub-object detector.
        
        Args:
            config (Dict): Configuration dictionary containing model parameters
        """
        self.config = config
        self.model = YOLO(config['model']['subobject_detector']['model_path'])
        self.conf_threshold = config['model']['subobject_detector']['confidence_threshold']
        self.iou_threshold = config['model']['subobject_detector']['iou_threshold']
        self.relationships = config['relationships']
        self.subobject_count = {}
        
    def get_valid_subobjects(self, object_name: str) -> List[str]:
        """
        Get list of valid sub-objects for a given object.
        
        Args:
            object_name (str): Name of the main object
            
        Returns:
            List[str]: List of valid sub-object names
        """
        return self.relationships.get(object_name, [])
        
    def detect_subobjects(self, frame: np.ndarray, detection: Dict) -> Dict:
        """
        Detect sub-objects within the region of a detected object.
        
        Args:
            frame (np.ndarray): Input frame
            detection (Dict): Main object detection
            
        Returns:
            Dict: Updated detection with sub-object information
        """
        object_name = detection['object']
        valid_subobjects = self.get_valid_subobjects(object_name)
        
        if not valid_subobjects:
            return detection
            
        # Extract object region
        bbox = detection['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        object_region = frame[y1:y2, x1:x2]
        
        # Detect sub-objects in the region
        results = self.model(object_region, conf=self.conf_threshold, iou=self.iou_threshold)[0]
        
        best_subobject = None
        best_conf = 0
        
        for box in results.boxes:
            class_id = int(box.cls)
            conf = float(box.conf)
            subobj_bbox = box.xyxy[0].cpu().numpy()
            
            subobj_name = results.names[class_id]
            
            if subobj_name in valid_subobjects and conf > best_conf:
                # Adjust bbox coordinates relative to full frame
                adjusted_bbox = [
                    subobj_bbox[0] + x1,
                    subobj_bbox[1] + y1,
                    subobj_bbox[2] + x1,
                    subobj_bbox[3] + y1
                ]
                
                # Generate unique ID for the sub-object
                if subobj_name not in self.subobject_count:
                    self.subobject_count[subobj_name] = 0
                self.subobject_count[subobj_name] += 1
                
                best_subobject = {
                    "object": subobj_name,
                    "id": self.subobject_count[subobj_name],
                    "bbox": adjusted_bbox.tolist(),  # Ensure this is a list
                    "confidence": conf
                }
                best_conf = conf
        
        detection['subobject'] = best_subobject
        return detection
    
    def process_detections(self, frame: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """
        Process all detections to find sub-objects.
        
        Args:
            frame (np.ndarray): Input frame
            detections (List[Dict]): List of main object detections
            
        Returns:
            List[Dict]: Updated detections with sub-object information
        """
        return [self.detect_subobjects(frame, det) for det in detections]