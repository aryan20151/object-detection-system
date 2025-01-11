import cv2
import numpy as np
from typing import Dict, List, Tuple

class Visualizer:
    """Utility class for visualization of detections."""
    
    def __init__(self, config: Dict):
        """
        Initialize visualizer.
        
        Args:
            config (Dict): Configuration dictionary
        """
        self.config = config
        self.box_thickness = config['output']['visualization']['box_thickness']
        self.text_size = config['output']['visualization']['text_size']
        self.font_thickness = config['output']['visualization']['font_thickness']
        
        # Color mapping for different objects
        self.colors = {
            'person': (0, 255, 0),    # Green
            'car': (255, 0, 0),       # Blue
            'bicycle': (0, 0, 255),   # Red
            'helmet': (255, 255, 0),  # Cyan
            'tire': (255, 0, 255),    # Magenta
            'license_plate': (0, 255, 255),  # Yellow
        }
        
    def get_color(self, object_name: str) -> Tuple[int, int, int]:
        """
        Get color for object visualization.
        
        Args:
            object_name (str): Name of the object
            
        Returns:
            Tuple[int, int, int]: BGR color tuple
        """
        if object_name in self.colors:
            return self.colors[object_name]
        
        # Generate random color for unknown objects
        return tuple(np.random.randint(0, 255, 3).tolist())
        
    def draw_detection(self, frame: np.ndarray, detection: Dict) -> np.ndarray:
        """
        Draw single detection on frame.
        
        Args:
            frame (np.ndarray): Input frame
            detection (Dict): Detection dictionary
            
        Returns:
            np.ndarray: Frame with drawn detection
        """
        # Draw main object
        x1, y1, x2, y2 = map(int, detection['bbox'])
        color = self.get_color(detection['object'])
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.box_thickness)
        label = f"{detection['object']} {detection['id']}"
        
        # Add label
        (label_w, label_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, self.text_size, self.font_thickness
        )
        cv2.rectangle(frame, (x1, y1-label_h-10), (x1+label_w, y1), color, -1)
        cv2.putText(
            frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX,
            self.text_size, (255, 255, 255), self.font_thickness
        )
        
        # Draw sub-object if present
        if detection['subobject']:
            sx1, sy1, sx2, sy2 = map(int, detection['subobject']['bbox'])
            sub_color = self.get_color(detection['subobject']['object'])
            
            cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), sub_color, self.box_thickness)
            sub_label = f"{detection['subobject']['object']} {detection['subobject']['id']}"
            
            # Add sub-object label
            (sub_label_w, sub_label_h), _ = cv2.getTextSize(
                sub_label, cv2.FONT_HERSHEY_SIMPLEX, self.text_size, self.font_thickness
            )
            cv2.rectangle(
                frame, (sx1, sy1-sub_label_h-10),
                (sx1+sub_label_w, sy1), sub_color, -1
            )
            cv2.putText(
                frame, sub_label, (sx1, sy1-5), cv2.FONT_HERSHEY_SIMPLEX,
                self.text_size, (255, 255, 255), self.font_thickness
            )
            
            # Draw connection line between object and sub-object
            cv2.line(
                frame, (x1, y1), (sx1, sy1),
                sub_color, max(1, self.box_thickness-1)
            )
            
        return frame
        
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw all detections on frame.
        
        Args:
            frame (np.ndarray): Input frame
            detections (List[Dict]): List of detections
            
        Returns:
            np.ndarray: Frame with all detections drawn
        """
        for detection in detections:
            frame = self.draw_detection(frame, detection)
        return frame
        
    def save_visualization(self, frame: np.ndarray, filepath: str) -> None:
        """
        Save visualization to file.
        
        Args:
            frame (np.ndarray): Frame with visualizations
            filepath (str): Output file path
        """
        cv2.imwrite(filepath, frame)
