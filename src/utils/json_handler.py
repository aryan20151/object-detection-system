import json
from typing import Dict, List
import os

class JSONHandler:
    """Utility class for handling JSON output formatting and file operations."""
    
    def __init__(self, config: Dict):
        """
        Initialize JSON handler.
        
        Args:
            config (Dict): Configuration dictionary
        """
        self.config = config
        self.indent = config['output']['json_indent']
        self.results_dir = config['paths']['results_dir']
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
        
    def format_detection(self, detection: Dict) -> Dict:
        """
        Format detection dictionary to match required JSON structure.
        
        Args:
            detection (Dict): Raw detection dictionary
            
        Returns:
            Dict: Formatted detection dictionary
        """
        formatted = {
            "object": detection['object'],
            "id": detection['id'],
            "bbox": detection['bbox']
        }
        
        if detection['subobject']:
            formatted["subobject"] = {
                "object": detection['subobject']['object'],
                "id": detection['subobject']['id'],
                "bbox": detection['subobject']['bbox']
            }
            
        return formatted
        
    def save_detections(self, detections: List[Dict], frame_number: int = None) -> str:
        """
        Save detections to JSON file.
        
        Args:
            detections (List[Dict]): List of detections
            frame_number (int, optional): Frame number for video processing
            
        Returns:
            str: Path to saved JSON file
        """
        formatted_detections = [self.format_detection(det) for det in detections]
        
        # Generate filename
        filename = f"detections_{frame_number}.json" if frame_number is not None else "detections.json"
        filepath = os.path.join(self.results_dir, filename)
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(formatted_detections, f, indent=self.indent)
            
        return filepath
        
    def load_detections(self, filepath: str) -> List[Dict]:
        """
        Load detections from JSON file.
        
        Args:
            filepath (str): Path to JSON file
            
        Returns:
            List[Dict]: List of detections
        """
        with open(filepath, 'r') as f:
            return json.load(f)
            
    def save_video_detections(self, all_detections: List[List[Dict]], video_name: str) -> str:
        """
        Save all detections from video processing.
        
        Args:
            all_detections (List[List[Dict]]): List of detections for each frame
            video_name (str): Name of the processed video
            
        Returns:
            str: Path to saved JSON file
        """
        formatted_detections = {
            f"frame_{i}": [self.format_detection(det) for det in frame_dets]
            for i, frame_dets in enumerate(all_detections)
        }
        
        filepath = os.path.join(self.results_dir, f"{video_name}_detections.json")
        
        with open(filepath, 'w') as f:
            json.dump(formatted_detections, f, indent=self.indent)
            
        return filepath
