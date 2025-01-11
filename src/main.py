import argparse
import yaml
import cv2
import os
import time
from detector.object_detector import ObjectDetector
from detector.subobject_detector import SubObjectDetector
from utils.json_handler import JSONHandler
from utils.visualization import Visualizer

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def process_image(image_path: str, config: dict, output_dir: str) -> None:
    """Process single image."""
    # Initialize components
    object_detector = ObjectDetector(config)
    subobject_detector = SubObjectDetector(config)
    json_handler = JSONHandler(config)
    visualizer = Visualizer(config)
    
    # Read image
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Detect objects
    detections = object_detector.detect(frame)
    
    # Detect sub-objects
    detections = subobject_detector.process_detections(frame, detections)
    
    # Save JSON output
    json_path = os.path.join(output_dir, 'detections.json')
    json_handler.save_detections(detections)
    
    # Create visualization
    vis_frame = visualizer.draw_detections(frame.copy(), detections)
    vis_path = os.path.join(output_dir, 'visualization.jpg')
    visualizer.save_visualization(vis_frame, vis_path)

def process_video(video_path: str, config: dict, output_dir: str) -> None:
    """Process video file."""
    # Initialize components
    object_detector = ObjectDetector(config)
    subobject_detector = SubObjectDetector(config)
    json_handler = JSONHandler(config)
    visualizer = Visualizer(config)
    
    # Open video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer
    output_video_path = os.path.join(output_dir, 'output_video.mp4')
    writer = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )
    
    all_detections = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect objects
        detections = object_detector.detect(frame)
        
        # Detect sub-objects
        detections = subobject_detector.process_detections(frame, detections)
        
        # Store detections
        all_detections.append(detections)
        
        # Create visualization
        vis_frame = visualizer.draw_detections(frame.copy(), detections)
        writer.write(vis_frame)

        # Display the resulting frame
        cv2.imshow('Real-Time Detection', vis_frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
        
    # Release resources
    cap.release()
    writer.release()
    cv2.destroyAllWindows()  # Close the OpenCV window
    
    # Save all detections to JSON
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    json_handler.save_video_detections(all_detections, video_name)

def extract_subobject(video_path: str, config: dict, output_dir: str,
                     object_id: int, subobject_name: str) -> None:
    """Extract specific sub-object images."""
    # Initialize components
    object_detector = ObjectDetector(config)
    subobject_detector = SubObjectDetector(config)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect objects and sub-objects
        detections = object_detector.detect(frame)
        detections = subobject_detector.process_detections(frame, detections)
        
        # Find matching detection
        for detection in detections:
            if detection['id'] == object_id:
                subobj_img = object_detector.extract_subobject(
                    frame, detection, subobject_name
                )
                if subobj_img is not None:
                    # Save extracted sub-object image
                    output_path = os.path.join(
                        output_dir,
                        f'subobject_{subobject_name}_{frame_count}.jpg'
                    )
                    cv2.imwrite(output_path, subobj_img)
        
        frame_count += 1
    
    cap.release()

def benchmark_inference(video_path: str, config: dict) -> None:
    """Benchmark the inference speed on a video."""
    object_detector = ObjectDetector(config)  # Initialize here for use in loop
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect objects
        detections = object_detector.detect(frame)
        frame_count += 1

    end_time = time.time()
    cap.release()

    total_time = end_time - start_time
    fps = frame_count / total_time if total_time > 0 else 0
    print(f"Processed {frame_count} frames in {total_time:.2f} seconds. FPS: {fps:.2f}")

def main():
    parser = argparse.ArgumentParser(description='Object Detection System')
    parser.add_argument('--config', default='config/config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--input', required=True,
                      help='Path to input image or video')
    parser.add_argument('--output', default='data/output',
                      help='Path to output directory')
    parser.add_argument('--retrieve-subobject', action='store_true',
                      help='Extract specific sub-object images')
    parser.add_argument('--object-id', type=int,
                      help='Object ID for sub-object extraction')
    parser.add_argument('--subobject', type=str,
                      help='Sub-object name for extraction')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    if args.retrieve_subobject:
        if args.object_id is None or args.subobject is None:
            raise ValueError("Must specify --object-id and --subobject for retrieval")
        extract_subobject(args.input, config, args.output,
                         args.object_id, args.subobject)
    else:
        # Determine input type
        if args.input.lower().endswith(('.mp4', '.avi', '.mov')):
            process_video(args.input, config, args.output)
            benchmark_inference(args.input, config)  # Call benchmarking here
        else:
            process_image(args.input, config, args.output)

if __name__ == '__main__':
    main()