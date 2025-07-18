import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from pathlib import Path
import time
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from utils.plots import plot_one_box
from utils.torch_utils import select_device
import datetime
# Import Sort tracker
from customsort import Sort

class TwoStageDetector:
    def __init__(self, person_weights='yolov7.pt', weapon_weights='weapon.pt', 
                 img_size=640, conf_thres=0.4, iou_thres=0.45, device='0'):
        """
        Initialize two-stage YOLOv7 detector
        
        Args:
            person_weights (str): Path to person detection weights file
            weapon_weights (str): Path to weapon detection weights file
            img_size (int): Input image size
            conf_thres (float): Confidence threshold
            iou_thres (float): IoU threshold for NMS
            device (str): Device to run inference on ('cpu' or GPU index)
        """
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # Initialize device
        self.device = select_device(device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        
        # Load person detection model
        print(f"Loading person detection model: {person_weights}")
        self.person_model = attempt_load(person_weights, map_location=self.device)
        self.person_stride = int(self.person_model.stride.max())
        self.person_img_size = check_img_size(img_size, s=self.person_stride)
        if self.half:
            self.person_model.half()
        
        # Get person model class names
        self.person_names = self.person_model.module.names if hasattr(self.person_model, 'module') else self.person_model.names
        self.person_colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in self.person_names]
        
        # Load weapon detection model
        print(f"Loading weapon detection model: {weapon_weights}")
        self.weapon_model = attempt_load(weapon_weights, map_location=self.device)
        self.weapon_stride = int(self.weapon_model.stride.max())
        self.weapon_img_size = check_img_size(img_size, s=self.weapon_stride)
        if self.half:
            self.weapon_model.half()
            
        # Get weapon model class names
        self.weapon_names = self.weapon_model.module.names if hasattr(self.weapon_model, 'module') else self.weapon_model.names
        self.weapon_colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in self.weapon_names]
        
        # Filter for weapon classes
        self.weapon_filter = ['gun', 'rifle', 'knife']
        
        # Run inference once to initialize
        if self.device.type != 'cpu':
            self.person_model(torch.zeros(1, 3, self.person_img_size, self.person_img_size).to(self.device).type_as(next(self.person_model.parameters())))
            self.weapon_model(torch.zeros(1, 3, self.weapon_img_size, self.weapon_img_size).to(self.device).type_as(next(self.weapon_model.parameters())))
            
        # Initialize trackers for persons and weapons
        self.initialize_trackers()
        
        # For tracking objects
        self.track = {}
        self.scam = {}
    
    def initialize_trackers(self):
        """Initialize SORT trackers for person and weapon objects"""
        # Parameters for SORT
        sort_max_age = 5
        sort_min_hits = 2
        sort_iou_thresh = 0.2
        
        # Initialize trackers
        self.person_tracker = Sort(max_age=sort_max_age, 
                                  min_hits=sort_min_hits, 
                                  iou_threshold=sort_iou_thresh)
        
        self.weapon_tracker = Sort(max_age=sort_max_age, 
                                  min_hits=sort_min_hits, 
                                  iou_threshold=sort_iou_thresh)
    
    def preprocess_image(self, img0, model_stride, model_img_size):
        """
        Preprocess image for inference
        """
        # Padded resize
        img = letterbox(img0, model_img_size, stride=model_stride)[0]
        
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
            
        return img
    
    def detect_persons(self, img0):
        """
        Detect persons in the image
        
        Returns:
            list: List of person detections [x1, y1, x2, y2, conf, cls]
        """
        # Preprocess image
        img = self.preprocess_image(img0, self.person_stride, self.person_img_size)
        
        # Inference
        with torch.no_grad():
            pred = self.person_model(img)[0]
        
        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
        
        # Process detections
        persons = []
        
        for i, det in enumerate(pred):  # detections per image
            if len(det) == 0:
                continue
                
            # Rescale boxes from img_size to img0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            
            # Filter for person class (usually index 0, but let's double check)
            for idx, (*xyxy, conf, cls) in enumerate(det):
                cls_name = self.person_names[int(cls)]
                if cls_name.lower() == 'person':
                    persons.append(det[idx].cpu().numpy())
                
        return persons
    
    def detect_weapons(self, img0):
        """
        Detect weapons in the full image
        
        Args:
            img0: Original image
            
        Returns:
            list: List of weapon detections [x1, y1, x2, y2, conf, cls]
        """
        # Preprocess image for weapon detection
        img = self.preprocess_image(img0, self.weapon_stride, self.weapon_img_size)
        
        # Inference
        with torch.no_grad():
            pred = self.weapon_model(img)[0]
        
        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
        
        # Process detections
        weapons = []
        
        for i, det in enumerate(pred):  # detections per image
            if len(det) == 0:
                continue
                
            # Rescale boxes from img_size to img0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            
            # Filter for weapon classes
            for *xyxy, conf, cls in det:
                cls_int = int(cls.item())
                cls_name = self.weapon_names[cls_int].lower()
                
                if any(weapon in cls_name for weapon in self.weapon_filter):
                    weapons.append([
                        xyxy[0].item(), xyxy[1].item(), 
                        xyxy[2].item(), xyxy[3].item(), 
                        conf.item(), cls_int
                    ])
        
        return weapons
    
    def process_frame(self, frame):
        """
        Process a single frame with the two-stage detection and tracking
        
        Args:
            frame: Input frame
            
        Returns:
            annotated_frame: Frame with bounding boxes and tracking IDs
        """
        # Make a copy for annotations
        annotated_frame = frame.copy()
        
        # First, detect persons
        persons = self.detect_persons(frame)
        
        # Format for SORT tracker
        person_dets = np.empty((0, 6))
        for person in persons:
            x1, y1, x2, y2, conf, cls = person
            person_dets = np.vstack((person_dets, np.array([x1, y1, x2, y2, conf, cls])))
        
        # Update person tracker
        if len(person_dets) > 0:
            person_tracks = self.person_tracker.update(person_dets)
        else:
            person_tracks = np.empty((0, 9))
        
        # Detect weapons
        weapons = self.detect_weapons(frame)
        
        # Format for SORT tracker
        weapon_dets = np.empty((0, 6))
        for weapon in weapons:
            x1, y1, x2, y2, conf, cls = weapon
            weapon_dets = np.vstack((weapon_dets, np.array([x1, y1, x2, y2, conf, cls])))
        
        # Update weapon tracker
        if len(weapon_dets) > 0:
            weapon_tracks = self.weapon_tracker.update(weapon_dets)
        else:
            weapon_tracks = np.empty((0, 9))
        
        # Draw person bounding boxes with tracking IDs
        for track in person_tracks:
            x1, y1, x2, y2 = track[:4]
            track_id = int(track[8])
            cls = int(track[4])  # Get class from track
            
            # Update tracking dictionary
            centroid = (x1 + x2) / 2
            if track_id in self.track:
                if self.scam.get(track_id, 'False') == 'False':
                    self.track[track_id].append(centroid)
            else:
                self.track[track_id] = [centroid]
                self.scam[track_id] = 'False'
            
            person_label = f'Person #{track_id} {track[6]:.2f}'
            plot_one_box([x1, y1, x2, y2], annotated_frame, label=person_label, 
                         color=self.person_colors[cls % len(self.person_colors)], line_thickness=2)
        
        # Draw weapon bounding boxes with tracking IDs
        for track in weapon_tracks:
            x1, y1, x2, y2 = track[:4]
            track_id = int(track[8])
            cls = int(track[4])  # Get class from track
            
            # Update tracking dictionary
            centroid = (x1 + x2) / 2
            if track_id in self.track:
                if self.scam.get(track_id, 'False') == 'False':
                    self.track[track_id].append(centroid)
            else:
                self.track[track_id] = [centroid]
                self.scam[track_id] = 'False'
            
            try:
                weapon_label = f'{self.weapon_names[cls]} #{track_id} {track[6]:.2f}'
            except IndexError:
                weapon_label = f'Weapon #{track_id} {track[6]:.2f}'
                
            plot_one_box([x1, y1, x2, y2], annotated_frame, label=weapon_label,
                         color=self.weapon_colors[cls % len(self.weapon_colors)], line_thickness=3)
        
        # Check for threat: at least 2 persons and any weapon in the same frame
        is_threat = len(person_tracks) >= 2 and len(weapon_tracks) > 0
        
        # If threat detected, add warning text
        if is_threat:
            h, w = annotated_frame.shape[:2]
            # Create semi-transparent overlay for the warning text
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
            
            # Add threat text
            cv2.putText(overlay, "THREAT DETECTED", (w//2 - 200, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            
            # Add detailed warning
            weapon_names = []
            for track in weapon_tracks:
                cls = int(track[4])
                try:
                    weapon_name = self.weapon_names[cls]
                    if weapon_name not in weapon_names:
                        weapon_names.append(weapon_name)
                except IndexError:
                    if "Unknown" not in weapon_names:
                        weapon_names.append("Unknown")
            
            weapon_text = ", ".join(weapon_names)
            warning_text = f"Multiple persons ({len(person_tracks)}) with {weapon_text} detected"
            cv2.putText(overlay, warning_text, (20, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Apply the overlay with transparency
            alpha = 0.7
            cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)
            
            # Also add red border around the frame to indicate threat
            border_thickness = 10
            cv2.rectangle(annotated_frame, (0, 0), (w, h), (0, 0, 255), border_thickness)
        
        return annotated_frame

def process_camera(output_path, person_weights, weapon_weights, device='0', camera_id=0):
    """
    Process camera feed with the two-stage detection
    """
    # Initialize detector
    detector = TwoStageDetector(
        person_weights=person_weights, 
        weapon_weights=weapon_weights,
        device=device
    )
    
    # Open camera
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return
    
    # Add these lines to set a higher resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1400)  # Set width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000)  # Set height
    
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30  # Assume 30 FPS for camera
    
    # Initialize video writer if output path is provided
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process frames
    frame_count = 0
    processing_times = []
    last_annotated_frame = None  # Store the last annotated frame
    
    print(f"Processing camera feed (camera ID: {camera_id})")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera")
            break
        
        frame_count += 1
        display_frame = frame.copy()  # Create a copy for display
        
        # Process every 3rd frame for real-time performance
        process_this_frame = frame_count % 3 == 0
        
        if process_this_frame:
            start_time = time.time()
            
            # Process frame
            annotated_frame = detector.process_frame(frame)
            last_annotated_frame = annotated_frame.copy()  # Save the annotated frame
            
            # Calculate and display FPS
            process_time = time.time() - start_time
            processing_times.append(process_time)
            avg_time = sum(processing_times[-20:]) / min(len(processing_times), 20)
            fps_text = f"FPS: {1/avg_time:.2f}"
            cv2.putText(last_annotated_frame, fps_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Add timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(last_annotated_frame, timestamp, (width - 200, height - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            display_frame = last_annotated_frame  # Use the newly annotated frame for display
        else:
            # For non-processed frames, use the original frame for recording
            # but use the last annotated frame for display if available
            if last_annotated_frame is not None:
                # Update only the timestamp on the last annotated frame
                display_copy = last_annotated_frame.copy()
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(display_copy, timestamp, (width - 200, height - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                display_frame = display_copy
        
        # Write the appropriate frame to video
        if out:
            # For recording, we want to capture the actual frames
            # You can choose to record either:
            # 1. The original frame (more accurate timing but no annotations)
            # 2. The display frame (with annotations but may have duplicates)
            out.write(display_frame)  # Using display_frame to include annotations
                
        # Display frame
        cv2.imshow('YOLOv7 Detection', display_frame)
        
        # Break if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    
    # Print statistics
    if processing_times:
        avg_process_time = sum(processing_times) / len(processing_times)
        print(f"\nProcessing complete. Average processing time per frame: {avg_process_time:.4f}s ({1/avg_process_time:.2f} FPS)")
        if output_path:
            print(f"Output saved to: {output_path}")
            
def process_image(input_path, output_path, person_weights, weapon_weights, device='0'):
    """
    Process a single image with the two-stage detection
    """
    # Initialize detector
    detector = TwoStageDetector(
        person_weights=person_weights, 
        weapon_weights=weapon_weights,
        device=device
    )
    
    # Read image
    frame = cv2.imread(input_path)
    if frame is None:
        print(f"Error: Could not read image {input_path}")
        return
    
    # Process frame
    start_time = time.time()
    annotated_frame = detector.process_frame(frame)
    process_time = time.time() - start_time
    
    # Add processing time
    fps_text = f"Process time: {process_time:.4f}s ({1/process_time:.2f} FPS)"
    cv2.putText(annotated_frame, fps_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Save output
    cv2.imwrite(output_path, annotated_frame)
    
    # Display image
    cv2.imshow('Detection', annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print(f"Processing complete. Time: {process_time:.4f}s")
    print(f"Output saved to: {output_path}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Two-stage YOLOv7 detection (Person → Weapons)')
    parser.add_argument('--output', type=str, default='', help='Output video path (optional)')
    parser.add_argument('--person-weights', type=str, default='yolov7.pt', help='Person detection model weights')
    parser.add_argument('--weapon-weights', type=str, default='weapon.pt', help='Weapon detection model weights')
    parser.add_argument('--device', type=str, default='0', help='Device to use (e.g., cpu or 0 for GPU)')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID (default: 0)')
    
    args = parser.parse_args()
    
    # Process camera feed
    process_camera(args.output, args.person_weights, args.weapon_weights, args.device, args.camera)