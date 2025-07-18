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

class TwoStageDetector:
    def __init__(self, person_weights='yolov7.pt', weapon_weights='weapon.pt', 
                 img_size=640, conf_thres=0.5, iou_thres=0.45, device='0'):
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
        self.weapon_filter = ['gun', 'rifle', 'knife', 'baseball']
        
        # Run inference once to initialize
        if self.device.type != 'cpu':
            self.person_model(torch.zeros(1, 3, self.person_img_size, self.person_img_size).to(self.device).type_as(next(self.person_model.parameters())))
            self.weapon_model(torch.zeros(1, 3, self.weapon_img_size, self.weapon_img_size).to(self.device).type_as(next(self.weapon_model.parameters())))
    
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
        Process a single frame with the two-stage detection
        
        Args:
            frame: Input frame
            
        Returns:
            annotated_frame: Frame with bounding boxes
        """
        # Make a copy for annotations
        annotated_frame = frame.copy()
        
        # First, detect persons
        persons = self.detect_persons(frame)
        
        # If no persons detected, no need to run weapon detection
        if not persons:
            return annotated_frame
        
        # Since at least one person is detected, run weapon detection on full frame
        weapons = self.detect_weapons(frame)
        
        # Draw person bounding boxes
        for person in persons:
            x1, y1, x2, y2, conf, cls = person
            person_label = f'Person {conf:.2f}'
            plot_one_box([x1, y1, x2, y2], annotated_frame, label=person_label, 
                         color=self.person_colors[int(cls)], line_thickness=2)
        
        # Draw weapon bounding boxes
        for weapon in weapons:
            x1w, y1w, x2w, y2w, confw, clsw = weapon
            weapon_label = f'{self.weapon_names[int(clsw)]} {confw:.2f}'
            plot_one_box([x1w, y1w, x2w, y2w], annotated_frame, label=weapon_label,
                         color=self.weapon_colors[int(clsw)], line_thickness=3)
        
        return annotated_frame

def process_video(input_path, output_path, person_weights, weapon_weights, device='0'):
    """
    Process a video with the two-stage detection
    """
    # Initialize detector
    detector = TwoStageDetector(
        person_weights=person_weights, 
        weapon_weights=weapon_weights,
        device=device
    )
    
    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process frames
    frame_count = 0
    processing_times = []
    
    print(f"Processing video: {input_path}")
    print(f"Total frames: {total_frames}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 10 == 0:  # Process every 10th frame for speed
            start_time = time.time()
            
            # Process frame
            annotated_frame = detector.process_frame(frame)
            
            # Calculate and display FPS
            process_time = time.time() - start_time
            processing_times.append(process_time)
            avg_time = sum(processing_times[-20:]) / min(len(processing_times), 20)
            fps_text = f"FPS: {1/avg_time:.2f}"
            cv2.putText(annotated_frame, fps_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Display progress
            progress = f"Frame: {frame_count}/{total_frames} ({100*frame_count/total_frames:.1f}%)"
            print(progress, end='\r')
            
            # Write frame
            out.write(annotated_frame)
            
            # Display frame
            cv2.imshow('Detection', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            # For skipped frames, just write the original frame
            out.write(frame)
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Print statistics
    avg_process_time = sum(processing_times) / len(processing_times) if processing_times else 0
    print(f"\nProcessing complete. Average processing time per frame: {avg_process_time:.4f}s ({1/avg_process_time:.2f} FPS)")
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
    parser.add_argument('--input', type=str, required=True, help='Input image or video path')
    parser.add_argument('--output', type=str, required=True, help='Output image or video path')
    parser.add_argument('--person-weights', type=str, default='yolov7.pt', help='Person detection model weights')
    parser.add_argument('--weapon-weights', type=str, default='weapon.pt', help='Weapon detection model weights')
    parser.add_argument('--device', type=str, default='0', help='Device to use (e.g., cpu or 0 for GPU)')
    
    args = parser.parse_args()
    
    # Check if input is image or video
    is_video = args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    
    if is_video:
        process_video(args.input, args.output, args.person_weights, args.weapon_weights, args.device)
    else:
        process_image(args.input, args.output, args.person_weights, args.weapon_weights, args.device)