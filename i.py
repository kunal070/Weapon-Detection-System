import cv2
import numpy as np
import torch
from ultralytics import YOLO

class WeaponDetector:
    def __init__(self, person_model_path, weapon_model_path):
        """Initialize with person and weapon detection models"""
        # Load models with explicit configuration
        self.person_model = YOLO(person_model_path)
        self.weapon_model = YOLO(weapon_model_path)
        
        # Set device automatically
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def calculate_centroid(self, bbox):
        """Calculate center point of bounding box"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def is_threat(self, person_bbox, weapon_bbox, threshold=100):
        """Check if weapon is close enough to person"""
        person_center = self.calculate_centroid(person_bbox)
        weapon_center = self.calculate_centroid(weapon_bbox)
        
        distance = np.sqrt(
            (person_center[0] - weapon_center[0])**2 + 
            (person_center[1] - weapon_center[1])**2
        )
        return distance < threshold
    
    def detect_threats(self, video_path, output_path, display=False):
        """Process video and detect threats"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file {video_path}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer
        out = cv2.VideoWriter(output_path, 
                             cv2.VideoWriter_fourcc(*'mp4v'), 
                             fps, 
                             (width, height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect persons with explicit parameters
            person_results = self.person_model.predict(
                source=frame, 
                stream=False,  # Process single frame
                conf=0.5,  # Confidence threshold
                classes=[0]  # Only detect persons (COCO class 0)
            )
            
            for person in person_results[0].boxes:
                person_bbox = person.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, person_bbox)
                person_roi = frame[y1:y2, x1:x2]
                
                if person_roi.size == 0:
                    continue
                
                # Detect weapons in person ROI
                weapon_results = self.weapon_model.predict(
                    source=person_roi, 
                    stream=False,
                    conf=0.5
                )
                
                for weapon in weapon_results[0].boxes:
                    weapon_bbox = weapon.xyxy[0].cpu().numpy()
                    # Convert to full frame coordinates
                    weapon_bbox[0] += x1
                    weapon_bbox[1] += y1
                    weapon_bbox[2] += x1
                    weapon_bbox[3] += y1
                    
                    if self.is_threat(person_bbox, weapon_bbox):
                        # Draw threat boxes
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.rectangle(frame, 
                                     (int(weapon_bbox[0]), int(weapon_bbox[1])),
                                     (int(weapon_bbox[2]), int(weapon_bbox[3])),
                                     (0, 0, 255), 2)
                        cv2.putText(frame, "THREAT DETECTED",
                                   (50, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX,
                                   1, (0, 0, 255), 2)
            
            out.write(frame)
            
            if display:
                cv2.imshow('Weapon Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        out.release()
        if display:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    # Initialize detector
    detector = WeaponDetector(
        person_model_path='yolov8n.pt',  # Using a standard YOLO model
        weapon_model_path='yolov8n.pt'   # You may need to replace with your specific weapon model
    )
    
    # Process video
    detector.detect_threats(
        video_path='b.mp4',
        output_path='output_video.mp4',
        display=True
    )