import cv2
import numpy as np
import torch
from ultralytics import YOLO
import sys
import ultralytics

class WeaponDetector:
    def __init__(self, person_model_path, weapon_model_path):
        """Initialize with person and weapon detection models"""
        try:
            # Load models with specific inference settings
            self.person_model = YOLO(person_model_path)
            self.weapon_model = YOLO(weapon_model_path)
        except Exception as e:
            print(f"Error loading models: {e}")
            raise
        
        # Specific weapon classes to detect
        self.weapon_classes = ['gun', 'rifle', 'knife', 'baseball bat']
        
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
            
            # Detect persons 
            try:
                # Use inference parameters directly
                person_results = self.person_model.predict(
                    frame, 
                    conf=0.5,  # Confidence threshold
                    iou=0.45,  # IoU threshold
                    verbose=False  # Disable print output
                )
            except Exception as e:
                print(f"Error in person detection: {e}")
                print(f"Full traceback: {sys.exc_info()}")
                continue
            
            # For each detected person
            for person in person_results[0].boxes:
                # Get bounding box coordinates
                person_bbox = person.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, person_bbox)
                
                # Annotate person with blue bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # Extract person region of interest
                person_roi = frame[y1:y2, x1:x2]
                
                if person_roi.size == 0:
                    continue
                
                # Detect weapons 
                try:
                    weapon_results = self.weapon_model.predict(
                        person_roi, 
                        conf=0.5,
                        iou=0.45,
                        verbose=False
                    )
                except Exception as e:
                    print(f"Error in weapon detection: {e}")
                    print(f"Full traceback: {sys.exc_info()}")
                    continue
                
                # Check for weapons
                threat_detected = False
                for weapon in weapon_results[0].boxes:
                    weapon_bbox = weapon.xyxy[0].cpu().numpy()
                    # Convert weapon bbox to full frame coordinates
                    weapon_bbox[0] += x1
                    weapon_bbox[1] += y1
                    weapon_bbox[2] += x1
                    weapon_bbox[3] += y1
                    
                    # Get weapon class name
                    weapon_class = self.weapon_model.names[int(weapon.cls)]
                    
                    # Only proceed if it's one of our specified weapon classes
                    if weapon_class.lower() not in [w.lower() for w in self.weapon_classes]:
                        continue
                    
                    # Draw weapon bounding box in red
                    cv2.rectangle(frame, 
                                 (int(weapon_bbox[0]), int(weapon_bbox[1])),
                                 (int(weapon_bbox[2]), int(weapon_bbox[3])),
                                 (0, 0, 255), 2)
                    
                    # Add weapon class label
                    cv2.putText(frame, 
                               f"Weapon: {weapon_class}", 
                               (int(weapon_bbox[0]), int(weapon_bbox[1])-10),
                               cv2.FONT_HERSHEY_SIMPLEX,
                               0.9, (0, 0, 255), 2)
                    
                    # Check if weapon is a threat
                    if self.is_threat(person_bbox, weapon_bbox):
                        threat_detected = True
                
                # Display threat detected if close proximity
                if threat_detected:
                    cv2.putText(frame, "THREAT DETECTED",
                               (50, 50),
                               cv2.FONT_HERSHEY_SIMPLEX,
                               1.5, (0, 0, 255), 3)
            
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
    print(f"Python Version: {sys.version}")
    print(f"Ultralytics YOLO Version: {ultralytics.__version__}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"OpenCV Version: {cv2.__version__}")
    
    # Initialize detector
    detector = WeaponDetector(
        person_model_path='yolov7-tiny.pt',  # YOLOv7 tiny model for persons
        weapon_model_path='Main.pt'      # Your weapon detection model
    )
    
    # Process video
    detector.detect_threats(
        video_path='b.mp4',
        output_path='output_video.mp4',
        display=True
    )