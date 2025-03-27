import cv2
import torch
import numpy as np
import traceback

def preprocess_image(img, model, device):
    """
    Preprocess the input image with device-specific handling
    """
    try:
        # Try to get input size from model's stride
        stride = max(int(model.stride.max()), 32)
        
        # Calculate new size that's divisible by stride
        height, width = img.shape[:2]
        new_height = int(np.ceil(height / stride) * stride)
        new_width = int(np.ceil(width / stride) * stride)
        
        # Resize image
        img_resized = cv2.resize(img, (new_width, new_height), 
                                 interpolation=cv2.INTER_LINEAR)
    except Exception as e:
        print(f"Error determining model input size: {e}")
        # Fallback to a standard size if determination fails
        img_resized = cv2.resize(img, (640, 640))
    
    # Convert BGR to RGB
    img_rgb = img_resized[:, :, ::-1].transpose(2, 0, 1)
    
    # Make contiguous and convert to tensor
    img_tensor = torch.from_numpy(np.ascontiguousarray(img_rgb)).float()
    
    # Normalize
    img_tensor /= 255.0
    
    # Add batch dimension and move to correct device
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    return img_tensor

def detailed_type_inspection(pred):
    """
    Perform a detailed inspection of the prediction object
    """
    print("\n--- Prediction Detailed Inspection ---")
    print(f"Type of prediction: {type(pred)}")
    
    # Inspect prediction attributes and methods
    print("\nAvailable attributes:")
    for attr in dir(pred):
        if not attr.startswith('__'):
            try:
                attr_value = getattr(pred, attr)
                print(f"{attr}: {type(attr_value)}")
            except Exception as e:
                print(f"{attr}: Unable to access - {e}")
    
    # If it's a tensor or has tensor-like properties
    if isinstance(pred, (torch.Tensor, list)):
        print("\nTensor/List Details:")
        if isinstance(pred, torch.Tensor):
            print(f"Shape: {pred.shape}")
            print(f"Dtype: {pred.dtype}")
        elif isinstance(pred, list):
            print(f"Length: {len(pred)}")
            for i, item in enumerate(pred):
                print(f"Item {i}: Type {type(item)}")
                if isinstance(item, torch.Tensor):
                    print(f"  Shape: {item.shape}")

def weapon_detection(source, model, device):
    """
    Perform weapon detection on video input with extensive debugging
    """
    # Open video capture
    cap = cv2.VideoCapture(source)
    
    while True:
        # Read frame
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Preprocess image
        try:
            # Preprocess with device-specific handling
            processed_img = preprocess_image(frame, model, device)
            
            # Inference
            with torch.no_grad():
                # Capture the raw prediction
                pred = model(processed_img)
                
                # Perform detailed type inspection
                detailed_type_inspection(pred)
            
            # Pause to allow inspection
            input("Press Enter to continue...")
        
        except Exception as e:
            print(f"Error during inference: {e}")
            traceback.print_exc()
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

def main():
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    try:
        from ultralytics import YOLO
        
        # Load YOLOv8 model
        model_path = 'Main.pt'  # Replace with your model path
        model = YOLO(model_path)
        
        # Convert model to the correct device
        model.to(device)
        
        # Video source
        video_source = 'b.mp4'  # Replace with your video path
        
        # Perform detection
        # Use the underlying PyTorch model for more direct access
        weapon_detection(video_source, model.model, device)
    
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    main()