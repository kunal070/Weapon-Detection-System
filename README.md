# **Weapon Detection System**

This project implements an advanced **object detection model** using **YOLOv7** to identify and classify weapons such as **guns, rifles, knives**, and other objects (e.g., baseball equipment) in real-time images and videos. It leverages PyTorch, OpenCV, and pre-trained YOLOv7 weights, with fine-tuning on a custom dataset.

---

## **Features**
- Detects **weapons** (gun, rifle, knife) and non-weapon objects.
- Real-time detection from:
  - Webcam
  - Image/Video files
  - RTSP/HTTP streams
- Visualizes bounding boxes with class labels and confidence scores.
- Tracks model metrics such as precision, recall, and F1-score.
- Generates evaluation plots:
  - Confusion matrix
  - Precision-Recall curves
  - Confidence vs Precision/Recall/F1

---

## **Sample Results**

### **Confusion Matrix**
![Confusion Matrix](confusion_matrix.png)

### **Model Metrics**
- **Precision Curve**  
  ![Precision Curve](P_curve.png)
- **Recall Curve**  
  ![Recall Curve](R_curve.png)
- **Precision-Recall (PR) Curve**  
  ![PR Curve](PR_curve.png)
- **F1 Score vs Confidence**  
  ![F1 Curve](F1_curve.png)

### **Training Results**
![Training Results](results.png)

---

## **Detection Examples**
**Ground Truth vs Predictions:**
- **Labeled Test Batch**  
  ![Test Labels](test_batch0_labels.jpg)
- **Predicted Test Batch**  
  ![Test Predictions](test_batch0_pred.jpg)

---

## **Installation**

### **Requirements**
- Python 3.8+
- PyTorch (with CUDA if GPU available)
- OpenCV
- Other dependencies in `requirements.txt`

Install dependencies:
```bash
pip install -r requirements.txt
