import argparse
import time
from pathlib import Path
import math
import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size,  non_max_suppression,  \
    scale_coords,  set_logging

from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from customsort import *
sort_tracker = None
 
def trackerr():
    global sort_tracker
    tracks = sort_tracker.getTrackers()
    print('inside the tracker')
    for i,track in enumerate(tracks):
        print(f"inside the loop : {track.centroidarr[-1]}")
        try:
            # dist = math.dist(track.centroidarr[-1][0], track.centroidarr[0][0])
            # print(dist)
            if track.centroidarr[-1][0] > 1244 and track.centroidarr[0][0] < 1244:
                print('Arrival yard')
                return 1
            
            elif track.centroidarr[-1][0] < 1244 and track.centroidarr[0][0] > 1244:
                print('Depature yard')
                return -1
        except:
            pass
            print('waiting')

def draw_boxes(img, bbox, identities=None, categories=None, names=None, offset=(0, 0), rs=None):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
            
        label = names[cat]
        colorcode=[(80,255, 30),(86, 86, 255), (80,255, 30), (40,255,249),  (86, 86, 253), (40,255,105)]
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        if cat!=0 or cat!=1 or cat!=2:
            cv2.rectangle(img, (x1, y1), (x2, y2), colorcode[cat], 2)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + w + 10, y1), colorcode[cat], -1)
            cv2.putText(img, f'{id}:{label}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [0, 0, 0], 2)
    return img
    

def video_detection(rtsp):
    global sort_tracker
    weights = r"vehicle.pt"
    kernel = np.array([[0, -1, 0],
                          [-1, 5, -1],
                          [0, -1, 0]])
    # Initialize
    classes= [0,1,2,3,4,5]
    device = select_device('0')
    half = device.type != 'cpu'  # half precision only supported on CUDA
    model = attempt_load(weights, map_location=device)
     
    
    if half:
        model.half() 
    
 # load FP32 model
    stride = int(model.stride.max())  # model stride
    #imgsz = check_img_size(imgsz, s=stride)  # check img_size
    imgsz = 640
    
    sort_max_age = 5
    sort_min_hits = 2
    sort_iou_thresh = 0.2
    sort_tracker = Sort(max_age=sort_max_age,
                        min_hits=sort_min_hits,
                        iou_threshold=sort_iou_thresh)
    
    cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = []
    dataset = LoadStreams(rtsp, img_size=imgsz, stride=stride)
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    
    for path, img, im0s, vid_cap in dataset:
        #print(vid_cap)
        #img = cv2.filter2D(src = img, kernel = kernel, ddepth = -1 )
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        cv2.line(img,(1244,0),(1244,1440),(255,255,0),2) #line
        
        # Inference
        torch.cuda.empty_cache()

        # with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
        pred = model(img,augment=False)[0]
        

        # Apply NMS
        pred = non_max_suppression(pred,conf_thres=0.50, iou_thres=0.45, classes=classes)
        

        print(pred,rtsp)
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            
            p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            p = Path(p)  # to Path
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy,conf,cls in reversed(det):
                        object_coordinates = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                        x=str(int(xyxy[0])),str(int(xyxy[1]))
                        y=str(int(xyxy[2])),str(int(xyxy[3]))
                dets_to_sort = np.empty((0, 6))

                # NOTE: We send in detected object class too
                for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
                    dets_to_sort = np.vstack((dets_to_sort,
                                            np.array([x1, y1, x2, y2, conf, detclass])))

                tracked_dets = sort_tracker.update(dets_to_sort)
                if len(tracked_dets) > 0:
                    bbox_xyxy = tracked_dets[:, :4]
                    identities = tracked_dets[:, 8]
                    categories = tracked_dets[:, 4]
                    im0,=draw_boxes(im0, bbox_xyxy, identities, categories, names)
            yield im0
            
            

   


