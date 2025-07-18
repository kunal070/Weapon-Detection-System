import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from customsort import *
import _thread, time

def tracker_initialize():
    global sort_tracker
    sort_max_age = 5
    sort_min_hits = 2
    sort_iou_thresh = 0.2
    sort_tracker = Sort(max_age=sort_max_age,
                        min_hits=sort_min_hits,
                        iou_threshold=sort_iou_thresh)
    return sort_tracker

def trackerr():
    global track         
    for key, values in track.items():
        if values:  # Check if the list is not empty
            first_value = values[0]
            last_value = values[-1]
            if(scam[key] == 'False') : 
                if first_value < 998 and last_value > 998:
                    print('Arrival yard')
                    scam[key] = 'True'
                    print ('id : ' , scam[key])
                    return 1
                elif first_value > 998 and last_value < 998:
                    print('Depature yard')
                    scam[key] = 'True'
                    print ('id : ' , scam[key])
                    return -1
        else:
            print(f"For key '{key}': The list is empty")

def LoadModel(weights):
    device = select_device('0')
    half = True  # half precision only supported on CUDA
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = 640  # check img_size
    if half:
        model.half()  # to FP16
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1
    return model, names, colors, device
 
def image_preprocess(img, device):
    img = letterbox(img, 640, 32)[0]
        # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img
 
def bbox_rel(*xyxy):
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

truck_record_log,tractor_record_log,crane_record_log,jcb_record_log,pickup_record_log,cement_record_log  = [], [], [], [], [], []



def draw_boxes(img, bbox, identities=None, categories=None, names=None, offset=(0, 0), rs=None):
    t = 0
    pick_flag = False               # vehicledict[id]=
    truck_count,tractor_count,pickup_count,jcb_count,crane_count,cement_count = 0,0,0,0,0,0
    global truck_record_log,tractor_record_log,crane_record_log,jcb_record_log,pickup_record_log,cement_record_log,tracking_vehi
    global vehi,track,scam
    centroid = 0
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        label = names[cat]
        centroid = (x1 + x2)/2
        if id in track:
            if(scam[id] == 'False'):
                track[id].append(centroid)
        else:
            track[id] = [centroid]
            scam[id] = 'False'
            try :
                if(id==1):
                    pass
                else: 
                    del track[id-1]
                    del scam[id-1] 
            except():
                pass 
        
       
    return img, truck_count,tractor_count,pickup_count,jcb_count,crane_count,cement_count
 
def annotations(pred, img, im0, sort_tracker, names):
    global truck_count, tractor_count, pickup_count, cement_count, jcb_count, crane_count
  
    for i, det in enumerate(pred):  # detections per image
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        overlay = im0.copy()
        # cv2.rectangle(overlay, (20,170),(365, 360), (0, 0, 0), -1)
        # cv2.rectangle(im0, (20,170),(365, 360), (0, 255, 255), 1)
        alpha = 0.8 # Transparency factor.
        im0 = cv2.addWeighted(overlay, alpha, im0, 1 - alpha, 0)
        
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            dets_to_sort = np.empty((0, 6))
            for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
                dets_to_sort = np.vstack((dets_to_sort,
                                            np.array([x1, y1, x2, y2, conf, detclass])))
            tracked_dets = sort_tracker.update(dets_to_sort)
            if len(tracked_dets) > 0:
                bbox_xyxy = tracked_dets[:, :4]
                identities = tracked_dets[:, 8]
                categories = tracked_dets[:, 4]
                im0,truck_count,tractor_count,pickup_count,jcb_count,crane_count,cement_count = draw_boxes(im0, bbox_xyxy, identities, categories, names)
        if len(det)==0:
            truck_count,tractor_count,pickup_count,jcb_count,crane_count,cement_count = 0,0,0,0,0,0

    return im0,truck_count,tractor_count,pickup_count,jcb_count,crane_count,cement_count

def create_table(vehicle_type, in_count, out_count, im0):
    # Define column widths and heights
    col_widths = [200, 100, 100]
    row_height = 50

    # Define column headers
    headers = ['Vehicle Type', 'In', 'Out']

    # Draw column headers and set the background to black
    cv2.rectangle(im0, (30, 200), (30 + sum(col_widths), 200 + row_height), (0, 0, 0), -1)
    for i, header in enumerate(headers):
        text_size = cv2.getTextSize(header, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.putText(im0, header, (30 + int((col_widths[i] - text_size[0]) / 2) + sum(col_widths[:i]), 200 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Draw data rows and set the background to black
    for i, (vt, inc, outc) in enumerate(zip(vehicle_type, in_count, out_count), start=1):
        cv2.rectangle(im0, (30, 200 + i * row_height), (30 + sum(col_widths), 200 + (i + 1) * row_height), (0, 0, 0), -1)
        cv2.putText(im0, vt, (30 + 20, 200 + i * row_height + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(im0, str(inc), (30 + col_widths[0] + 20, 200 + i * row_height + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(im0, str(outc), (30 + sum(col_widths[:2]) + 20, 200 + i * row_height + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Draw white borders around the entire table
    cv2.rectangle(im0, (30, 200), (30 + sum(col_widths), 200 + (len(vehicle_type) + 1) * row_height), (255, 255, 255), 2)
     # Draw white borders between columns
    for i in range(1, len(col_widths)):
        cv2.rectangle(im0, (30 + sum(col_widths[:i]), 200), (30 + sum(col_widths[:i]) + 2, 200 + (len(vehicle_type) + 1) * row_height), (255, 255, 255), -1)

    # Draw white borders between rows
    for i in range(1, len(vehicle_type) + 1):
        cv2.rectangle(im0, (30, 200 + i * row_height), (30 + sum(col_widths), 200 + i * row_height + 2), (255, 255, 255), -1)

    return im0
 
def detect(source):
    source, weights = source, "vehi.pt"
    sort_tracker = tracker_initialize()
    model, names, color, device = LoadModel(weights)
    cap = cv2.VideoCapture(source)
    framenum = 0
    print('ok')
    while True:
        ret, img = cap.read()
        if ret:
            im0 = img
            framenum += 1
            # print(framenum)
            if framenum % 10 ==0: 
                img = image_preprocess(img, device)
                # Inference
                pred = model(img)[0]
                # Apply NMS
                if len(pred)!=0:
                    pred = non_max_suppression(pred, 0.5, 0.5)  
                    im0,truck_count,tractor_count,pickup_count,jcb_count,crane_count,cement_count  = annotations(pred, img, im0, sort_tracker, names)
                    #im0 = trackerr(im0)
                    # txt1 = f'Truck Count : {len(truck_record_log)} '
                    # List of counts and corresponding texts
                    counts = [truck_count, tractor_count, jcb_count, cement_count, crane_count, pickup_count]
                    texts = ['Truck Count', 'Tractor Count', 'Jcb Count', 'Cement Mixer Count', 'Crane Count', 'pickup_truck Count']
                    

                    
                    area = [(924, 0), (924, 1440)]
                    key_names = []
                    zeros_count = []
                    ones_count = []
                    for key, value in vehi.items():
                        # Append the key name to the key_names list
                        key_names.append(key)
                        
                        # Count the number of zeros and ones in the value list
                        zeros = value.count(0)
                        ones = value.count(1)
                        
                        # Append the counts to the respective lists
                        zeros_count.append(zeros)
                        ones_count.append(ones)
                        
                    if len(vehi) == 0:
                        print('no')
                    else :
                        im0 = create_table(key_names,zeros_count,ones_count,im0)
                        print(vehi)    

                    cv2.line(im0, area[0], area[1], (0, 255, 0), 2)  # (0, 255, 0) represents the color green, 2 is the thickness
                    #vidsave.addframe(cv2.resize(im0, dsize=(0,0),fx = 0.5, fy = 0.5))
                    # vidsave.addframe(im0)
                    cv2.imshow('Vehicle Analytics', cv2.resize(im0, dsize=(0,0),fx = 0.5, fy = 0.5))

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break  # 1 millisecond

            #time.sleep(0.5)
        else:
            
            break
 
if __name__ == '__main__':
    source1 = r"roi_cement.mp4"
     
    with torch.no_grad():
        detect(source1)