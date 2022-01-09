import glob
import os
import cv2
import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torchvision

from matplotlib import pyplot as plt

from driver import io_shape_dict
from driver_base import FINNExampleOverlay
from utils import (clip_coords, scale_coords, letterbox, 
                   xywh2xyxy, non_max_suppression,  
                   visualize_boxes)
from models import Detect


def get_driver(bitfile, platform, weights):
    driver = FINNExampleOverlay(
        bitfile_name=bitfile,
        platform=platform,
        io_shape_dict=io_shape_dict,
        batch_size=1,
        runtime_weight_dir=weights,
    )
    return driver

def load_images(source):
    img_paths = glob.glob(source + "*.jpg")
    return img_paths

def infer_and_save_results(driver, test_img_paths, img_size, conf_thres,
                          output_folder):
    anchors = np.array([[10,14,23,27,37,58]]) / np.array([32])
    nc = 1
    scale = np.load("scale.npy")
    detect_head = Detect(nc, anchors)
    len_img_paths = len(test_img_paths)

    for number, test_img_path in enumerate(test_img_paths):
        print(f"[INFO] processing {number+1}/{len_img_paths}")
        img_org = cv2.imread(test_img_path)
        img = img_org.copy()

        h, w, _ = img_org.shape
        img, ratio, (dw, dh) = letterbox(img, img_size, auto=False)

        img = img[:, :, ::-1]
        img = np.ascontiguousarray(img)
        img = img.astype(np.uint8)

        driver_in = np.expand_dims(img, 0)

        #t1 = time.time()
        output = driver.execute(driver_in)
        #t2 = time.time()
        output = scale.transpose(1,2,3,0)*output
        output = output.transpose(0,3,1,2)
        #print(f"Time passed for driver execution: {t2-t1} sec")
        #t3 = time.time()
        output = torch.from_numpy(output)
        pred = detect_head([output])[0]
        #t4 = time.time()
        #print(f"Time passed for sigmoid execution: {t4-t3} sec")

        # Get detected boxes_detected, labels, confidences, class-scores.
        #t5 = time.time()
        pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=0.10, classes=None, max_det=1000)
        #t6 = time.time()
        #print(f"Time passed for decode execution: {t6-t5} sec")

        boxes_detected, class_names_detected, probs_detected = [], [], []
        
        folder_name, file_name = test_img_path.split("/")[-2:]
        save_folder = output_folder + folder_name 
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_path = save_folder + "/" + file_name.replace(".jpg", ".txt")
        f = open(save_path, "w")
        det = pred[0] if len(pred)>0 else []
        
        # Process predictions
        if len(det):
            f.write(file_name+"\n")
            f.write(str(len(det))+"\n")
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape, det[:, :4], img_org.shape).round()
            for d in det:
                f.write("%.5f %.5f %.5f %.5f %g\n"%(d[0], d[1], d[2]-d[0], d[3]-d[1], d[4]))
        f.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Produce results for evaluatıon of FINN-generated accelerator"
    )
    parser.add_argument(
        "--source", help="path to dataset", default="../../test_samples/"
    )
    parser.add_argument(
        "--platform", help="Target platform: zynq-iodma alveo", default="zynq-iodma"
    )
    parser.add_argument(
        "--bitfile", help='name of bitfile (i.e. "finn-accel.bit")', default="../bitfile/finn-accel.bit"
    )
    parser.add_argument(
        "--weightfolder", help='weights folder', default="runtime_weights/"
    )
    parser.add_argument(
        "--scaleparams", help='path to scale params (i.e. "scale.npy")', default="scale.npy"
    )
    parser.add_argument(
        "--threshold", help="confidence threshold", type=float, default=0.02
    )
    parser.add_argument(
        "--outputs", help='path for saving outputs', default="./outputs/"
    )
    parser.add_argument(
        "--img_size", help="tuple of input size", default=(416, 416)
    )
    # parse arguments
    args = parser.parse_args()
    
    source = args.source
    conf_threshold = args.threshold
    bitfile = args.bitfile
    platform = args.platform
    weightfolder = args.weightfolder
    scaleparams = args.scaleparams
    outputs = args.outputs
    threshold = args.threshold
    img_size = args.img_size
    
    driver = get_driver(bitfile, platform, weightfolder)
    print("driver is loaded")
    val_img_paths = load_images(source)
    print("validation files are loaded")
    infer_and_save_results(driver, val_img_paths, img_size, threshold, outputs)