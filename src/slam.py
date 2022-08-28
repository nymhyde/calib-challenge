#!/usr/bin/env python3

import time
import cv2
from display import Display
from extractor import Extractor
import numpy as np
import pandas as pd


pitch, yaw = [], []
pred = np.zeros((1196,2))

W = 1920//2
H = 1080//2

F = 910
K = np.array([[F,0,W//2],[0,F,H//2], [0,0,1]])
print(K)

disp = Display(W, H)
fe = Extractor(K)

def process_frame(img):
    img = cv2.resize(img, (W,H))
    matches, pose, PP, YY = fe.extract(img)

    pitch.append(PP)
    yaw.append(PP)

    if pose is None:
        return

    print(f">> {len(matches)} matches Found ! <<")
    print(pose)
    print(f"Pitch angle :: {PP:.4f}")
    print(f"Yaw angle :: {YY:.4f}\n\n")

    for pt1, pt2 in matches:
        u1,v1 = fe.denormalize(pt1)
        u2,v2 = fe.denormalize(pt2)
        cv2.circle(img, (u1, v1), color=(255,0,0), radius=4)
        cv2.line(img, (u1, v1), (u2, v2), color=(0,0,255))

    disp.paint(img)

if __name__ == "__main__":
    cap = cv2.VideoCapture("../labeled/4.hevc")

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            frame = frame[:600,:]
            process_frame(frame)
        else:
            pred[:,0], pred[:,1] = pitch, yaw
            np.savetxt('4.txt', pred)
            break

