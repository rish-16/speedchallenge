import os
import numpy as np
import cv2
from PIL import Image

from models import SpeedChallengeModel

vid_path = "../data/train.mp4"
speed_path = "../data/train.txt"

with open(speed_path) as f:
    y_train = f.read().strip().split("\n")
    y_train = [float(y_train[i]) for i in range(len(y_train))]
    
x_train = []
    
vidcap = cv2.VideoCapture(vid_path)
suc, img = vidcap.read()

while suc:
    suc, img = vidcap.read()
    x_train.append(img)
    
x_train = np.array(x_train)    
y_train = np.array(y_train)

print (x_train.shape, y_train.shape)

np.save("../data/x_train_dump.npy", x_train)
np.save("../data/y_train_dump.npy", y_train)