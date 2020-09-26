import os
import numpy as np
import cv2
from PIL import Image

train_vid_path = "../data/train/"
train_speed_path = "../data/train.txt"
test_vid_path = "../data/test/"

with open(train_speed_path) as f:
    y_train = f.read().strip().split("\n")
    y_train = [float(y_train[i]) for i in range(len(y_train))]
    
for i in range(len(os.listdir(train_vid_path))):
    path = train_vid_path + os.listdir(train_vid_path)[i]
    
    frames = []
    
    vidcap = cv2.VideoCapture(path)
    suc, img = vidcap.read()
    
    while suc:
        suc, img = vidcap.read()
        frames.append(img)
    
    frames = np.array(frames)
    np.save("../data/train_segments/x_train_segment_{}.npy".format(str(i+1)), frames, allow_pickle=True)

y_train = np.array(y_train)
np.save("../data/y_train_dump.npy", y_train)