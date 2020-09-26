import numpy as np

path = "../data/train_segments/x_train_segment_1.npy"

x = np.load(path, allow_pickle=True)
print (x.shape)