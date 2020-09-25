# comma.ai Programming Challenge

---

### Why

I've begun my search for Summer 2021 internships. So, I thought of trying my hand at comma.ai as well. This repo contains my submission to the Speed Challenge where the aim is to build a model that's able to predict the speed of a vehicle solely based on a continuous camera feed.

The training data is a 17 minute long video containing of ~20400 frames. The ground truth labels are the exact speeds of the associated vehicle at each timestep/frame.

---

### My Approach

While too much convolution can cause more harm than good, I've decided to stick to sequential models rather than spatial ones. This way, I'm able to capture the nuances of the problem and build an accurate regression model.

I've decided to use an DeepConv Encoder to grab each frame and convert it into a low dimensional code that's finally passed into a LSTM that spits out the real speed value pertaining to said input frame.

---

### Stack

My solution is fully built using TensorFlow v2.0. The image preprocessing steps have been taken care of using `opencv` and `pillow`. The choice of hyperparameters is completely original with no help of any Auto-ML based solutions. Midway, I did, however, think of using a hyperparameter tuning tool like Keras Tuner but decided not to because of unncessary experimentation overhead (my laptop would have given up).