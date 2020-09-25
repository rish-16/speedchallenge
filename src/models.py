import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, LSTM, Flatten, Dropout

class DeepConvEncoder:
    def __init__(self, dim):
        self.dim = dim
        self.model = self.build(dim)
        
    def build(self, dim):
        input_img = Input(shape=dim)
        x = Conv2D(256, (3,3), activation="relu")(input_img)
        x = MaxPool2D(pool_size=(2,2))(x)
        x = Conv2D(128, (3,3), activation="relu")(x)
        x = MaxPool2D(pool_size=(2,2))(x)
        x = Dropout(0.25)(x)
        x = Conv2D(64, (3,3), activation="relu")(x)
        x = Flatten()(x)
        x = Dropout(0.25)(x)
        code = Dense(32, activation="relu")(x)
        
        model = Model(inputs=input_img, outputs=code)
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["mse", "acc"])
        
        return model
        
    def inference(self, x):
        code = self.model.predict(x)
        return code
        
class LSTMDecoder:
    def __init__(self, dim):
        self.dim = dim
        self.model = self.build(dim)
        
    def build(self, dim):
        input_code = Input(shape=dim)
        x = LSTM(100, activation="tanh")(input_code)
        y = Dense(1, activation="linear")(x) 
        
        model = Model(inputs=input_code, outputs=y)
        model.compile(loss="mean_squared_error", optimizer="adam")
        
        return model
        
    def inference(self, code):
        speed = self.model.predict(code)
        return speed
        
class SpeedChallengeModel:
    def __init__(self, input_dim, code_dim):
        self.encoder = DeepConvEncoder(input_dim)
        self.decoder = LSTMDecoder(code_dim)
        self.model = self.build()
        
    def build(self, frame_dim):
        img = Input(shape=frame_dim)
        code = self.encoder(img)
        speed = self.decoder(code)
        
        model = Model(inputs=img, outputs=speed)
        model.compile(loss="mean_squared_error", optimizer="adam")
        
        return model
        
    def train(self, x_train, y_train):
        hist = self.model.fit(x_train, y_train, batch_size=64, epochs=10)
        
        return hist
        
    def inference(self, frame):
        speed = self.model.predict(frame)
        
        return speed
        
        