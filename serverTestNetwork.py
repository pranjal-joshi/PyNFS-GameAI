
import time
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import backend as k
import socket
import sys
import numpy as np
import pickle
import base64
import os

MYIP = "192.168.0.4"
WIDTH = 80
HEIGHT = 60
fname = "keras-trained-E12.h5"
MAX_CLASSIFIERS = 3

delta_time = 0.3

def createConvNet(w,h):
	model = Sequential()

	model.add(Convolution2D(32,3,3,input_shape=(w,h,1)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Convolution2D(32,3,3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Convolution2D(64,3,3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Convolution2D(64,3,3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Flatten())

	model.add(Dense(80))
	model.add(Activation('relu'))
	model.add(Dropout(0.4))

	model.add(Dense(64))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	model.add(Dense(MAX_CLASSIFIERS))
	model.add(Activation('softmax'))
	return model

os.system("clear")
print("PyNFS-AutoDrive server")
serverSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddress = (MYIP,12345)
print("Server at: ",serverAddress)
serverSocket.bind(serverAddress)


print("Loading & compiling pre-trained network..")
network = createConvNet(WIDTH,HEIGHT)
#network = load_model(fname)
network.load_weights(fname)
network.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

print("Server is ready & waiting for data...")

while True:
	data,addr = serverSocket.recvfrom(1024*100)
	data = pickle.loads(data)
	data = data.reshape(-1,WIDTH,HEIGHT,1)

	preds = network.predict(data)
	print(preds)

	data = pickle.dumps(preds,protocol=1)
	resp = serverSocket.sendto(data,addr)
