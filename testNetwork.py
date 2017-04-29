
import time
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import backend as k
import numpy as np
import cv2
import pandas as pd
import os
from sendKeys import PressKey, ReleaseKey, A, D
from printScreen import screenshotMethod
from getKeys import checkKeys

WIDTH = 80
HEIGHT = 60
fname = "keras-trained-E14.h5"
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

def left():
	PressKey(A)
	print("left")
	time.sleep(delta_time)
	ReleaseKey(A)

def right():
	PressKey(D)
	print("right")
	time.sleep(delta_time)
	ReleaseKey(D)

print("Loading & compiling pre-trained network..")
network = createConvNet(WIDTH,HEIGHT)
#network = load_model(fname)
network.load_weights(fname)
network.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

print("Staring AI in ...")
for i in range(1,6)[::-1]:
	print(i)

paused = False


while True:
	if not paused:
		try:
			scr = screenshotMethod(WIDTH,HEIGHT)
			scr = scr.reshape(-1,WIDTH,HEIGHT,1)
			preds = network.predict(scr)
			print(preds)
		except Exception as e:
			print(str(e))

		if int(round(preds[0][0])) == 1:
			left()
		elif int(round(preds[0][1])) == 1:
			right()

	keys = checkKeys()

	if 'P' in keys:
		if paused:
			print("Resuming..")
			paused = False
		else:
			paused = True
			print("Pausing..")
			ReleaseKey(A)
			ReleaseKey(D)
		time.sleep(1)

'''
inp = np.load('train_data.npy')
df = pd.DataFrame(inp)

x = []
for i in inp:
	if i[0] == None:
		pass
	else:
		x.append(i[0])

x = np.array(x,dtype='uint8')
#x = x.reshape(-1,WIDTH,HEIGHT,1)

for i in x:
    z = i.copy()
    pred = network.predict(z.reshape(-1,WIDTH,HEIGHT,1))
    print("%d %d %d" % (round(pred[0][0]), round(pred[0][1]), round(pred[0][2])))
    img = cv2.resize(i,(320,240),1)
    cv2.imshow("train_data",img)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
    #time.sleep(0.005)
    #except Exception as e:
        #raise e

'''