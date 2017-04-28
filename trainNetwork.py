
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import backend as k
import numpy as np
import pandas as pd
import os

os.system("cls")

WIDTH = 80
HEIGHT = 60
MAX_CLASSIFIERS = 3 		# left & right

alpha = 0.03
EPOCHS = 8
BATCH = 10

fname = 'keras-trained-E%d.h5' % EPOCHS

def createConvNet(w,h):

	model = Sequential()

	model.add(Conv2D(32,(3,3),input_shape=(w,h,1)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Conv2D(32,(3,3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Conv2D(64,(3,3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Conv2D(64,(3,3)))
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

	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	return model

print "Loading numpy data..."
data = np.load('final_train_data.npy')

print "Splitting train-test sets..."
train_test_split_factor = int(round(0.1*len(data)))
print "train_test_split_factor = %d" % train_test_split_factor

train = data[:-train_test_split_factor]
test = data[-train_test_split_factor:]


q = []
y = []
for i in range(train.shape[0]):
	try:
		t = train[i]
		x = t[0].reshape(WIDTH,HEIGHT,1)
		q.append(x)
		y.append(t[1])
	except Exception as e:
		print str(e)
x = np.array(q,dtype='uint8')
print x.shape
print len(y)
#x = np.array([train[0][i] for i in train])
#x = x.reshape(-1,WIDTH,HEIGHT,1)
#y = [train[1][i] for i in train]
'''
x = []
y = []
for i in train:
	x.append(i[0])
	y.append(i[1])

x = np.array(x,dtype='uint8')
x = x.reshape(-1,WIDTH,HEIGHT,1)
'''

x_test = []
y_test = []
for i in test:
	x_test.append(i[0])
	y_test.append(i[1])

x_test = np.array(x_test,dtype='uint8')
x_test = x_test.reshape(-1,WIDTH,HEIGHT,1)
print x_test.shape

print "Creating neural network..."
network = createConvNet(WIDTH,HEIGHT)

print "training network..."
network.fit(x,y,batch_size=BATCH,epochs=EPOCHS,verbose=1,validation_data=(x_test,y_test))

score = network.evaluate(x_test,y_test,verbose=2)
print 'Test loss: ',score[0]
print 'Test accuracy: ', score[1]

network.save(fname)
