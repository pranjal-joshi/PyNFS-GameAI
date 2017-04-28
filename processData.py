import numpy as np
import pandas as pd
import cv2
from collections import Counter
from random import shuffle
import time
import os

os.system("cls")

VISUALIZE = False or True
VISUALIZE_OUTPUT = True

train_data = np.load('train_data.npy')

print "loading train_data.npy"
df = pd.DataFrame(train_data)
print df.head()
print Counter(df[1].apply(str))

if VISUALIZE:
	for i in range(len(df)):
		try:
			z = df[0][i]
			z = cv2.resize(z,(320,240),1)
			cv2.imshow("train_data",z)
			if cv2.waitKey(25) & 0xFF == ord('q'):
				cv2.destroyAllWindows()
				break
			time.sleep(0.01)
		except:
			pass

left = []
right = []
fwd = []

shuffle(train_data)

for data in train_data:
	i = data[0]
	label = data[1]

	if label == [1,0,0]:
		fwd.append([i,label])
	elif label == [0,1,0]:
		left.append([i,label])
	elif label == [0,0,1]:
		right.append([i,label])

MIN = min(len(left),len(right))
MIN = min(MIN,len(fwd))
print "MIN = %d" % MIN
left = left[:MIN]
right = right[:MIN]
fwd = fwd[:MIN]

final_train_data = left + right + fwd
shuffle(final_train_data)

np.save('final_train_data.npy',final_train_data)

if VISUALIZE_OUTPUT:
	df = pd.DataFrame(final_train_data)
	for i in range(len(final_train_data)):
		try:
			z = df[0][i]
			z = cv2.resize(z,(320,240),1)
			cv2.imshow("final_train_data",z)
			if cv2.waitKey(25) & 0xFF == ord('q'):
				cv2.destroyAllWindows()
				break
			time.sleep(0.01)
		except:
			pass
