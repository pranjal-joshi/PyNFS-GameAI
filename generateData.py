
from getKeys import checkKeys
from printScreen import screenshotMethod
import numpy as np
import cv2
import pandas as pd
import time
import os

def mapKeys(keys):
	label = [0,0]
	if 'A' in keys:
		label = [1,0]
	elif 'D' in keys:
		label = [0,1]
	return label

if __name__ == '__main__':

	fname = "train_data.npy"
	train_data = []
	print "Starting in"
	for i in range(1,6)[::-1]:
		print i
		time.sleep(1)

	paused = False

	while True:
		if not paused:
			screenshot = screenshotMethod(80,60)
			keys = checkKeys()
			output_keys = mapKeys(keys)
			train_data.append([screenshot,output_keys])
			time.sleep(0.025)
			print len(train_data)

		k = checkKeys()
		if 'P' in k:
			if paused:
				paused = False
			else:
				paused = True
			time.sleep(1)

		if 'O' in k:
			np.save(fname,train_data)
			print "Quitting..."
			break
