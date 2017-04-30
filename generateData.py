
from getKeys import checkKeys
from printScreen import screenshotMethod
import numpy as np
import pandas as pd
import time
import os
import cv2

WIDTH = 80
HEIGHT = 60

def mapKeys(keys):
	label = [0,0,0]
	if 'W' in keys:
		label = [1,0,0]
	if 'K' in keys:				# press K while turning left
		label = [0,1,0]
	if 'L' in keys:				# press L while turning right
		label = [0,0,1]
	return label

def maskImage(img):
	uz = np.zeros((25,80),dtype='uint8')
	o = np.ones((25,80),dtype='uint8')
	lz = np.zeros((10,80),dtype='uint8')
	mask = np.vstack((uz,o))
	mask = np.vstack((mask,lz))
	masked = cv2.bitwise_and(img,img,mask=mask)
	masked = masked[25:51,:]
	masked = cv2.resize(masked,(WIDTH,HEIGHT),1)
	return masked

if __name__ == '__main__':

	fname = "train_data.npy"
	train_data = []
	print("Starting in")
	for i in range(1,6)[::-1]:
		print(i)
		time.sleep(1)

	paused = False

	while True:
		if not paused:
			try:
				screenshot = maskImage(screenshotMethod(WIDTH,HEIGHT))
				keys = checkKeys()
				output_keys = mapKeys(keys)
				train_data.append([screenshot,output_keys])
				time.sleep(0.025)
				print(len(train_data))
			except Exception as e:
				print(str(e))

		k = checkKeys()
		if 'P' in k:
			if paused:
				paused = False
			else:
				paused = True
			time.sleep(1)

		if 'O' in k:
			np.save(fname,train_data)
			print("Quitting...")
			break
