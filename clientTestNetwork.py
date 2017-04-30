
import time
import numpy as np
import cv2
import pandas as pd
import os
from sendKeys import PressKey, ReleaseKey, A, D
from printScreen import screenshotMethod, maskImage
from getKeys import checkKeys
import socket
import sys
import pickle
import base64

THRESHOLD = 0.9
WIDTH = 80
HEIGHT = 60
delta_time = 0.03
SERVER_IP = "192.168.0.4"			#"192.168.0.50"


try:
	clientSocket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
	serverAddress = (SERVER_IP,12345)
	print("Connecting to AutoDrive server at ",serverAddress)
except:
	sys.exit("Failed to connect AutoDrive server! Quitting...")


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


print("Staring AI in ...")
for i in range(1,6)[::-1]:
	print(i)
	time.sleep(1)

paused = False

while True:
	if not paused:
		try:
			scr = maskImage(screenshotMethod(WIDTH,HEIGHT))
			scr = pickle.dumps(scr,protocol=1)
			sendScr = clientSocket.sendto(scr,serverAddress)

			preds, server = clientSocket.recvfrom(1024*100)
			preds = pickle.loads(preds,encoding="latin1").tolist()
			print(preds)

			if preds[0][1] > THRESHOLD:
				left()
			elif preds[0][2] > THRESHOLD:
				right()
		except Exception as e:
			#raise e
			print(str(e))


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