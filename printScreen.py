from PIL import ImageGrab, Image
import time
from sendKeys import PressKey, ReleaseKey, PRINT_SCREEN
import win32api, win32con
import numpy as np
import cv2

WIDTH = 80
HEIGHT = 60

def grabImage(x,cy,cx,saveimg=False):
	try:
		i = ImageGrab.grabclipboard()
		if isinstance(i, Image.Image):
			i = np.array(i,dtype="uint8")
			i = cv2.resize(i,(cy,cx),1)
			i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
			i = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
			if saveimg:
				cv2.imwrite('sampleData/captured%d.bmp' % int(x),i)
			return i
	except Exception as e:
		print(str(e))
		pass

def screenshotMethod(y,x):
	win32api.keybd_event(win32con.VK_SNAPSHOT,1)
	i = grabImage(0,y,x,saveimg=False)
	return i

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

'''
time.sleep(8)
while True:
#for z in range(0,200):
	try:
		t = time.time()
		win32api.keybd_event(win32con.VK_SNAPSHOT,1)
		i = grabImage(0,320,240,saveimg=False)
		cv2.imshow("printScreen",i)
		if(cv2.waitKey(25) & 0xFF == ord('q')):
			cv2.destroyAllWindows()
			break
		print(1/(time.time()-t))
	except Exception as e:
		print(str(e))
		pass
'''