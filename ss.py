#!/usr/bin/python

import cv2
from PIL import ImageGrab as scr
import numpy as np
import time

def screenCast():
    while True:
        t = time.time()
        i = scr.grab(bbox=(25,25,650,505))
        i = np.array(i,dtype='uint8')
        i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
        i = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
        cv2.putText(i,"FPS: %s" % str(int(round(1/(time.time()-t)))), (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
        cv2.imshow("ScreenCapture",i)
        #print "FPS: %s" % str(1/(time.time()-t))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

def screenShot():
    print "Capturing screenshot in"
    for i in range(1,6)[::-1]:
        time.sleep(1)
        print i
    i = np.array(scr.grab(),dtype='uint8')
    i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
    i = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("screenShot.png",i)

screenShot()
