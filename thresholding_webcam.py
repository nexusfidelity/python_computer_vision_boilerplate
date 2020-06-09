# import the necessary packages

import time
import cv2
import numpy as np


 
# initialize the camera and grab a reference to the raw camera capture


 
# allow the camera to warmup
time.sleep(0.1)

cap = cv2.VideoCapture(0)
 
# capture frames from the camera
while(True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    

    ret, image = cap.read()

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
    ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    
    # show the frame
    cv2.imshow("original", image)
    cv2.imshow("grayscale", gray)
    cv2.imshow("thresh", thresh)
    
    
    
    key = cv2.waitKey(1) & 0xFF
 
    # clear the stream in preparation for the next frame
 
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

cv2.destroyAllWindows()
