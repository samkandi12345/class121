import cv2
import time
import numpy as np

fourcc = cv2.VideoWriter_fourcc(*"XVID")
output_file = cv2.VideoWriter("output.avi",fourcc,20,(640,480))
Cap = cv2.VideoCapture(0)
time.sleep(2)
bg = 0

for i in range(10):
    ret,bg = Cap.read()

bg = np.flip(bg,axis=1)
while(Cap.isOpened()):
    ret,img = Cap.read()
    if not ret:
        break
    img = np.flip(img,axis=1)
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lower_red = np.array([240,0,0])
    upper_red = np.array([10,255,255])
    mask1 = cv2.inRange(hsv,lower_red,upper_red)
    lower_red = np.array([170,120,70])
    upper_red = np.array([120,255,255])
    mask2 = cv2.inRange(hsv,lower_red,upper_red)
    mask1 += mask2
    mask1 = cv2.morphologyEx(mask1,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    mask1 = cv2.morphologyEx(mask1,cv2.MORPH_DILATE,np.ones((3,3),np.uint8))
    mask2 = cv2.bitwise_not(mask1)
    res1 = cv2.bitwise_and(img,img,mask=mask2)
    res2 = cv2.bitwise_and(img,img,mask=mask1)
    finaloutput = cv2.addWeighted(res1,1,res2,1,0)
    output_file.write(finaloutput)
    cv2.imshow("magic",finaloutput)
    cv2.waitKey(1)
Cap.release()
finaloutput.release()
cv2.destroyAllWindows()

