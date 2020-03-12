#!/usr/bin/env python

import cv2
import numpy as np
import matplotlib.pyplot as plt

user_input = input('Enter the path of the video to rectify. for eg. /home/user/Downloads/Night Drive - 2689.mp4: ')
print('The path you entered is:', user_input)
print('')

user_input1 = input('Enter the path where you want to save the rendered video. for eg. /home/user/Downloads/: ')
print('The path you entered is:', user_input1)
print('')


cap = cv2.VideoCapture(user_input)

if cap.isOpened() == False:
    print('Error Loading the video!')


def gamma_correction(img,gamma):
    #Output = Input^(1/gamma)
    #Scale the input from (0 to 256) to (0 to 1)
    #Apply gamma correction
    #Scale back to original values
    gamma = 1/gamma
    lT =[]
    for i in np.arange(0,256).astype(np.uint8):
        lT.append(np.uint8(((i/255)**gamma)*255))
    lookup = np.array(lT)
    #Creating the lookup table, cv can find the gamma corrected value of each pixel value
    corrected = cv2.LUT(img,lookup)
    return corrected
    
out = cv2.VideoWriter(user_input1 + 'Video Enhancement.avi',cv2.VideoWriter_fourcc(*'XVID'),30,(960,540))
Frame = 0
while cap.isOpened():
    ret, frame = cap.read()
    if ret == False:
        break
    img = cv2.resize(frame, (0,0),fx=0.5,fy=0.5)
    corrected_frame = gamma_correction(img,2)
    sharpened_img = cv2.convertScaleAbs(corrected_frame,-1,0.8,3)
	#Alpha = 0.8 for contrast
        #Beta = 3 for additional brigthness (scaling)
	#Output image flag set to -1
    #cv2.imshow('Cust2',eq6)
    out.write(sharpened_img)

    
    Frame+=1
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()




