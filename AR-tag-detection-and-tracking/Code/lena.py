import cv2
import numpy as np
import math
from scipy.spatial import distance as dist
import mylib
import time

start_time = time.time()


user_input = input('Which video do you want to superimpose the cube on? Your options are: 1.Tag0 2.Tag1 3.Tag2 4. multipleTags:   ')
print('You chose:', user_input)
print('Nice choice!')
print('')

user_input1 = input('Please specify the path from where I should pick the videos and the image from in this format, for eg. /home/user/Downloads:   ')
print('The path you entered is:', user_input1)
print('')

user_input2 = input('Which image do you want to superimpose on the Video?, Lena.png:   ')
print('The path you entered is:', user_input2)
print('')


# Assigning the functions from utils.py to a local name
order_pts = mylib.order_pts
homography = mylib.homography
warpTag = mylib.warpTag
warpLena = mylib.warpLena
decodeTag = mylib.decodeTag
reorient = mylib.reorient
contour_gen = mylib.contour_gen

if user_input == 'Tag0':
	cap = cv2.VideoCapture(user_input1 + '/' + user_input + '.mp4')
elif user_input == 'Tag1':
	cap = cv2.VideoCapture(user_input1 + '/' + user_input + '.mp4')
elif user_input == 'Tag2':
	cap = cv2.VideoCapture(user_input1 + '/' + user_input + '.mp4')
else:
	cap = cv2.VideoCapture(user_input1 + '/' + user_input + '.mp4')


# Specify path for the image to be superimposed here
frame2 = cv2.imread(user_input1 + '/' + user_input2)

lena = cv2.resize(frame2, (200,200))


if cap.isOpened() == False:
    print("Error loading")
Frame = 0

p2 = np.array([[0,0],[200,0],[200,200],[0,200]]).reshape(4,2)


if cap.isOpened() == False:
    print("Error loading")
Frame = 0

p2 = np.array([[0,0],[200,0],[200,200],[0,200]]).reshape(4,2)

lena_list = list()

# Specify the path of the output video to be rendered
out = cv2.VideoWriter(user_input1 + '/Lena.avi',cv2.VideoWriter_fourcc(*'XVID'), 30, (960,540))

#Iterating through all the frames in the Video
print('Video Rendering started ...')
start_time = time.time()
while cap.isOpened():
    ret,frame = cap.read()
    print("Frame:",Frame)
    if ret == False:
         break
    img = cv2.resize(frame, (0,0),fx=0.5,fy=0.5)
    #Finding the contours of the AR Tag 
    contour, edge = contour_gen(img)
  
    for i in range(len(contour)):
        H = homography(contour[i],p2)
        tag = warpTag(img, H, (200,200))
        ang,o,ID = decodeTag(tag)
        print("TAG:",ang,",",o,",",ID)
        lena_pose = reorient(o)
        H_lena = homography(lena_pose,contour[i])
        lena_warped = warpLena(lena, H_lena, (img.shape[1],img.shape[0]),img)
        out.write(lena_warped)
    Frame+=1
    
    if cv2.waitKey(5) & 0xFF == 27:
         break




out.release()            
cap.release()
cv2.destroyAllWindows()
end_time = time.time()
time_taken = end_time - start_time
print('Video Rendered, Time it took:',time_taken)
print('All is well! Have a nice day!')

