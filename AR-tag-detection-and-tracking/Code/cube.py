import cv2
import numpy as np
import math
from scipy.spatial import distance as dist
import mylib
import time



k=np.array([[1406.08415449821,0,0],
    [2.20679787308599, 1417.99930662800,0],
    [1014.13643417416, 566.347754321696,1]]).T

user_input = input('Which video do you want to superimpose the cube on? Your options are: 1.Tag0 2.Tag1 3.Tag2 4. multipleTags:   ')
print('You chose:', user_input)
print('Nice choice!')
print('')

user_input1 = input('Please specify the path from where I should pick the videos and the image from in this format, for eg. /home/user/Downloads:   ')
print('The path you entered is:', user_input1)
print('')


# Assigning the functions from utils.py to a local name
drawCube = mylib.drawCube
getProjection = mylib.getProjection
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



if cap.isOpened() == False:
    print("Error loading")
Frame = 0

p2 = np.array([[0,0],[200,0],[200,200],[0,200]]).reshape(4,2)

lena_list = list()
out = cv2.VideoWriter(user_input1 + '/Cube.avi',cv2.VideoWriter_fourcc(*'XVID'), 30, (960,540))
print('Video Rendering started ...')
start_time = time.time()
while cap.isOpened():
    ret,frame = cap.read()
    print("Frame:",Frame)
    if ret == False:
         break
    img = cv2.resize(frame, (0,0),fx=0.5,fy=0.5)
    contour, edge = contour_gen(img)

    for i in range(len(contour)):
        H = homography(contour[i],p2)

        tag = warpTag(img, H, (200,200))
        ang,o,ID = decodeTag(tag)
        print("TAG:",ang,",",o,",",ID)

        lena_pose = reorient(o)
        H_lena = homography(lena_pose,contour[i])
        P = getProjection(H_lena,k)
        drawCube(img, P)
        out.write(img)
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

