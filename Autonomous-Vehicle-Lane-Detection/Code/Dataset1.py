import numpy as np
from scipy.interpolate import interp1d
import cv2
import matplotlib.pyplot as plt
import glob

def preprocess(img,k,p):
    h,w = img.shape[:2]

    ## UNDISTORTING THE IMAGE AND SELECTING ROI ##
    dst = cv2.undistort(img,k,p, None)
        
    return dst

def lane_process(img):
    hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
    
    #Seperate yellow
    lower_yellow = np.array([20,100,100],dtype=np.uint8)
    upper_yellow = np.array([40,220,255],dtype=np.uint8)
    yellow_mask = cv2.inRange(hls_img,lower_yellow,upper_yellow)
    
      
    #Seperate White
    lower_white = np.array([0,200,0],dtype=np.uint8)
    upper_white = np.array([255,255,255],dtype = np.uint8)
    white_mask = cv2.inRange(hls_img,lower_white,upper_white)
    
    
    
    #Create thesholded image based on yellow OR white color
    combined_frames = np.zeros((img.shape[0], img.shape[1]))
    for row in range(0, hls_img.shape[0]):
        for col in range(0, hls_img.shape[1]):
            if(yellow_mask[row, col] > 170 or white_mask[row, col] > 170):
                combined_frames[row,col] = 255
                
    
    return combined_frames

# Function to warp the lane
def image_warp(img,src_pts):
    dest_pts = np.float32([[0,0],[300,0],[0,300],[300,300]])
    H = cv2.getPerspectiveTransform(src_pts,dest_pts)
    out = cv2.warpPerspective(img,H,(300,300))
    return out


def lane_line_fit(lanes):
    
    #Using histogram to get pixels with max value
    hist = np.sum(lanes,axis=0) #Calculating histogram
    channel_out = np.dstack((lanes,lanes,lanes))*255
    midpoint = int(hist.shape[0]/2)
    image_center = int(lanes.shape[1]/2)
    #getting the max values from Left and right lane 
    left_lane_ix = np.argmax(hist[:midpoint]) #getting index of the maximum pixel in hist(0 to midpoint) : (white pixel)
    right_lane_ix = np.argmax(hist[midpoint:])+midpoint #Adding midpoint for bias from 0

    #Applying the sliding window concept
    nwindows = 10
    window_height = np.int(lanes.shape[0]/nwindows)
    margin=100
    nonzero_pts = lanes.nonzero()
    nonzero_x = np.array(nonzero_pts[1])
    nonzero_y = np.array(nonzero_pts[0])
    
    left_shift = left_lane_ix
    right_shift = right_lane_ix
    leftLane_px = []
    rightLane_px = []

    for window in range(nwindows):
        window_left_border_LL = left_shift - margin
        window_right_border_LL = left_shift + margin
        window_left_border_RL = right_shift - margin
        window_right_border_RL = right_shift + margin
        window_top_border = lanes.shape[0]-  window*window_height
        window_bottom_border = lanes.shape[0] -(window+1)*window_height

        desired_pixels_leftLane = ((nonzero_y >= window_bottom_border) & (nonzero_y < window_top_border) & (nonzero_x >= window_left_border_LL) & (nonzero_x < window_right_border_LL)).nonzero()[0]
        desired_pixels_rightLane = ((nonzero_y >= window_bottom_border) & (nonzero_y < window_top_border) & (nonzero_x >= window_left_border_RL) & (nonzero_x < window_right_border_RL)).nonzero()[0]
        leftLane_px.append(desired_pixels_leftLane)
        rightLane_px.append(desired_pixels_rightLane)
     
        if len(desired_pixels_leftLane) > 50:
            left_shift = int(np.mean(nonzero_x[desired_pixels_leftLane]))
       
        if len(desired_pixels_rightLane) > 50:
            right_shift = int(np.mean(nonzero_x[desired_pixels_rightLane]))

    leftLane_px = np.concatenate(leftLane_px)
    rightLane_px = np.concatenate(rightLane_px)

    #Left lane pixels are given as
    Leftx = nonzero_x[leftLane_px]
    Lefty = nonzero_y[leftLane_px]
    
    #Right lane pixels are
    Rightx = nonzero_x[rightLane_px]
    Righty = nonzero_y[rightLane_px]
    

    right_fit = np.polyfit(Righty,Rightx,2)
    left_fit = np.polyfit(Lefty,Leftx,2)

    left_line_x = []
    left_line_y = []

    for i in range(lanes.shape[0]):
      y1 = left_fit[0]*i**2 + left_fit[1]*i + left_fit[2]
      left_line_x.append(y1)
      left_line_y.append(i)
    
    right_line_x = []
    right_line_y = []
    for i in range(300):
      y2 = right_fit[0]*i**2 + right_fit[1]*i + right_fit[2]
      if int(y2)< 300:
        right_line_x.append(y2)
        right_line_y.append(i)

    
    left_pts = np.array([np.transpose(np.vstack([left_line_x, left_line_y]))])
    right_pts = np.array([np.flipud(np.transpose(np.vstack([right_line_x, right_line_y])))])

    pts = np.hstack((left_pts, right_pts))
    #Fill poly must come here on channel_out
    cv2.fillPoly( channel_out, np.int_([pts]), (255,0, 0))
    cv2.polylines(channel_out, np.int32([left_pts]), isClosed=False, color=(0,0,255), thickness=10)
    cv2.polylines(channel_out, np.int32([right_pts]), isClosed=False, color=(0,0,255), thickness=10)
    #Fitting the curve on these points
    # Pixel to meters
    xm = 3.65/270
    ym = 30/300
    right_fit_meters = np.polyfit(Righty*ym,Rightx*xm,2)
    left_fit_meters = np.polyfit(Lefty*ym,Leftx*xm,2)
    # Curvature of left and right lanes
    left_radius = ((1 + (2*left_fit_meters[0]*ym*300 + left_fit_meters[1])**2)**1.5) / np.absolute(2*left_fit_meters[0])
    right_radius = ((1 + (2*right_fit_meters[0]*ym*300 +right_fit_meters[1])**2)**1.5) / np.absolute(2*right_fit_meters[0])
    roc = (left_radius+right_radius)/2
    return right_fit,left_fit,channel_out, roc

def dewarp(img, src_pts):
    dest_pts = np.float32([[270,140],[380,140],[70,253],[493,253]])
    H1= cv2.getPerspectiveTransform(src_pts,dest_pts)
    out= cv2.warpPerspective(img, H1, (696,256))

    return out
###########################################-MAIN-##################################################
cam_mtx = np.array([[9.037596e+02, 0.000000e+00, 6.957519e+02] ,
                    [0.000000e+00, 9.019653e+02, 2.242509e+02], 
                    [0.000000e+00, 0.000000e+00 ,1.000000e+00]],dtype=np.int32)
dist_mtx = np.array([-3.639558e-01, 1.788651e-01, 6.029694e-04, -3.922424e-04, -5.382460e-02],dtype=np.int32)


user_input = input('Enter the path of the folder from which you want read the images for eg. /home/user/Downloads/data ')
print('The path you entered is:', user_input)
print('')

user_input1 = input('Enter the path where you want to save the rendered video. for eg. /home/user/Downloads/: ')
print('The path you entered is:', user_input1)
print('')


#imgs = glob.iglob(user_input+"/*")
out = cv2.VideoWriter(user_input1 + 'Dataset1.avi',cv2.VideoWriter_fourcc(*'XVID'), 30, (696,256))
Frame = 0
for pic in sorted(glob.glob(user_input+ "/*")):

  print(pic)
  img = cv2.imread(pic)
  img = cv2.resize(img,(0,0),fx=0.5,fy=0.5)

  print(img.shape)
  cv2.waitKey(0)
  pp_img = preprocess(img,cam_mtx,dist_mtx)
  img_process = lane_process(pp_img)
  lanes = image_warp(img_process,np.float32([[270,140],[380,140],[70,253],[493,253]]))
  
  try:
    right_fit, left_fit, out_img, roc = lane_line_fit(lanes)
  
    lane_dewarp = dewarp(out_img, np.float32([[0,0],[300,0],[0,300],[300,300]]))
    
    final_frame = cv2.addWeighted(np.uint8(img), 1, np.uint8(lane_dewarp), 0.5, 0)
   
    cv2.putText(final_frame,'Radius of Curvature: ' + str(int(roc/1000))+' Km', (30,70),         cv2.FONT_HERSHEY_SIMPLEX, 0.7,
    (0,0,255), 2)
    if (int(roc/1000)) > 5:
      pass
    else:
      cv2.putText(final_frame,'<-- Turn left', (30,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
      (255,0,0), 2)
      print('<-- Turn left')
    out.write(np.uint8(final_frame))
   

    print('Rendered frame no:', Frame)
    print('ROC is:', roc/1000, 'Km')
    Frame+=1
  except:
    print('Error!')
    out.write(np.uint8(img))
   
out.release()
cv2.destroyAllWindows()


