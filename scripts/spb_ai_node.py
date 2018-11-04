#!/usr/bin/env python
import rospy
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from std_msgs.msg import UInt16
from cv_bridge import CvBridge
import cv2
import pickle

# import serial

class surface_line(object):
    def __init__(self):
        self.locked_on = False
        self.detected = False #was the most recent frame detected
        self.y_val = 0
        self.confidence = 0
        self.iteration = 0
        self.mv_avg = 5

        self.max_confidence = 20
        self.locked_thresh = 5

    def update_sl(self,lvl): #update the tracker
        if lvl > 0:  #if we find a line, raise the confidence
            self.confidence = min(self.max_confidence, self.confidence + 1)
            self.iteration = min(self.mv_avg,self.iteration + 1)
            self.y_val = self.y_val*(self.iteration - 1) / self.iteration + lvl / (self.iteration)
        else:    #if we don't find a line, lower the confidence
            self.confidence = max(0, self.confidence - 1)

        if self.confidence > self.locked_thresh:
            self.locked_on = True
        else:
            self.locked_on = False



class BeerDetector(object):
    def __init__(self):
        rospy.init_node('beer_detector')

        #variables
        self.camera_image = None
        self.bridge = CvBridge()

        #subscribers & publishers
        sub1 = rospy.Subscriber('/camera/image_color', Image, self.image_cb, queue_size=1)
        self.pub1 = rospy.Publisher('/spb/image_lines', Image, queue_size=1)
        self.pub2 = rospy.Publisher('/spb/level_lines', Int32, queue_size=1)
        self.pub3 = rospy.Publisher('/spb/sobely', Image, queue_size=1)
        self.pub4 = rospy.Publisher('/spb/houghlines', Image, queue_size=1)
        self.pub5 = rospy.Publisher('/spb/level_area', Int32, queue_size=1)
        self.pub6 = rospy.Publisher('/spb/image_area', Image, queue_size=1)
        self.pub7 = rospy.Publisher('/spb/lvl', UInt16, queue_size=1)

        #load camera calibration
        camera_cal = pickle.load( open( "/home/robbie/spb_ws/src/spb_camera/camera_cal2.p", "rb" ) )
        self.ret = camera_cal[0]
        self.mtx = camera_cal[1]
        self.dist = camera_cal[2]
        self.line_tracker = surface_line()
        self.area_tracker = surface_line()

        self.cropTop = 400
        self.cropBot = 1000

        # self.ser = ser = serial.Serial('/dev/ttyACM0',115200, timeout=0)

        rospy.spin()

    def pix2dist(self,lvl_pix):

        y1 = 130.
        x1 = 300.
        y2 = 193.
        x2 = 700.

        lvl = lvl_pix + self.cropTop

        m = (y2-y1)/(x2-x1)
        b = 130 - m * 300
        return m*lvl+b


    def image_cb(self, msg):


        #convert ROS img msg to OpenCV
        cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        #undistort
        img_dst = cv2.undistort(cv_img, self.mtx, self.dist, None, self.mtx)
        #crop
        img_crop = img_dst[self.cropTop:self.cropBot,400:900]
        #grayscale
        img_gray = cv2.cvtColor(img_crop,cv2.COLOR_BGR2GRAY)
        #HLS
        img_hls = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HLS).astype(np.float)
        #HSV
        img_hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)


        #Find the Lines --------------
        img_out_hough, range_sobely, lvl_lines = self.find_lines(img_hls,img_crop)

        line_img1 = np.zeros((img_gray.shape[0],img_gray.shape[1],3), dtype=np.uint8)
        cv2.line(line_img1,(0,lvl_lines),(650,lvl_lines),(255,255,0),5)
        # img_out_lines = cv2.addWeighted(np.uint8(img_crop*255.0),0.8,line_img1, 1.,0.)
        img_out_lines = cv2.addWeighted(img_crop,0.8,line_img1, 1.,0.)
        range_sobely_col = np.dstack([range_sobely*0,range_sobely,range_sobely])

        #Area filter ------------------
        lvl_area, mask_and_3 = self.find_area(img_hsv, img_gray)

        line_img2 = np.zeros((img_gray.shape[0],img_gray.shape[1],3), dtype=np.uint8)
        cv2.line(line_img2,(0,lvl_area),(line_img2.shape[0],lvl_area),(255,0,0),5)
        img_out_area = cv2.addWeighted(img_crop,0.8,line_img2, 1.,0.)

        img_out_area = cv2.addWeighted(img_out_area,0.8,mask_and_3, 1.,0.)


        #combine surface lvls from lines and area filters
        self.line_tracker.update_sl(lvl_lines)
        self.area_tracker.update_sl(lvl_area)

        #Take the output of the tracker
        if self.line_tracker.locked_on == True and self.area_tracker.locked_on == True:
            output = (self.line_tracker.y_val + self.area_tracker.y_val)*0.5
        elif self.line_tracker.locked_on == True and self.area_tracker.locked_on == False:
            output = self.line_tracker.y_val
        elif self.line_tracker.locked_on == False and self.area_tracker.locked_on == True:
            output = self.area_tracker.y_val
        else:
             output = 0

        output = self.pix2dist(output)
        print(output)
        # self.ser.write(str(output)+"\n")



        #convert cv2 image to ROS img
        # ros_img_lines = self.bridge.cv2_to_imgmsg(img_out_lines, "bgr8")
        ros_img_lines = self.bridge.cv2_to_imgmsg(img_out_lines, "bgr8")
        ros_hough_img = self.bridge.cv2_to_imgmsg(img_out_hough, "bgr8")
        ros_sobel_img = self.bridge.cv2_to_imgmsg(range_sobely_col, "bgr8")
        ros_img_area = self.bridge.cv2_to_imgmsg(img_out_area, "bgr8")

        #publishers
        self.pub1.publish(ros_img_lines)
        self.pub2.publish(lvl_lines)
        self.pub3.publish(ros_sobel_img)
        self.pub4.publish(ros_hough_img)
        self.pub5.publish(lvl_area)
        self.pub6.publish(ros_img_area)
        self.pub7.publish(output)


    def reject_outliers(self,data, m = 1.):
        d = np.abs(data - np.median(data))
        mdev = np.median(d)
        s = d/mdev if mdev else 0.
        return data[s<m]

    def find_lines(self,img_hls,img_crop):
        #HLS: 0: HUE, 1: LIGHTNESS, 2: SATURATION
        L_y = len(img_hls[:,0])
        L_x = len(img_hls[0,:])

        #Sobel Y---------
        sobel_kernel = 11
        kernel_size = 3
        #Gaussian Blur
        blur_gray = cv2.GaussianBlur(img_hls[:,:,1],(kernel_size, kernel_size),0)
        #Sobel Gradient in Y
        sobely = cv2.Sobel(blur_gray, cv2.CV_64F,0,1,ksize = sobel_kernel)
        abs_sobely = np.absolute(sobely)
        scaled_sobely=np.uint8(255*abs_sobely/np.max(abs_sobely))

        #range filter
        range_sobely = cv2.inRange(scaled_sobely,45,255)

        #Hough lines
        rho = 1 # distance resolution in pixels of the Hough grid
        theta = np.pi/180 # angular resolution in radians of the Hough grid
        threshold = 12    # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 45 #minimum number of pixels making up a line
        max_line_gap = 10   # maximum gap in pixels between connectable line segments
        lines = cv2.HoughLinesP(range_sobely, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
        #img with just one line
        # line_img = np.zeros((range_sobely.shape[0],range_sobely.shape[1],3), dtype=np.uint8)
        #img with all hough lines
        line_img_hough = np.zeros((range_sobely.shape[0],range_sobely.shape[1],3), dtype=np.uint8)
        #     print(lines)
        Y = [] #yvalues
        W = [] #weights
        mY = 0

        #iterate through the lines and filter by the slope
        if lines is not None:
            if len(lines) > 0:
                for line in lines:
                    x1,y1,x2,y2 = line[0]
                    #filter by slope
                    m = (y2-y1)/(x2-x1)
                    if abs(m) < 0.27 and x2 > L_x / 2.0:
                        cv2.line(line_img_hough,(x1,y1),(x2,y2),(255,0,255),2)
                        Y.append(y1)
                        Y.append(y2) #finding more lines in the same area might weight the result in that region
                # plt.imshow(line_img)
        #find the mean Y value
        if len(Y) > 0:
            # mY = int(np.mean(self.reject_outliers(np.array(Y))))  # average of all Y values
            mY = int(np.mean(Y))
        # cv2.line(line_img,(0,mY),(650,mY),(255,255,0),5)
        # img_out = cv2.addWeighted(np.uint8(img_crop*255.0),0.8,line_img, 1.,0.)
        img_out_hough = cv2.addWeighted(img_crop,0.8,line_img_hough, 1.,0.)

        return img_out_hough, range_sobely, mY

    def find_area(self,img_hsv, img_gray):
        #threshold in HSV
        lower_hsv = np.array([150, 0, 0])
        upper_hsv = np.array([200, 255, 255]) #red
        mask_hsv = cv2.inRange(img_hsv, lower_hsv, upper_hsv)
        mask_hsv_3 = np.dstack([mask_hsv*0,mask_hsv,mask_hsv])
        #threshold in grayscale
        lower_gray = 200
        upper_gray = 255
        mask_gray = cv2.inRange(np.uint8(img_gray*255),lower_gray,upper_gray)
        mask_gray_3 = np.dstack([mask_gray*0,mask_gray*0,mask_gray*1])
        #combine masks
        mask_and = cv2.bitwise_and(mask_gray,mask_hsv)
        mask_and_3 = np.dstack([mask_and*0,mask_and*0,mask_and*1])
        #match filter - overkill, could just add lines
        # target = mask_and.astype(np.float)
        # target = mask_gray.astype(np.float)
        target = mask_hsv.astype(np.float)

        L_y = len(target[:,0])
        L_x = len(target[0,:])
        # print(L_x,L_y)
        win_y = 5
        win_x = L_x

        #detection threshold = increase probability of line detection, sensitivity to noise
        peak_thresh = 10 #threshold for detecting a peak in a single convolution line Note win_x size = 20  12/20 = 60% of pixels
        peak_thresh = peak_thresh*win_y  #threshold for detecting a peak over the y_window

        x_pos = list()
        y_pos = list()

        #convolution window. zeros with a ones in the center of length win_x
        window = np.zeros_like(target[:,0])

        window[int(L_y/2-win_y/2):int(L_y/2+win_y/2)] = 1.0

        conv=np.zeros_like(window)
        # for xi in range(int(L_x/2-1),int(L_x/2+1)):
        for xi in range(L_x):
            line_conv = np.convolve(target[:,xi],window,mode='same')
            conv = conv + line_conv
        #     plt.plot(line_conv)

        line_thresh = 0.8
        beer_line = 0
        for yi in range(L_y):
            if (conv[yi] > (line_thresh * 255.0 * L_x * win_y/2.0)):
                beer_line = yi;
                break


        # return beer_line,mask_and_3
        # mask_gray_3
        # return beer_line,mask_gray_3
        return beer_line,mask_hsv_3


if __name__ == '__main__':
    try:
        BeerDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start Beer Detector node.')
