#!/usr/bin/env python
import rospy
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from std_msgs.msg import UInt16
from cv_bridge import CvBridge
import cv2
import pickle
import time
# import serial

class surface_line(object):
    def __init__(self):
        self.locked_on = False
        self.detected = False #was the most recent frame detected
        self.y_val = 0
        self.confidence = 0
        self.iteration = 0
        self.mv_avg = 4

        self.max_confidence = 20
        self.locked_thresh = 5

    def update_sl(self,lvl): #update the tracker
        if lvl > 0:  #if we find a line, raise the confidence
            self.confidence = min(self.max_confidence, self.confidence + 1)
            self.iteration = min(self.mv_avg,self.iteration + 1)
            #Moving average
            self.y_val = self.y_val*(self.iteration - 1) / self.iteration + lvl / (self.iteration)
            #last lvl
            # self.y_val = lvl
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


        self.USE_LINES = False
        self.USE_AREA = True
        self.USE_AI = False

        if self.USE_AI:
            self.beerClassifier = beerClassifier()
            from beer_classifier import beerClassifier
            while self.beerClassifier.model_ready == False:
                time.sleep(0.5)

        #subscribers & publishers
        sub1 = rospy.Subscriber('/camera/image_color', Image, self.image_cb, queue_size=1)
        if self.USE_LINES:
            self.pub1 = rospy.Publisher('/spb/image_lines', Image, queue_size=1)
            self.pub2 = rospy.Publisher('/spb/level_lines', Int32, queue_size=1)
            self.pub3 = rospy.Publisher('/spb/sobely', Image, queue_size=1)
            self.pub4 = rospy.Publisher('/spb/houghlines', Image, queue_size=1)
        if self.USE_AREA:
            self.pub5 = rospy.Publisher('/spb/level_area', Int32, queue_size=1)
            self.pub6 = rospy.Publisher('/spb/image_area', Image, queue_size=1)
        self.pub7 = rospy.Publisher('/spb/lvl', UInt16, queue_size=1)
        self.pub8 = rospy.Publisher('/spb/image_output', Image, queue_size=1)
        if self.USE_AI:
            self.pub9 = rospy.Publisher('/spb/image_ai_beer', Image, queue_size=1)
            self.pub10 = rospy.Publisher('/spb/image_ai_foam', Image, queue_size=1)
            self.pub11 = rospy.Publisher('/spb/image_ai_glass', Image, queue_size=1)
        #load camera calibration
        camera_cal = pickle.load( open( "/home/robbie/spb_ws/src/spb_camera/camera_cal2.p", "rb" ) )
        self.ret = camera_cal[0]
        self.mtx = camera_cal[1]
        self.dist = camera_cal[2]
        self.line_tracker = surface_line()
        self.area_tracker = surface_line()

        self.img_shape = (964,1296)
        self.cropTop = 400
        self.cropBot = 1000
        self.cropLeft = 400
        self.cropRight = 750



        rate = rospy.Rate(20)
        rate.sleep()
        rospy.spin()

    def pix2dist(self,lvl_pix):
        #converts the pixel measurement from the camera into a distance in mm from the top of the machine
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
        if self.USE_AI:
            cv_img_rgb = self.bridge.imgmsg_to_cv2(msg, "rgb8") #for AI
        # if self.USE_LINES or self.USE_AREA:
        #undistort
        img_dst = cv2.undistort(cv_img, self.mtx, self.dist, None, self.mtx)
        #crop
        img_crop = img_dst[self.cropTop:self.cropBot,self.cropLeft:self.cropRight]
        #grayscale
        img_gray = cv2.cvtColor(img_crop,cv2.COLOR_BGR2GRAY)
        #HLS
        img_hls = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HLS).astype(np.float)
        #HSV
        img_hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)


        #Gradient Line filter --------------
        if self.USE_LINES:
            img_out_hough, range_sobely, lvl_lines = self.find_lines(img_hls,img_crop)
            self.line_tracker.update_sl(lvl_lines)

            #make an image and add the level line to it
            line_img1 = np.zeros((img_gray.shape[0],img_gray.shape[1],3), dtype=np.uint8)
            cv2.line(line_img1,(0,lvl_lines),(650,lvl_lines),(255,255,0),5)
            # img_out_lines = cv2.addWeighted(np.uint8(img_crop*255.0),0.8,line_img1, 1.,0.)
            img_out_lines = cv2.add(np.uint8(img_crop*255.0),line_img1)
            # img_out_lines = cv2.addWeighted(img_crop,0.8,line_img1, 1.,0.)
            # img_out_lines = cv2.addWeighted(img_crop,0.8,line_img1, 1.,0.)
            range_sobely_col = np.dstack([range_sobely*0,range_sobely,range_sobely])

        #Area filter ------------------
        if self.USE_AREA:
            lvl_area, mask_and_3 = self.find_area(img_hsv, img_gray)
            self.area_tracker.update_sl(lvl_area)

            #make an image and add the level line to it
            area_img = np.zeros((img_gray.shape[0],img_gray.shape[1],3), dtype=np.uint8)
            cv2.line(area_img,(0,lvl_area),(area_img.shape[0],lvl_area),(255,0,0),5)
            # img_out_area = cv2.addWeighted(img_crop,0.8,area_img, 1.,0.)
            img_out_area = cv2.add(img_crop,area_img)
            # img_out_area = cv2.addWeighted(img_out_area,0.8,mask_and_3, 1.,0.)
            img_out_area = cv2.add(img_out_area,mask_and_3)

        #AI (semantic segmentation) filter
        if self.USE_AI:
            img_beer, img_foam, img_glass = self.beerClassifier.run_classifier(cv_img_rgb)
            #resize the classified image, undistort it, crop it, find the top of the beer

            img_beer = np.array(img_beer)
            img_foam = np.array(img_foam)
            img_glass = np.array(img_glass)
            # Convert RGB to BGR
            img_beer = img_beer[:, :, ::-1].copy()
            img_foam = img_foam[:, :, ::-1].copy()
            img_glass = img_glass[:, :, ::-1].copy()


            # img_beer_res = cv2.resize(img_beer,self.img_shape)
            # img_beer_dst = cv2.undistort(img_beer_res, self.mtx, self.dist, None, self.mtx)
            # img_beer_crop = img_beer_dst[self.cropTop:self.cropBot,self.cropLeft:self.cropRight]



        #combine surface lvls from lines and area filters

        #Take the output of the tracker
        if self.line_tracker.locked_on == True and self.area_tracker.locked_on == True:
            pix_output = (self.line_tracker.y_val + self.area_tracker.y_val)*0.5
        elif self.line_tracker.locked_on == True and self.area_tracker.locked_on == False:
            pix_output = self.line_tracker.y_val
        elif self.line_tracker.locked_on == False and self.area_tracker.locked_on == True:
            pix_output = self.area_tracker.y_val
        else:
             pix_output = 0

        #make the final output image
        out_img = np.zeros((img_crop.shape[0],img_crop.shape[1],3), dtype=np.uint8)
        cv2.line(out_img,(0,int(pix_output)),(650,int(pix_output)),(250,0,180),5)
        img_output = cv2.addWeighted(img_crop,0.8,out_img, 1.,0.)

        # output = self.area_tracker.y_val
        output_mm = self.pix2dist(pix_output)

        #temporary fix for area only output
        if output_mm <146.0: output_mm = 250
        print output_mm



        #convert cv2 image to ROS img
        if self.USE_LINES:
            ros_img_lines = self.bridge.cv2_to_imgmsg(img_out_lines, "bgr8")
            ros_hough_img = self.bridge.cv2_to_imgmsg(img_out_hough, "bgr8")
            ros_sobel_img = self.bridge.cv2_to_imgmsg(range_sobely_col, "bgr8")
        if self.USE_AREA:
            ros_img_area = self.bridge.cv2_to_imgmsg(img_out_area, "bgr8")
        if self.USE_AI:
            ros_img_ai_beer = self.bridge.cv2_to_imgmsg(img_beer, "bgr8")
            ros_img_ai_foam = self.bridge.cv2_to_imgmsg(img_foam, "bgr8")
            ros_img_ai_glass = self.bridge.cv2_to_imgmsg(img_glass, "bgr8")

        ros_img_output = self.bridge.cv2_to_imgmsg(img_output, "bgr8")

        #publishers

        if self.USE_LINES:
            self.pub1.publish(ros_img_lines)
            self.pub2.publish(lvl_lines)
            self.pub3.publish(ros_sobel_img)
            self.pub4.publish(ros_hough_img)
        if self.USE_AREA:
            self.pub5.publish(lvl_area)
            self.pub6.publish(ros_img_area)
        if self.USE_AI:
            self.pub9.publish(ros_img_ai_beer)
            self.pub10.publish(ros_img_ai_foam)
            self.pub11.publish(ros_img_ai_glass)

        # if not self.USE_AI:
        self.pub7.publish(output_mm)
        self.pub8.publish(ros_img_output)


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
                    if abs(m) < 0.20 and x2 > L_x / 2.0:
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
        #red: h = 150:200
        lower_hsv = np.array([0, 0, 0])
        upper_hsv = np.array([255, 150, 150]) #red
        mask_hsv = cv2.inRange(img_hsv, lower_hsv, upper_hsv)
        mask_hsv_3 = np.dstack([mask_hsv*0,mask_hsv,mask_hsv])
        #threshold in grayscale
        # lower_gray = 200
        # upper_gray = 255
        # mask_gray = cv2.inRange(np.uint8(img_gray*255),lower_gray,upper_gray)
        # mask_gray_3 = np.dstack([mask_gray*0,mask_gray*0,mask_gray*1])
        #combine masks
        # mask_and = cv2.bitwise_and(mask_gray,mask_hsv)
        # mask_and_3 = np.dstack([mask_and*0,mask_and*0,mask_and*1])
        #match filter - overkill, could just add lines
        # target = mask_and.astype(np.float)
        # target = mask_gray.astype(np.float)
        # target = mask_hsv.astype(np.float)
        target = mask_hsv
        L_y = len(target[:,0])
        L_x = len(target[0,:])
        # print(L_x,L_y)
        # win_y = 5
        # win_x = L_x

        #detection threshold = increase probability of line detection, sensitivity to noise
        # peak_thresh = 10 #threshold for detecting a peak in a single convolution line Note win_x size = 20  12/20 = 60% of pixels
        # peak_thresh = peak_thresh*win_y  #threshold for detecting a peak over the y_window

        # x_pos = list()
        # y_pos = list()

        line_thresh = 0.7
        beer_line = 0

        ## Convolution based line finding
        #convolution window. zeros with a ones in the center of length win_x
        # window = np.zeros_like(target[:,0])

        # window[int(L_y/2-win_y/2):int(L_y/2+win_y/2)] = 1.0

        # conv=np.zeros_like(window)
        # for xi in range(int(L_x/2-1),int(L_x/2+1)):
        # for xi in range(L_x):
            # line_conv = np.convolve(target[:,xi],window,mode='same')
            # conv = conv + line_conv
        #     plt.plot(line_conv)


        # for yi in range(L_y):
        #     if (conv[yi] > (line_thresh * 255.0 * L_x * win_y/2.0)):
        #         beer_line = yi;
        #         break

        ##Simple pixel based line finding - IMPLEMENT BETTER SEARCH
        for yi in range(0,L_y,4):
            # print sum(target[yi,:])
            # print line_thresh * L_x * 255.0
            if (sum(target[yi,:]) > line_thresh * L_x * 255.0):
                beer_line = yi
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
