#!/usr/bin/env python
import rospy
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from cv_bridge import CvBridge
import cv2
import pickle

class BeerDetector(object):
    def __init__(self):
        rospy.init_node('beer_detector')

        #variables
        self.camera_image = None
        self.bridge = CvBridge()

        #subscribers & publishers
        sub1 = rospy.Subscriber('/camera/image_color', Image, self.image_cb, queue_size=1)
        self.pub1 = rospy.Publisher('/spb/image_color', Image, queue_size=1)
        self.pub2 = rospy.Publisher('/spb/level', Int32, queue_size=1)
        self.pub3 = rospy.Publisher('/spb/sobely', Image, queue_size=1)
        self.pub4 = rospy.Publisher('/spb/houghlines', Image, queue_size=1)

        #load camera calibration
        camera_cal = pickle.load( open( "/home/robbie/spb_ws/src/spb_camera/camera_cal2.p", "rb" ) )
        self.ret = camera_cal[0]
        self.mtx = camera_cal[1]
        self.dist = camera_cal[2]

        rospy.spin()

    def image_cb(self, msg):

        #convert ROS img msg to OpenCV
        cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        #undistort
        img_dst = cv2.undistort(cv_img, self.mtx, self.dist, None, self.mtx)
        #crop
        img_crop = img_dst[200:800,300:1000]
        #grayscale
        img_gray = cv2.cvtColor(img_crop,cv2.COLOR_BGR2GRAY)
        #HLS
        img_hls = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HLS).astype(np.float)

        #Sobel Y
        sobel_kernel = 11
        kernel_size = 5
        blur_gray = cv2.GaussianBlur(img_hls[:,:,1],(kernel_size, kernel_size),0)
        sobely = cv2.Sobel(blur_gray, cv2.CV_64F,0,1,ksize = sobel_kernel)
        abs_sobely = np.absolute(sobely)
        scaled_sobely=np.uint8(255*abs_sobely/np.max(abs_sobely))

        #range filter
        range_sobely = cv2.inRange(scaled_sobely,165,255)

        #Hough lines
        rho = 1 # distance resolution in pixels of the Hough grid
        theta = np.pi/180 # angular resolution in radians of the Hough grid
        threshold = 12    # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 45 #minimum number of pixels making up a line
        max_line_gap = 20    # maximum gap in pixels between connectable line segments
        lines = cv2.HoughLinesP(range_sobely, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
        #img with just one line
        line_img = np.zeros((range_sobely.shape[0],range_sobely.shape[1],3), dtype=np.uint8)
        #img with all hough lines
        line_img_hough = np.zeros((range_sobely.shape[0],range_sobely.shape[1],3), dtype=np.uint8)
        #     print(lines)
        Y = []
        mY = 0

        if lines is not None:
            if len(lines) > 0:
                for line in lines:
                    x1,y1,x2,y2 = line[0]
                    #filter by slope
                    m = (y2-y1)/(x2-x1)
                    if abs(m) < 0.23:
                        cv2.line(line_img_hough,(x1,y1),(x2,y2),(255,0,255),2)
                        Y.append(y1)
                        Y.append(y2)
                # plt.imshow(line_img)
        #find the mean Y value
        if len(Y) > 0:
            mY = int(np.mean(Y))
        cv2.line(line_img,(0,mY),(650,mY),(255,255,0),2)
        img_out = cv2.addWeighted(np.uint8(img_crop*255.0),0.8,line_img, 1.,0.)
        img_out_hough = cv2.addWeighted(np.uint8(img_crop*255.0),0.8,line_img_hough, 1.,0.)




        #convert cv2 image to ROS img
        ros_img = self.bridge.cv2_to_imgmsg(img_out, "bgr8")
        ros_hough_img = self.bridge.cv2_to_imgmsg(img_out_hough, "bgr8")

        range_sobely_col = np.dstack([range_sobely*0,range_sobely,range_sobely])
        ros_sobel_img = self.bridge.cv2_to_imgmsg(range_sobely_col, "bgr8")

        #publishers
        self.pub1.publish(ros_img)
        self.pub2.publish(mY)
        self.pub3.publish(ros_sobel_img)
        self.pub4.publish(ros_hough_img)


if __name__ == '__main__':
    try:
        BeerDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start Beer Detector node.')
