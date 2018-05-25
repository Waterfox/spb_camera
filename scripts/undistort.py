#!/usr/bin/env python
import rospy
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
        dst = cv2.undistort(cv_img, self.mtx, self.dist, None, self.mtx)
        #convert cv2 image to ROS img
        ros_img = self.bridge.cv2_to_imgmsg(dst, "bgr8")
        #publishers
        self.pub1.publish(ros_img)


if __name__ == '__main__':
    try:
        BeerDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start Beer Detector node.')
