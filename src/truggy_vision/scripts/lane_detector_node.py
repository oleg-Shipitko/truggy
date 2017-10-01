#!/usr/bin/env python

import sys
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class LaneDetector(object):
    """
    Lane detector class used to detect road markup
    Based on: https://github.com/naokishibuya/car-finding-lane-lines
    """
    def __init__(self):
        self.bridge = CvBridge()
        self.img_sub = rospy.Subscriber("/cv_camera/image_raw", Image, self.image_callback)
        self.canny_low = rospy.get_param('canny_low', 25)
        self.canny_high = rospy.get_param('canny_high', 75)


    def image_callback(self, ros_img):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(ros_img, "bgr8")
        except CvBridgeError as err:
            rospy.logwarn("CvBridge error: %s", err)
        img_gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
        img_blurred = cv2.GaussianBlur(img_gray, (15, 15), 0)
        img_edges = cv2.Canny(img_blurred, self.canny_low, self.canny_high)
        cv2.imshow("Edges", img_edges)
        img_edges_roi = self.select_region(img_edges)
        cv2.imshow("Edges roi", img_edges_roi)
        lines = cv2.HoughLinesP(img_edges_roi, 1, 3.14/180, 20,  minLineLength=100, maxLineGap=30)
        if lines is None:
            rospy.logwarn("Lines were not found, skipping image")
            return
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(cv_img,(x1,y1),(x2,y2),(0,0,255),2)
        cv2.imshow("Result", cv_img)
        cv2.waitKey(3)


    def filter_region(self, image, vertices):
        """
        Create the mask using the vertices and apply it to the input image
        """
        mask = np.zeros_like(image)
        cv2.imshow("mask", mask)
        cv2.fillPoly(mask, vertices, 255)        
        return cv2.bitwise_and(image, mask)

    
    def select_region(self, image):
        """
        It keeps the region surrounded by the `vertices` (i.e. polygon).  Other area is set to 0 (black).
        """
        # first, define the polygon by vertices
        rows, cols = image.shape[:2]
        bottom_left  = [cols*0.05, rows*1]
        top_left     = [cols*0.3, rows*0]
        bottom_right = [cols*0.95, rows*1]
        top_right    = [cols*0.7, rows*0] 
        # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
        vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
        return self.filter_region(image, vertices)


def main(args):
    lane_detector = LaneDetector()
    rospy.init_node('lane_detector')
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)