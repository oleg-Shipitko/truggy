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
        self.canny_low = rospy.get_param('~canny_low', 25)
        self.canny_high = rospy.get_param('~canny_high', 75)


    def image_callback(self, ros_img):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(ros_img, "bgr8")
        except CvBridgeError as err:
            rospy.logwarn("CvBridge error: %s", err)
        orig_img = cv_img.copy()

        # black color mask in HSV
        # lower = np.uint8([0, 0, 0])
        # upper = np.uint8([255, 255, 75])
        # hsv_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2HSV)
        # black_mask = cv2.inRange(hsv_img, lower, upper)
        # masked_img = cv2.bitwise_and(hsv_img, hsv_img, mask = black_mask)
        # cv2.imshow("Black mask", masked_img)
        # cv_img = cv2.cvtColor(masked_img, cv2.COLOR_HSV2RGB)
        # cv2.imshow("Masked RGB", cv_img)
        img_gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
        img_blurred = cv2.GaussianBlur(img_gray, (15, 15), 0)
        img_edges = cv2.Canny(img_blurred, self.canny_low, self.canny_high)
        # cv2.imshow("Edges", img_edges)
        img_edges_roi = self.select_region(img_edges)
        # cv2.imshow("Edges roi", img_edges_roi)
        lines = cv2.HoughLinesP(img_edges_roi, 1, 3.14/180, 20,  minLineLength=100, maxLineGap=30)
        if lines is None:
            rospy.logwarn("Lines were not found, skipping image")
            return
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(cv_img,(x1,y1),(x2,y2),(0,0,255),2)
        # cv2.imshow("Lines", cv_img)
        left_lane, right_lane = self.lines_averaging(lines)
        lanes_img = self.draw_lanes(orig_img, left_lane, right_lane)
        if lanes_img is not None:
            cv2.imshow("Result", lanes_img)
        cv2.waitKey(3)


    def filter_region(self, image, vertices):
        """
        Create the mask using the vertices and apply it to the input image
        """
        mask = np.zeros_like(image)
        # cv2.imshow("mask", mask)
        cv2.fillPoly(mask, vertices, 255)        
        return cv2.bitwise_and(image, mask)

    
    def select_region(self, image):
        """
        It keeps the region surrounded by the `vertices` (i.e. polygon).  Other area is set to 0 (black).
        """
        # first, define the polygon by vertices
        rows, cols = image.shape[:2]
        bottom_left  = [cols*0.05, rows*1]
        top_left     = [cols*0.3, rows*0.5]
        bottom_right = [cols*0.95, rows*1]
        top_right    = [cols*0.7, rows*0.5] 
        # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
        vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)        
        return self.filter_region(image, vertices)


    def lines_averaging(self, lines):
        """
        Computes average lines.
        """
        left_lines    = [] # (slope, intercept)
        left_weights  = [] # (length)
        right_lines   = [] # (slope, intercept)
        right_weights = [] # (length)
        for line in lines:
            for x1,y1,x2,y2 in line:
                if x1 == x2:
                    continue
                if abs(y1 - y2) < 20:
                    continue
                slope = (y2-y1)/(x2-x1)
                intercept = y1 - slope*x1
                length = np.sqrt((y2-y1)**2+(x2-x1)**2)
                if slope < 0: # y is reversed in image
                    left_lines.append((slope, intercept))
                    left_weights.append((length))
                else:
                    right_lines.append((slope, intercept))
                    right_weights.append((length))
    
        # add more weight to longer lines    
        left_lane  = np.dot(left_weights,  left_lines) /np.sum(left_weights)  if len(left_weights) > 0 else None
        right_lane = np.dot(right_weights, right_lines)/np.sum(right_weights) if len(right_weights) > 0 else None
        # rospy.loginfo("%s", left_lane)
        # rospy.loginfo("%s", right_lane)
        return left_lane, right_lane
    
    def draw_lanes(self, image, left_lane, right_lane):
        """
        Draw lanes represented in slope and intercept
        """
        if left_lane is None or right_lane is None:
            return None
        
        y1 = image.shape[0] # bottom of the image
        y2 = 0.5 * y1        # slightly lower than the middle

        try:
            slope_l, intercept_l = left_lane
            x1_l = int((y1 - intercept_l)/slope_l)
            x2_l = int((y2 - intercept_l)/slope_l)
            y1_l = int(y1)
            y2_l = int(y2)

            slope_r, intercept_r = right_lane
            x1_r = int((y1 - intercept_r)/slope_r)
            x2_r = int((y2 - intercept_r)/slope_r)
            y1_r = int(y1)
            y2_r = int(y2)
        except ArithmeticError as err:
           rospy.loginfo("%s", err)
           return None

        left_line = ((x1_l, y1_l), (x2_l, y2_l))
        right_line = ((x1_r, y1_r), (x2_r, y2_r))

        color = [0, 255, 0]
        thickness = 20

        # make a separate image to draw lines and combine with the orignal later
        line_image = np.zeros_like(image)
        cv2.line(line_image, (x1_l, y1_l), (x2_l, y2_l), color, thickness)
        cv2.line(line_image, (x1_r, y1_r), (x2_r, y2_r),  color, thickness)
        # image1 * a + image2 * b + c
        # image1 and image2 must be the same shape.
        return cv2.addWeighted(image, 1.0, line_image, 0.95, 0.0)

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