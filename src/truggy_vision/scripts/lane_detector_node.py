#!/usr/bin/env python

import sys
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class LaneDetector(object):
    """
    Lane detector class used to detect road markup
    Based on: https://medium.com/computer-car/
    my-lane-detection-project-for-the-self-driving-
    car-nanodegree-by-udacity-36a230553bd3
    """
    def __init__(self):
        self.bridge = CvBridge()
        self.img_sub = rospy.Subscriber("/cv_camera/image_raw", Image, self.image_callback)

    def image_callback(self, ros_img):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(ros_img, "bgr8")
        except CvBridgeError as e:
            print(e)
        blurred = cv2.GaussianBlur(cv_image, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 100)
        minLineLength = 300
        maxLineGap = 10
        lines = cv2.HoughLinesP(edges,1,3.14/180,1,minLineLength,maxLineGap)
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(cv_image,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.imshow("Result", cv_image)
        cv2.waitKey(3)

def main(args):
  lane_detector = LaneDetector()
  rospy.init_node('lane_detector')
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)