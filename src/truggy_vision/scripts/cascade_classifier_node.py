#!/usr/bin/env python

import rospy
import cv2
import os.path
import sys
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge, CvBridgeError

class CascadeClassifier(object):
    """
    Class for road sign detection using cascade classifier
    """
    def __init__(self):
        self.pub_topic = rospy.get_param('~pub_topic', '')
        self.sub_topic = rospy.get_param('~sub_ropic', '/cv_camera/image_raw')
        self.cascade_xml = rospy.get_param('~cascade_xml', '')
        
        if self.sub_topic and self.pub_topic: 
            self.img_sub = rospy.Subscriber(self.sub_topic, Image, self.image_callback)
            self.detect_pub = rospy.Publisher(self.pub_topic, Bool)
        else:
            rospy.logerr("Topics for subscription and publishing" + 
                         "were not specified:\n %s\n %s", 
                         self.sub_topic,
                         self.pub_topic)
            sys.exit()

        self.bridge = CvBridge()
        if self.cascade_xml and \
           os.path.exists(self.cascade_xml) and \
           os.path.isfile(self.cascade_xml):
            self.cascade = cv2.CascadeClassifier(self.cascade_xml)
        else:
            rospy.logerr("Wrong xml file was specified: %s", self.cascade_xml)
            sys.exit()
            
    def image_callback(self, ros_img):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(ros_img, "bgr8")
        except CvBridgeError as err:
            rospy.logwarn("CvBridge error: %s", err)
    
        gray_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        signs = self.cascade.detectMultiScale(gray_img, 1.2, 3)

        if signs:
            sign_detected = Bool()
            sign_detected.data = True
            self.detect_pub.publish(sign_detected)

        for (x,y,w,h) in signs:
            cv2.rectangle(cv_img,(x,y),(x+w,y+h),(255,0,0),2)

        cv2.imshow('img',cv_img)
        cv2.waitKey(3)
    
def main(args):
    cascade_classifier = CascadeClassifier()
    rospy.init_node('cascade_classifier')
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
