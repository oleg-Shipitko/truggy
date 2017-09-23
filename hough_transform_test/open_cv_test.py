import cv2
import numpy as np

img = cv2.imread('road_test_1.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,450,700,apertureSize = 3)
cv2.imwrite('edges.jpg',edges)
minLineLength = 300
maxLineGap = 10
lines = cv2.HoughLinesP(edges,1,np.pi/180,1,minLineLength,maxLineGap)
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imwrite('houghlines.jpg',img)