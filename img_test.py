#!/usr/bin/env python3

import cv2

# img = cv2.imread("robot_logo.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.imread("snowleopard_cropped.jpeg", cv2.IMREAD_GRAYSCALE)
# sift = cv2.SIFT_create()
# kp_img, des_img = sift.detectAndCompute(img, None)
# img = cv2.drawKeypoints(img, kp_img, img)
cv2.imshow("Image", cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
cv2.waitKey(0)

# cv2.imwrite("snowleopard_cropped.jpeg", img[:, 50:380])