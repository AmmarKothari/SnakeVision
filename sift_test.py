import cv2
import numpy as np
# import matplotlib.pyplot as plt


print(cv2.__version__)
img = cv2.imread('Image_01.png')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)
img=cv2.drawKeypoints(gray,kp)
cv2.imwrite('sift_keypoints.jpg',img)



# plt.imshow(img)
# plt.show()
cv2.imwshow('sift_keypoints.jpg',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
