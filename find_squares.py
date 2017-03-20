# import the necessary packages
import numpy as np
import cv2
import pdb


def find_squares(frame):
	# load the games image
	# image = cv2.imread(Image_name)
	# pdb.set_trace()
	# find the red color game in the image
	upper = np.array([50, 50, 50])
	lower = np.array([0, 0, 0])
	mask = cv2.inRange(image, lower, upper)
	output = cv2.bitwise_and(image, image, mask = mask)
	 
	# find contours in the masked image and keep the largest one
	(_, cnts, _) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	c = max(cnts, key=cv2.contourArea)
	d = sorted(cnts, key=cv2.contourArea)

	pdb.set_trace()
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.05 * peri, True)

	approx_d = list()
	for d_sub in d:
		peri = cv2.arcLength(d_sub, True)
		approx_d.append(cv2.approxPolyDP(d_sub, 0.05 * peri, True))
	 
	# draw a green bounding box surrounding the red game
	# cv2.drawContours(image, [approx], -1, (0, 255, 0), 4)
	cv2.drawContours(image, approx_d[-10:-1], -1, (0, 255, 0), 4)
	new_img = image
	# new_img = np.hstack([image, output])
	cv2.imshow("Image", new_img)
	# cv2.imshow("Image", np.hstack([image, output])
	cv2.waitKey(0)
