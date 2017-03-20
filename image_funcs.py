import cv2
import numpy as np
import pdb












def Hough(img):
	gray = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
	gimg = cv2.medianBlur(gray,5)

	circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,50,
								param1=100,param2=40,minRadius=0,maxRadius=100)

	circles = np.uint16(np.around(circles))
	for i in circles[0,:]:
		# draw the outer circle
		cv2.circle(gimg,(i[0],i[1]),i[2],(0,255,0),2)
		# draw the center of the circle
		cv2.circle(gimg,(i[0],i[1]),2,(0,0,255),3)

	return cimg

def Harris(img):
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	gray = np.float32(gray)
	dst = cv2.cornerHarris(gray,2,3,0.04)

	#result is dilated for marking the corners, not important
	dst = cv2.dilate(dst,None)

	# Threshold for an optimal value, it may vary depending on the image.
	img[dst>0.01*dst.max()]=[0,0,255]

	return img

def find_squares(image):
	upper = np.array([50, 50, 50])
	lower = np.array([0, 0, 0])

	# upper = np.array([160, 170, 200])
	# lower = np.array([100,110, 180])
	mask = cv2.inRange(image, lower, upper)
	output = cv2.bitwise_and(image, image, mask = mask)
	 
	# find contours in the masked image and keep the largest one
	(_, cnts, _) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	c = max(cnts, key=cv2.contourArea)
	d = sorted(cnts, key=cv2.contourArea)

	# pdb.set_trace()
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
	
	return new_img

def SIFTdetect(image):
	# pdb.set_trace()
	gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	img = gray
	sift = cv2.xfeatures2d.SIFT_create()
	kp = sift.detect(gray,None) 
	cv2.drawKeypoints(gray,kp, img)
	# cv2.imwrite('sift_keypoints.jpg',img)
	return img


def SURFdetect(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	img = gray
	surfDetector = cv2.FeatureDetector_create("SURF")
	surfDescriptorExtractor = cv2.DescriptorExtractor_create("SURF")
	keypoints = surfDetector.detect(im)
	(keypoints, descriptors) = surfDescriptorExtractor.compute(im,keypoints)
	cv2.drawKeypoints(gray, keypoints, img)

	return img