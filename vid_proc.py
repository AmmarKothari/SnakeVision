import numpy as np
import cv2
import find_squares


BASIC_PLOT = False
HARRIS_PLOT = False
HOUGH_CIRCLE_PLOT = False
CONTOUR_BOUND = True

cap = cv2.VideoCapture('IMG_0344.1.m4v')

i_iter = 0
while(True):
	i_iter += 1
	# Capture frame-by-frame
	ret, frame = cap.read()

	# Our operations on the frame come here
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	if HARRIS_PLOT:
		dst = cv2.cornerHarris(gray,2,3,0.04)

		#result is dilated for marking the corners, not important
		dst = cv2.dilate(dst,None)

		# Threshold for an optimal value, it may vary depending on the image.
		frame[dst>0.01*dst.max()]=[0,0,255]

	if HOUGH_CIRCLE_PLOT:
		img = cv2.medianBlur(frame,5)
		# cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
		cimg = gray
		circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,50,
							param1=100,param2=40,minRadius=0,maxRadius=100)
		circles = np.uint16(np.around(circles))
		for i in circles[0,:]:
			# draw the outer circle
			cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
			# draw the center of the circle
			cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

	if CONTOUR_BOUND:
		find_squares(frame)

	# Display the resulting frame
	if BASIC_PLOT:
		cv2.imshow('frame',frame)
	elif HARRIS_PLOT:
		cv2.imshow('frame', gray)
	elif HOUGH_CIRCLE_PLOT:
		cv2.imshow('Detected Circles', cimg)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	if i_iter == 1:
		cv2.imwrite('Image_01.png', frame)
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()