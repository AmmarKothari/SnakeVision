import numpy as np
import cv2

cap = cv2.VideoCapture('IMG_0344.1.m4v')

i_iter = 0
while(True):
	i_iter += 1
	# Capture frame-by-frame
	ret, frame = cap.read()

	# Our operations on the frame come here
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Display the resulting frame
	cv2.imshow('frame',gray)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	if i_iter == 1:
		cv2.imwrite('Image_01.png', gray)
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()