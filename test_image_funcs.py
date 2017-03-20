import image_funcs as IMS
import cv2
import pdb


# filename = 'Image_01.png'
# img = cv2.imread(filename)

cap = cv2.VideoCapture('IMG_0344.1.m4v')

while(True):

	# Capture frame-by-frame
	ret, frame = cap.read()

	# img_t = IMS.Hoguh(img)
	# img_t = IMS.Harris(frame) #
	# img_t = IMS.find_squaers(frame)
	# img_t = IMS.SIFTdetect(frame)
	img_t = IMS.SURFdetect(frame)



	cv2.imshow('detected circles',img_t)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cv2.waitKey(0)
cv2.destroyAllWindows()

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()