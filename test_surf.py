import cv2
# import cv2.cv as cv
import numpy as np

import pdb

filename = 'Image_01.png'



img = cv2.imread(filename)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
surfDetector = cv2.FeatureDetector_create("SURF")
surfDescriptorExtractor = cv2.DescriptorExtractor_create("SURF")
keypoints = surfDetector.detect(im)
(keypoints, descriptors) = surfDescriptorExtractor.compute(im,keypoints)