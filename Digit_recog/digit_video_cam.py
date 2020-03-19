# -*- coding: utf-8 -*-
"""
@author: aman_pal
"""
# Import the modules
import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np

SZ = 28

#area of rectangle
def areaOfRectangle(width, length):
    area = float(width) * float(length)
    return area

#deskew
def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img,M,(SZ, SZ))
    return img

# Load the classifier
clf = joblib.load("digits_cls.pkl")

# Read the input image 
#im = cv2.imread("two2.jpg")

'''


This is warning 

'''
im = cv2.imread("two2.jpg")
# Video Camera testing
cap = cv2.VideoCapture(0)

while(1):
    ret, im = cap.read()
    # Convert to grayscale and apply Gaussian filtering
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
    # Threshold the image
    ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the image
    _, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get rectangles contains each contour
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]

    #deskew function calling
    im_gray = deskew(im_gray)
    # For each rectangular region, calculate HOG features and predict
    # the digit using Linear SVM.
    for rect in rects:
        # Draw the rectangles
        if areaOfRectangle(rect[2], rect[3]) > 100:
            cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
            # Make the rectangular region around the digit
            leng = int(rect[3] * 1.6)
            pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
            pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
            roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
            # Resize the image
            try:
                roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
                roi = cv2.dilate(roi, (3, 3))
                roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(7, 7), cells_per_block=(1, 1), visualize=False)
                nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
                cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
            except Exception as e: 
                print(str("Error"))

    # Display the frame
    cv2.imshow('Resulting Image with Rectangular ROIs',im) 
	
    # Wait for 25ms
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
		
# release the camera from video capture
cap.release() 

# De-allocate any associated memory usage 
cv2.destroyAllWindows() 

'''


This is warning 

'''