import imutils
import cv2

image = cv2.imread("tetrominoes.png")
# cv2.imshow("Image Before", image)
# cv2.waitKey(0)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Image After", gray_image)
cv2.waitKey(0)

# mess around w the minVal and maxVal
# edged_image = cv2.Canny(gray_image, 30, 150)
# cv2.imshow("Edged Image", edged_image)
# cv2.waitKey(0)

# (img, thresh, maxVal, type)
# THRESH_BINARY if pixel color is > thresh ? maxVal : 0
# THRESH_BINARY_INV if pixel color is > thresh ? 0 : maxVal
thresh_image = cv2.threshold(gray_image, 250, 255, cv2.THRESH_BINARY_INV)[1]
cv2.imshow("Thresh", thresh_image)
cv2.waitKey(0)

# threshold image, hierarchy type, type of storing contours (boundary points)
# cv2.RETR_LIST draws contours for both internal and external objects
# RETR_EXTERNAL draws contours for external objects

# findContours searches for white pixels in the foreground
# cnts = cv2.findContours(thresh_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# print(len(cnts))
# cnts = imutils.grab_contours(cnts)
# print(len(cnts))
#
# output = image.copy()
#
# for c in cnts:
#     cv2.drawContours(output, [c], -1, (240, 0, 159), 3)
#     cv2.imshow("Contours", output)
#     cv2.waitKey(0)

# Reduces size around the objects
mask = thresh_image.copy()
mask = cv2.erode(mask, None, iterations=5)
cv2.imshow("MASK", mask)
cv2.waitKey(0)

dilate = thresh_image.copy()
dilate = cv2.dilate(dilate, None, iterations=1)
cv2.imshow("Dilate", dilate)
cv2.waitKey(0)

mask = thresh_image.copy()
output = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("Mask", output)
cv2.waitKey()

# LEARNING
# Convert colored picture to grayscale picture using cv2.cvtColor
# We can edge the picture using cv2.Canny
# We can find the threshold of the picture (basically make objects a different color from background) yusing cv2.thresh
# We can apply contours with the help of the grayscale picture cv2.findContours and imutils.grap_contours and cv2.drawContours
