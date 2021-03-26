import cv2
import imutils

# Read the Image
image = cv2.imread('shapes.jpg')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Find the threshold
# thresh_image = cv2.threshold(gray_image, 243, 255, cv2.THRESH_BINARY_INV)[1]

# Using Canny Detection
canny = cv2.Canny(gray_image, 160, 180)
canny_dilated = cv2.dilate(canny, (11,11), iterations=2)

cv2.imshow("Canny", canny_dilated)
cv2.waitKey(0)

# Remove any noise if present
# no_noise_image = thresh_image.copy()
#
# no_noise_image = cv2.erode(no_noise_image, None, iterations=1)
# cv2.imshow("Clean", no_noise_image)
# cv2.waitKey(0)

# Use the threshold to find the contours
cnts = cv2.findContours(canny_dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

print(f'There are {len(cnts)} shapes')

# Draw the contours
output = image.copy()

for c in cnts:
    cv2.drawContours(output, [c], -1, (0, 0, 255), 3)
    cv2.imshow("Outline!", output)

cv2.waitKey(0)