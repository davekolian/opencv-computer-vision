import imutils
import cv2

image = cv2.imread("byoda.jpeg")
# image is a numpy array and thats why we get height before width since we need rows and cols
(h, w, d) = image.shape
print(f"w={w}, h= {h}, d={d}")

# h = no. of rows, w = no. of cols d = no. of channels (d=3 -> rgb, d=4 -> rgba)
# x, y (axis) -> w, h = no. of cols, no. of rows

(b, g, r) = image[100, 50]  # image[y, x] or image[row, col]


print(f"{b} {g} {r}")

# Printing a 100x100 ROI (region of interest) starting from the middle of the picture
mw = w // 2
mh = h // 2

roi = image[mh:mh + 100, mw:mw + 100]  # image[startY:endY, startX:endX]

# cv2.imshow("Image", image)
# cv2.waitKey(0)

# cv2.imshow("ROI", roi)
# cv2.waitKey(0)

# resized_image = cv2.resize(image, (200, 300)) # (x, y)
# resizing an image and keeping the aspect ratio
# aspect_ratio = 300 / w
# new_height = int(aspect_ratio * h)
#
# resized_image = imutils.resize(image, width=300)

# cv2.imshow("Resized", resized_image)
# cv2.waitKey(0)

# Rotate an image 45 deg clockwise

# rotated_image = imutils.rotate_bound(image, 45)
#
# cv2.imshow("Rotated image", rotated_image)
# cv2.waitKey(0)

# blurred_image = cv2.GaussianBlur(image, (11, 11), 10)
#
# cv2.imshow("Blur3rr", blurred_image)
# cv2.waitKey(0)
#
# blurred_image = cv2.GaussianBlur(image, (11, 11), 20)
#
# cv2.imshow("Blurrr", blurred_image)
# cv2.waitKey(0)

# ORIGIN IS TO_LEFT

image_copy = image.copy()
# (x,y) top-left, bottom, right, BGR tuple, thickness
cv2.rectangle(image_copy, (320, 60), (420, 160), (0, 0, 255), 2)
cv2.putText(image_copy, "Hello World!", (320, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)

cv2.imshow("RECT", image_copy)
cv2.waitKey(0)
