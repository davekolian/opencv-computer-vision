import cv2 as cv

main_image = cv.imread('cup.jpg', cv.IMREAD_GRAYSCALE)
main_image = cv.resize(main_image, (0, 0), fx=0.25, fy=0.25)
blur_image = cv.GaussianBlur(main_image, (15,15), 25)
canny_image = cv.Canny(blur_image, 25, 35)

cv.imshow("Main", blur_image)
cv.imshow("Canny", canny_image)

cv.waitKey(0)
