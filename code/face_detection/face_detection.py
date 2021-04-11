import cv2 as cv
import dlib
import numpy as np
import matplotlib.pyplot as plt
from imutils import face_utils

main_image = cv.imread('face.jpg')
gray = cv.imread('face.jpg', cv.IMREAD_GRAYSCALE)

video_capture = cv.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    face_detect = dlib.get_frontal_face_detector()
    rects = face_detect(gray_frame, 1)
    for (i, rect) in enumerate(rects):
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 3)
    cv.imshow("Face detection!", frame)

    if cv.waitKey(1) == ord('q'):
        break
