import cv2 as cv
import dlib
from imutils import face_utils

# Connects the the camera which is the 0th camera accessible to the computer | instead of 0 put path to video
video_capture = cv.VideoCapture(0)


# Checks if the camera can be opened or not
if not video_capture.isOpened():
    print("Camera cannot be opened!")
    exit()

while True:
    ret, frame = video_capture.read()
    # ret is a boolean which tells if the frame is read correctly or not

    if not ret:
        print("Cannot read frame, exiting!")
        break

    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    face_detect = dlib.get_frontal_face_detector()
    rects = face_detect(gray_frame, 1)
    for rect in rects:
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 3)
    cv.imshow("Face detection!", frame)

    if cv.waitKey(1) == ord('q'):
        break


video_capture.release()
cv.destroyAllWindows()