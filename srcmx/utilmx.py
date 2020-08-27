import cv2
from numba import jit


def FaceDetect(frame, facecenter):
    face_cascade = cv2.CascadeClassifier()
    profileface_cascade = cv2.CascadeClassifier()

    # load the cascades
    face_cascade.load(cv2.samples.findFile('./modelmx/haarcascade_frontalface_alt.xml'))
    profileface_cascade.load(cv2.samples.findFile('./modelmx/haarcascade_profileface.xml'))

    # convert the frame into grayhist
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)

    # detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    if len(faces) == 0:
        faces = profileface_cascade.detectMultiScale(frame_gray)
    
    for x, y, w, h in faces:
        cx, cy = x + w//2, y+h//2
        if abs(cy-facecenter[1]) <= 40:
            facecenter = (cx, cy)
            cv2.rectangle(frame, (x, y), (x+w, y+h), [0, 255, 0], 2)
    (cx, cy) = facecenter
    lux, rbx = max(0, cx-150), min(380, cx+150)
    cv2.rectangle(frame, (lux, 0), (rbx, 280), [0, 0, 255], 2)
    return facecenter, frame