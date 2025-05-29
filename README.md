import cv2
import cv2.data

modelPath = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
trainedMachine = cv2.CascadeClassifier(modelPath)

camera = cv2.VideoCapture(0)
while True:
    status, frame = camera.read()
    faces = trainedMachine.detectMultiScale(frame, 1.3, 5)
    for face in faces:
        x, y, w, h = face
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 4)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow("All faces", frame)
    cv2.waitKey(1)
