import cv2
import numpy as np
import os
from datetime import datetime

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Prepare training data
path = 'images'
faces = []
labels = []
names = {}
label_id = 0

for file in os.listdir(path):
    img_path = os.path.join(path, file)
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in face_rects:
        faces.append(gray[y:y+h, x:x+w])
        labels.append(label_id)

    names[label_id] = os.path.splitext(file)[0]
    label_id += 1

# Train model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))

# Attendance
def markAttendance(name):
    with open('attendance.csv', 'a') as f:
        now = datetime.now()
        dt = now.strftime('%H:%M:%S')
        f.write(f"{name},{dt}\n")

# Webcam
cap = cv2.VideoCapture(0)

marked = set()

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces_rect = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces_rect:
        face_roi = gray[y:y+h, x:x+w]

        label, confidence = recognizer.predict(face_roi)

        name = names[label]

        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255,255,255), 2)

        if name not in marked:
            markAttendance(name)
            marked.add(name)

    cv2.imshow("Face Recognition Attendance", frame)

    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()
