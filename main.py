import cv2
import numpy as np
import os
from datetime import datetime
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

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

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))

# Attendance
def markAttendance(name):
    with open('attendance.csv', 'a') as f:
        now = datetime.now()
        dt = now.strftime('%H:%M:%S')
        f.write(f"{name},{dt}\n")

# GUI Setup
root = tk.Tk()
root.title("Face Recognition Attendance System")
root.geometry("800x600")

label = tk.Label(root)
label.pack()

cap = cv2.VideoCapture(0)
marked = set()

def start_camera():
    def update_frame():
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces_rect = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces_rect:
                face_roi = gray[y:y+h, x:x+w]
                label_id, confidence = recognizer.predict(face_roi)

                name = names[label_id]

                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(frame, name, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (255,255,255), 2)

                if name not in marked:
                    markAttendance(name)
                    marked.add(name)

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            label.imgtk = imgtk
            label.configure(image=imgtk)

        label.after(10, update_frame)

    update_frame()

def exit_app():
    cap.release()
    root.destroy()

# Buttons
btn_start = tk.Button(root, text="Start Camera", command=start_camera, font=("Arial", 14))
btn_start.pack(pady=10)

btn_exit = tk.Button(root, text="Exit", command=exit_app, font=("Arial", 14))
btn_exit.pack(pady=10)

root.mainloop()
