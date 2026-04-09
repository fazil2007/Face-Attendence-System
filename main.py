import cv2
from datetime import datetime

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Start webcam
cap = cv2.VideoCapture(0)

# Create file if not exists
with open("attendance.csv", "a") as f:
    pass

def markAttendance():
    with open("attendance.csv", "r+") as f:
        lines = f.readlines()

        # Add header if empty
        if len(lines) == 0:
            f.write("Status,Date Time\n")

        # Mark only once
        if len(lines) < 2:
            now = datetime.now()
            dtString = now.strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"Present,{dtString}\n")

already_marked = False

while True:
    success, img = cap.read()

    if not success:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Draw rectangle
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display text
        cv2.putText(img, "Face Detected", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

        # Mark attendance only once
        if not already_marked:
            markAttendance()
            already_marked = True

    cv2.imshow("Face Attendance System", img)

    # Press ENTER to exit
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()
