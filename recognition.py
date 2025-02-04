import cv2
import numpy as np
import os

# Load trained recognizer model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('H:/sem-6/trainer/trainer.yml') 

# Load Haar cascade for face detection
cascadePath = "H:/sem-6/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# Define font for text overlay
font = cv2.FONT_HERSHEY_SIMPLEX

# ID counter (number of persons included)
id = 5  # Adjust according to the dataset

# List of names corresponding to recognized IDs (index 0 left empty)
names = ['', 'bhavya', 'anju','Hamza','Megha','Dev']

# Initialize webcam
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # Set video width
cam.set(4, 480)  # Set video height

# Define min window size to be recognized as a face
minW = int(0.1 * cam.get(3))
minH = int(0.1 * cam.get(4))

while True:
    ret, img = cam.read()
    if not ret:
        print("\n[ERROR] Failed to capture image")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(minW, minH)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        # Confidence level check (lower confidence means better match)
        if confidence < 100:
            name = names[id] if id < len(names) else "Unknown"
            confidence_text = " {0}%".format(round(100 - confidence))
        else:
            name = "Unknown"
            confidence_text = " {0}%".format(round(100 - confidence))

        # Display name and confidence on screen
        cv2.putText(img, str(name), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence_text), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    cv2.imshow('camera', img)

    # Press 'ESC' to exit
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

# Cleanup
print("\n[INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
