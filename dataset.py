import cv2
import os

# Initialize webcam
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # Set video width
cam.set(4, 480)  # Set video height

# Load Haar cascade classifier for face detection
face_detector = cv2.CascadeClassifier('H:/sem-6/haarcascade_frontalface_default.xml')

# Get user input for face ID
face_id = input('\nEnter user ID and press <Return>: ')

print("\n[INFO] Initializing face capture. Look at the camera and wait...")

# Initialize individual sampling face count
count = 0

# Start face detection and capture 30 images
while True:
    ret, img = cam.read()
    if not ret:
        print("\n[ERROR] Failed to capture image")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1

        # Save captured face images in the dataset folder
        cv2.imwrite(f"H:/sem-6/dataset/User.{face_id}.{count}.jpg", gray[y:y+h, x:x+w])

    cv2.imshow('image', img)

    # Press 'ESC' to exit
    k = cv2.waitKey(100) & 0xff
    if k == 27 or count >= 30:
        break

# Cleanup
print("\n[INFO] Exiting Program and cleaning up...")
cam.release()
cv2.destroyAllWindows()
