import cv2
import numpy as np
from PIL import Image  # Import from Pillow package
import os

# Path for face image database
path = 'H:/sem-6/dataset'

# Initialize the recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load Haar cascade for face detection
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


# Function to get the images and label data
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:
        # Convert image to grayscale
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')

        # Extract ID from the image filename
        id = int(os.path.split(imagePath)[-1].split(".")[1])

        # Detect faces
        faces = detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)

    return faceSamples, ids


print("\n[INFO] Training faces. This may take a few seconds...")

# Get training data
faces, ids = getImagesAndLabels(path)

# Train the recognizer
recognizer.train(faces, np.array(ids))

# Save the trained model
recognizer.write('trainer/trainer.yml')  # `recognizer.save()` works on Mac, but not on Raspberry Pi

# Print the number of faces trained
print("\n[INFO] {0} faces trained. Exiting Program.".format(len(np.unique(ids))))
