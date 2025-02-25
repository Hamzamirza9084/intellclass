import cv2
import os
from facenet_pytorch import MTCNN

# Initialize MTCNN for face detection
mtcnn = MTCNN()

# Directory to store dataset
dataset_path = "dataset"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Get user name for dataset
user_name = input("Enter your name: ")
user_folder = os.path.join(dataset_path, user_name)
if not os.path.exists(user_folder):
    os.makedirs(user_folder)

# Open webcam
cap = cv2.VideoCapture(0)
count = 0

while count < 50:  # Capture 50 images per user
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face, _ = mtcnn(frame_rgb, return_prob=True)

    if face is not None:
        img_path = os.path.join(user_folder, f"{count}.jpg")
        cv2.imwrite(img_path, frame)
        count += 1
        print(f"Saved {img_path}")

    cv2.imshow("Face Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Dataset collection complete!")
