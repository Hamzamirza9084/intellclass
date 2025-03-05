import cv2
import os
from facenet_pytorch import MTCNN

def get_next_unique_id(counter_file):
    """Reads the counter from file, increments it, and returns the new unique ID."""
    if os.path.exists(counter_file):
        with open(counter_file, 'r') as f:
            try:
                unique_id = int(f.read().strip())
            except ValueError:
                unique_id = 0
    else:
        unique_id = 0
    unique_id += 1
    with open(counter_file, 'w') as f:
        f.write(str(unique_id))
    return unique_id

# Directory to store dataset and counter file
dataset_path = "teacher_dataset"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
counter_file = os.path.join(dataset_path, "counter.txt")

# Generate a unique ID starting from 1 and incrementing
unique_id = get_next_unique_id(counter_file)

# Get teacher details for the dataset
professor_name = input("Enter Professor Name: ")
department = input("Enter Department: ")

# Create a structured folder path with the auto-generated unique ID and teacher details
teacher_folder = os.path.join(dataset_path, f"{unique_id}_{professor_name}_{department}")
if not os.path.exists(teacher_folder):
    os.makedirs(teacher_folder)

# Initialize MTCNN for face detection
mtcnn = MTCNN()

# Open webcam
cap = cv2.VideoCapture(0)
count = 0

while count < 50:  # Capture 50 images per professor
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face, _ = mtcnn(frame_rgb, return_prob=True)

    if face is not None:
        img_path = os.path.join(teacher_folder, f"{unique_id}_{count}.jpg")
        cv2.imwrite(img_path, frame)
        count += 1
        print(f"Saved {img_path}")

    cv2.imshow("Professor Face Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Professor dataset collection complete!")
