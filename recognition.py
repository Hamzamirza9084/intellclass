import numpy as np
import cv2
import torch
import csv
import time
from datetime import datetime
from facenet_pytorch import InceptionResnetV1, MTCNN
from numpy.linalg import norm

# Initialize MTCNN for face detection and InceptionResnetV1 for embeddings
mtcnn = MTCNN()
facenet = InceptionResnetV1(pretrained='vggface2').eval()

# Load saved embeddings dictionary
saved_embeddings = np.load("face_embeddings.npy", allow_pickle=True).item()

def cosine_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2.T) / (norm(embedding1) * norm(embedding2))

# Prepare attendance CSV file
attendance_file = "attendance.csv"
attendance_header = ["UID", "Name", "Course", "Year", "Division", "RollNo", "Timestamp"]

# Initialize a set to keep track of users already marked present
attendance_set = set()

# If file doesn't exist, create it with header
try:
    with open(attendance_file, 'x', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(attendance_header)
except FileExistsError:
    # File already exists
    pass

cap = cv2.VideoCapture(0)

start_time = time.time()  # Capture the start time

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Check if 1 minute has passed
    if time.time() - start_time > 60:
        print("Time limit reached. Closing the application.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face, prob = mtcnn(frame_rgb, return_prob=True)

    if face is not None and prob > 0.90:  # proceed if face detected with high probability
        new_embedding = facenet(face.unsqueeze(0)).detach().numpy().flatten()
        best_match = None
        max_similarity = 0

        # Iterate over stored embeddings to find the best match
        for person, embeddings in saved_embeddings.items():
            for stored_embedding in embeddings:
                similarity = cosine_similarity(new_embedding, stored_embedding.flatten())

                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match = person

        # Check threshold to determine recognition
        if max_similarity > 0.7:
            text = f"Recognized: {best_match}"
            color = (0, 255, 0)

            # Parse details from folder name (format: uid_name_course_year_div_rollno)
            try:
                uid, name, course, year, division, roll_no = best_match.split("_")
            except Exception as e:
                print(f"Error parsing user details from folder name {best_match}: {e}")
                uid = name = course = year = division = roll_no = "Unknown"

            # Mark attendance if not already marked
            if best_match not in attendance_set:
                attendance_set.add(best_match)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(attendance_file, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([uid, name, course, year, division, roll_no, timestamp])
                print(f"Attendance marked for {name}")
        else:
            text = "Unknown Face"
            color = (0, 0, 255)

        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
