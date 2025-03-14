import numpy as np
import cv2
import torch
import csv
import time
import sys
from datetime import datetime, timedelta
from facenet_pytorch import InceptionResnetV1, MTCNN
from numpy.linalg import norm
from pymongo import MongoClient  # New import for MongoDB Atlas

if len(sys.argv) < 2:
    print("Error: No subject name provided.")
    sys.exit(1)

subject_name = sys.argv[1]
print(f"Running attendance system for subject: {subject_name}")

# Function to upload CSV file to MongoDB Atlas
def upload_csv_to_mongo(csv_filename):
    # Replace with your actual MongoDB Atlas connection string
    connection_string = "mongodb+srv://hamzamirza9084:surge7698302331@cluster0.ixn2m.mongodb.net/"
    client = MongoClient(connection_string)
    db = client["attendance_db"]  # Replace with your database name

    # Dynamic collection name based on professor_csv_filename
    professor_collection_name = csv_filename.replace(".csv", "")  # Remove .csv extension
    professor_collection = db[professor_collection_name]

    # Fixed collection for all professor attendances
    main_collection = db["professor_attendances"]

    with open(csv_filename, 'r', newline='') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        data = list(csv_reader)

    if data:
        # Insert into the professor's specific collection
        professor_collection.insert_many(data)
        print(f"Uploaded {len(data)} records from {csv_filename} to MongoDB Atlas in collection '{professor_collection_name}'.")

        # Insert into the main professor_attendances collection as well
        main_collection.insert_many(data)
        print(f"Uploaded {len(data)} records from {csv_filename} to MongoDB Atlas in collection 'professor_attendances'.")

    else:
        print(f"No data found in {csv_filename} to upload.")

# Initialize MTCNN for face detection and InceptionResnetV1 for embeddings
mtcnn = MTCNN()
facenet = InceptionResnetV1(pretrained='vggface2').eval()

# Load saved embeddings for professors and students
professor_embeddings = np.load("professor_face_embeddings.npy", allow_pickle=True).item()
student_embeddings = np.load("face_embeddings.npy", allow_pickle=True).item()

# Function to calculate cosine similarity between embeddings
def cosine_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2.T) / (norm(embedding1) * norm(embedding2))

# Initialize sets to keep track of detected professors and students
detected_professors = set()
attendance_set = set()

# Get subject name from the user
#subject_name = input("Enter Subject Name: ")

# Attendance CSV for both professors and students
attendance_file = "attendance.csv"
attendance_header = ["UID", "Name", "Course", "Year", "Division", "RollNo", "Subject", "Timestamp"]

# Create the CSV file if it doesn't exist
try:
    with open(attendance_file, 'x', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(attendance_header)
except FileExistsError:
    pass

cap = cv2.VideoCapture(0)

# Phase 1: Professor Recognition (30 seconds)
start_time = datetime.now()
duration = timedelta(seconds=30)  # Run first phase for 30 seconds
professor_csv_filename = None  # Store professor attendance file name

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face, _ = mtcnn(frame_rgb, return_prob=True)

    if face is not None:
        new_embedding = facenet(face.unsqueeze(0)).detach().numpy().flatten()
        best_match = None
        max_similarity = 0

        # Compare detected face with saved professor embeddings
        for person, embeddings in professor_embeddings.items():
            for stored_embedding in embeddings:
                similarity = cosine_similarity(new_embedding, stored_embedding.flatten())
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match = person

        # Display result and mark attendance if the professor is recognized
        if max_similarity > 0.7:  # Recognition threshold
            text = f"Professor Detected: {best_match}"
            color = (0, 255, 0)

            # Create the CSV file once per detected professor
            if best_match not in detected_professors:
                detected_professors.add(best_match)
                detection_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                professor_csv_filename = f"{best_match}.csv"

                with open(professor_csv_filename, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["UID", "Name", "Course", "Year", "Division", "RollNo", "Subject", "Timestamp"])
                    writer.writerow([best_match, best_match, "N/A", "N/A", "N/A", "N/A", subject_name, detection_time])
                print(f"Professor CSV created: {professor_csv_filename}")
                
                # Upload the professor CSV file to MongoDB Atlas immediately after creation
                upload_csv_to_mongo(professor_csv_filename)

                # Mark attendance for professor in the main attendance file
                with open(attendance_file, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([best_match, best_match, "N/A", "N/A", "N/A", "N/A", subject_name, detection_time])
                print(f"Attendance marked for Professor {best_match}")

        else:
            text = "Unknown Face"
            color = (0, 0, 255)

        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Recognition System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if datetime.now() - start_time > duration:
        print("Switching to Student Attendance Mode...")
        break

# Phase 2: Student Attendance (1 minute)
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if time.time() - start_time > 60:
        print("Time limit reached for student attendance. Closing the application.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face, prob = mtcnn(frame_rgb, return_prob=True)

    if face is not None and prob > 0.90:
        new_embedding = facenet(face.unsqueeze(0)).detach().numpy().flatten()
        best_match = None
        max_similarity = 0

        for person, embeddings in student_embeddings.items():
            for stored_embedding in embeddings:
                similarity = cosine_similarity(new_embedding, stored_embedding.flatten())
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match = person

        if max_similarity > 0.7:
            text = f"Recognized: {best_match}"
            color = (0, 255, 0)

            try:
                uid, name, course, year, division, roll_no = best_match.split("_")
            except Exception as e:
                print(f"Error parsing details from {best_match}: {e}")
                uid = name = course = year = division = roll_no = "Unknown"

            if best_match not in attendance_set:
                attendance_set.add(best_match)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Save in main attendance file
                with open(attendance_file, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([uid, name, course, year, division, roll_no, subject_name, timestamp])

                # Also append to the professor's CSV file
                if professor_csv_filename:
                    with open(professor_csv_filename, 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([uid, name, course, year, division, roll_no, subject_name, timestamp])

                print(f"Attendance marked for {name}")
        else:
            text = "Unknown Face"
            color = (0, 0, 255)

        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Recognition System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Upload the final updated professor CSV file to MongoDB Atlas
if professor_csv_filename:
    print("Uploading the final updated version of the professor CSV file...")
    upload_csv_to_mongo(professor_csv_filename)

cap.release()
cv2.destroyAllWindows() 