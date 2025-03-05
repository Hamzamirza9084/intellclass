import os
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN

# Initialize MTCNN for face detection and InceptionResnetV1 for face embeddings
mtcnn = MTCNN()  
facenet = InceptionResnetV1(pretrained='vggface2').eval()

def get_embedding(image_path):
    """Extracts face embedding from the given image."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not read image {image_path}")
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Detect face in the image
    face, prob = mtcnn(img_rgb, return_prob=True)
    
    if face is not None and prob > 0.90:  # Adjust probability threshold if needed
        embedding = facenet(face.unsqueeze(0)).detach().numpy()
        return embedding
    else:
        print(f"Face not detected or low probability in {image_path}")
    return None

# Define the dataset path for professors
dataset_path = "teacher_dataset"
embeddings = {}

# Iterate over each professor's folder in the dataset
for professor in os.listdir(dataset_path):
    professor_path = os.path.join(dataset_path, professor)
    
    # Process only directories
    if not os.path.isdir(professor_path):
        continue

    embeddings[professor] = []
    print(f"Processing images for: {professor}")

    for img_file in os.listdir(professor_path):
        img_path = os.path.join(professor_path, img_file)
        embedding = get_embedding(img_path)
        
        if embedding is not None:
            embeddings[professor].append(embedding)
            print(f"Processed {img_file} for {professor}")
        else:
            print(f"Skipping {img_file} for {professor}")

# Save the embeddings to a .npy file
np.save("professor_face_embeddings.npy", embeddings)
print("Professor face embeddings saved successfully!")
