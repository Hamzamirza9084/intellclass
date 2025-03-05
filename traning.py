import os
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN

# Initialize MTCNN for face detection and InceptionResnetV1 for face embeddings
mtcnn = MTCNN()  
facenet = InceptionResnetV1(pretrained='vggface2').eval()

def get_embedding(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not read image {image_path}")
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Detect face in the image
    face, prob = mtcnn(img_rgb, return_prob=True)
    
    if face is not None and prob > 0.90:  # You can adjust the threshold as needed
        embedding = facenet(face.unsqueeze(0)).detach().numpy()
        return embedding
    else:
        print(f"Face not detected or low probability in {image_path}")
    return None

dataset_path = "dataset"
embeddings = {}

# Iterate over each person folder in the dataset
for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)
    
    # Process only directories
    if not os.path.isdir(person_path):
        continue

    embeddings[person] = []
    print(f"Processing images for: {person}")

    for img_file in os.listdir(person_path):
        img_path = os.path.join(person_path, img_file)
        embedding = get_embedding(img_path)
        
        if embedding is not None:
            embeddings[person].append(embedding)
            print(f"Processed {img_file} for {person}")
        else:
            print(f"Skipping {img_file} for {person}")

np.save("face_embeddings.npy", embeddings)
print("Face embeddings saved successfully!")
