import os
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN


mtcnn = MTCNN()  
facenet = InceptionResnetV1(pretrained='vggface2').eval()


def get_embedding(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    face, _ = mtcnn(img_rgb, return_prob=True)
    
    if face is not None:
        embedding = facenet(face.unsqueeze(0)).detach().numpy()
        return embedding
    return None


dataset_path = "dataset"
embeddings = {}


for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)
    embeddings[person] = []

    for img_file in os.listdir(person_path):
        img_path = os.path.join(person_path, img_file)
        embedding = get_embedding(img_path)
        
        if embedding is not None:
            embeddings[person].append(embedding)


np.save("face_embeddings.npy", embeddings)
print("Face embeddings saved successfully!")
