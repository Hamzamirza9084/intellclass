import numpy as np
import cv2
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from numpy.linalg import norm


mtcnn = MTCNN()


facenet = InceptionResnetV1(pretrained='vggface2').eval()


saved_embeddings = np.load("face_embeddings.npy", allow_pickle=True).item()


def cosine_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2.T) / (norm(embedding1) * norm(embedding2))


cap = cv2.VideoCapture(0)

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


        for person, embeddings in saved_embeddings.items():
            for stored_embedding in embeddings:
                similarity = cosine_similarity(new_embedding, stored_embedding.flatten())

                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match = person


        if max_similarity > 0.7:  
            text = f"Recognized: {best_match}"
            color = (0, 255, 0)
        else:
            text = "Unknown Face"
            color = (0, 0, 255)

        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
