from pymongo import MongoClient
import gridfs
import cv2
import numpy as np

# Connect to MongoDB Atlas (replace with your credentials)
client = MongoClient("mongodb+srv://hamzamirza9084:surge7698302331@cluster0.ixn2m.mongodb.net/")
db = client['face_recognition']
fs = gridfs.GridFS(db)

# Retrieve all stored images from GridFS
files = list(fs.find({}))

if not files:
    print("No stored images found.")
else:
    for file in files:
        # Read the binary data and convert to a numpy array
        data = file.read()
        nparr = np.frombuffer(data, np.uint8)
        # Decode the image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Check if image decoding was successful
        if img is not None:
            window_name = f"{file.filename}"
            cv2.imshow(window_name, img)
            print(f"Displaying image: {file.filename}")
            cv2.waitKey(0)  # Wait for a key press to proceed to the next image
            cv2.destroyWindow(window_name)
        else:
            print(f"Could not decode image: {file.filename}")

cv2.destroyAllWindows()
