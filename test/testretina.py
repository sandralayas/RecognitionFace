import cv2
from retinaface import RetinaFace
import matplotlib.pyplot as plt

# --- Configuration ---
image_path = "faceDataset\imgunclear.jpg" 

# --- 1. Load the image ---
img = cv2.imread(image_path)

if img is None:
    print(f"Error: Could not load image from {image_path}")
    print("Please ensure the image file exists and the path is correct.")
    exit()

# RetinaFace models often expect RGB, while OpenCV reads BGR.
# Converting is good practice, though 'retina-face' library might handle it internally.
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# --- 2. Detect faces ---
# The detect_faces method returns a dictionary of detected faces.
# Each key ('face_1', 'face_2', etc.) corresponds to a detected face.
# The value is a dictionary containing 'facial_area', 'landmarks', 'score'.
detections = RetinaFace.detect_faces(img_rgb)

# --- 3. Process and visualize detections ---
if isinstance(detections, dict):
    print(f"Found {len(detections)} face(s) in the image.")
    for face_id, face_data in detections.items():
        # Get bounding box coordinates
        x1, y1, x2, y2 = face_data['facial_area']
        
        # Get confidence score
        confidence = face_data['score']

        # Get landmarks
        landmarks = face_data['landmarks']

        print(f"\n--- {face_id} (Confidence: {confidence:.2f}) ---")
        print(f"  Bounding Box: ({x1}, {y1}) to ({x2}, {y2})")

        # Draw bounding box
        # Color in BGR: (0, 255, 0) is Green
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2) # (image, start_point, end_point, color, thickness)

        # Draw landmarks
        # Color in BGR: (0, 0, 255) is Red
        for landmark_name, coords in landmarks.items():
            print(f"  {landmark_name.replace('_', ' ').title()}: ({coords[0]}, {coords[1]})")
            cv2.circle(img, (int(coords[0]), int(coords[1])), 3, (0, 0, 255), -1) # (image, center, radius, color, thickness=-1 for filled)
            
    # --- 4. Display the result ---
    # Convert image back to RGB for matplotlib display (as it expects RGB)
    img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_display)
    plt.title("Face Detection with RetinaFace")
    plt.axis('off') # Hide axes
    plt.show()

else:
    print("No faces detected in the image.")