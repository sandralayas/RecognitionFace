import cv2
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# 1. Load an image
# Replace 'path/to/your/image.jpg' with the actual path to an image file
# You can use a sample image from the internet or your local machine.
# For example, if you have 'sample.jpg' in the same directory as your script.
image_path = 'image.jpg'
try:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    # MTCNN expects RGB images, but OpenCV reads BGR by default
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
except FileNotFoundError as e:
    print(e)
    print("Please make sure you have an image file (e.g., sample.jpg) in the correct path.")
    print("For demonstration, creating a dummy black image.")
    img_rgb = np.zeros((400, 600, 3), dtype=np.uint8) # Create a black image
    cv2.putText(img_rgb, "No Image Found - Dummy Display", (50, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR) # Convert back for imshow later

# 2. Initialize the MTCNN detector
detector = MTCNN()

# 3. Detect faces in the image
# The detect_faces method returns a list of dictionaries, one for each detected face.
# Each dictionary contains 'box' (bounding box) and 'keypoints' (facial landmarks).
faces = detector.detect_faces(img_rgb)

print(f"Detected {len(faces)} face(s).")

# 4. Display the image and draw bounding boxes/landmarks
fig, ax = plt.subplots(1, figsize=(10, 8))
ax.imshow(img_rgb) # Use the RGB image for display

# Iterate over each detected face
for face in faces:
    x, y, width, height = face['box']
    
    # Draw bounding box
    rect = patches.Rectangle((x, y), width, height, 
                             linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    
    # Draw keypoints (facial landmarks: left_eye, right_eye, nose, mouth_left, mouth_right)
    keypoints = face['keypoints']
    ax.plot(keypoints['left_eye'][0], keypoints['left_eye'][1], 'go', markersize=8)
    ax.plot(keypoints['right_eye'][0], keypoints['right_eye'][1], 'go', markersize=8)
    ax.plot(keypoints['nose'][0], keypoints['nose'][1], 'bo', markersize=8)
    ax.plot(keypoints['mouth_left'][0], keypoints['mouth_left'][1], 'yo', markersize=8)
    ax.plot(keypoints['mouth_right'][0], keypoints['mouth_right'][1], 'yo', markersize=8)

    # Optional: Add confidence score
    confidence = face['confidence']
    ax.text(x, y - 10, f'{confidence:.2f}', bbox=dict(facecolor='red', alpha=0.5), fontsize=10, color='white')

ax.axis('off') # Hide axes
plt.title('MTCNN Face Detection')
plt.show()

# If you also want to show using OpenCV (less common for quick display)
# cv2.imshow('Detected Faces', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
# cv2.waitKey(0)
# cv2.destroyAllWindows()