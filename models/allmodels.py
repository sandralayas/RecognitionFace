# # all models

from deepface import DeepFace

import cv2
import matplotlib.pyplot as plt
import os
from deepface import DeepFace

def visualize_verification_result(img1_path, img2_path, verification_result):
    """
    Visualizes the verification result by drawing bounding boxes,
    keypoints, and verification status on the images.

    Args:
        img1_path (str): Path to the first image.
        img2_path (str): Path to the second image.
        verification_result (dict): The dictionary returned by DeepFace.verify().
    """
    # Load images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None:
        print(f"Error: Could not load image from {img1_path}")
        return
    if img2 is None:
        print(f"Error: Could not load image from {img2_path}")
        return

    # Convert BGR to RGB for matplotlib display
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # Get facial areas information
    face_areas = verification_result.get('facial_areas', {})
    img1_face = face_areas.get('img1')
    img2_face = face_areas.get('img2')

    # Define colors for drawing
    verified_color = (0, 255, 0)  # Green for verified
    not_verified_color = (255, 0, 0) # Red for not verified
    line_thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    text_thickness = 2

    # Draw on img1
    if img1_face:
        x, y, w, h = img1_face['x'], img1_face['y'], img1_face['w'], img1_face['h']
        left_eye = img1_face.get('left_eye')
        right_eye = img1_face.get('right_eye')

        color = verified_color if verification_result['verified'] else not_verified_color
        cv2.rectangle(img1_rgb, (x, y), (x + w, y + h), color, line_thickness)
        if left_eye:
            cv2.circle(img1_rgb, left_eye, 5, (0, 255, 255), -1) # Yellow for eyes
        if right_eye:
            cv2.circle(img1_rgb, right_eye, 5, (0, 255, 255), -1)

    # Draw on img2
    if img2_face:
        x, y, w, h = img2_face['x'], img2_face['y'], img2_face['w'], img2_face['h']
        left_eye = img2_face.get('left_eye')
        right_eye = img2_face.get('right_eye')

        color = verified_color if verification_result['verified'] else not_verified_color
        cv2.rectangle(img2_rgb, (x, y), (x + w, y + h), color, line_thickness)
        if left_eye:
            cv2.circle(img2_rgb, left_eye, 5, (0, 255, 255), -1)
        if right_eye:
            cv2.circle(img2_rgb, right_eye, 5, (0, 255, 255), -1)

    # Prepare status text
    status_text = f"Verified: {verification_result['verified']}"
    distance_text = f"Distance: {verification_result['distance']:.3f}"
    threshold_text = f"Threshold: {verification_result['threshold']:.3f}"
    model_text = f"Model: {verification_result['model']}"
    time_text = f"Time: {verification_result['time']:.2f}s"

    # Create a figure to display both images
    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    plt.imshow(img1_rgb)
    plt.title(f"Image 1 ({os.path.basename(img1_path)})")
    plt.axis('off')
    cv2.putText(img1_rgb, status_text, (50, 50), font, font_scale, color, text_thickness, cv2.LINE_AA)


    plt.subplot(1, 2, 2)
    plt.imshow(img2_rgb)
    plt.title(f"Image 2 ({os.path.basename(img2_path)})")
    plt.axis('off')
    cv2.putText(img2_rgb, status_text, (50, 50), font, font_scale, color, text_thickness, cv2.LINE_AA)


    # Add general info below the images (using plt.figtext for common text)
    plt.figtext(0.5, 0.01,
                f"{model_text} | {distance_text} | {threshold_text} | {time_text} | Similarity: {verification_result['similarity_metric'].capitalize()}",
                ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.7, "pad":5})

    plt.tight_layout(rect=[0, 0.05, 1, 1]) # Adjust rect to make space for figtext
    plt.show()

# List of available models in deepface
models = [
    "VGG-Face",
    # "FaceNet",
    # "FaceNet512", # often a separate option for the 512-dim embedding version
    "OpenFace",
    "DeepFace",
    "DeepID",
    "ArcFace",
    # "Dlib",
    "SFace",
    "GhostFaceNet",
]

# Example of using ArcFace for verification

img1_path=r"C:\Users\sandr\Documents\pix\FaceRec\models\corpus\Sophy1.PNG"
img2_path=r"C:\Users\sandr\Documents\pix\FaceRec\models\corpus\Sophy2.PNG"
        
for model in models:
    model_name = model
    print(f"Using model: {model_name}")
    result = DeepFace.verify(
        img1_path=img1_path,
        img2_path=img2_path,
        model_name=model_name
    )
    visualize_verification_result(img1_path, img2_path, result)
    print(result)