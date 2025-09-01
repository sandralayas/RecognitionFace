

'''
this is modified accroding to the fast api pipeline 
'''

import cv2
import insightface
import numpy as np
from insightface.app import FaceAnalysis
import time
import math

# Initialize the InsightFace model
app = FaceAnalysis(root='/root/app', providers=['CPUExecutionProvider'])
# app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])  # Use 'CUDAExecutionProvider' for GPU
app.prepare(ctx_id=-1)  # ctx_id=-1 for CPU, 0 for GPU

def image_face_embedding(image_cv):
    """Extract face embedding from an image cv format"""
    img = image_cv
    if img is None:
        raise ValueError(f"Could not read image")
    
    faces = app.get(img)
    
    if len(faces) < 1:
        raise ValueError("No faces detected in the image")
    if len(faces) > 1:
        print("Warning: Multiple faces detected. Using first detected face")
    return faces[0].embedding

def get_person_gender(image):
    faces = app.get(image)
    if faces:
        main_face = faces[0]
        estimated_gender = main_face.gender
    return estimated_gender

def get_person_age(image):
    faces = app.get(image)

    if faces:
        main_face = faces[0]
        estimated_age = main_face.age # Access the 'age' attribute
        
        return estimated_age

def compare_faces(emb1, emb2, threshold=0.65): # Adjust this threshold according to your usecase.
    """Compare two embeddings using cosine similarity"""
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return similarity

def get_face_embedding(image_path):
    """Extract face embedding from an image"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    faces = app.get(img)
    
    if len(faces) < 1:
        raise ValueError("No faces detected in the image")
    if len(faces) > 1:
        print("Warning: Multiple faces detected. Using first detected face")
    
    return faces[0].embedding