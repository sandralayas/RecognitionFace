
# C:\Users\sandr\Documents\pix\FaceRec\models\insightcode.py

import cv2
import insightface
import numpy as np
from insightface.app import FaceAnalysis
import time
import math

from imageQuality import preprocessing

# import logging
# logger = logging.getLogger('insightface')
# logger.setLevel(logging.WARNING)

# Initialize face analysis model
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])  # Use 'CUDAExecutionProvider' for GPU
app.prepare(ctx_id=-1)  # ctx_id=-1 for CPU, 0 for GPU

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

def compare_faces(emb1, emb2, threshold=0.65): # Adjust this threshold according to your usecase.
    """Compare two embeddings using cosine similarity"""
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return similarity

file=open('preprocess.txt','r')
# file=open('similar.txt','r')
# file=open('dissimilar.txt','r')
whole=file.read()
imgpaths=whole.split('\n')

# imgpaths=['Divas2.jpg - Divas4.jpg','nikitha1.png - nikitha2.jpg']
# imgpaths = ['eveena1.jpg - eveena2.jpg']
# imgpaths = ['test1.jpg - test2.jpg']


'''
Yellowing/Browning
Fading (Desaturation)
Low Resolution/Sharpness
Poor Lighting
Outdated Appearance hair
Outdated Appearance age
Reflections/Glare (from lamination)
Posture
'''

timelist=[]

for imgpair in imgpaths:
    lazypath=(imgpair).split(" - ")

    img1_path="corpus/"+str(lazypath[0])
    img2_path="corpus/"+str(lazypath[1])

    preprocessingmodel='combained'
    
    image1=cv2.imread(img1_path)
    image1=preprocessing(image1,preprocessingmodel)
    cv2.imwrite("image1.png", image1)
    image2=cv2.imread(img2_path)
    image2=preprocessing(image2,preprocessingmodel)
    cv2.imwrite("image2.png", image2)

    # Paths to your Indian face images
    image1_path = "image1.png"
    image2_path = "image2.png"

    try:

        # Get embeddings
        emb1 = get_face_embedding(image1_path)
        emb2 = get_face_embedding(image2_path)
        
        # emb1 = get_face_embedding(img1_path)
        # emb2 = get_face_embedding(img2_path)
        
        # Compare faces
        start=time.time()
        similarity_score = compare_faces(emb1, emb2)
        end=time.time()
        
        # timelist.append(end-start)
        
        similarity_score=(similarity_score+1)*50
        # similarity_score=100/(1+math.exp(-1*(20*(similarity_score))))
        
        print(f"{similarity_score:.4f} %")
        
    except Exception as e:
        print(f"Error: {str(e)}")

# for i in timelist:print(round(i,2))