# # all models

from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import os

from imageQuality import illumination_normalization
from detector import feature5

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

# Example of using ArcFace for verification - 
# img1_path="output.png"

# - 

img1_path=r"corpus\Deepthi1.jpg"
img2_path=r"corpus\Deepthi2.jpg"

image1=cv2.imread(img1_path)
image2=cv2.imread(img2_path)

# cv2.namedWindow('My Resizable Window',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('My Resizable Window', 800, 600)
# cv2.imshow('My Resizable Window',image1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.namedWindow('My Resizable Window',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('My Resizable Window', 800, 600)
# cv2.imshow('My Resizable Window',image2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

accoutput=[]
modeloutput=[]
featureemd=[feature5(img1_path),feature5(img2_path)]

for model in models:
    
    model_name = model
    # print(f"{model_name}")
    result = DeepFace.verify(
        img1_path=img1_path,
        img2_path=img2_path,
        model_name=model_name
    )
    value=((1-result['distance']) /2)*(100)
    if value<0:
        value=((result['distance']-1) /2)*(100)
    
    accoutput.append(str(round(value,3))+" % "+str(result['verified']))
    modeloutput.append(result)
    
for i in accoutput:print(i)
# for i in modeloutput:print(i)
for i in featureemd:print(i)