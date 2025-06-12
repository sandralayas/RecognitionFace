# # all models

from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import os

from vizualize import visualize_verification_result
from imageQuality import illumination_normalization,imgShow
from detector import feature5

# List of available models in deepface
models = [
    # "VGG-Face",
    "Facenet",
    # "FaceNet512", # often a separate option for the 512-dim embedding version
    # "OpenFace",
    # "DeepFace",
    # "DeepID",
    # "ArcFace",
    "Dlib",
    # "SFace",
    # "GhostFaceNet",
]

# Example of using ArcFace for verification - 
# img1_path="output.png"

# file=open('files.txt','r')
# whole=file.read()
# imgpaths=whole.split('\n')

# imgpaths=['Divas2.jpg - Divas4.jpg','nikitha1.png - nikitha2.jpg']
imgpaths = ['eveena1.jpg - eveena2.jpg']

outputallmaodels=[]

for imgpair in imgpaths:
    lazypath=(imgpair).split(" - ")

    img1_path="corpus/"+str(lazypath[0])
    img2_path="corpus/"+str(lazypath[1])

    image1=cv2.imread(img1_path)
    image2=cv2.imread(img2_path)
    
    # imgShow(image1,image2)

    accoutput=[]
    modeloutput=[]
    featureemd=[feature5(img1_path),feature5(img2_path)]

    for model in models:
        
        model_name = model
        # print(f"{model_name}")
        try:
            result = DeepFace.verify(
                img1_path=img1_path,
                img2_path=img2_path,
                model_name=model_name,
                detector_backend='retinaface'
            )
            if result['verified']==True:
                value=((1-result['distance']))*(100)
            else:value=0#((result['distance']-1))*(100)
            
            accoutput.append(str(round(value,3))+" % ")#+str(result['verified']))
            modeloutput.append(result)
            # visualize_verification_result(img1_path, img2_path, result)
        except ValueError:
            print(imgpair,' face is not detected')       
     
    outputallmaodels.append([imgpair,accoutput])
    # for i in featureemd:print(i)

for i in modeloutput:print(i)

for singlemodel in outputallmaodels:
    # print(singlemodel[0])
    for pair in singlemodel[1]:
        print(pair)
