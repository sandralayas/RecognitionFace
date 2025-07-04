# # all models

from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import os

from vizualize import visualize_verification_result
from imageQuality import illumination_normalization,imgShow
from detector import feature5

file=open('similarpairs.txt','r')
whole=file.read()
imgpaths=whole.split('\n')

outputalldetectors=[]

for imgpair in imgpaths:
    lazypath=(imgpair).split(" - ")

    img1_path="corpus/"+str(lazypath[0])
    img2_path="corpus/"+str(lazypath[1])

    image1=cv2.imread(img1_path)
    image2=cv2.imread(img2_path)
    
    # imgShow(image1,image2)
    
    featureemd=[feature5(img1_path),feature5(img2_path)]

    detectoroutput=[]
    
    backend=['opencv','mtcnn','retinaface','ssd']
    for detectors in backend:
        try:
            result = DeepFace.verify(
                img1_path=img1_path,
                img2_path=img2_path,
                model_name='Dlib',
                detector_backend=detectors
            )
            if result['verified']==True:
                value=((1-result['distance']))*(100)
            else:value=0#((result['distance']-1))*(100)

            detectoroutput.append(str(round(value,3))+" % ")
            
            # visualize_verification_result(img1_path, img2_path, result)
        except ValueError:
            print(imgpair,' face is not detected')
    outputalldetectors.append(detectoroutput)
for i in(outputalldetectors):
    print()
    for j in i:print(j)
    
