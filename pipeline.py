'''
1. the best model so far is ArcFace
2. get the face embedding
3. 
'''

# from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import os
import insightcode

photo=os.listdir('bestphoto')
id=os.listdir('bestid')

# file=open('similar.txt','r')
# # file=open('dissimilar.txt','r')
# whole=file.read()
# imgpaths=whole.split('\n')

for imgpair in range(7):
    # lazypath=(imgpair).split(" - ")

    img1_path="bestphoto/"+str(photo[imgpair])
    img2_path="bestid/"+str(id[imgpair])

    print(img1_path,img2_path)
    
    image1=cv2.imread(img1_path)
    image2=cv2.imread(img2_path)

    # dict1 = DeepFace.represent(
    #     img_path=img1_path,
    #     model_name='ArcFace',
    #     detector_backend='opencv'
    # )
    # dict2 = DeepFace.represent(
    #     img_path=img2_path,
    #     model_name='ArcFace',
    #     detector_backend='opencv'
    # )
    
    dict1=dict1[0]
    dict2=dict2[0]
    
    face1=dict1['embedding']
    face2=dict2['embedding']
    
    face1=insightcode.get_face_embedding(img1_path)
    face2=insightcode.get_face_embedding(img2_path)
    
    cosinedist=insightcode.cosine(face1,face2)
    euclideandist=insightcode.euclidian(face1,face2)
    
    cosinedist=round((cosinedist+1)*50,2)
    euclideandist=round(100-min(euclideandist,100),2)
    
    print('Cosine :',cosinedist)
    print('Euclidean :',euclideandist)
    # print('InsightFace',len(insightface1),len(insightface2))
    