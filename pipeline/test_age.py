# test_age


import cv2

from insightcode_modified import get_face_embedding,compare_faces,get_person_age,get_person_gender
from imageQuality import preprocessing,find_which_preprocess,imgShow

file=open(r"C:\Users\sandr\Documents\pix\FaceRec\models\similar.txt",'r')
# # file=open('similar.txt','r')
# # file=open('dissimilar.txt','r')
whole=file.read()
imgpaths=whole.split('\n')

# imgpaths=['Divas2.jpg - Divas4.jpg','nikitha1.png - nikitha2.jpg']
# imgpaths = ['eveena1.jpg - eveena2.jpg']
# imgpaths = ['test1.jpg - test2.jpg']
# imgpaths = ['alex.png - Divas5.jpg']



similarity_score_list=[]

count=0
total_pairs=0

for imgpair in imgpaths:
    total_pairs+=1
    lazypath=(imgpair).split(" - ")

    image1_path="C:/Users/sandr/Documents/pix/FaceRec/corpus/"+str(lazypath[0])
    image2_path="C:/Users/sandr/Documents/pix/FaceRec/corpus/"+str(lazypath[1])
    
    image1=cv2.imread(image1_path)
    image2=cv2.imread(image2_path)

    # print(lazypath)

    age1=get_person_age(image1)
    age2=get_person_age(image2)
    # print(age1,age2)
    
    diff=max(age1,age2)-min(age1,age2)
    
    # print(age1,age2,diff)
    # if age1 and age2 in range(mid-5,mid+5):
    
    if diff<10:
        count+=1
    else:pass

print('\nAccurracy :',round(count/total_pairs*100,2),'%\nTotal number of pictures :',total_pairs,'\nNew pictures correct :',count,'\nOld pictures :',total_pairs-count)