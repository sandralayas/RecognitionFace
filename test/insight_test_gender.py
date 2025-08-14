# test_gender

import cv2


from insightcode_modified import get_person_gender

file=open('corpuslist.txt','r')
whole=file.read()
imgpaths=whole.split('\n')

count=0
total=len(imgpaths)

for imgpair in imgpaths:
    lazypath=(imgpair).split(" - ")

    # print(lazypath)

    image1_path="C:/Users/sandr/Documents/pix/FaceRec/corpus/"+str(lazypath[1])
    image1=cv2.imread(image1_path)
    
    if get_person_gender(image1)==int(lazypath[0]):
        count+=1
        
print('\nAccurracy :',round(count/total*100,2),'%\nTotal number of pictures :',total,'\nPredicted correct :',count)