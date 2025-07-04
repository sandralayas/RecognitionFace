
import cv2


from insightcode_modified import get_face_embedding,compare_faces,get_person_age,get_person_gender
from imageQuality import preprocessing,find_which_preprocess,imgShow

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
Reflections/Glare (from lamination)S
Posture
'''

similarity_score_list=[]

for imgpair in imgpaths:
    lazypath=(imgpair).split(" - ")

    image1_path="C:/Users/sandr/Documents/pix/FaceRec/corpus/"+str(lazypath[0])
    image2_path="C:/Users/sandr/Documents/pix/FaceRec/corpus/"+str(lazypath[1])
    
    image1=cv2.imread(image1_path)
    image2=cv2.imread(image2_path)

    # age1=get_person_age(image1)
    # age2=get_person_age(image2)
    # print(age1,age2)

    # age
    # if get_person_gender(image1)!=get_person_gender(image2):
    #     print(lazypath)

    # preprocessing

    preprocess_1=find_which_preprocess(image1)
    preprocess_2=find_which_preprocess(image2)
    
    # print(preprocess_1,preprocess_2)
    
    image1=preprocessing(image1,preprocess_1)
    cv2.imwrite("image1.png", image1)
    image2=preprocessing(image2,preprocess_2)
    cv2.imwrite("image2.png", image2)
    image1_path = "image1.png"
    image2_path = "image2.png"

    # #display the images
    # imgShow(image1,image2)
    
    try:

        # Get embeddings
        emb1 = get_face_embedding(image1_path)
        emb2 = get_face_embedding(image2_path)
        
        # Compare faces
        similarity_score = compare_faces(emb1, emb2)
        
        # timelist.append(end-start)
        
        similarity_score=(similarity_score+1)*50
        # similarity_score=100/(1+math.exp(-1*(20*(similarity_score))))
        
        similarity_score_list.append(similarity_score)
        # print(f"{similarity_score:.4f} %")
        
    except Exception as e:
        print(f"Error: {str(e)}")

for i in similarity_score_list:print(round(i,2),'%')