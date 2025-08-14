#A Gender and Age Detection program by Mahesh Sawant

import cv2
import math
import argparse
import numpy as np # Import numpy for array operations

def highlightface1(net, frame1, conf_threshold=0.7):
    # This function should receive a 3-channel image.
    # We'll ensure the input 'frame1' is 3-channel *before* calling this function.

    frame1OpencvDnn = frame1.copy()
    frame1Height = frame1OpencvDnn.shape[0]
    frame1Width = frame1OpencvDnn.shape[1]

    # The original line: frame1 = cv2.cvtColor(frame1, cv2.COLOR_RGBA2RGB)
    # is now handled outside this function for the main 'frame1',
    # and specifically for 'face1' ROIs below.
    # No need for it here if the input 'frame1' is already guaranteed 3-channel.

    blob = cv2.dnn.blobFromImage(frame1OpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    face1Boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > conf_threshold:
            x1 = int(detections[0,0,i,3] * frame1Width)
            y1 = int(detections[0,0,i,4] * frame1Height)
            x2 = int(detections[0,0,i,5] * frame1Width)
            y2 = int(detections[0,0,i,6] * frame1Height)
            face1Boxes.append([x1,y1,x2,y2])
            cv2.rectangle(frame1OpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frame1Height/150)), 8)
    return frame1OpencvDnn, face1Boxes

def give_Age_Gender(frame1):
    
    padding=20
    
    if frame1.ndim == 3 and frame1.shape[-1] == 4: # If RGBA (4 channels)
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_RGBA2RGB)
    elif frame1.ndim == 2: # If Grayscale (1 channel)
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_GRAY2BGR)
    elif frame1.ndim != 3 or frame1.shape[-1] != 3:
        print(f"WARNING: Main frame1 has unexpected number of channels: {frame1.shape}. Skipping frame1.")

    resultImg, face1Boxes = highlightface1(faceNet, frame1)

    if not face1Boxes:
        print("No face1 detected")

    for face1Box in face1Boxes:
        
        face1 = frame1[max(0, face1Box[1]-padding):min(face1Box[3]+padding, frame1.shape[0]-1),
                        max(0, face1Box[0]-padding):min(face1Box[2]+padding, frame1.shape[1]-1)]

        if face1.ndim == 3 and face1.shape[-1] == 4:
            face1 = cv2.cvtColor(face1, cv2.COLOR_RGBA2RGB)
        elif face1.ndim == 2:
            face1 = cv2.cvtColor(face1, cv2.COLOR_GRAY2BGR)
        elif face1.ndim != 3 or face1.shape[-1] != 3:
            print(f"WARNING: Extracted face1 ROI has unexpected number of channels: {face1.shape}. Skipping this face1 for gender/age detection.")
            continue
        
        if face1.shape[0] == 0 or face1.shape[1] == 0:
            print("WARNING: Empty face1 ROI. Skipping.")
            continue

        blob1 = cv2.dnn.blobFromImage(face1, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    
    return blob1

parser = argparse.ArgumentParser()
parser.add_argument('--image', help='Path to image file. If not specified, webcam will be used.')
# You can uncomment and use '--video' if you want to explicitly handle video files
# parser.add_argument('--video', help='Path to video file. If not specified, webcam will be used.')

args = parser.parse_args()

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male','Female']

faceNet = cv2.dnn.readNet(faceModel,faceProto)
ageNet = cv2.dnn.readNet(ageModel,ageProto)
genderNet = cv2.dnn.readNet(genderModel,genderProto)


file=open(r"C:\Users\sandr\Documents\pix\faceRec\models\similar.txt",'r')
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
count_gender=0
total_pairs=0

for imgpair in imgpaths:
    total_pairs+=1
    lazypath=(imgpair).split(" - ")

    image1_path="C:/Users/sandr/Documents/pix/faceRec/corpus/"+str(lazypath[0])
    image2_path="C:/Users/sandr/Documents/pix/faceRec/corpus/"+str(lazypath[1])
    
    image1=cv2.imread(image1_path)
    image2=cv2.imread(image2_path)
    
    frame1=image1
    frame2=image2
    
    blob1 = give_Age_Gender(frame1)

    genderNet.setInput(blob1)
    genderPreds = genderNet.forward()
    gender1 = genderList[genderPreds[0].argmax()]

    ageNet.setInput(blob1)
    agePreds = ageNet.forward()
    age1 = ageList[agePreds[0].argmax()]
    
    blob2 = give_Age_Gender(frame2)

    genderNet.setInput(blob2)
    genderPreds = genderNet.forward()
    gender2 = genderList[genderPreds[0].argmax()]

    ageNet.setInput(blob2)
    agePreds = ageNet.forward()
    age2 = ageList[agePreds[0].argmax()]
    
    if age1==age2:
        count+=1
        
    if gender1==gender2:
        count_gender+=1
    
print('\nAccuracy :',round(count/total_pairs*100,2),'%\nTotal number of pictures :',total_pairs,'\nNew pictures correct :',count,'\nOld pictures :',total_pairs-count)
print('\nAccuracy :',round(count_gender/total_pairs*100,2),'%\nTotal number of pictures :',total_pairs,'\nPredicted correct :',count_gender)