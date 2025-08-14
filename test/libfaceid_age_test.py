from enum import Enum
import cv2
import os
from libfaceid.detector import FaceDetector, FaceDetectorModels
from libfaceid.encoder import FaceEncoder, FaceEncoderModels

# --- Your Custom Gender and Age Estimator Classes ---
# Paste these directly or ensure they are imported from their respective files

class FaceGenderEstimatorModels(Enum):
    CV2CAFFE = 0
    DEFAULT = CV2CAFFE

class FaceGenderEstimator:
    def __init__(self, model=FaceGenderEstimatorModels.DEFAULT, path=None):
        self._base = None
        if model == FaceGenderEstimatorModels.CV2CAFFE:
            self._base = FaceGenderEstimator_CV2CAFFE(path)

    def estimate(self, frame, face_image):
        return self._base.estimate(frame, face_image)

class FaceGenderEstimator_CV2CAFFE:
    def __init__(self, path):
        self._mean_values = (78.4263377603, 87.7689143744, 114.895847746)
        # Ensure 'path' ends with a slash if not already handled by os.path.join
        self._classifier = cv2.dnn.readNetFromCaffe(os.path.join(path, 'gender_deploy.prototxt'), os.path.join(path, 'gender_net.caffemodel'))
        self._selection = ['Male', 'Female']

    def estimate(self, frame, face_image):
        # Resize face_image to (227, 227) if it's not already, as expected by the model
        resized_face = cv2.resize(face_image, (227, 227))
        blob = cv2.dnn.blobFromImage(resized_face, 1, (227, 227), self._mean_values, swapRB=False)
        self._classifier.setInput(blob)
        prediction = self._classifier.forward()
        return self._selection[prediction[0].argmax()]


class FaceAgeEstimatorModels(Enum):
    CV2CAFFE = 0
    DEFAULT = CV2CAFFE

class FaceAgeEstimator:
    def __init__(self, model=FaceAgeEstimatorModels.DEFAULT, path=None):
        self._base = None
        if model == FaceAgeEstimatorModels.CV2CAFFE:
            self._base = FaceAgeEstimator_CV2CAFFE(path)

    def estimate(self, frame, face_image):
        return self._base.estimate(frame, face_image)

class FaceAgeEstimator_CV2CAFFE:
    def __init__(self, path):
        self._mean_values = (78.4263377603, 87.7689143744, 114.895847746)
        # Ensure 'path' ends with a slash if not already handled by os.path.join
        self._classifier = cv2.dnn.readNetFromCaffe(os.path.join(path, 'age_deploy.prototxt'), os.path.join(path, 'age_net.caffemodel'))
        self._selection = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']

    def estimate(self, frame, face_image):
        # Resize face_image to (227, 227) if it's not already, as expected by the model
        resized_face = cv2.resize(face_image, (227, 227))
        blob = cv2.dnn.blobFromImage(resized_face, 1, (227, 227), self._mean_values, swapRB=False)
        self._classifier.setInput(blob)
        prediction = self._classifier.forward()
        return self._selection[prediction[0].argmax()]

# --- Configuration ---
IMAGE_FOLDER = "C:/Users/sandr/Documents/pix/FaceRec/corpus"
MODELS_DIR = r"C:\Users\sandr\Documents\git\libfaceid_model\models\estimation" # Where your .prototxt and .caffemodel files are

INPUT_DIR_MODEL_ENCODING = os.path.join(MODELS_DIR, "dlib")
INPUT_DIR_MODEL_TRAINING = os.path.join(MODELS_DIR, "lbph")

# --- Initialize libfaceid components ---
try:
    print(f"Initializing FaceDetector (MTCNN)...")
    face_detector = FaceDetector(model=FaceDetectorModels.MTCNN, path=MODELS_DIR)

    print(f"Initializing Custom Gender Estimator...")
    gender_estimator = FaceGenderEstimator(model=FaceGenderEstimatorModels.DEFAULT, path=MODELS_DIR)

    print(f"Initializing Custom Age Estimator...")
    age_estimator = FaceAgeEstimator(model=FaceAgeEstimatorModels.DEFAULT, path=MODELS_DIR)


    # If you intend to do face encoding/recognition, initialize FaceEncoder.
    # print(f"Initializing FaceEncoder (DEFAULT)...")
    # face_encoder = FaceEncoder(model=FaceEncoderModels.DEFAULT, path=INPUT_DIR_MODEL_ENCODING, path_training=INPUT_DIR_MODEL_TRAINING, training=False)

except Exception as e:
    print(f"Error initializing libfaceid components: {e}")
    print("Please ensure your model paths are correct and all required model files are present.")
    exit()

'==========================================================================================='
file=open(r"C:\Users\sandr\Documents\pix\faceRec\models\similar.txt",'r')
# # file=open('similar.txt','r')
# # file=open('dissimilar.txt','r')
whole=file.read()
imgpaths=whole.split('\n')

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

    try:
        if image1 is None:
            continue
        faces = face_detector.detect(image1)

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                padding = 20
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(image1.shape[1], x + w + padding)
                y2 = min(image1.shape[0], y + h + padding)

                face_roi = image1[y1:y2, x1:x2]


                if face_roi.shape[0] > 0 and face_roi.shape[1] > 0:
                    gender1 = gender_estimator.estimate(image1, face_roi)
                    age1 = age_estimator.estimate(image1, face_roi)

        if image2 is None:
            continue
        faces = face_detector.detect(image2)

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                padding = 20
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(image2.shape[1], x + w + padding)
                y2 = min(image2.shape[0], y + h + padding)

                face_roi = image2[y1:y2, x1:x2]


                if face_roi.shape[0] > 0 and face_roi.shape[1] > 0:
                    gender2 = gender_estimator.estimate(image1, face_roi)
                    age2 = age_estimator.estimate(image1, face_roi)
        
        if age1==age2:
            count+=1
            
        if gender1==gender2:
            count_gender+=1
            

    except Exception as e:
            pass
        

print('\nAccuracy :',round(count/total_pairs*100,2),'%\nTotal number of pictures :',total_pairs,'\nNew pictures correct :',count,'\nOld pictures :',total_pairs-count)
print('\nAccuracy :',round(count_gender/total_pairs*100,2),'%\nTotal number of pictures :',total_pairs,'\nPredicted correct :',count_gender)
  
'=========================================================================================='
'=========================================================================================='
# for filename in os.listdir(IMAGE_FOLDER):
#     if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
#         image_path = os.path.join(IMAGE_FOLDER, filename)
#         print(f"\n--- Processing: {filename} ---")

#         try:
#             image = cv2.imread(image_path)
#             if image is None:
#                 print(f"Warning: Could not read image {filename}. Skipping.")
#                 continue

#             # 1. Detect faces
#             faces = face_detector.detect(image)
#             print(f"Found {len(faces)} face(s) in {filename}")

#             if len(faces) > 0:
#                 for (x, y, w, h) in faces:
#                     # Draw a rectangle around the detected face
#                     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

#                     # 2. Estimate Age and Gender for each detected face
#                     # Extract the face region of interest (ROI)
#                     # Add some padding to the face ROI to ensure the model gets enough context
#                     # The padding values can be adjusted
#                     padding = 20
#                     x1 = max(0, x - padding)
#                     y1 = max(0, y - padding)
#                     x2 = min(image.shape[1], x + w + padding)
#                     y2 = min(image.shape[0], y + h + padding)

#                     face_roi = image[y1:y2, x1:x2]


#                     if face_roi.shape[0] > 0 and face_roi.shape[1] > 0: # Ensure ROI is not empty
#                         gender = gender_estimator.estimate(image, face_roi) # Pass original frame and face_roi
#                         age = age_estimator.estimate(image, face_roi) # Pass original frame and face_roi
#                         print(f"   Face at ({x},{y}): Age = {age}, Gender = {gender}")

#                         # Put text on the image
#                         label = f"{gender}, {age}"
#                         cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#                     else:
#                         print(f"   Warning: Empty or invalid face ROI for a detected face in {filename}.")

#             processed_count += 1

#         except Exception as e:
#             print(f"Error processing {filename}: {e}")

# print(f"\nFinished processing {processed_count} images.")
# cv2.destroyAllWindows()
'===================================================================================================================================='
# # import sys
# # import argparse
# # import cv2
# # import datetime
# # from libfaceid.detector import FaceDetectorModels, FaceDetector
# # from libfaceid.age import FaceAgeEstimatorModels, FaceAgeEstimator

# # INPUT_DIR_MODEL_DETECTION       = "models/detection/"

# # detector=FaceDetectorModels.MTCNN

# # face_detector = FaceDetector(model=detector, path=INPUT_DIR_MODEL_DETECTION, minfacesize=120)

# # path=r"C:\Users\sandr\Documents\pix\FaceRec\corpus\Abhiram1.PNG"
# # frame=cv2.imread(path)
# # faces = face_detector.detect(frame)

# import cv2
# from libfaceid.detector import FaceDetectorModels, FaceDetector
# from libfaceid.encoder  import FaceEncoderModels, FaceEncoder

# from facial_recognition import label_face

# INPUT_DIR_MODEL_DETECTION = "models/detection/"
# INPUT_DIR_MODEL_ENCODING  = "models/encoding/"
# INPUT_DIR_MODEL_TRAINING  = "models/training/"

# imagePath=r"C:\Users\sandr\Documents\pix\FaceRec\corpus\Abhiram1.PNG"

# image = cv2.VideoCapture(imagePath)
# face_detector = FaceDetector(model=FaceDetectorModels.DEFAULT, path=INPUT_DIR_MODEL_DETECTION)
# face_encoder = FaceEncoder(model=FaceEncoderModels.DEFAULT, path=INPUT_DIR_MODEL_ENCODING, path_training=INPUT_DIR_MODEL_TRAINING, training=False)

# frame = image.read()
# faces = face_detector.detect(frame)
# for (index, face) in enumerate(faces):
#     face_id, confidence = face_encoder.identify(frame, face)
#     label_face(frame, face, face_id, confidence)
# cv2.imshow(frame)
# cv2.waitKey(5000)

# image.release()
# cv2.destroyAllWindows()