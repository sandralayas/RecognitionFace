from mtcnn import MTCNN
import cv2

def feature5(img_path):

    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    detector = MTCNN()
    faces = detector.detect_faces(image_rgb)
    landmarks_list = []
    if faces:
        for i, face_data in enumerate(faces):
            
            landmarks = face_data['keypoints']
            
            left_eye = landmarks['left_eye']
            right_eye = landmarks['right_eye']
            nose = landmarks['nose']
            mouth_left = landmarks['mouth_left']
            mouth_right = landmarks['mouth_right']

            landmarks_list.append([left_eye, right_eye, nose, mouth_left, mouth_right])
        
    return landmarks_list
        
# for i in landmarks_list:print(i)