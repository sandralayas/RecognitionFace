import cv2
import numpy as np
import os
import base64
from insightface.app import FaceAnalysis


app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1)

path=r"C:\Users\sandr\Documents\database\base64imagecorpus"
photo=os.listdir(path)
laplacian_var_list=[]

for i in range(len(photo)):
    with open(os.path.join(path,photo[i]), "r") as file:
        photo_string = file.read()
    
    photo_bytes = base64.b64decode(photo_string.encode('utf-8'))
    
    photo_array = np.frombuffer(photo_bytes, np.uint8)
    
    image = cv2.imdecode(photo_array, cv2.IMREAD_COLOR)
    
    faces = app.get(image)
    if len(faces) != 1:
        print(0)
        laplacian_var_list.append(0)
        continue
    bbox = faces[0].bbox.astype(int)
    x1, y1, x2, y2 = bbox
    image = image[y1:y2, x1:x2]
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    average_intensity = np.mean(gray_image)
    laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    
    print(average_intensity)
    laplacian_var_list.append(laplacian_var)
    # print(f"{photo[i]} - Brightness: {average_intensity:.2f}, Blur: {laplacian_var:.2f}")
print("Laplacian Variance List:")
for i in laplacian_var_list:
    print(i)