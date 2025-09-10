# Face Recognition models comparison

## Progress so far
- Face Recognition model Comparison 
- Face Detection Model Comparison
- Face Match base code
- Gender detection filter 
- Age detection filter 
- Age detection fine-tuning 
- Face Match Final code 
- Project Presentation 
- Local host Docker Deployment 
- Pixel Service Docker Deployment
- Logging and Monitor Memory leaks
- QA testing Repository Creation 
- QA testing and Initial Deployment 

## **Preprocessing Challenges**
- Poor lighting filter 
- Age detection filter
  - To filter out the ID photo which is older than 15 years
- Gender Detection Filter
  - To skip the initial filter


## **Face Recognition Models**
- VGG-Face
- Facenet
- OpenFace
- DeepFace
- DeepID
- **ArcFace**
- *Dlib*

## **Face Detection Models**
- Opencv
- mtcnn
- retinaface
- ssd

**OpenCV (Open Source Computer Vision Library)**

It is a comprehensive and widely-used open-source library that provides a rich set of tools and functions for computer vision and machine learning tasks. Developed primarily in C++ with interfaces for Python, Java, and MATLAB, OpenCV offers functionalities ranging from basic image and video processing operations (like filtering, transformations, and color manipulation) to more advanced algorithms for object detection, facial recognition, motion tracking, and augmented reality. Its extensive capabilities, cross-platform compatibility, and active community make it an indispensable resource for researchers and developers working in the field of computer vision.

**Multi-task Cascaded Convolutional Networks - mtcnn**

[Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://arxiv.org/pdf/1604.02878)

Multi-task Cascaded Convolutional Networks (MTCNN) is a popular and robust deep learning-based framework primarily used for face detection and facial landmark alignment. It operates in a cascaded three-stage architecture (P-Net, R-Net, O-Net) that progressively refines bounding box proposals and predicts five facial landmarks (eyes, nose, mouth corners) in a coarse-to-fine manner. This multi-task learning approach allows MTCNN to achieve high accuracy in detecting faces even in challenging "in-the-wild" conditions, making it a common choice for the initial stage of many face recognition pipelines.

![Screenshot 2025-05-29 114019](https://github.com/user-attachments/assets/cd127c28-8735-4dc4-9a6c-495f9e823a29)
![Screenshot 2025-05-29 114350](https://github.com/user-attachments/assets/1d7fb5e3-dd70-4979-a3e5-9214a9110988)

**RetinaFace**

RetinaFace is a cutting-edge, single-stage face detection algorithm that leverages deep learning to accurately identify faces and localize their key facial landmarks (like eyes, nose, and mouth) even in challenging "in-the-wild" conditions. It achieves high performance by combining face box prediction, 2D facial landmark localization, and 3D vertices regression into a unified, efficient framework. Trained on large datasets like WIDER FACE, RetinaFace is widely used in various applications requiring robust and fast face detection.

**Single Shot MultiBox Detector (SSD)**

It is a popular deep learning model for object detection that prioritizes both accuracy and speed. Unlike two-stage detectors that first propose regions and then classify them, SSD directly predicts bounding box offsets and class probabilities for multiple objects within a single forward pass of a convolutional neural network. It achieves this by using a set of default bounding boxes (priors) of varying scales and aspect ratios across different feature map layers, allowing it to detect objects at various sizes and locations efficiently. Its "single shot" nature makes it well-suited for real-time applications.
