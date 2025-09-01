import cv2
import insightface
import numpy as np
from insightface.app import FaceAnalysis
import time
import math
import torch
import torch.nn as nn
from torchvision import transforms

# the url download error fix
insightface.model_zoo.set_model_path('/app')
model = insightface.model_zoo.get_model('buffalo_l')

# --- Your fine-tuned model architecture ---
# You need to define the model architecture that matches the one
# you used to create `age_estimator_finetuned.pth`.
# This is a placeholder example. Replace this with your actual model.
class FineTunedAgeEstimator(nn.Module):
    def __init__(self):
        super(FineTunedAgeEstimator, self).__init__()
        # Define your layers here
        # Example: self.layer1 = nn.Linear(512, 256)
        # self.output_layer = nn.Linear(256, 1)

    def forward(self, x):
        # Define the forward pass
        # Example: x = self.layer1(x)
        # return self.output_layer(x)
        pass

# --- Load the fine-tuned model and its weights ---
def load_finetuned_model(model_path):
    # This is a placeholder. You need to instantiate your model and load the weights.
    model = FineTunedAgeEstimator()
    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
    model.eval()  # Set the model to evaluation mode
    return model

# Load the model once when the script starts
try:
    fine_tuned_age_model = load_finetuned_model('gpu_finetuned_1.pth')
    print("Fine-tuned age model loaded successfully!")
except Exception as e:
    print(f"Error loading fine-tuned model: {e}")
    fine_tuned_age_model = None

# Your existing insightface app setup
app = FaceAnalysis(root='/root/app', providers=['CPUExecutionProvider'])
# app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1)

# --- The modified get_person_age function ---
def get_person_age(image):
    faces = app.get(image)

    if faces:
        main_face = faces[0]
        
        # --- Use the fine-tuned model if available ---
        if fine_tuned_age_model:
            # 1. Get the face embedding from insightface.
            # Your fine-tuned model might expect the embedding as input.
            face_embedding = main_face.embedding

            # 2. Run inference with the fine-tuned model.
            # The input format for your model might be different. 
            # You might need to preprocess the face image instead of the embedding.
            # This is a placeholder showing how to use the embedding.
            with torch.no_grad():
                input_tensor = torch.from_numpy(face_embedding).float().unsqueeze(0)
                estimated_age = fine_tuned_age_model(input_tensor).item()
            
            return math.floor(estimated_age)
        
        else:
            # Fallback to the default insightface age estimation
            estimated_age = main_face.age
            return estimated_age
    
    return None

# The rest of your code remains the same
def image_face_embedding(image_cv):
    img = image_cv
    if img is None:
        raise ValueError(f"Could not read image")
    
    faces = app.get(img)
    
    if len(faces) < 1:
        raise ValueError("No faces detected in the image")
    if len(faces) > 1:
        print("Warning: Multiple faces detected. Using first detected face")
    return faces[0].embedding

def get_person_gender(image):
    faces = app.get(image)
    if faces:
        main_face = faces[0]
        estimated_gender = main_face.gender
    return estimated_gender

def compare_faces(emb1, emb2, threshold=0.65):
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return similarity

def get_face_embedding(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    faces = app.get(img)
    
    if len(faces) < 1:
        raise ValueError("No faces detected in the image")
    if len(faces) > 1:
        print("Warning: Multiple faces detected. Using first detected face")
    
    return faces[0].embedding