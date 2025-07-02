import io
from typing import List, Dict, Any

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
from insightface.app import FaceAnalysis

app = FastAPI(
    title="InsightFace Face Matching API",
    description="API for face matching two images (casual and ID) using InsightFace.",
    version="1.0.0",
)

# Initialize InsightFace model globally for efficiency
# 'buffalo_l' is a commonly used model for face recognition.
# You might need to download the model if it's not present locally (~/.insightface/models)
# The first time you run this, InsightFace might download the model.

try:
    face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=(640, 640))
except Exception as e:
    raise RuntimeError(f"Failed to load InsightFace model: {e}. "
                       "Ensure the 'buffalo_l' model is available or downloaded.")

async def read_image_from_uploadfile(file: UploadFile) -> np.ndarray:
    """Reads an image from an UploadFile and converts it to a NumPy array (BGR format)."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not decode image from {file.filename}. Please ensure it's a valid image file."
        )
    return img

@app.post("/face-match/")
async def face_match(
    casual_image: UploadFile = File(..., description="A casual photograph of the person."),
    id_image: UploadFile = File(..., description="An ID photograph of the person.")
) -> Dict[str, Any]:
    """
    Performs face matching between a casual image and an ID image.

    Args:
        casual_image (UploadFile): The casual image file.
        id_image (UploadFile): The ID image file.

    Returns:
        Dict[str, Any]: A dictionary containing matching status, confidence score, and messages.
    """
    try:
        # 1. Image Preprocessing
        img_casual = await read_image_from_uploadfile(casual_image)
        img_id = await read_image_from_uploadfile(id_image)

        # 2. Face Detection for casual image
        faces_casual = face_app.get(img_casual)
        if not faces_casual:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No face detected in the casual image."
            )
        if len(faces_casual) > 1:
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={"status": "multiple_faces_casual",
                         "message": "Multiple faces detected in the casual image. Please provide an image with a single clear face.",
                         "confidence": None}
            )
        face_casual = faces_casual[0].embedding

        # 3. Face Detection for ID image
        faces_id = face_app.get(img_id)
        if not faces_id:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No face detected in the ID image."
            )
        if len(faces_id) > 1:
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={"status": "multiple_faces_id",
                         "message": "Multiple faces detected in the ID image. Please provide an image with a single clear face.",
                         "confidence": None}
            )
        face_id = faces_id[0].embedding

        # 4. Face Embedding Extraction (InsightFace does this as part of `get()` and stores in face.embedding)
        # embedding_casual = face_casual.embedding
        # embedding_id = face_id.embedding

        # print('embedding_casual',embedding_casual,'\nembedding_id',embedding_id)
        
        # 5. Face Matching (Cosine Similarity)
        # InsightFace provides a function for similarity calculation
        similarity_score = np.dot(face_casual, face_id) / (np.linalg.norm(face_casual) * np.linalg.norm(face_id))

        print('\nSIMILARITY',similarity_score,'\n')
        
        # You can define a threshold for "matching"
        matching_threshold = 0.5  # This threshold can be fine-tuned based on your specific needs

        if similarity_score >= matching_threshold:
            match_status = "match"
            message = "Faces match with high confidence."
        else:
            match_status = "no_match"
            message = "Faces do not match based on the set threshold."

        return {
            "status": match_status,
            "confidence": similarity_score,
            "message": message
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )

# Optional: Root endpoint for health check
@app.get("/")
async def read_root():
    return {"message": "InsightFace Face Matching API is running!"}