# app.py
# uvicorn main:app --reload 

import io
from typing import Dict, Any

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, status, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# Assuming these modules are in your project directory
# from insightcode_modified import image_face_embedding, compare_faces
from fintuned_model import image_face_embedding, compare_faces
from imageQuality import preprocessing, find_which_preprocess
from age_gender_preprocess_function import filtering_preprocess
import time

# Set up FastAPI
app = FastAPI(
    title="InsightFace Face Matching API",
    description="API for face matching two images (casual and ID) using InsightFace.",
    version="1.0.0",
)

# This assumes your HTML file is in a 'static' directory
# Create a folder named 'static' and save the HTML file inside it
BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

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

    try:
        # 1. Read the images as NumPy arrays
        img_casual = await read_image_from_uploadfile(casual_image)
        img_id = await read_image_from_uploadfile(id_image)
        
        if img_casual is None or img_id is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="One or both images could not be processed."
            )

        start_time = time.time()
        # 2. Apply preprocessing for face quality and embeddings
        #    NOTE: This is the critical step to ensure consistency.
        #    The preprocessing must happen BEFORE getting embeddings.
        img_casual_preprocessed = preprocessing(img_casual, find_which_preprocess(img_casual))
        img_id_preprocessed = preprocessing(img_id, find_which_preprocess(img_id))
        
        # 3. Perform age and gender filtering
        passed_filter, filter_match_status, filter_message = filtering_preprocess(img_casual_preprocessed, img_id_preprocessed)

        end_time = time.time()
        
        if not passed_filter:
            return {
                "filter":passed_filter,
                "status": filter_match_status,
                "confidence": 0.0,
                "message": filter_message,
                "time": end_time-start_time
            }

        # 4. If filters pass, get embeddings from the PREPROCESSED images
        embd_casual = image_face_embedding(img_casual_preprocessed)
        embd_id = image_face_embedding(img_id_preprocessed)

        # 5. Compare the face embeddings
        similarity = compare_faces(embd_casual, embd_id)
        similarity_score = float(similarity)

        matching_threshold = 0.4

        if similarity_score >= matching_threshold:
            match_status = "match"
            message = "Faces match with high confidence and passed age/gender checks."
        else:
            match_status = "no_match"
            message = "Faces do not match based on the set threshold, but passed age/gender checks."

        end_time = time.time()
        
        return {
            "filter":passed_filter,
            "status": match_status,
            "confidence": similarity_score,
            "message": message,
            "time": end_time-start_time
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )
        

# New endpoint to serve the UI
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serves the HTML UI for the face-matching tool."""
    with open("static/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)