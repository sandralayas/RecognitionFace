import io
from typing import List, Dict, Any

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse

from insightcode_modified import image_face_embedding,compare_faces,get_person_gender,get_person_age
from imageQuality import preprocessing,find_which_preprocess
from age_gender_preprocess_function import filtering_preprocess # Import filtering_preprocess

app = FastAPI(
    title="InsightFace Face Matching API",
    description="API for face matching two images (casual and ID) using InsightFace.",
    version="1.0.0",
)

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
        # 1. Image Preprocessing
        img_casual = await read_image_from_uploadfile(casual_image)
        img_id = await read_image_from_uploadfile(id_image)

        # Apply general preprocessing for face embedding and quality checks
        img_casual_preprocessed = preprocessing(img_casual, find_which_preprocess(img_casual))
        img_id_preprocessed = preprocessing(img_id, find_which_preprocess(img_id))

        # Perform age and gender filtering first
        passed_filter, filter_match_status, filter_message = filtering_preprocess(img_casual_preprocessed, img_id_preprocessed)

        if not passed_filter:
            return {
                "status": filter_match_status,
                "confidence": 0.0, # Or some appropriate value for non-match due to filter
                "message": filter_message
            }

        # If age and gender filter passes, proceed with face embedding and comparison
        embd_casual = image_face_embedding(img_casual_preprocessed)
        embd_id = image_face_embedding(img_id_preprocessed)

        similarity = compare_faces(embd_casual, embd_id)
        similarity_score = float(similarity)

        matching_threshold = 0.4

        if similarity_score >= matching_threshold:
            match_status = "match"
            message = "Faces match with high confidence and passed age/gender checks."
        else:
            match_status = "no_match"
            message = "Faces do not match based on the set threshold, but passed age/gender checks."

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
'----------------------------------------------------------------------------------------------------------------------'
# import io
# from typing import List, Dict, Any

# import cv2
# import numpy as np
# from fastapi import FastAPI, File, UploadFile, HTTPException, status
# from fastapi.responses import JSONResponse

# from insightcode_modified import image_face_embedding,compare_faces,get_person_gender,get_person_age
# from imageQuality import preprocessing,find_which_preprocess
# from age_gender_preprocess_function import filtering_preprocess

# app = FastAPI(
#     title="InsightFace Face Matching API",
#     description="API for face matching two images (casual and ID) using InsightFace.",
#     version="1.0.0",
# )

# async def read_image_from_uploadfile(file: UploadFile) -> np.ndarray:
#     """Reads an image from an UploadFile and converts it to a NumPy array (BGR format)."""
#     contents = await file.read()
#     nparr = np.frombuffer(contents, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     if img is None:
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail=f"Could not decode image from {file.filename}. Please ensure it's a valid image file."
#         )
#     return img

# @app.post("/face-match/")
# async def face_match(
#     casual_image: UploadFile = File(..., description="A casual photograph of the person."),
#     id_image: UploadFile = File(..., description="An ID photograph of the person.")
# ) -> Dict[str, Any]:
    
#     try:
#         # 1. Image Preprocessing
#         img_casual = await read_image_from_uploadfile(casual_image)
#         img_id = await read_image_from_uploadfile(id_image)
        
#         embd_casual=image_face_embedding(img_casual)
#         embd_id=image_face_embedding(img_id)
        
#         img_casual=preprocessing(img_casual,find_which_preprocess(img_casual))
#         img_id=preprocessing(img_id,find_which_preprocess(img_id))
        
#         # passed,match_status,message=filtering_preprocess(img_casual,img_id)
#         passed=True
        
#         if passed:
#             similarity = compare_faces(embd_casual,embd_id)
#             similarity_score=float(similarity)
            
#             matching_threshold = 0.4
            
#             if similarity_score >= matching_threshold:
#                 match_status = "match"
#                 message = "Faces match with high confidence."
#             else:
#                 match_status = "no_match"
#                 message = "Faces do not match based on the set threshold."

#             return {
#                 "status": match_status,
#                 "confidence": similarity_score,
#                 "message": message
#             }
#     except HTTPException as e:
#         raise e
#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"An unexpected error occurred: {str(e)}"
#         )