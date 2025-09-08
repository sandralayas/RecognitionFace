import io
from typing import List, Dict, Any

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse

from insightcode_modified import crop_face,get_person_gender,get_person_age
from imageQuality import preprocessing,find_which_preprocess

def filtering_preprocess(img_casual, img_id):
    """
    Applies preprocessing filters to two images and checks for age/gender consistency.
    """
    # 1. Process and filter the casual image
    cropped_casual = crop_face(img_casual)
    if cropped_casual is None:
        return False, "no_match", "No face detected in the casual image."

    # 2. Process and filter the ID image
    cropped_id = crop_face(img_id)
    if cropped_id is None:
        return False, "no_match", "No face detected in the ID image."

    # 3. Check for image quality
    for cropped_img in [cropped_casual, cropped_id]:
        image_defect = find_which_preprocess(cropped_img)
        if image_defect != 'normal':
            message = f"Poor image quality detected: {image_defect}."
            return False, "no_match", message

    # 4. Perform age and gender checks
    age1 = get_person_age(img_casual)
    age2 = get_person_age(img_id)

    # Add a safety check in case get_person_age returns None
    if age1 is None or age2 is None:
        return False, "no_match", "Failed to determine age from one or both images."

    diff = abs(age1 - age2)

    if diff >= 15:
        message = 'The detected ages differ significantly.'
        return False, "no_match", message

    gender1 = get_person_gender(img_casual)
    gender2 = get_person_gender(img_id)

    if gender1 != gender2:
        message = 'The detected genders do not match.'
        return False, "no_match", message

    # 5. All checks passed
    return True, "match", "The detected age and gender pass initial filtering."