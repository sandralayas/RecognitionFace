import io
from typing import List, Dict, Any

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse

from insightcode_modified import image_face_embedding,compare_faces,get_person_gender,get_person_age
from imageQuality import preprocessing,find_which_preprocess

def filtering_preprocess(img_casual,img_id):

    if get_person_gender(img_casual)!=get_person_gender(img_id):
        match_status = "no_match"
        message='The found gender does not match'
        return False,match_status,message

    age1=get_person_age(img_casual)
    age2=get_person_age(img_id)

    diff=max(age1,age2)-min(age1,age2)

    if diff<15:
        match_status = "no_match"
        message='The found gender does not match'
        return False,match_status,message

    match_status = "match"
    message='The passed age and filter'
    return True,match_status,message