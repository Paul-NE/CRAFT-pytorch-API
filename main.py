import io
import pickle

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from fastapi import FastAPI
import PIL.ImageOps
import numpy as np
import PIL.Image
import cv2

import init_net

class BoxesResponse(BaseModel):
    boxes: list[list[list[float]]]  # Define a 2D array

app = FastAPI()
net, refine_net = init_net.craft_factory()

origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=methods,
    allow_headers=headers
)

@app.post("/text-found",  response_model=BoxesResponse)
async def get_prediction(file: UploadFile = File(...)):
    content = await file.read()
    pil_image = PIL.Image.open(io.BytesIO(content)).convert("RGB")
    # pil_image =  pil_image.resize((360, 360), PIL.ANTIALIAS)
    img_array = np.array(pil_image)
    print(img_array.shape)
    
    bboxes, polys, score_text = init_net.process_image(net, img_array, refine_net)
    return {"boxes": bboxes}