from fastapi import FastAPI
from pydantic import BaseModel
import base64
import cv2
import numpy as np

app = FastAPI()


class Point(BaseModel):
    x: float
    y: float


class Corners(BaseModel):
    tl: Point
    tr: Point
    br: Point
    bl: Point


class CropRequest(BaseModel):
    imageBase64: str
    corners: Corners


def _distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


@app.post("/api/document/crop")
def crop_document(req: CropRequest):
    # 1) Decode base64 image
    img_data = base64.b64decode(req.imageBase64)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return {"base64": None}

    h, w = img.shape[:2]
    c = req.corners

    # 2) Source points (in pixels) from normalized corners (0–1)
    src = np.array(
        [
            [c.tl.x * w, c.tl.y * h],
            [c.tr.x * w, c.tr.y * h],
            [c.br.x * w, c.br.y * h],
            [c.bl.x * w, c.bl.y * h],
        ],
        dtype="float32",
    )

    # 3) Bepaal outputgrootte (rechthoek)
    widt

