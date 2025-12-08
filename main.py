from fastapi import FastAPI
from pydantic import BaseModel
import base64
import cv2
import numpy as np

app = FastAPI()


class Corners(BaseModel):
    tl: dict
    tr: dict
    br: dict
    bl: dict


class CropRequest(BaseModel):
    imageBase64: str
    corners: Corners


@app.post("/api/document/crop")
def crop_document(req: CropRequest):
    img_data = base64.b64decode(req.imageBase64)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"base64": None}

    h, w = img.shape[:2]
    c = req.corners

    src = np.array(
        [
            [c.tl["x"] * w, c.tl["y"] * h],
            [c.tr["x"] * w, c.tr["y"] * h],
            [c.br["x"] * w, c.br["y"] * h],
            [c.bl["x"] * w, c.bl["y"] * h],
        ],
        dtype="float32",
    )

    width_top = np.linalg.norm(src[0] - src[1])
    width_bottom = np.linalg.norm(src[3] - src[2])
    max_width = int(max(width_top, width_bottom))

    height_left = np.linalg.norm(src[0] - src[3])
    height_right = np.linalg.norm(src[1] - src[2])
    max_height = int(max(height_left, height_right))

    if max_width < 10 or max_height < 10:
        return {"base64": None}

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (max_width, max_height))

    ok, buf = cv2.imencode(".jpg", warped, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return {"base64": None}

    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return {"base64": b64}
