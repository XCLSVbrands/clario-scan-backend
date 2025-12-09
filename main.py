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
    # 1) Decode base64 naar een BGR-beeld
    img_data = base64.b64decode(req.imageBase64)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return {"base64": None}

    h, w = img.shape[:2]
    c = req.corners

    # 2) Source-punten (pixels) uit genormaliseerde hoeken
    src = np.array(
        [
            [c.tl.x * w, c.tl.y * h],
            [c.tr.x * w, c.tr.y * h],
            [c.br.x * w, c.br.y * h],
            [c.bl.x * w, c.bl.y * h],
        ],
        dtype="float32",
    )

    # 3) Doel-rechthoek (uitgangsformaat)
    width_top = _distance(src[0], src[1])
    width_bottom = _distance(src[3], src[2])
    height_left = _distance(src[0], src[3])
    height_right = _distance(src[1], src[2])

    max_width = int(max(width_top, width_bottom))
    max_height = int(max(height_left, height_right))

    max_width = max(200, max_width)
    max_height = max(200, max_height)

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    # 4) Perspective-correctie (document recht trekken)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (max_width, max_height))

    # 5) Lichte verbetering: kleur en contrast, maar geen harde threshold
    #    a) LAB + CLAHE op L-kanaal (lokale contrastverbetering)
    lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    #    b) milde ruisreductie (beter dan sterke blur)
    enhanced = cv2.bilateralFilter(enhanced, d=5, sigmaColor=25, sigmaSpace=25)

    #    c) lichte sharpen (unsharp mask)
    blur = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    sharp = cv2.addWeighted(enhanced, 1.2, blur, -0.2, 0)

    # 6) Encode naar JPEG met redelijke kwaliteit
    success, buf = cv2.imencode(
        ".jpg", sharp, [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    )
    if not success:
        return {"base64": None}

    out_b64 = base64.b64encode(buf).decode("utf-8")
    return {"base64": out_b64}
