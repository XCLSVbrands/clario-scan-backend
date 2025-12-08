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
    # 1) Decode base64
    try:
        img_data = base64.b64decode(req.imageBase64)
    except Exception:
        return {"base64": None}

    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return {"base64": None}

    h, w = img.shape[:2]
    c = req.corners

    # 2) Source points (pixels) from normalized corners
    src = np.array(
        [
            [c.tl.x * w, c.tl.y * h],
            [c.tr.x * w, c.tr.y * h],
            [c.br.x * w, c.br.y * h],
            [c.bl.x * w, c.bl.y * h],
        ],
        dtype="float32",
    )

    # 3) Output size (rechte rechthoek)
    width_top = _distance(src[0], src[1])
    width_bottom = _distance(src[3], src[2])
    height_left = _distance(src[0], src[3])
    height_right = _distance(src[1], src[2])

    max_width = int(max(width_top, width_bottom))
    max_height = int(max(height_left, height_right))

    max_width = max(100, max_width)
    max_height = max(100, max_height)

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    # 4) Perspective transform → document recht trekken
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (max_width, max_height))

    # -------------------------------------------------
    # 5) KLEUR-SCANNER EFFECT  (option C)
    #    - witbalans
    #    - lokale contrastboost
    #    - lichte denoise
    #    - verscherpen
    #    -> kleur blijft behouden (logo blijft rood)
    # -------------------------------------------------

    # 5a) Simple gray-world white balance
    result = warped.astype(np.float32)

    avg_b = np.mean(result[:, :, 0])
    avg_g = np.mean(result[:, :, 1])
    avg_r = np.mean(result[:, :, 2])
    avg_gray = (avg_b + avg_g + avg_r) / 3.0 + 1e-6

    result[:, :, 0] *= avg_gray / (avg_b + 1e-6)
    result[:, :, 1] *= avg_gray / (avg_g + 1e-6)
    result[:, :, 2] *= avg_gray / (avg_r + 1e-6)

    result = np.clip(result, 0, 255).astype(np.uint8)

    # 5b) LAB + CLAHE voor lokale contrastverbetering
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)

    lab2 = cv2.merge((l2, a, b))
    result = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    # 5c) Lichte kleur-denoise
    result = cv2.fastNlMeansDenoisingColored(result, None, 5, 5, 7, 21)

    # 5d) Unsharp mask voor scherpe letters
    blur = cv2.GaussianBlur(result, (0, 0), sigmaX=1.0)
    result = cv2.addWeighted(result, 1.5, blur, -0.5, 0)

    # 5e) Klein beetje extra helderheid
    result = cv2.convertScaleAbs(result, alpha=1.05, beta=5)

    # 6) Encode terug naar JPEG → base64
    success, buf = cv2.imencode(
        ".jpg", result, [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    )
    if not success:
        return {"base64": None}

    out_b64 = base64.b64encode(buf).decode("utf-8")
    return {"base64": out_b64}

