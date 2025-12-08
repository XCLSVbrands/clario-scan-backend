from fastapi import FastAPI
from pydantic import BaseModel
import base64
import cv2
import numpy as np

app = FastAPI()


# --------- MODELS ---------

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


class DetectRequest(BaseModel):
    imageBase64: str


# --------- HELPERS ---------

def _distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _order_points(pts: np.ndarray) -> np.ndarray:
    # pts shape (4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]      # tl: kleinste x+y
    rect[2] = pts[np.argmax(s)]      # br: grootste x+y
    rect[1] = pts[np.argmin(diff)]   # tr: kleinste (x-y)
    rect[3] = pts[np.argmax(diff)]   # bl: grootste (x-y)
    return rect


# --------- DOC CROP + SCAN EFFECT ---------

@app.post("/api/document/crop")
def crop_document(req: CropRequest):
    # 1) Decode base64
    img_data = base64.b64decode(req.imageBase64)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return {"base64": None}

    h, w = img.shape[:2]
    c = req.corners

    # 2) Source points in pixels
    src = np.array(
        [
            [c.tl.x * w, c.tl.y * h],
            [c.tr.x * w, c.tr.y * h],
            [c.br.x * w, c.br.y * h],
            [c.bl.x * w, c.bl.y * h],
        ],
        dtype="float32",
    )

    # 3) Doel-afmeting
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

    # 4) Perspectief-correctie
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (max_width, max_height))

    # 5) "Scanner look" maar niet keihard zwart/wit
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    # lichte denoise
    gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # contrast verbeteren met CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # unsharp mask (scherpere letters)
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
    sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

    # terug naar 3-kanalen zodat het “normaal” oogt
    final = cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)

    # 6) Encode naar JPEG
    success, buf = cv2.imencode(".jpg", final, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not success:
        return {"base64": None}

    out_b64 = base64.b64encode(buf).decode("utf-8")
    return {"base64": out_b64}


# --------- AUTO CORNER DETECTION ---------

@app.post("/api/document/detect_corners")
def detect_corners(req: DetectRequest):
    img_data = base64.b64decode(req.imageBase64)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return {"corners": None}

    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    doc_cnt = None
    max_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.1 * w * h:  # te klein, skip
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4 and area > max_area:
            doc_cnt = approx
            max_area = area

    if doc_cnt is None:
        # fallback: hele beeld
        tl = {"x": 0.05, "y": 0.05}
        tr = {"x": 0.95, "y": 0.05}
        br = {"x": 0.95, "y": 0.95}
        bl = {"x": 0.05, "y": 0.95}
    else:
        pts = doc_cnt.reshape(4, 2).astype("float32")
        ordered = _order_points(pts)

        tl_pt, tr_pt, br_pt, bl_pt = ordered

        tl = {"x": float(tl_pt[0] / w), "y": float(tl_pt[1] / h)}
        tr = {"x": float(tr_pt[0] / w), "y": float(tr_pt[1] / h)}
        br = {"x": float(br_pt[0] / w), "y": float(br_pt[1] / h)}
        bl = {"x": float(bl_pt[0] / w), "y": float(bl_pt[1] / h)}

    return {"corners": {"tl": tl, "tr": tr, "br": br, "bl": bl}}

