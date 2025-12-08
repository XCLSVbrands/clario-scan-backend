from fastapi import FastAPI
from pydantic import BaseModel
import base64
import cv2
import numpy as np

app = FastAPI()

# ------------ MODELS ------------

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


class DetectResponse(BaseModel):
    corners: Corners


# ------------ HELPERS ------------

def _distance(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


def _order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]      # tl
    rect[2] = pts[np.argmax(s)]      # br
    rect[1] = pts[np.argmin(diff)]   # tr
    rect[3] = pts[np.argmax(diff)]   # bl

    return rect


# ------------ AUTO-DETECT ENDPOINT ------------

@app.post("/api/document/detect-corners", response_model=DetectResponse)
def detect_corners(req: DetectRequest):
    img_data = base64.b64decode(req.imageBase64)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        tl = Point(x=0.05, y=0.05)
        tr = Point(x=0.95, y=0.05)
        br = Point(x=0.95, y=0.95)
        bl = Point(x=0.05, y=0.95)
        return DetectResponse(corners=Corners(tl=tl, tr=tr, br=br, bl=bl))

    h, w = img.shape[:2]

    # verkleinen voor detectie
    scale = 800 / max(h, w)
    small = cv2.resize(img, (int(w*scale), int(h*scale))) if scale < 1 else img.copy()

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 60, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    quad = None
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.015 * peri, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2)
            break

    if quad is None:
        tl = Point(x=0.05, y=0.05)
        tr = Point(x=0.95, y=0.05)
        br = Point(x=0.95, y=0.95)
        bl = Point(x=0.05, y=0.95)
        return DetectResponse(corners=Corners(tl=tl, tr=tr, br=br, bl=bl))

    quad = quad / scale
    ordered = _order_points(quad)

    tl = Point(x=float(ordered[0, 0] / w), y=float(ordered[0, 1] / h))
    tr = Point(x=float(ordered[1, 0] / w), y=float(ordered[1, 1] / h))
    br = Point(x=float(ordered[2, 0] / w), y=float(ordered[2, 1] / h))
    bl = Point(x=float(ordered[3, 0] / w), y=float(ordered[3, 1] / h))

    return DetectResponse(corners=Corners(tl=tl, tr=tr, br=br, bl=bl))


# ------------ CROP + SCAN ENDPOINT ------------

@app.post("/api/document/crop")
def crop_document(req: CropRequest):
    img_data = base64.b64decode(req.imageBase64)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return {"base64": None}

    h, w = img.shape[:2]
    c = req.corners

    src = np.array([
        [c.tl.x * w, c.tl.y * h],
        [c.tr.x * w, c.tr.y * h],
        [c.br.x * w, c.br.y * h],
        [c.bl.x * w, c.bl.y * h],
    ], dtype="float32")

    width_top = _distance(src[0], src[1])
    width_bottom = _distance(src[3], src[2])
    height_left = _distance(src[0], src[3])
    height_right = _distance(src[1], src[2])

    max_width = max(int(width_top), int(width_bottom), 100)
    max_height = max(int(height_left), int(height_right), 100)

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (max_width, max_height))

    # ------ improved document scan ------
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, 12, 7, 21)

    # very light blur
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # hybrid threshold
    T, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    bw = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        25,
        -3
    )

    # clean dots
    kernel = np.ones((2, 2), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)

    success, buf = cv2.imencode(".jpg", bw, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not success:
        return {"base64": None}

    out = base64.b64encode(buf).decode("utf-8")
    return {"base64": out}
