from fastapi import FastAPI
from pydantic import BaseModel
import base64
import cv2
import numpy as np

app = FastAPI()


# ---------- Models ----------

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


# ---------- Helpers ----------

def _distance(a, b):
    return float(np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))


def _order_points(pts):
    """
    Neem 4 punten (x, y) en orden ze als tl, tr, br, bl.
    """
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # tl
    rect[2] = pts[np.argmax(s)]  # br

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # tr
    rect[3] = pts[np.argmax(diff)]  # bl

    return rect


# ---------- /api/document/crop ----------

@app.post("/api/document/crop")
def crop_document(req: CropRequest):
    # 1) Decode base64 image
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

    # 2) Source points (pixels from normalized corners)
    src = np.array(
        [
            [c.tl.x * w, c.tl.y * h],
            [c.tr.x * w, c.tr.y * h],
            [c.br.x * w, c.br.y * h],
            [c.bl.x * w, c.bl.y * h],
        ],
        dtype="float32",
    )

    # 3) Output size (straight rectangle)
    width_top = _distance(src[0], src[1])
    width_bottom = _distance(src[3], src[2])
    height_left = _distance(src[0], src[3])
    height_right = _distance(src[1], src[2])

    max_width = int(max(width_top, width_bottom))
    max_height = int(max(height_left, height_right))

    max_width = max(300, max_width)
    max_height = max(400, max_height)

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    # 4) Perspective transform: maak de pagina recht
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (max_width, max_height))

    # ---------- "Scanner" effect ----------
    # 5) Grijs + local contrast (CLAHE) + lichte sharpening
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    # Adaptive contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Light denoise
    enhanced = cv2.fastNlMeansDenoising(enhanced, None, h=8, templateWindowSize=7, searchWindowSize=21)

    # Mild sharpening
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], dtype="float32")
    sharp = cv2.filter2D(enhanced, -1, kernel)

    # Maak er 3-kanalen beeld van voor nette JPEG
    out = cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)

    # 6) Encode to JPEG -> base64
    success, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not success:
        return {"base64": None}

    out_b64 = base64.b64encode(buf).decode("utf-8")
    return {"base64": out_b64}


# ---------- /api/document/detect ----------

@app.post("/api/document/detect")
def detect_document(req: DetectRequest):
    """
    Vind document-contour en geef corners terug als normalized coords (0..1).
    """
    try:
        img_data = base64.b64decode(req.imageBase64)
    except Exception:
        return {"corners": None}

    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return {"corners": None}

    h, w = img.shape[:2]

    # 1) Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2) Edges + contours
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {"corners": None}

    # 3) Pak grootste 4-hoek
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    doc_cnt = None
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            doc_cnt = approx
            break

    if doc_cnt is None:
        return {"corners": None}

    pts = doc_cnt.reshape(4, 2).astype("float32")
    ordered = _order_points(pts)

    # 4) Normaliseer naar 0..1
    tl, tr, br, bl = ordered
    corners = {
        "tl": {"x": float(tl[0] / w), "y": float(tl[1] / h)},
        "tr": {"x": float(tr[0] / w), "y": float(tr[1] / h)},
        "br": {"x": float(br[0] / w), "y": float(br[1] / h)},
        "bl": {"x": float(bl[0] / w), "y": float(bl[1] / h)},
    }

    return {"corners": corners}
