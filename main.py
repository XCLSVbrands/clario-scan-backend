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


class DetectRequest(BaseModel):
    imageBase64: str


def _distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _order_points(pts: np.ndarray) -> np.ndarray:
    # pts: shape (4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect


@app.post("/api/document/detect")
def detect_document(req: DetectRequest):
    """
    Zoek automatisch de document-rand en geef genormaliseerde hoeken terug
    (waarden tussen 0 en 1).
    """
    img_data = base64.b64decode(req.imageBase64)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        # fallback: hele beeld als document
        return {
            "corners": {
                "tl": {"x": 0.05, "y": 0.05},
                "tr": {"x": 0.95, "y": 0.05},
                "br": {"x": 0.95, "y": 0.95},
                "bl": {"x": 0.05, "y": 0.95},
            }
        }

    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 75, 200)

    contours, _ = cv2.findContours(
        edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    doc_cnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            doc_cnt = approx.reshape(4, 2)
            break

    if doc_cnt is None:
        # niets gevonden → hele beeld
        doc_cnt = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")

    rect = _order_points(doc_cnt.astype("float32"))

    # normaliseer 0–1
    norm = rect.copy()
    norm[:, 0] /= float(w)
    norm[:, 1] /= float(h)

    return {
        "corners": {
            "tl": {"x": float(norm[0, 0]), "y": float(norm[0, 1])},
            "tr": {"x": float(norm[1, 0]), "y": float(norm[1, 1])},
            "br": {"x": float(norm[2, 0]), "y": float(norm[2, 1])},
            "bl": {"x": float(norm[3, 0]), "y": float(norm[3, 1])},
        }
    }


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

    # 2) Source points (in pixels) from normalized corners
    src = np.array(
        [
            [c.tl.x * w, c.tl.y * h],
            [c.tr.x * w, c.tr.y * h],
            [c.br.x * w, c.br.y * h],
            [c.bl.x * w, c.bl.y * h],
        ],
        dtype="float32",
    )

    # 3) Compute output size (straight rectangle)
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

    # 4) Perspective transform
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (max_width, max_height))

    # 5) "Scanner look": grijs + denoise + scherpte + lichte boost
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    gray = cv2.fastNlMeansDenoising(
        gray, None, h=10, templateWindowSize=7, searchWindowSize=21
    )

    # unsharp mask (scherper)
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

    # iets lichter / meer contrast
    sharp = cv2.convertScaleAbs(sharp, alpha=1.1, beta=10)

    # Hier GEEN harde black/white threshold → tekst blijft leesbaar
    final_img = sharp

    # 6) Encode naar JPEG en terug naar base64
    success, buf = cv2.imencode(".jpg", final_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not success:
        return {"base64": None}

    out_b64 = base64.b64encode(buf).decode("utf-8")
    return {"base64": out_b64}



