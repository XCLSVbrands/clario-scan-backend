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


# ---------- Helpers ----------

def _order_points(pts: np.ndarray) -> np.ndarray:
    """
    Sorteer 4 punten: tl, tr, br, bl
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def _find_document_corners(img: np.ndarray) -> np.ndarray | None:
    """
    Probeer automatisch de grootste 4-hoek (document) te vinden.
    Retourneert 4 punten in (x, y) of None.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 60, 150)
    edges = cv2.dilate(edges, None, iterations=2)
    edges = cv2.erode(edges, None, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        # 4 hoekpunten = kandidaat document
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype("float32")
            return _order_points(pts)

    return None


# ---------- Endpoint ----------

@app.post("/api/document/crop")
def crop_document(req: CropRequest):
    """
    1) Decode base64 → OpenCV image
    2) Probeer document automatisch te vinden (4-hoek)
       - als dat lukt, gebruiken we die corners
       - anders gebruiken we de corners uit de app
    3) Pas perspectief-correctie toe
    4) Maak "scanner" versie (meer contrast / witter papier)
    5) Encode terug naar base64 JPEG
    """
    # ---- 1. Decode ----
    try:
        img_bytes = base64.b64decode(req.imageBase64)
    except Exception:
        return {"base64": None}

    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return {"base64": None}

    h, w = img.shape[:2]

    # ---- 2. Auto-detect corners ----
    auto_corners = _find_document_corners(img)

    if auto_corners is not None:
        src = auto_corners
    else:
        # fallback: gebruik corners van de gebruiker
        c = req.corners
        src = np.array(
            [
                [c.tl.x * w, c.tl.y * h],
                [c.tr.x * w, c.tr.y * h],
                [c.br.x * w, c.br.y * h],
                [c.bl.x * w, c.bl.y * h],
            ],
            dtype="float32",
        )
        src = _order_points(src)

    # ---- 3. Perspectief-correctie ----
    width_top = np.linalg.norm(src[0] - src[1])
    width_bottom = np.linalg.norm(src[3] - src[2])
    height_left = np.linalg.norm(src[0] - src[3])
    height_right = np.linalg.norm(src[1] - src[2])

    max_width = int(max(width_top, width_bottom))
    max_height = int(max(height_left, height_right))

    max_width = max(400, max_width)
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

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (max_width, max_height))

    # ---- 4. "Scanner" effect ----
    # We houden het bij grijs + contrast verbetering.
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    # zachte denoising tegen korrel
    gray = cv2.fastNlMeansDenoising(
        gray, None, h=10, templateWindowSize=7, searchWindowSize=21
    )

    # lokale contrastverhoging
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # lichte adaptive threshold (niet zo extreem als eerst)
    bw = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,  # blokgrootte
        5,   # offset, lager = minder hard contrast
    )

    # terug naar 3-kanaals beeld
    out = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

    # ---- 5. Encode naar base64 ----
    ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return {"base64": None}

    out_b64 = base64.b64encode(buf).decode("utf-8")
    return {"base64": out_b64}


