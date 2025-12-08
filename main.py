from fastapi import FastAPI
from pydantic import BaseModel
import base64
import cv2
import numpy as np

app = FastAPI()


# ---------- Data models ----------

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


# ---------- Helpers ----------

def _distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _order_points(pts: np.ndarray) -> np.ndarray:
    """
    Orden de 4 punten als: top-left, top-right, bottom-right, bottom-left.
    pts: shape (4, 2)
    """
    rect = np.zeros((4, 2), dtype="float32")

    # som en verschil van coördinaten
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]      # tl = kleinste som
    rect[2] = pts[np.argmax(s)]      # br = grootste som
    rect[1] = pts[np.argmin(diff)]   # tr = kleinste (x - y)
    rect[3] = pts[np.argmax(diff)]   # bl = grootste (x - y)

    return rect


# ---------- API: detect corners ----------

@app.post("/api/document/detect-corners", response_model=DetectResponse)
def detect_corners(req: DetectRequest):
    """
    Zoekt automatisch de randen van het document.
    Geeft genormaliseerde hoeken (0–1) terug voor tl/tr/br/bl.
    """

    # Decode base64
    img_data = base64.b64decode(req.imageBase64)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        # Fallback: hele beeld als rechthoek
        h, w = 1, 1
    else:
        h, w = img.shape[:2]

    if img is None:
        # volledige afbeelding
        tl = Point(x=0.0, y=0.0)
        tr = Point(x=1.0, y=0.0)
        br = Point(x=1.0, y=1.0)
        bl = Point(x=0.0, y=1.0)
        return DetectResponse(corners=Corners(tl=tl, tr=tr, br=br, bl=bl))

    # verklein voor sneller detecteren
    scale = 800.0 / max(h, w)
    if scale < 1.0:
        resized = cv2.resize(img, (int(w * scale), int(h * scale)))
    else:
        resized = img.copy()

    rh, rw = resized.shape[:2]

    # grijs + blur + randen
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    doc_quad = None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            doc_quad = approx.reshape(4, 2)
            break

    if doc_quad is None:
        # fallback: hele beeld
        tl = Point(x=0.0, y=0.0)
        tr = Point(x=1.0, y=0.0)
        br = Point(x=1.0, y=1.0)
        bl = Point(x=0.0, y=1.0)
        return DetectResponse(corners=Corners(tl=tl, tr=tr, br=br, bl=bl))

    # punten ordenen
    ordered = _order_points(doc_quad)  # shape (4,2)

    # schaal terug naar originele resolutie
    if scale < 1.0:
        ordered = ordered / scale

    # normaliseren 0–1
    norm = np.zeros_like(ordered)
    norm[:, 0] = ordered[:, 0] / float(w)
    norm[:, 1] = ordered[:, 1] / float(h)

    tl = Point(x=float(norm[0, 0]), y=float(norm[0, 1]))
    tr = Point(x=float(norm[1, 0]), y=float(norm[1, 1]))
    br = Point(x=float(norm[2, 0]), y=float(norm[2, 1]))
    bl = Point(x=float(norm[3, 0]), y=float(norm[3, 1]))

    return DetectResponse(corners=Corners(tl=tl, tr=tr, br=br, bl=bl))


# ---------- API: crop + "scan" ----------

@app.post("/api/document/crop")
def crop_document(req: CropRequest):
    """
    Cropt het document met perspectief-correctie en geeft
    een zwart/wit 'gescande' versie terug.
    """

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

    # 4) Perspective transform (maakt de pagina recht / plat)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (max_width, max_height))

    # 5) Mild document-scan: grijs + kleine blur + OTSU
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    _, bw = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    # lichte morphology om kleine puntjes weg te halen
    kernel = np.ones((2, 2), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)

    # 6) Encode naar JPEG en terug naar base64 voor de app
    success, buf = cv2.imencode(".jpg", bw, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not success:
        return {"base64": None}

    out_b64 = base64.b64encode(buf).decode("utf-8")
    return {"base64": out_b64}
