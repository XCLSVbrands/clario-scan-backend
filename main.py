from fastapi import FastAPI
from pydantic import BaseModel
import base64
import cv2
import numpy as np

app = FastAPI()


# ---------- Pydantic modellen ----------

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
    success: bool
    corners: Corners


# ---------- Hulpfuncties ----------

def _distance(a, b):
    return float(np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))


def _order_points(pts: np.ndarray) -> np.ndarray:
    """
    Sorteer 4 punten naar (tl, tr, br, bl).
    """
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # tl
    rect[2] = pts[np.argmax(s)]  # br

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # tr
    rect[3] = pts[np.argmax(diff)]  # bl

    return rect


def _default_corners(w: int, h: int) -> Corners:
    # Veilig fallback-gebied in het midden van de foto
    return Corners(
        tl=Point(x=0.1, y=0.1),
        tr=Point(x=0.9, y=0.1),
        br=Point(x=0.9, y=0.9),
        bl=Point(x=0.1, y=0.9),
    )


def _scan_postprocess(warped: np.ndarray) -> np.ndarray:
    """
    Maakt een 'scanner look': witte achtergrond, donkere tekst, redelijk scherp.
    Geeft een enkelkanaals (grijs) plaatje terug.
    """
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    # lichte denoise + blur
    gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # unsharp mask voor wat extra scherpte
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    sharp = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)

    # Otsu threshold -> wit papier, zwarte tekst
    _, bw = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return bw


# ---------- Hoek-detectie endpoint ----------

@app.post("/api/document/detect-corners", response_model=DetectResponse)
def detect_corners(req: DetectRequest):
    # Base64 decoderen
    try:
        img_data = base64.b64decode(req.imageBase64)
    except Exception:
        # kan niet decoderen -> fallback
        c = _default_corners(1, 1)
        return DetectResponse(success=False, corners=c)

    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        c = _default_corners(1, 1)
        return DetectResponse(success=False, corners=c)

    h, w = img.shape[:2]

    # verklein voor detectie (sneller, minder ruis)
    scale = 800.0 / max(w, h)
    if scale < 1.0:
        resized = cv2.resize(img, (int(w * scale), int(h * scale)))
    else:
        resized = img.copy()
        scale = 1.0

    rh, rw = resized.shape[:2]

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 50, 150)

    # randen een beetje dikker maken
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    doc_cnt = None
    max_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (rw * rh * 0.2):  # te klein -> negeren
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4 and area > max_area:
            doc_cnt = approx
            max_area = area

    if doc_cnt is None:
        # geen goede contour gevonden -> fallback
        c = _default_corners(w, h)
        return DetectResponse(success=False, corners=c)

    pts = doc_cnt.reshape(4, 2).astype("float32")
    pts = _order_points(pts)

    # terugschalen naar originele resolutie
    pts[:, 0] = pts[:, 0] / scale
    pts[:, 1] = pts[:, 1] / scale

    # normaliseren naar [0,1] voor frontend
    norm = np.zeros_like(pts)
    norm[:, 0] = pts[:, 0] / float(w)
    norm[:, 1] = pts[:, 1] / float(h)

    corners = Corners(
        tl=Point(x=float(norm[0, 0]), y=float(norm[0, 1])),
        tr=Point(x=float(norm[1, 0]), y=float(norm[1, 1])),
        br=Point(x=float(norm[2, 0]), y=float(norm[2, 1])),
        bl=Point(x=float(norm[3, 0]), y=float(norm[3, 1])),
    )

    return DetectResponse(success=True, corners=corners)


# ---------- Crop + scan endpoint ----------

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

    # 4) Perspective transform → pagina recht
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (max_width, max_height))

    # 5) Scanner-look (wit papier, donkere tekst)
    scanned = _scan_postprocess(warped)

    # 6) Encode naar JPEG en terug naar base64 voor de app
    success, buf = cv2.imencode(".jpg", scanned, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not success:
        return {"base64": None}

    out_b64 = base64.b64encode(buf).decode("utf-8")
    return {"base64": out_b64}
