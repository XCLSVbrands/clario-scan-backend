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


class AutoCornersRequest(BaseModel):
    imageBase64: str


# ---------- Hulpfuncties ----------

def _distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _order_points(pts):
    """Sorteer 4 punten naar (tl, tr, br, bl)."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]     # tl
    rect[2] = pts[np.argmax(s)]     # br
    rect[1] = pts[np.argmin(diff)]  # tr
    rect[3] = pts[np.argmax(diff)]  # bl
    return rect


def _shrink_quad(pts, factor=0.02):
    """
    Trek de vierhoek een klein beetje naar binnen
    zodat de crop strakker rond het document ligt.
    factor = 0.02 betekent ~2% naar binnen.
    """
    if pts.shape != (4, 2):
        return pts

    cx = np.mean(pts[:, 0])
    cy = np.mean(pts[:, 1])
    centered = pts - np.array([[cx, cy]])
    shrunk = centered * (1.0 - factor) + np.array([[cx, cy]])
    return shrunk


def _enhance_document_bw(img):
    """
    Maak van een foto een 'echte scan':
    - grijs
    - Otsu threshold -> bijna wit papier, donkere tekst
    - kleine ruis weg
    Retourneert een BGR-image zodat we als kleur-JPEG kunnen encoden.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # lichte blur tegen ruis
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # automatische drempel (zwart/wit)
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # kleine vlekjes weghalen
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # terug naar BGR zodat JPEG het snapt
    bw_bgr = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
    return bw_bgr


# ---------- ENDPOINT: perspectief-crop + scan-look ----------

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

    # trek de vierhoek een klein beetje naar binnen
    src = _shrink_quad(src, factor=0.02)

    # 3) Output size bepalen
    width_top = _distance(src[0], src[1])
    width_bottom = _distance(src[3], src[2])
    height_left = _distance(src[0], src[3])
    height_right = _distance(src[1], src[2])

    max_width = int(max(width_top, width_bottom))
    max_height = int(max(height_left, height_right))

    max_width = max(300, max_width)
    max_height = max(300, max_height)

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    # 4) Perspective transform (rechte, platte pagina)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (max_width, max_height))

    # 5) Document omzetten naar zwart/wit scan
    enhanced = _enhance_document_bw(warped)

    # 6) Encode naar JPEG en base64 teruggeven
    success, buf = cv2.imencode(".jpg", enhanced, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not success:
        return {"base64": None}

    out_b64 = base64.b64encode(buf).decode("utf-8")
    return {"base64": out_b64}


# ---------- ENDPOINT: auto-detect document corners ----------

@app.post("/api/document/autocorners")
def auto_detect_corners(req: AutoCornersRequest):
    img_data = base64.b64decode(req.imageBase64)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return {"corners": None}

    orig_h, orig_w = img.shape[:2]

    # 1) verklein voor snelheid
    max_side = max(orig_w, orig_h)
    scale = 1000.0 / max_side if max_side > 1000 else 1.0
    resized = cv2.resize(
        img,
        (int(orig_w * scale), int(orig_h * scale)),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )

    # 2) grijs + blur
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3) adaptive threshold om papier duidelijk te maken
    thr = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    # 4) randen + dichtmaken
    edges = cv2.Canny(thr, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    best_quad = None
    best_area = 0
    img_area = resized.shape[0] * resized.shape[1]

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.015 * peri, True)  # iets strakker

        if len(approx) != 4:
            continue

        area = cv2.contourArea(approx)
        area_ratio = area / float(img_area)

        # alleen grote vierhoeken (tussen 25% en 90% van het beeld)
        if area_ratio < 0.25 or area_ratio > 0.9:
            continue

        if area > best_area:
            best_area = area
            best_quad = approx

    if best_quad is None:
        # fallback: groot kader in het midden
        corners = {
            "tl": {"x": 0.08, "y": 0.08},
            "tr": {"x": 0.92, "y": 0.08},
            "br": {"x": 0.92, "y": 0.92},
            "bl": {"x": 0.08, "y": 0.92},
        }
        return {"corners": corners}

    # terugschalen naar originele resolutie
    pts = best_quad.reshape(4, 2).astype("float32")
    pts /= scale

    ordered = _order_points(pts)

    tl, tr, br, bl = ordered
    corners = {
        "tl": {"x": float(tl[0] / orig_w), "y": float(tl[1] / orig_h)},
        "tr": {"x": float(tr[0] / orig_w), "y": float(tr[1] / orig_h)},
        "br": {"x": float(br[0] / orig_w), "y": float(br[1] / orig_h)},
        "bl": {"x": float(bl[0] / orig_w), "y": float(bl[1] / orig_h)},
    }

    return {"corners": corners}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port="8000")
