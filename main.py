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
    factor = 0.02 = ~2% naar binnen.
    """
    if pts.shape != (4, 2):
        return pts

    cx = np.mean(pts[:, 0])
    cy = np.mean(pts[:, 1])
    centered = pts - np.array([[cx, cy]])
    shrunk = centered * (1.0 - factor) + np.array([[cx, cy]])
    return shrunk


def _enhance_document_color(img):
    """
    Kleur-document licht verbeteren:
    - beetje ruisreductie
    - contrast iets omhoog met CLAHE
    - subtiel verscherpen
    NIET naar zwart/wit, zodat tekst leesbaar blijft.
    """
    denoised = cv2.fastNlMeansDenoisingColored(
        img, None, h=5, hColor=5, templateWindowSize=7, searchWindowSize=21
    )

    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab_enhanced = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    gaussian = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    sharp = cv2.addWeighted(enhanced, 1.4, gaussian, -0.4, 0)

    return sharp


# ---------- ENDPOINT: perspectief-crop + lichte kleurbewerking ----------

@app.post("/api/document/crop")
def crop_document(req: CropRequest):
    img_data = base64.b64decode(req.imageBase64)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return {"base64": None}

    h, w = img.shape[:2]
    c = req.corners

    # normalized -> pixels
    src = np.array(
        [
            [c.tl.x * w, c.tl.y * h],
            [c.tr.x * w, c.tr.y * h],
            [c.br.x * w, c.br.y * h],
            [c.bl.x * w, c.bl.y * h],
        ],
        dtype="float32",
    )

    # vierhoek klein beetje naar binnen voor strakkere crop
    src = _shrink_quad(src, factor=0.02)

    # outputgrootte bepalen
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

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (max_width, max_height))

    enhanced = _enhance_document_color(warped)

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

    max_side = max(orig_w, orig_h)
    scale = 1000.0 / max_side if max_side > 1000 else 1.0
    resized = cv2.resize(
        img,
        (int(orig_w * scale), int(orig_h * scale)),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    thr = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

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
        approx = cv2.approxPolyDP(cnt, 0.015 * peri, True)

        if len(approx) != 4:
            continue

        area = cv2.contourArea(approx)
        area_ratio = area / float(img_area)

        if area_ratio < 0.25 or area_ratio > 0.9:
            continue

        if area > best_area:
            best_area = area
            best_quad = approx

    if best_quad is None:
        corners = {
            "tl": {"x": 0.08, "y": 0.08},
            "tr": {"x": 0.92, "y": 0.08},
            "br": {"x": 0.92, "y": 0.92},
            "bl": {"x": 0.08, "y": 0.92},
        }
        return {"corners": corners}

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
    uvicorn.run(app, host="0.0.0.0", port=8000)
