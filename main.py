# main.py

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import uvicorn

import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import base64
import pillow_heif

app = FastAPI()


# ----------------- Helpers -----------------


def read_image_bytes(content: bytes) -> np.ndarray:
    """
    Probeer eerst 'normale' JPEG/PNG via OpenCV.
    Als dat faalt (bijv. HEIC), gebruik pillow-heif.
    """
    np_img = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    if img is not None:
        return img

    # HEIC fallback
    heif_file = pillow_heif.read_heif(content)
    pil_img = Image.frombytes(
        heif_file.mode,
        heif_file.size,
        heif_file.data,
        "raw",
    )
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Sorteer 4 punten naar:
    top-left, top-right, bottom-right, bottom-left
    """
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left

    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Perspectief-correctie op basis van 4 punten.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # breedtes
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    # hoogtes
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array(
        [
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1],
        ],
        dtype="float32",
    )

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def detect_document_corners(image: np.ndarray):
    """
    Zoek het grootste 'document'-vlak (4-hoek).
    Retourneert 4 punten (tl, tr, br, bl) of None.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 50, 150)

    contours, _ = cv2.findContours(
        edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype("float32")
            return order_points(pts)

    return None


def enhance_document(image: np.ndarray) -> np.ndarray:
    """
    Maak tekst duidelijker zonder hard zwart/wit:
    - CLAHE op grijsbeeld
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    enhanced = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
    return enhanced


def encode_image_to_base64(image: np.ndarray, quality: int = 95) -> str:
    """
    Converteer een OpenCV image naar JPEG base64 string.
    """
    if len(image.shape) == 2:
        pil_img = Image.fromarray(image)
    else:
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    buf = BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ----------------- Endpoints -----------------


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """
    1) Ontvangt de originele foto.
    2) Detecteert document-hoeken.
    3) Stuurt width, height en corners.{topLeft, topRight, bottomRight, bottomLeft} terug.
    """
    content = await file.read()
    image = read_image_bytes(content)

    if image is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Could not decode image"},
        )

    h, w = image.shape[:2]
    corners = detect_document_corners(image)

    if corners is None:
        # fallback: hele afbeelding
        corners = np.array(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
            dtype="float32",
        )

    # corners is ge-ordered: tl, tr, br, bl
    tl, tr, br, bl = corners

    return {
        "width": w,
        "height": h,
        "corners": {
            "topLeft": {"x": float(tl[0]), "y": float(tl[1])},
            "topRight": {"x": float(tr[0]), "y": float(tr[1])},
            "bottomRight": {"x": float(br[0]), "y": float(br[1])},
            "bottomLeft": {"x": float(bl[0]), "y": float(bl[1])},
        },
    }


@app.post("/crop")
async def crop(
    file: UploadFile = File(...),
    topLeftX: float = Form(...),
    topLeftY: float = Form(...),
    topRightX: float = Form(...),
    topRightY: float = Form(...),
    bottomRightX: float = Form(...),
    bottomRightY: float = Form(...),
    bottomLeftX: float = Form(...),
    bottomLeftY: float = Form(...),
):
    """
    1) Ontvangt originele foto + definitieve hoeken (pixels).
    2) Doet perspectief-correctie.
    3) Verbetert leesbaarheid licht met CLAHE.
    4) Stuurt scherpe JPEG als base64 terug.
    """
    content = await file.read()
    image = read_image_bytes(content)

    if image is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Could not decode image"},
        )

    pts = np.array(
        [
            [topLeftX, topLeftY],
            [topRightX, topRightY],
            [bottomRightX, bottomRightY],
            [bottomLeftX, bottomLeftY],
        ],
        dtype="float32",
    )

    warped = four_point_transform(image, pts)
    enhanced = enhance_document(warped)

    image_base64 = encode_image_to_base64(enhanced, quality=95)

    return {"image_base64": image_base64}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
