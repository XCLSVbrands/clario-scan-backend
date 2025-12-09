# main.py

from fastapi import FastAPI
from pydantic import BaseModel
import base64
import cv2
import numpy as np

app = FastAPI()


# ---------- Models ----------

class Point(BaseModel):
    x: float  # genormaliseerd 0..1
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

def _distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _unsharp_mask(bgr, amount=1.0, radius=3, threshold=0):
    """
    Simpele unsharp mask in kleur.
    amount   = hoeveel scherpte (0.5–1.5 is normaal)
    radius   = blur kernel (3 of 5)
    threshold= niet gebruiken (laten op 0)
    """
    blurred = cv2.GaussianBlur(bgr, (radius * 2 + 1, radius * 2 + 1), 0)
    # versterk verschil tussen origineel en blur
    sharpened = cv2.addWeighted(bgr, 1 + amount, blurred, -amount, 0)

    # clip naar geldige range
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    return sharpened


def _enhance_document_color(bgr):
    """
    Houdt het document in kleur, maakt achtergrond witter
    en tekst wat duidelijker, zonder harde zwart/wit threshold.
    """

    # 1) lichte ruisonderdrukking in kleur
    #    h=5 is mild. Als het nog te "vlekkerig" is, verlaag naar 3.
    denoised = cv2.fastNlMeansDenoisingColored(
        bgr, None,
        h=5,         # luminantie ruis
        hColor=5,    # kleur ruis
        templateWindowSize=7,
        searchWindowSize=21,
    )

    # 2) Werk in LAB-kleurruimte: L = lichtheid,
    #    daarop doen we lokale contrastverbetering (CLAHE)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE = adaptieve contrast-herverdeling
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)

    lab2 = cv2.merge((l2, a, b))
    enhanced = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    # 3) milde sharpening voor tekst
    sharpened = _unsharp_mask(enhanced, amount=0.8, radius=2)

    return sharpened


# ---------- Routes ----------

@app.get("/")
def health():
    return {"status": "ok"}


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

    # 2) Source points (in pixels) van genormaliseerde hoeken
    src = np.array(
        [
            [c.tl.x * w, c.tl.y * h],
            [c.tr.x * w, c.tr.y * h],
            [c.br.x * w, c.br.y * h],
            [c.bl.x * w, c.bl.y * h],
        ],
        dtype="float32",
    )

    # 3) Bepaal rechte doel-rect
    width_top = _distance(src[0], src[1])
    width_bottom = _distance(src[3], src[2])
    height_left = _distance(src[0], src[3])
    height_right = _distance(src[1], src[2])

    max_width = int(max(width_top, width_bottom))
    max_height = int(max(height_left, height_right))

    # minimale grootte voor veiligheid
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

    # 4) Perspectief-correctie (document recht trekken)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (max_width, max_height))

    # 5) Document-verbetering in kleur
    doc_color = _enhance_document_color(warped)

    # 6) Eventueel verkleinen voor filesize (optioneel)
    #    Max breedte 1600px
    max_side = 1600
    h_doc, w_doc = doc_color.shape[:2]
    scale = min(1.0, float(max_side) / max(w_doc, h_doc))
    if scale < 1.0:
        new_w = int(w_doc * scale)
        new_h = int(h_doc * scale)
        doc_color = cv2.resize(doc_color, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 7) Encode naar JPEG en terug naar base64
    success, buf = cv2.imencode(
        ".jpg",
        doc_color,
        [int(cv2.IMWRITE_JPEG_QUALITY), 90],
    )

    if not success:
        return {"base64": None}

    out_b64 = base64.b64encode(buf).decode("utf-8")
    return {"base64": out_b64}
