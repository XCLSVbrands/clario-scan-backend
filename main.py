from fastapi import FastAPI
from pydantic import BaseModel
import base64
import cv2
import numpy as np

app = FastAPI()


# ---------- MODELS ----------

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


# ---------- HELPERS ----------

def _distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _order_points(pts):
    # pts: (4,2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]      # tl
    rect[2] = pts[np.argmax(s)]      # br
    rect[1] = pts[np.argmin(diff)]   # tr
    rect[3] = pts[np.argmax(diff)]   # bl
    return rect


# ---------- DETECT ENDPOINT ----------

@app.post("/api/document/detect")
def detect_document(req: DetectRequest):
  try:
      img_data = base64.b64decode(req.imageBase64)
      np_arr = np.frombuffer(img_data, np.uint8)
      img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
  except Exception:
      return {"corners": None}

  if img is None:
      return {"corners": None}

  h, w = img.shape[:2]

  # verklein voor snelheid, maar onthoud schaal
  scale = 600.0 / max(h, w)
  if scale < 1.0:
      img_small = cv2.resize(img, (int(w * scale), int(h * scale)))
  else:
      img_small = img.copy()
      scale = 1.0

  gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
  gray = cv2.GaussianBlur(gray, (5, 5), 0)

  # Canny + contours
  edged = cv2.Canny(gray, 50, 150)
  contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

  if not contours:
      return {"corners": None}

  contours = sorted(contours, key=cv2.contourArea, reverse=True)

  doc_cnt = None
  for c in contours:
      peri = cv2.arcLength(c, True)
      approx = cv2.approxPolyDP(c, 0.02 * peri, True)
      if len(approx) == 4:
          doc_cnt = approx
          break

  if doc_cnt is None:
      return {"corners": None}

  pts = doc_cnt.reshape(4, 2).astype("float32")

  # schaal terug naar originele resolutie
  pts /= scale

  rect = _order_points(pts)  # tl, tr, br, bl

  (tl, tr, br, bl) = rect
  # normaliseer naar 0–1
  tl_n = {"x": float(tl[0] / w), "y": float(tl[1] / h)}
  tr_n = {"x": float(tr[0] / w), "y": float(tr[1] / h)}
  br_n = {"x": float(br[0] / w), "y": float(br[1] / h)}
  bl_n = {"x": float(bl[0] / w), "y": float(bl[1] / h)}

  return {"corners": {"tl": tl_n, "tr": tr_n, "br": br_n, "bl": bl_n}}


# ---------- CROP ENDPOINT ----------

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

    # 4) Perspective transform (document recht maken)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (max_width, max_height))

    # 5) "Scanner effect" in kleur:
    #    - lichte denoise
    #    - lokale contrastverbetering (CLAHE op L-kanaal)
    denoised = cv2.fastNlMeansDenoisingColored(
        warped, None, 5, 5, 7, 21
    )

    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)

    lab2 = cv2.merge((l2, a, b))
    enhanced = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    # optioneel: klein beetje verscherpen
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], dtype="float32")
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    # 6) Encode naar JPEG en terug als base64
    success, buf = cv2.imencode(".jpg", sharpened, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not success:
        return {"base64": None}

    out_b64 = base64.b64encode(buf).decode("utf-8")
    return {"base64": out_b64}
