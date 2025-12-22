# detector.py

import cv2
import json
import numpy as np
from PIL import Image
from ultralytics import YOLO
import io
import os

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
L_SHAPE_FILE = os.path.join(BASE_DIR, "lshape_data", "c_lshape.json")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")


# Load L-shape ROI
with open(L_SHAPE_FILE, "r") as f:
    data = json.load(f)

def to_np(pt): return np.array(pt, dtype=np.int32)

# Load Model Once
model = YOLO(MODEL_PATH)

# === Precompute static points ===
A = to_np(data["A"])
B = to_np(data["B"])
C = to_np(data["C"])
Q = to_np(data["Q"]) if data["Q"] else None
P = to_np(data["P"]) if data["P"] else None
O = to_np(data["O"]) if data["O"] else None
S = to_np(data["S"])
R = to_np(data["R"])
BOTTOM_RIGHT = to_np(data["bottom_right"])
BOTTOM_LEFT = to_np(data["bottom_left"])

def detect_and_classify(image_bytes):
    npimg = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    height, width = frame.shape[:2]

    # === Create zone mask ===
    zone_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.line(zone_mask, (A[0], height - A[1]), (B[0], height - B[1]), 1, 5)
    cv2.line(zone_mask, (B[0], height - B[1]), (C[0], height - C[1]), 1, 5)

    if Q is not None and P is not None and O is not None:
        triangle = np.array([
            (Q[0], height - Q[1]),
            (P[0], height - P[1]),
            (O[0], height - O[1])
        ])
        cv2.fillPoly(zone_mask, [triangle], 1)

    bc_shade = np.array([
        (S[0], height - S[1]),
        (R[0], height - R[1]),
        (BOTTOM_RIGHT[0], height - BOTTOM_RIGHT[1]),
        (BOTTOM_LEFT[0], height - BOTTOM_LEFT[1])
    ])
    cv2.fillPoly(zone_mask, [bc_shade], 1)

    # === Run YOLOv11 ===
    results = model(frame)[0]
    detections = []

    for idx, (mask, cls) in enumerate(zip(results.masks.data, results.boxes.cls)):
        class_id = int(cls.item())
        class_name = "Truck" if class_id == 0 else "Tyre"

        mask_np = mask.cpu().numpy()
        mask_resized = cv2.resize(mask_np, (width, height))
        mask_bin = (mask_resized > 0.5).astype(np.uint8) * 255

        # Check overlap
        contour_mask = np.zeros((height, width), dtype=np.uint8)
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contour_mask, contours, -1, 1, -1)
        overlap = cv2.bitwise_and(zone_mask, contour_mask)

        inside_lshape = bool(np.any(overlap))
        detections.append({
            "segment_index": idx,
            "class": class_name,
            "inside_lshape": inside_lshape
        })

    return detections
