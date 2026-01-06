# detector.py

import cv2
import json
import numpy as np
from PIL import Image
from ultralytics import YOLO
import io
import os
import base64

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
L_SHAPE_FILE = os.path.join(BASE_DIR, "lshape_data", "plain_lshape_test.json")
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
Q = to_np(data["Q"]) if data.get("Q") else None
P = to_np(data["P"]) if data.get("P") else None
O = to_np(data["O"]) if data.get("O") else None
S = to_np(data["S"])
R = to_np(data["R"])
BOTTOM_RIGHT = to_np(data.get("bottom_right") or data.get("BOTTOM_RIGHT"))
BOTTOM_LEFT = to_np(data.get("bottom_left") or data.get("BOTTOM_LEFT"))

def detect_and_classify(image_bytes):
    npimg = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    height, width = frame.shape[:2]

    # Create a copy for annotation
    annotated_frame = frame.copy()

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

    # === Draw shaded regions on annotated frame ===
    # Create overlay for shaded regions
    shade_overlay = annotated_frame.copy()
    shade_color = (255, 0, 255)  # Magenta for shaded area

    # Draw triangle region (upper shaded area)
    if Q is not None and P is not None and O is not None:
        triangle_pts = np.array([
            (Q[0], height - Q[1]),
            (P[0], height - P[1]),
            (O[0], height - O[1])
        ], dtype=np.int32)
        cv2.fillPoly(shade_overlay, [triangle_pts], shade_color)

    # Draw BC shade region (lower shaded area)
    bc_shade_pts = np.array([
        (S[0], height - S[1]),
        (R[0], height - R[1]),
        (BOTTOM_RIGHT[0], height - BOTTOM_RIGHT[1]),
        (BOTTOM_LEFT[0], height - BOTTOM_LEFT[1])
    ], dtype=np.int32)
    cv2.fillPoly(shade_overlay, [bc_shade_pts], shade_color)

    # Blend shaded overlay with annotated frame (0.3 opacity)
    annotated_frame = cv2.addWeighted(shade_overlay, 0.3, annotated_frame, 0.7, 0)

    # === Draw L-shape zone on annotated frame ===
    # Draw the L-shape lines (A -> B -> C)
    pt_a = (A[0], height - A[1])
    pt_b = (B[0], height - B[1])
    pt_c = (C[0], height - C[1])

    cv2.line(annotated_frame, pt_a, pt_b, (0, 255, 0), 3)  # Green line A-B
    cv2.line(annotated_frame, pt_b, pt_c, (0, 255, 0), 3)  # Green line B-C

    # Draw corner points with labels
    for name, pt in [("A", pt_a), ("B", pt_b), ("C", pt_c)]:
        cv2.circle(annotated_frame, pt, 8, (0, 0, 255), -1)  # Red filled circle
        cv2.putText(annotated_frame, name, (pt[0] + 10, pt[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # === Run YOLOv11 ===
    results = model(frame)[0]
    detections = []

    if results.masks is not None:
        for idx, (mask, cls, box) in enumerate(zip(results.masks.data, results.boxes.cls, results.boxes.xyxy)):
            class_name = "Tyre"  # Model only has 1 class: tyre (class_id = 0)

            mask_np = mask.cpu().numpy()
            mask_resized = cv2.resize(mask_np, (width, height))
            mask_bin = (mask_resized > 0.5).astype(np.uint8) * 255

            # Get contours from mask
            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Check if polygon contour points are in shaded area (Tyres only)
            any_point_in_shade = False
            all_points_in_shade = False
            if class_name == "Tyre" and contours:
                # Get all polygon contour points
                all_contour_points = np.vstack(contours).reshape(-1, 2)

                # Check each contour point
                points_in_shade = [
                    zone_mask[int(y), int(x)] == 1
                    for x, y in all_contour_points
                    if 0 <= int(x) < width and 0 <= int(y) < height
                ]

                any_point_in_shade = any(points_in_shade)
                all_points_in_shade = all(points_in_shade)

            # === Draw detection on annotated frame ===
            # INSIDE (green) if NO points in shaded area, OUTSIDE (red) if ANY point in shaded
            if not any_point_in_shade:
                color = (0, 255, 0)  # Green - no points in shaded area = INSIDE
                status = "INSIDE"
            else:
                color = (0, 0, 255)  # Red - any point in shaded area = OUTSIDE
                status = "OUTSIDE"

            # Draw mask overlay
            colored_mask = np.zeros_like(annotated_frame)
            colored_mask[mask_bin > 0] = color
            annotated_frame = cv2.addWeighted(annotated_frame, 1, colored_mask, 0.3, 0)

            # Draw polygon (contour) instead of bounding box
            if contours:
                # Draw all contours for this detection as polygons
                cv2.drawContours(annotated_frame, contours, -1, color, 2)

            # Draw label at the top of the bounding box position
            x1, y1, x2, y2 = map(int, box)
            label = f"{class_name} - {status}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            detections.append({
                "segment_index": idx,
                "class": class_name,
                "inside_lshape": not any_point_in_shade  # True if NO points in shaded area
            })

    # === Encode annotated image to base64 ===
    _, buffer = cv2.imencode('.jpg', annotated_frame)
    annotated_base64 = base64.b64encode(buffer).decode('utf-8')

    return detections, annotated_base64
