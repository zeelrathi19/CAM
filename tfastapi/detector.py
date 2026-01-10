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
CONFIG_FILE = os.path.join(BASE_DIR, "lshape_data", "config.json")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")

def to_np(pt):
    return np.array(pt, dtype=np.int32) if pt else None

# Load Model Once (outside function for performance)
model = YOLO(MODEL_PATH)

def load_config():
    """Load configuration from config.json file

    Returns:
        Tuple: A, B, C, ab_left, ab_right, ab_top, bc_left, bc_right, bc_top, ab_bottom, bc_bottom, restricted_areas, config_type
        Note: C and bc_* values will be None for straight_line mode
    """
    with open(CONFIG_FILE, "r") as f:
        config = json.load(f)

    # Determine config type (backward compatibility)
    config_type = config.get("config_type", "lshape")

    # Get appropriate data key
    if config_type == "straight_line":
        data = config.get("straight_line_data", {})
    else:
        data = config.get("lshape_data", {})

    restricted_areas = config.get("restricted_areas", [])

    # Convert points to numpy arrays
    A = to_np(data.get("A"))
    B = to_np(data.get("B"))
    C = to_np(data.get("C"))  # Will be None for straight_line
    ab_left = to_np(data.get("ab_left"))
    ab_right = to_np(data.get("ab_right"))
    ab_top = to_np(data.get("ab_top"))
    bc_left = to_np(data.get("bc_left"))  # None for straight_line
    bc_right = to_np(data.get("bc_right"))  # None for straight_line
    bc_top = to_np(data.get("bc_top"))  # None for straight_line
    ab_bottom = to_np(data.get("ab_bottom"))
    bc_bottom = to_np(data.get("bc_bottom"))  # None for straight_line

    return A, B, C, ab_left, ab_right, ab_top, bc_left, bc_right, bc_top, ab_bottom, bc_bottom, restricted_areas, config_type

def point_below_line(point, line_left, line_right, height):
    """Check if a point is below a line defined by left and right edge intersections
    Note: line_left and line_right are in math coordinates (origin bottom-left)
    point is also in math coordinates
    """
    x, y = point

    # Calculate line equation: y = mx + c
    x1, y1 = line_left
    x2, y2 = line_right

    if x2 - x1 == 0:  # Vertical line
        return False

    m = (y2 - y1) / (x2 - x1)
    c = y1 - m * x1

    # Point on line at x coordinate (in math coordinates)
    line_y_at_x = m * x + c

    # In math coordinates, point is below line if point_y < line_y_at_x
    return y < line_y_at_x

def check_point_in_restricted_areas(point, height, width, restricted_areas, ab_left, ab_right, bc_left, bc_right, config_type):
    """Check if a point falls in any of the restricted areas

    For straight_line mode: checks above/below single line
    For lshape mode: checks 4 quadrants based on two lines
    """
    # Check position relative to AB line
    is_below_ab = point_below_line(point, ab_left, ab_right, height) if ab_left is not None and ab_right is not None else None

    # Straight line mode: only check above/below AB line
    if config_type == "straight_line":
        if is_below_ab is None:
            return False

        for area in restricted_areas:
            if area == "below_ab" and is_below_ab == True:
                return True
            elif area == "above_ab" and is_below_ab == False:
                return True
        return False

    # L-shape mode: check 4 quadrants based on both lines
    else:
        is_below_bc = point_below_line(point, bc_left, bc_right, height) if bc_left is not None and bc_right is not None else None

        # DEBUG: Disabled point-level debug
        # print(f"Point {point}: below_ab={is_below_ab}, below_bc={is_below_bc}, restricted={restricted_areas}")

        # Return False if we couldn't determine position (missing line data)
        if is_below_ab is None or is_below_bc is None:
            return False

        for area in restricted_areas:
            # Area 1 (below_ab): Bottom-Left quadrant
            # Below AB line AND Above BC line
            if area == "below_ab":
                if is_below_ab == True and is_below_bc == False:
                    return True

            # Area 2 (above_ab): Top-Left quadrant
            # Above AB line AND Above BC line (left of BC)
            elif area == "above_ab":
                if is_below_ab == False and is_below_bc == False:
                    return True

            # Area 3 (above_bc): Top-Right quadrant
            # Above AB line AND Below BC line (right of BC)
            elif area == "above_bc":
                if is_below_ab == False and is_below_bc == True:
                    return True

            # Area 4 (below_bc): Bottom-Right quadrant
            # Below AB line AND Below BC line
            elif area == "below_bc":
                if is_below_ab == True and is_below_bc == True:
                    return True

        return False

def detect_and_classify(image_bytes):
    # Load fresh configuration from file
    A, B, C, ab_left, ab_right, ab_top, bc_left, bc_right, bc_top, ab_bottom, bc_bottom, restricted_areas, config_type = load_config()

    npimg = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    height, width = frame.shape[:2]

    # Create a copy for annotation
    annotated_frame = frame.copy()

    # === Draw ONLY the selected restricted areas on annotated frame ===

    if config_type == "straight_line":
        # Straight line mode: Draw 2 areas (Above line / Below line)
        if B is not None and ab_left is not None and ab_right is not None:
            # Convert line points to image coordinates
            ab_left_img = (int(ab_left[0]), height - int(ab_left[1]))
            ab_right_img = (int(ab_right[0]), height - int(ab_right[1]))

            # Area 1: Above line (Magenta) - if "above_ab" selected
            if "above_ab" in restricted_areas:
                area1_overlay = np.zeros_like(annotated_frame)
                # Polygon vertices in clockwise order: ab_left -> top-left -> top-right -> ab_right
                area1 = np.array([
                    ab_left_img,
                    (0, 0),  # Top-left corner
                    (width, 0),  # Top-right corner
                    ab_right_img
                ], dtype=np.int32)
                cv2.fillPoly(area1_overlay, [area1], (255, 0, 255))  # Magenta
                annotated_frame = cv2.addWeighted(annotated_frame, 0.6, area1_overlay, 0.4, 0)

            # Area 2: Below line (Yellow) - if "below_ab" selected
            if "below_ab" in restricted_areas:
                area2_overlay = np.zeros_like(annotated_frame)
                # Polygon vertices in clockwise order: ab_left -> ab_right -> bottom-right -> bottom-left
                area2 = np.array([
                    ab_left_img,
                    ab_right_img,
                    (width, height),  # Bottom-right corner
                    (0, height)  # Bottom-left corner
                ], dtype=np.int32)
                cv2.fillPoly(area2_overlay, [area2], (0, 255, 255))  # Yellow
                annotated_frame = cv2.addWeighted(annotated_frame, 0.6, area2_overlay, 0.4, 0)

    else:
        # L-shape mode: Draw 4 areas (quadrants)
        # Draw areas as quadrants (clockwise from bottom-left: 1→2→3→4)
        if B is not None and ab_left is not None and ab_right is not None and bc_left is not None and bc_right is not None:
            # Convert B to image coordinates
            B_img = (int(B[0]), height - int(B[1]))

            # Convert line points to image coordinates
            ab_left_img = (int(ab_left[0]), height - int(ab_left[1]))
            ab_right_img = (int(ab_right[0]), height - int(ab_right[1]))
            ab_top_img = (int(ab_top[0]), height - int(ab_top[1]))
            ab_bottom_img = (int(ab_bottom[0]), height - int(ab_bottom[1]))
            bc_left_img = (int(bc_left[0]), height - int(bc_left[1]))
            bc_right_img = (int(bc_right[0]), height - int(bc_right[1]))
            bc_top_img = (int(bc_top[0]), height - int(bc_top[1]))
            bc_bottom_img = (int(bc_bottom[0]), height - int(bc_bottom[1]))

            # Area 1: Bottom-Left (Magenta) - only if "below_ab" selected
            if "below_ab" in restricted_areas:
                area1_overlay = np.zeros_like(annotated_frame)
                # Use same polygon as config.py: B -> ab_left -> bottom-left -> bc_left
                area1 = np.array([
                    B_img,
                    ab_left_img,
                    (0, height),
                    bc_left_img
                ], dtype=np.int32)
                cv2.fillPoly(area1_overlay, [area1], (255, 0, 255))
                annotated_frame = cv2.addWeighted(annotated_frame, 0.6, area1_overlay, 0.4, 0)

            # Area 2: Top-Left (Yellow) - only if "above_ab" selected
            if "above_ab" in restricted_areas:
                area2_overlay = np.zeros_like(annotated_frame)
                # Top-left quadrant bounded by extended AB and BC lines
                area2 = np.array([
                    B_img,
                    ab_left_img,
                    (0, 0),
                    bc_top_img
                ], dtype=np.int32)
                cv2.fillPoly(area2_overlay, [area2], (0, 255, 255))  # BGR: Yellow
                annotated_frame = cv2.addWeighted(annotated_frame, 0.6, area2_overlay, 0.4, 0)

            # Area 3: Top-Right (Cyan) - only if "above_bc" selected
            if "above_bc" in restricted_areas:
                area3_overlay = np.zeros_like(annotated_frame)
                # Top-right quadrant bounded by extended AB and BC lines
                area3 = np.array([
                    B_img,
                    ab_right_img,
                    (width, 0),
                    bc_right_img
                ], dtype=np.int32)
                cv2.fillPoly(area3_overlay, [area3], (255, 255, 0))  # BGR: Cyan
                annotated_frame = cv2.addWeighted(annotated_frame, 0.6, area3_overlay, 0.4, 0)

            # Area 4: Bottom-Right (Green) - only if "below_bc" selected
            if "below_bc" in restricted_areas:
                area4_overlay = np.zeros_like(annotated_frame)
                # Bottom-right quadrant bounded by extended AB and BC lines
                area4 = np.array([
                    B_img,
                    bc_bottom_img,
                    (width, height),
                    ab_bottom_img
                ], dtype=np.int32)
                cv2.fillPoly(area4_overlay, [area4], (0, 255, 0))
                annotated_frame = cv2.addWeighted(annotated_frame, 0.6, area4_overlay, 0.4, 0)

    # === Draw boundary lines ===
    # Draw line AB (for both modes)
    if A is not None and B is not None:
        pt_a = (A[0], height - A[1])
        pt_b = (B[0], height - B[1])
        cv2.line(annotated_frame, pt_a, pt_b, (0, 255, 0), 3)
        cv2.circle(annotated_frame, pt_a, 8, (0, 0, 255), -1)
        cv2.circle(annotated_frame, pt_b, 8, (0, 0, 255), -1)
        cv2.putText(annotated_frame, "A", (pt_a[0] + 10, pt_a[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_frame, "B", (pt_b[0] + 10, pt_b[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Draw line BC (only for lshape mode)
    if config_type == "lshape" and B is not None and C is not None:
        pt_b = (B[0], height - B[1])
        pt_c = (C[0], height - C[1])
        cv2.line(annotated_frame, pt_b, pt_c, (0, 255, 0), 3)
        cv2.circle(annotated_frame, pt_c, 8, (0, 0, 255), -1)
        cv2.putText(annotated_frame, "C", (pt_c[0] + 10, pt_c[1] - 10),
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

            # Check if any polygon contour points are in restricted areas
            any_point_in_restricted = False
            if contours:
                # Get all polygon contour points
                all_contour_points = np.vstack(contours).reshape(-1, 2)

                # Check each contour point against restricted areas
                for x, y in all_contour_points:
                    if 0 <= int(x) < width and 0 <= int(y) < height:
                        # Convert to math coordinates (origin bottom-left)
                        math_point = (int(x), height - int(y))
                        if check_point_in_restricted_areas(math_point, height, width, restricted_areas, ab_left, ab_right, bc_left, bc_right, config_type):
                            any_point_in_restricted = True
                            break

            # === Draw detection on annotated frame ===
            # INSIDE (green) if NO points in restricted areas, OUTSIDE (red) if ANY point in restricted
            if not any_point_in_restricted:
                color = (0, 255, 0)  # Green - no points in restricted area = INSIDE
                status = "INSIDE"
            else:
                color = (0, 0, 255)  # Red - any point in restricted area = OUTSIDE
                status = "OUTSIDE"

            # Draw mask overlay
            colored_mask = np.zeros_like(annotated_frame)
            colored_mask[mask_bin > 0] = color
            annotated_frame = cv2.addWeighted(annotated_frame, 1, colored_mask, 0.3, 0)

            # Draw polygon (contour) instead of bounding box
            if contours:
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
                "inside_lshape": not any_point_in_restricted  # True if NO points in restricted areas
            })

    # === Encode annotated image to base64 ===
    _, buffer = cv2.imencode('.jpg', annotated_frame)
    annotated_base64 = base64.b64encode(buffer).decode('utf-8')

    return detections, annotated_base64
