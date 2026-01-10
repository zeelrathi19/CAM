# pages/config.py - Configuration page

import streamlit as st
import json
import cv2
import numpy as np
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

def line_equation(p1, p2):
    """Calculate line equation from two points"""
    x1, y1 = p1
    x2, y2 = p2
    if x2 - x1 == 0:
        return None, None, f"x = {x1}"
    m = (y2 - y1) / (x2 - x1)
    c = y1 - m * x1
    return m, c, f"y = {m:.2f}x + {c:.2f}"

def compute_lshape_data(A, B, C, width, height):
    """Compute L-shape configuration from 3 points (same logic as Jupyter notebook)"""
    m_ab, c_ab, eq_ab = line_equation(A, B)
    m_bc, c_bc, eq_bc = line_equation(B, C)

    result = {
        "A": list(A), "B": list(B), "C": list(C),
        "line_AB": eq_ab, "line_BC": eq_bc
    }

    # Extended LINE AB intersections
    if m_ab is not None:
        ab_left = [0, c_ab]
        ab_right = [width, m_ab * width + c_ab]
        ab_bottom = [-c_ab / m_ab, 0]
        ab_top = [(height - c_ab) / m_ab, height]
        result.update({
            "ab_left": ab_left, "ab_right": ab_right,
            "ab_bottom": ab_bottom, "ab_top": ab_top
        })
    else:
        x_ab = A[0]
        result.update({
            "ab_left": None, "ab_right": None,
            "ab_bottom": [x_ab, 0], "ab_top": [x_ab, height]
        })

    # Extended LINE BC intersections
    if m_bc is not None:
        bc_left = [0, c_bc]
        bc_right = [width, m_bc * width + c_bc]
        bc_bottom = [-c_bc / m_bc, 0]
        bc_top = [(height - c_bc) / m_bc, height]
        result.update({
            "bc_left": bc_left, "bc_right": bc_right,
            "bc_bottom": bc_bottom, "bc_top": bc_top
        })
    else:
        x_bc = B[0]
        result.update({
            "bc_left": None, "bc_right": None,
            "bc_bottom": [x_bc, 0], "bc_top": [x_bc, height]
        })

    return result

def compute_straightline_data(A, B, width, height):
    """Compute straight line configuration from 2 points

    Args:
        A, B: Two points defining the line in math coordinates
        width, height: Image dimensions

    Returns:
        Dictionary with line data and edge intersections
    """
    m_ab, c_ab, eq_ab = line_equation(A, B)

    result = {
        "A": list(A),
        "B": list(B),
        "line_AB": eq_ab
    }

    # Extended LINE AB intersections
    if m_ab is not None:
        ab_left = [0, c_ab]
        ab_right = [width, m_ab * width + c_ab]
        ab_bottom = [-c_ab / m_ab, 0] if m_ab != 0 else None
        ab_top = [(height - c_ab) / m_ab, height] if m_ab != 0 else None
        result.update({
            "ab_left": ab_left,
            "ab_right": ab_right,
            "ab_bottom": ab_bottom,
            "ab_top": ab_top
        })
    else:  # Vertical line
        x_ab = A[0]
        result.update({
            "ab_left": None,
            "ab_right": None,
            "ab_bottom": [x_ab, 0],
            "ab_top": [x_ab, height]
        })

    return result

def draw_shaded_areas(image, json_data):
    """Draw 4 colored areas on the image based on JSON data"""
    img_array = np.array(image)
    height, width = img_array.shape[:2]

    # Create overlay
    overlay = img_array.copy()

    # Get line data
    ab_left = json_data.get("ab_left")
    ab_right = json_data.get("ab_right")
    ab_top = json_data.get("ab_top")
    ab_bottom = json_data.get("ab_bottom")
    bc_left = json_data.get("bc_left")
    bc_right = json_data.get("bc_right")
    bc_top = json_data.get("bc_top")
    bc_bottom = json_data.get("bc_bottom")

    # Draw all 4 areas as quadrants formed by lines AB and BC intersection at point B
    # Areas arranged clockwise starting from bottom-left: 1 → 2 → 3 → 4
    # BGR color format: (Blue, Green, Red)

    B_point = json_data.get("B")
    if ab_left and ab_right and bc_left and bc_right and B_point:
        # Convert B to image coordinates
        B_img = (int(B_point[0]), height - int(B_point[1]))

        # Convert line points to image coordinates
        ab_left_img = (int(ab_left[0]), height - int(ab_left[1]))
        ab_right_img = (int(ab_right[0]), height - int(ab_right[1]))
        ab_top_img = (int(ab_top[0]), height - int(ab_top[1]))
        ab_bottom_img = (int(ab_bottom[0]), height - int(ab_bottom[1]))
        bc_left_img = (int(bc_left[0]), height - int(bc_left[1]))
        bc_right_img = (int(bc_right[0]), height - int(bc_right[1]))
        bc_top_img = (int(bc_top[0]), height - int(bc_top[1]))
        bc_bottom_img = (int(bc_bottom[0]), height - int(bc_bottom[1]))

        # Area 1: Bottom-Left - Magenta
        area1_overlay = np.zeros_like(img_array)
        area1 = np.array([
            B_img,
            ab_left_img,
            (0, height),
            bc_left_img
        ], dtype=np.int32)
        cv2.fillPoly(area1_overlay, [area1], (255, 0, 255))  # Magenta
        overlay = cv2.addWeighted(overlay, 1.0, area1_overlay, 0.3, 0)
        cv2.putText(overlay, "Area 1", (50, height - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        # Area 2: Top-Left - Yellow (clockwise from Area 1)
        area2_overlay = np.zeros_like(img_array)
        area2 = np.array([
            B_img,
            ab_left_img,
            (0, 0),
            bc_top_img
        ], dtype=np.int32)
        cv2.fillPoly(area2_overlay, [area2], (255, 255, 0))  # Yellow in BGR
        overlay = cv2.addWeighted(overlay, 1.0, area2_overlay, 0.3, 0)
        cv2.putText(overlay, "Area 2", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Area 3: Top-Right - Cyan (clockwise from Area 2)
        area3_overlay = np.zeros_like(img_array)
        area3 = np.array([
            B_img,
            ab_right_img,
            (width, 0),
            bc_right_img
        ], dtype=np.int32)
        cv2.fillPoly(area3_overlay, [area3], (0, 255, 255))  # Cyan in BGR
        overlay = cv2.addWeighted(overlay, 1.0, area3_overlay, 0.3, 0)
        cv2.putText(overlay, "Area 3", (width - 150, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Area 4: Bottom-Right - Green (clockwise from Area 3)
        area4_overlay = np.zeros_like(img_array)
        area4 = np.array([
            B_img,
            bc_bottom_img,
            (width, height),
            ab_bottom_img
        ], dtype=np.int32)
        cv2.fillPoly(area4_overlay, [area4], (0, 255, 0))  # Green
        overlay = cv2.addWeighted(overlay, 1.0, area4_overlay, 0.3, 0)
        cv2.putText(overlay, "Area 4", (width - 150, height - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Draw lines AB and BC
    A = json_data.get("A")
    B = json_data.get("B")
    C = json_data.get("C")

    if A and B:
        cv2.line(overlay, (int(A[0]), height - int(A[1])),
                 (int(B[0]), height - int(B[1])), (255, 0, 0), 3)
        cv2.circle(overlay, (int(A[0]), height - int(A[1])), 10, (255, 0, 0), -1)
        cv2.circle(overlay, (int(B[0]), height - int(B[1])), 10, (255, 0, 0), -1)
        cv2.putText(overlay, "A", (int(A[0]) + 15, height - int(A[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(overlay, "B", (int(B[0]) + 15, height - int(B[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if B and C:
        cv2.line(overlay, (int(B[0]), height - int(B[1])),
                 (int(C[0]), height - int(C[1])), (255, 0, 0), 3)
        cv2.circle(overlay, (int(C[0]), height - int(C[1])), 10, (255, 0, 0), -1)
        cv2.putText(overlay, "C", (int(C[0]) + 15, height - int(C[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Return the overlay (areas already blended with transparency)
    return Image.fromarray(overlay)

def draw_shaded_areas_straightline(image, json_data):
    """Draw 2 colored areas on the image for straight line mode

    Areas:
    - Area 1 (Above line): Magenta
    - Area 2 (Below line): Yellow
    """
    img_array = np.array(image)
    height, width = img_array.shape[:2]

    overlay = img_array.copy()

    # Get line data
    ab_left = json_data.get("ab_left")
    ab_right = json_data.get("ab_right")
    ab_top = json_data.get("ab_top")
    ab_bottom = json_data.get("ab_bottom")
    A = json_data.get("A")
    B = json_data.get("B")

    if ab_left and ab_right:
        # Convert to image coordinates
        ab_left_img = (int(ab_left[0]), height - int(ab_left[1]))
        ab_right_img = (int(ab_right[0]), height - int(ab_right[1]))

        # Handle ab_top and ab_bottom (may be None for horizontal lines)
        if ab_top:
            ab_top_img = (int(ab_top[0]), height - int(ab_top[1]))
        else:
            ab_top_img = (0, 0)  # Top-left corner fallback

        if ab_bottom:
            ab_bottom_img = (int(ab_bottom[0]), height - int(ab_bottom[1]))
        else:
            ab_bottom_img = (0, height)  # Bottom-left corner fallback

        # Area 1: Above line - Magenta (255, 0, 255) in BGR
        # Polygon vertices in clockwise order: ab_left -> top-left -> top-right -> ab_right
        area1 = np.array([
            ab_left_img,
            (0, 0),  # Top-left corner
            (width, 0),  # Top-right corner
            ab_right_img
        ], dtype=np.int32)
        cv2.fillPoly(overlay, [area1], (255, 0, 255))  # Magenta BGR

        # Area 2: Below line - Yellow in BGR
        # Polygon vertices in clockwise order: ab_left -> ab_right -> bottom-right -> bottom-left
        area2 = np.array([
            ab_left_img,
            ab_right_img,
            (width, height),  # Bottom-right corner
            (0, height)  # Bottom-left corner
        ], dtype=np.int32)
        cv2.fillPoly(overlay, [area2], (0, 255, 255))  # Yellow in BGR

    # Apply transparency to the entire overlay
    overlay = cv2.addWeighted(img_array, 0.7, overlay, 0.3, 0)

    # Draw the line AB
    if A and B:
        pt_a = (int(A[0]), height - int(A[1]))
        pt_b = (int(B[0]), height - int(B[1]))
        cv2.line(overlay, pt_a, pt_b, (0, 255, 0), 3)  # Green line
        cv2.circle(overlay, pt_a, 10, (0, 0, 255), -1)
        cv2.circle(overlay, pt_b, 10, (0, 0, 255), -1)
        cv2.putText(overlay, "A", (pt_a[0] + 15, pt_a[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(overlay, "B", (pt_b[0] + 15, pt_b[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Convert BGR to RGB for PIL
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return Image.fromarray(overlay_rgb)

st.title("⚙️ Boundary Configuration")
st.markdown("Define your boundary by clicking points on an uploaded image, or upload an existing JSON configuration.")

# Initialize session state for points
if 'clicked_points' not in st.session_state:
    st.session_state.clicked_points = []
if 'just_reset' not in st.session_state:
    st.session_state.just_reset = False
if 'config_type' not in st.session_state:
    st.session_state.config_type = "lshape"

# Choose configuration mode
config_mode = st.radio(
    "Configuration Mode:",
    ["L-Shape (3 points, 4 areas)", "Straight Line (2 points, 2 areas)"],
    help="Choose the boundary configuration type"
)

# Update config_type in session state
if config_mode == "L-Shape (3 points, 4 areas)":
    st.session_state.config_type = "lshape"
    max_points = 3
else:
    st.session_state.config_type = "straight_line"
    max_points = 2

# Choose configuration method
config_method = st.radio(
    "Configuration Method:",
    [f"Click {max_points} Points on Image", "Upload Existing JSON"],
    help="Choose how to configure the boundary"
)

if config_method == f"Click {max_points} Points on Image":
    # Full width layout for clicking mode
    st.header(f"1. Define {config_mode.split(' (')[0]} Configuration")
    st.markdown("📸 **Step 1:** Upload an image")
    uploaded_image = st.file_uploader(
        "Choose Reference Image",
        type=["jpg", "jpeg", "png"],
        key="image_for_clicking",
        help="Upload the image to define L-shape points"
    )

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        img_array = np.array(image)
        height, width = img_array.shape[:2]

        if max_points == 3:
            st.markdown(f"📐 **Step 2:** Click 3 points (A → B → C) on the image below")
        else:
            st.markdown(f"📐 **Step 2:** Click 2 points (A → B) on the image below")
        st.markdown(f"**Points clicked:** {len(st.session_state.clicked_points)}/{max_points}")

        # Calculate display dimensions - fit to viewport width (typically 1200px max for Streamlit)
        max_display_width = 1200
        if width > max_display_width:
            scale_factor = max_display_width / width
            display_width = max_display_width
            display_height = int(height * scale_factor)
        else:
            scale_factor = 1.0
            display_width = width
            display_height = height

        # Resize image for display if needed
        if scale_factor != 1.0:
            display_img_base = cv2.resize(img_array, (display_width, display_height))
        else:
            display_img_base = img_array.copy()

        # Draw existing points on display image (scaled)
        display_img = display_img_base.copy()
        for idx, (x, y) in enumerate(st.session_state.clicked_points):
            # Scale points for display
            display_x = int(x * scale_factor)
            display_y = int(y * scale_factor)
            cv2.circle(display_img, (display_x, display_y), 10, (0, 0, 255), -1)
            label = ['A', 'B', 'C'][idx]
            cv2.putText(display_img, label, (display_x + 15, display_y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        # Draw lines if we have 2+ points
        if len(st.session_state.clicked_points) >= 2:
            pt0_x = int(st.session_state.clicked_points[0][0] * scale_factor)
            pt0_y = int(st.session_state.clicked_points[0][1] * scale_factor)
            pt1_x = int(st.session_state.clicked_points[1][0] * scale_factor)
            pt1_y = int(st.session_state.clicked_points[1][1] * scale_factor)
            cv2.line(display_img, (pt0_x, pt0_y), (pt1_x, pt1_y), (0, 255, 0), 3)

        if len(st.session_state.clicked_points) == 3:
            pt1_x = int(st.session_state.clicked_points[1][0] * scale_factor)
            pt1_y = int(st.session_state.clicked_points[1][1] * scale_factor)
            pt2_x = int(st.session_state.clicked_points[2][0] * scale_factor)
            pt2_y = int(st.session_state.clicked_points[2][1] * scale_factor)
            cv2.line(display_img, (pt1_x, pt1_y), (pt2_x, pt2_y), (0, 255, 0), 3)

        # Interactive image with click coordinates
        coords = streamlit_image_coordinates(
            Image.fromarray(display_img),
            width=display_width,
            key="image_coords"
        )

        # Handle new clicks - convert back to original coordinates
        # Skip processing coords if we just reset (prevents re-adding old coords)
        if st.session_state.just_reset:
            st.session_state.just_reset = False
        elif coords is not None and len(st.session_state.clicked_points) < max_points:
            # Convert display coordinates back to original image coordinates
            original_x = int(coords["x"] / scale_factor)
            original_y = int(coords["y"] / scale_factor)
            new_point = (original_x, original_y)
            # Check if this is a new click (not the same as last point)
            if len(st.session_state.clicked_points) == 0 or new_point != st.session_state.clicked_points[-1]:
                st.session_state.clicked_points.append(new_point)
                st.rerun()

        # Action buttons
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🔄 Reset Points"):
                st.session_state.clicked_points = []
                st.session_state.just_reset = True
                st.rerun()

        with col_btn2:
            if st.button("✅ Generate Config", disabled=len(st.session_state.clicked_points) != max_points):
                if len(st.session_state.clicked_points) == max_points:
                    # Convert image coordinates to math coordinates (origin bottom-left)
                    points_math = []
                    for (x, y) in st.session_state.clicked_points:
                        math_y = height - y
                        points_math.append((x, math_y))

                    if max_points == 3:
                        # L-Shape mode: 3 points
                        A, B, C = points_math
                        config_data = compute_lshape_data(A, B, C, width, height)
                        st.session_state.config_type = "lshape"
                        st.success("✅ L-shape configuration generated!")
                    else:
                        # Straight line mode: 2 points
                        A, B = points_math
                        config_data = compute_straightline_data(A, B, width, height)
                        st.session_state.config_type = "straight_line"
                        st.success("✅ Straight line configuration generated!")

                    st.session_state.lshape_config = config_data
                    st.session_state.config_uploaded = True
                    st.rerun()

        # Show visualization and restricted area selection after config is generated
        if st.session_state.config_uploaded:
            st.markdown("---")
            st.subheader(f"📊 {config_mode.split(' (')[0]} Visualization & Configuration")

            # Create two columns for visualization and area selection
            viz_col1, viz_col2 = st.columns([2, 1])

            with viz_col1:
                st.markdown("**Area Visualization**")
                # Draw shaded areas on the image using correct function based on mode
                if st.session_state.config_type == "lshape":
                    visualized_image = draw_shaded_areas(image, st.session_state.lshape_config)
                    caption = "4 Shaded Areas"
                    legend = """
                    **Legend (Clockwise from Bottom-Left):**
                    - 🟣 **Area 1**: Bottom-Left (Magenta)
                    - 🟡 **Area 2**: Top-Left (Yellow)
                    - 🔵 **Area 3**: Top-Right (Cyan)
                    - 🟢 **Area 4**: Bottom-Right (Green)
                    """
                else:
                    visualized_image = draw_shaded_areas_straightline(image, st.session_state.lshape_config)
                    caption = "2 Shaded Areas"
                    legend = """
                    **Legend:**
                    - 🟣 **Area 1**: Above Line (Magenta)
                    - 🟡 **Area 2**: Below Line (Yellow)
                    """

                st.image(visualized_image, caption=caption, use_container_width=True)
                st.markdown(legend)

            with viz_col2:
                st.markdown("**Select Restricted Areas**")
                st.markdown("""
                Select which areas should be considered **OUTSIDE** (restricted zones) for tyre detection.
                """)

                # Area options based on config type
                if st.session_state.config_type == "lshape":
                    area_options = {
                        "Area 1: Bottom-Left (Magenta)": "below_ab",
                        "Area 2: Top-Left (Yellow)": "above_ab",
                        "Area 3: Top-Right (Cyan)": "above_bc",
                        "Area 4: Bottom-Right (Green)": "below_bc"
                    }
                else:
                    area_options = {
                        "Area 1: Above Line (Magenta)": "above_ab",
                        "Area 2: Below Line (Yellow)": "below_ab"
                    }

                # Multi-select for restricted areas
                selected_areas = st.multiselect(
                    "Choose restricted areas:",
                    options=list(area_options.keys()),
                    default=st.session_state.restricted_areas if st.session_state.restricted_areas else [],
                    help="Tyres in selected areas will be marked as OUTSIDE"
                )

                # Update session state
                st.session_state.restricted_areas = selected_areas

                # Display selected configuration
                if selected_areas:
                    st.success(f"✅ {len(selected_areas)} area(s) selected")
                    st.markdown("**Restricted Areas:**")
                    for area in selected_areas:
                        area_code = area_options[area]
                        st.markdown(f"- {area}")
                else:
                    st.warning("⚠️ No restricted areas selected")

                # Save configuration button
                st.markdown("---")
                if st.button("💾 Save Configuration", type="primary", key="save_from_clicking"):
                    # Save configuration to a file
                    # Use appropriate data key based on config type
                    data_key = st.session_state.config_type + "_data" if st.session_state.config_type == "straight_line" else "lshape_data"
                    config_to_save = {
                        "config_type": st.session_state.config_type,
                        data_key: st.session_state.lshape_config,
                        "restricted_areas": [area_options[area] for area in selected_areas]
                    }

                    # Save to a JSON file
                    try:
                        with open("lshape_data/config.json", "w") as f:
                            json.dump(config_to_save, f, indent=2)
                        st.success("✅ Configuration saved successfully!")
                        st.balloons()
                    except Exception as e:
                        st.error(f"❌ Error saving configuration: {str(e)}")

else:  # Upload Existing JSON
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("📄 **Upload your pre-generated JSON file**")
        uploaded_json = st.file_uploader(
            "Choose JSON file",
            type=["json"],
            help="Upload the L-shape configuration JSON file generated from final.ipynb"
        )

        uploaded_image = st.file_uploader(
            "Choose Reference Image",
            type=["jpg", "jpeg", "png"],
            key="image_for_json",
            help="Upload the image corresponding to this configuration"
        )

        if uploaded_json is not None:
            try:
                # Read and parse JSON
                json_data = json.load(uploaded_json)

                # Detect config type from JSON (backward compatibility)
                detected_config_type = json_data.get("config_type", "lshape")
                st.session_state.config_type = detected_config_type

                # Extract the data based on config type
                if detected_config_type == "straight_line":
                    st.session_state.lshape_config = json_data.get("straight_line_data", {})
                else:
                    st.session_state.lshape_config = json_data.get("lshape_data", {})

                st.session_state.config_uploaded = True

                st.success(f"✅ JSON file loaded successfully! (Mode: {detected_config_type})")

                # Display JSON preview
                with st.expander("📄 View JSON Data"):
                    st.json(json_data)

            except json.JSONDecodeError:
                st.error("❌ Invalid JSON file. Please upload a valid JSON file.")
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")

        # Visualize shaded areas if configuration is ready
        if st.session_state.config_uploaded and uploaded_image is not None:
            try:
                # Load image
                image = Image.open(uploaded_image)

                st.markdown("---")
                st.subheader("📊 Area Visualization")

                # Draw shaded areas using correct function based on mode
                if st.session_state.config_type == "lshape":
                    visualized_image = draw_shaded_areas(image, st.session_state.lshape_config)
                    caption = "4 Shaded Areas"
                    legend = """
                    **Legend (Clockwise from Bottom-Left):**
                    - 🟣 **Area 1**: Bottom-Left (Magenta)
                    - 🟡 **Area 2**: Top-Left (Yellow)
                    - 🔵 **Area 3**: Top-Right (Cyan)
                    - 🟢 **Area 4**: Bottom-Right (Green)
                    """
                else:
                    visualized_image = draw_shaded_areas_straightline(image, st.session_state.lshape_config)
                    caption = "2 Shaded Areas"
                    legend = """
                    **Legend:**
                    - 🟣 **Area 1**: Above Line (Magenta)
                    - 🟡 **Area 2**: Below Line (Yellow)
                    """

                st.image(visualized_image, caption=caption, use_container_width=True)
                st.markdown(legend)

            except Exception as e:
                st.error(f"❌ Error visualizing areas: {str(e)}")

    with col2:
        st.header("2. Select Restricted Areas")

        if st.session_state.config_uploaded:
            st.markdown("""
            Select which areas should be considered **OUTSIDE** (restricted zones) for tyre detection.
            Tyres detected in these areas will be marked as **OUTSIDE**.
            """)

            # Area options based on config type
            if st.session_state.config_type == "lshape":
                area_options = {
                    "Area 1: Bottom-Left (Magenta)": "below_ab",
                    "Area 2: Top-Left (Yellow)": "above_ab",
                    "Area 3: Top-Right (Cyan)": "above_bc",
                    "Area 4: Bottom-Right (Green)": "below_bc"
                }
            else:
                area_options = {
                    "Area 1: Above Line (Magenta)": "above_ab",
                    "Area 2: Below Line (Yellow)": "below_ab"
                }

            # Multi-select for restricted areas
            selected_areas = st.multiselect(
                "Choose restricted areas:",
                options=list(area_options.keys()),
                default=st.session_state.restricted_areas if st.session_state.restricted_areas else [],
                help="Tyres in selected areas will be marked as OUTSIDE"
            )

            # Update session state
            st.session_state.restricted_areas = selected_areas

            # Display selected configuration
            if selected_areas:
                st.success(f"✅ {len(selected_areas)} area(s) selected as restricted zones")

                st.markdown("**Restricted Areas:**")
                for area in selected_areas:
                    area_code = area_options[area]
                    st.markdown(f"- {area}")
            else:
                st.warning("⚠️ No restricted areas selected. All areas will be considered INSIDE.")

            # Save configuration button
            st.markdown("---")
            if st.button("💾 Save Configuration", type="primary"):
                # Save configuration to a file
                # Use appropriate data key based on config type
                data_key = st.session_state.config_type + "_data" if st.session_state.config_type == "straight_line" else "lshape_data"
                config_to_save = {
                    "config_type": st.session_state.config_type,
                    data_key: st.session_state.lshape_config,
                    "restricted_areas": [area_options[area] for area in selected_areas]
                }

                # Save to a JSON file
                try:
                    with open("lshape_data/config.json", "w") as f:
                        json.dump(config_to_save, f, indent=2)
                    st.success("✅ Configuration saved successfully!")
                    st.balloons()
                except Exception as e:
                    st.error(f"❌ Error saving configuration: {str(e)}")
        else:
            st.info("👈 Please upload a JSON file first")

# Display current configuration status
st.markdown("---")
st.header("Current Configuration Status")

col_status1, col_status2 = st.columns(2)
with col_status1:
    if st.session_state.config_uploaded:
        st.metric("JSON Configuration", "✅ Loaded")
    else:
        st.metric("JSON Configuration", "❌ Not Loaded")

with col_status2:
    st.metric("Restricted Areas", f"{len(st.session_state.restricted_areas)} selected")
