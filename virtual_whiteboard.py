import cv2
import numpy as np
import os
import mediapipe as mp
from datetime import datetime
from collections import deque
import math
import time


class VirtualWhiteboard:
    def __init__(self):
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")

        # Set camera resolution for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Get camera frame dimensions
        _, frame = self.cap.read()
        self.height, self.width = frame.shape[:2]

        # Initialize MediaPipe Hands for detecting two hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Initialize canvas
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.canvas.fill(255)  # White background

        # Drawing settings
        self.drawing_enabled = True
        self.colors = [
            {"name": "Red", "bgr": (0, 0, 255)},
            {"name": "Orange", "bgr": (0, 128, 255)},
            {"name": "Yellow", "bgr": (0, 255, 255)},
            {"name": "Green", "bgr": (0, 255, 0)},
            {"name": "Cyan", "bgr": (255, 255, 0)},
            {"name": "Blue", "bgr": (255, 0, 0)},
            {"name": "Purple", "bgr": (255, 0, 255)},
            {"name": "Black", "bgr": (0, 0, 0)},
            {
                "name": "Custom",
                "bgr": (0, 0, 255),
            },  # Will be updated during calibration
        ]
        self.current_color_index = 0
        self.drawing_color = self.colors[self.current_color_index]["bgr"]
        self.pen_thickness = 5
        self.eraser_mode = False
        self.eraser_size = 20

        # Whiteboard mode
        self.whiteboard_mode = False

        # For smoothing movement
        self.points_buffer = deque(maxlen=5)

        # UI settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_color = (0, 0, 0)
        self.font_thickness = 2

        # Create output directory
        self.output_dir = "whiteboard_captures"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # For tracking FPS
        self.prev_frame_time = 0

        # For color-based tracking
        self.lower_color = np.array([100, 100, 100])
        self.upper_color = np.array([140, 255, 255])
        self.calibrated = False

        # UI state
        self.show_help = True

        # Touch detection settings
        self.touch_distance_threshold = 30
        self.is_touching = False
        self.was_touching = False
        self.finger_position = None
        self.object_position = None
        self.touch_frames = 0
        self.touch_stability_threshold = 2

        # Debug mode
        self.debug_mode = False

        # Hand registration variables
        self.drawing_hand = None  # 'left' or 'right'
        self.control_hand = None  # 'left' or 'right'
        self.hands_registered = False
        self.registration_stage = 0

        # Selector settings
        self.color_selector_active = False
        self.size_selector_active = False
        self.color_selector_alpha = 0.0
        self.size_selector_alpha = 0.0
        self.color_selector_target_alpha = 0.0
        self.size_selector_target_alpha = 0.0
        self.animation_speed = 0.15

        # Selector positions and dimensions
        self.color_selector_position = None
        self.size_selector_position = None
        self.selector_width = 250  # Smaller, more concise
        self.selector_height = 80

        # Current selection values
        self.color_selection_x = 0  # Horizontal position for color selection
        self.size_selection_x = 0  # Horizontal position for size selection

        # For gesture stability
        self.color_gesture_frames = 0
        self.size_gesture_frames = 0
        self.gesture_stability_threshold = 5

        # For tracking when a new drawing stroke starts
        self.new_stroke = True

        # Writing support features
        self.writing_support_enabled = False
        self.stroke_history = []
        self.current_stroke = []
        self.stroke_timeout = 1.0
        self.last_draw_time = None

        # Pre-rendered templates
        self.create_selector_templates()

        # Character templates for writing support
        self.character_templates = {
            "A": [[(0.0, 1.0), (0.5, 0.0), (1.0, 1.0)], [(0.25, 0.5), (0.75, 0.5)]],
            "B": [
                [(0.0, 0.0), (0.0, 1.0)],
                [(0.0, 0.0), (0.75, 0.0), (1.0, 0.25), (0.75, 0.5), (0.0, 0.5)],
                [(0.0, 0.5), (0.75, 0.5), (1.0, 0.75), (0.75, 1.0), (0.0, 1.0)],
            ],
            "C": [
                [
                    (1.0, 0.25),
                    (0.75, 0.0),
                    (0.25, 0.0),
                    (0.0, 0.25),
                    (0.0, 0.75),
                    (0.25, 1.0),
                    (0.75, 1.0),
                    (1.0, 0.75),
                ]
            ],
            "0": [[(0.5, 0.0), (0.0, 0.5), (0.5, 1.0), (1.0, 0.5), (0.5, 0.0)]],
            "1": [[(0.5, 0.0), (0.5, 1.0)], [(0.25, 0.25), (0.5, 0.0)]],
            "2": [
                [
                    (0.0, 0.25),
                    (0.25, 0.0),
                    (0.75, 0.0),
                    (1.0, 0.25),
                    (0.0, 1.0),
                    (1.0, 1.0),
                ]
            ],
            "3": [[(0.0, 0.0), (1.0, 0.0), (0.5, 0.5), (1.0, 1.0), (0.0, 1.0)]],
            "4": [[(0.75, 0.0), (0.75, 1.0)], [(0.75, 0.0), (0.0, 0.5), (1.0, 0.5)]],
            "5": [
                [
                    (1.0, 0.0),
                    (0.0, 0.0),
                    (0.0, 0.5),
                    (0.75, 0.5),
                    (1.0, 0.75),
                    (0.75, 1.0),
                    (0.0, 1.0),
                ]
            ],
        }

    def create_selector_templates(self):
        """Create templates for the color and size selectors"""
        # Color selector template
        width, height = self.selector_width, self.selector_height
        self.color_selector_template = np.zeros((height, width, 4), dtype=np.uint8)

        # Create a futuristic hologram-style background
        # Gradient background
        for x in range(width):
            alpha = int(180 * (1 - abs((x - width / 2) / (width / 2))))
            cv2.line(
                self.color_selector_template,
                (x, 0),
                (x, height),
                (50, 70, 120, alpha),
                1,
            )

        # Add grid lines for futuristic effect
        for i in range(0, width, 20):
            alpha = 100 if i % 40 == 0 else 50
            cv2.line(
                self.color_selector_template,
                (i, 0),
                (i, height),
                (100, 200, 255, alpha),
                1,
            )

        for i in range(0, height, 10):
            alpha = 100 if i % 20 == 0 else 50
            cv2.line(
                self.color_selector_template,
                (0, i),
                (width, i),
                (100, 200, 255, alpha),
                1,
            )

        # Add border with glowing effect
        cv2.rectangle(
            self.color_selector_template,
            (0, 0),
            (width - 1, height - 1),
            (100, 200, 255, 200),
            1,
        )

        # Add smaller inner border
        cv2.rectangle(
            self.color_selector_template,
            (5, 5),
            (width - 6, height - 6),
            (150, 220, 255, 150),
            1,
        )

        # Add futuristic corner elements
        # Top-left corner
        cv2.line(
            self.color_selector_template, (0, 10), (10, 0), (200, 230, 255, 220), 1
        )
        # Top-right corner
        cv2.line(
            self.color_selector_template,
            (width - 11, 0),
            (width - 1, 10),
            (200, 230, 255, 220),
            1,
        )
        # Bottom-left corner
        cv2.line(
            self.color_selector_template,
            (0, height - 11),
            (10, height - 1),
            (200, 230, 255, 220),
            1,
        )
        # Bottom-right corner
        cv2.line(
            self.color_selector_template,
            (width - 11, height - 1),
            (width - 1, height - 11),
            (200, 230, 255, 220),
            1,
        )

        # Add title text
        cv2.putText(
            self.color_selector_template,
            "COLOR SELECTION",
            (40, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255, 200),
            1,
        )

        # Create the size selector template (similar design but different title and color)
        self.size_selector_template = self.color_selector_template.copy()

        # Clear the title area and add new title
        cv2.rectangle(
            self.size_selector_template, (20, 5), (width - 20, 25), (0, 0, 0, 0), -1
        )
        cv2.putText(
            self.size_selector_template,
            "SIZE ADJUSTMENT",
            (40, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255, 200),
            1,
        )

    def overlay_color_selector(self, frame, center_position, selection_x=None):
        """Overlay the color selector with a futuristic holographic look"""
        if center_position is None:
            return frame

        # Get dimensions
        h, w = self.selector_height, self.selector_width

        # Calculate position (centered on hand position)
        x_offset = int(center_position[0] - w / 2)
        y_offset = int(center_position[1] - h / 2)

        # Make sure it stays within frame boundaries
        x_offset = max(10, min(self.width - w - 10, x_offset))
        y_offset = max(10, min(self.height - h - 10, y_offset))

        # Create a copy of the frame to work with
        result = frame.copy()

        # Apply scaling effect for animation
        scale_factor = min(1.0, self.color_selector_alpha * 1.5)  # Scale up faster

        # Scale the template if needed
        if scale_factor < 1.0:
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            scaled_template = cv2.resize(self.color_selector_template, (new_w, new_h))

            # Recalculate offsets to keep centered
            x_offset = int(center_position[0] - new_w / 2)
            y_offset = int(center_position[1] - new_h / 2)

            # Update dimensions
            h, w = new_h, new_w
        else:
            scaled_template = self.color_selector_template

        # Create a copy with adjusted alpha
        adjusted_template = scaled_template.copy()
        adjusted_template[:, :, 3] = (
            scaled_template[:, :, 3] * self.color_selector_alpha
        ).astype(np.uint8)

        # Add color swatches if fully visible
        if self.color_selector_alpha > 0.7:
            # Draw color swatches
            swatch_height = 30
            swatch_width = (w - 20) // len(self.colors)
            swatch_y = h // 2 - swatch_height // 2

            for i, color_info in enumerate(self.colors):
                # Calculate the x position for this color
                swatch_x = 10 + i * swatch_width

                # Draw color swatch on the template
                color_with_alpha = (*color_info["bgr"], 180)  # Add alpha channel

                # Create a swatch with gradient effect
                for y in range(swatch_height):
                    alpha_factor = 0.6 + 0.4 * (
                        1 - abs((y - swatch_height / 2) / (swatch_height / 2))
                    )
                    for x in range(swatch_width - 2):
                        if 0 <= swatch_x + x < w and 0 <= swatch_y + y < h:
                            adjusted_template[swatch_y + y, swatch_x + x] = (
                                color_info["bgr"][0],
                                color_info["bgr"][1],
                                color_info["bgr"][2],
                                int(180 * alpha_factor),
                            )

            # Draw selection indicator based on horizontal position
            if selection_x is not None:
                # Map the horizontal position to a color index
                rel_x = max(0, min(1.0, (selection_x - (x_offset + 10)) / (w - 20)))
                selected_index = min(
                    len(self.colors) - 1, int(rel_x * len(self.colors))
                )

                # Highlight the selected color
                highlight_x = 10 + selected_index * swatch_width
                highlight_y = swatch_y

                # Draw highlight box
                cv2.rectangle(
                    adjusted_template,
                    (highlight_x - 2, highlight_y - 2),
                    (highlight_x + swatch_width, highlight_y + swatch_height + 2),
                    (255, 255, 255, 220),
                    2,
                )

                # Add glow effect
                cv2.rectangle(
                    adjusted_template,
                    (highlight_x - 4, highlight_y - 4),
                    (highlight_x + swatch_width + 2, highlight_y + swatch_height + 4),
                    (200, 230, 255, 150),
                    1,
                )

                # Show selected color name
                selected_color = self.colors[selected_index]["name"]
                text_y = h - 10
                text_size = cv2.getTextSize(
                    selected_color, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )[0]
                text_x = w // 2 - text_size[0] // 2

                # Add text background
                cv2.rectangle(
                    adjusted_template,
                    (text_x - 5, text_y - 15),
                    (text_x + text_size[0] + 5, text_y + 5),
                    (0, 0, 0, 100),
                    -1,
                )

                cv2.putText(
                    adjusted_template,
                    selected_color,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255, 200),
                    1,
                )

        # Overlay the template onto the frame
        # Calculate region to overlay
        y_end = min(y_offset + h, self.height)
        x_end = min(x_offset + w, self.width)
        y_start = max(0, y_offset)
        x_start = max(0, x_offset)

        if y_end > y_start and x_end > x_start:
            # Get the visible region of the template
            template_y_start = y_start - y_offset if y_start > y_offset else 0
            template_x_start = x_start - x_offset if x_start > x_offset else 0
            template_y_end = template_y_start + (y_end - y_start)
            template_x_end = template_x_start + (x_end - x_start)

            template_region = adjusted_template[
                template_y_start:template_y_end, template_x_start:template_x_end
            ]
            frame_region = result[y_start:y_end, x_start:x_end]

            if template_region.shape[:2] == frame_region.shape[:2]:
                # Overlay using alpha blending
                alpha = template_region[:, :, 3] / 255.0
                for c in range(3):
                    frame_region[:, :, c] = (
                        frame_region[:, :, c] * (1 - alpha)
                        + template_region[:, :, c] * alpha
                    )

        return result

    def overlay_size_selector(self, frame, center_position, selection_x=None):
        """Overlay the size selector with a futuristic holographic look"""
        if center_position is None:
            return frame

        # Get dimensions
        h, w = self.selector_height, self.selector_width

        # Calculate position (centered on hand position)
        x_offset = int(center_position[0] - w / 2)
        y_offset = int(center_position[1] - h / 2)

        # Make sure it stays within frame boundaries
        x_offset = max(10, min(self.width - w - 10, x_offset))
        y_offset = max(10, min(self.height - h - 10, y_offset))

        # Create a copy of the frame to work with
        result = frame.copy()

        # Apply scaling effect for animation
        scale_factor = min(1.0, self.size_selector_alpha * 1.5)  # Scale up faster

        # Scale the template if needed
        if scale_factor < 1.0:
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            scaled_template = cv2.resize(self.size_selector_template, (new_w, new_h))

            # Recalculate offsets to keep centered
            x_offset = int(center_position[0] - new_w / 2)
            y_offset = int(center_position[1] - new_h / 2)

            # Update dimensions
            h, w = new_h, new_w
        else:
            scaled_template = self.size_selector_template

        # Create a copy with adjusted alpha
        adjusted_template = scaled_template.copy()
        adjusted_template[:, :, 3] = (
            scaled_template[:, :, 3] * self.size_selector_alpha
        ).astype(np.uint8)

        # Add size slider if fully visible
        if self.size_selector_alpha > 0.7:
            # Draw slider track
            track_y = h // 2
            track_width = w - 40
            track_x_start = 20
            track_x_end = track_x_start + track_width

            # Draw track line
            cv2.line(
                adjusted_template,
                (track_x_start, track_y),
                (track_x_end, track_y),
                (150, 150, 150, 180),
                2,
            )

            # Draw tick marks on track
            num_ticks = 10
            for i in range(num_ticks + 1):
                tick_x = track_x_start + (track_width * i // num_ticks)
                tick_height = 6 if i % 5 == 0 else 3
                cv2.line(
                    adjusted_template,
                    (tick_x, track_y - tick_height),
                    (tick_x, track_y + tick_height),
                    (200, 200, 200, 150),
                    1,
                )

            # Draw selection indicator based on horizontal position
            if selection_x is not None:
                # Map the horizontal position to a size value
                rel_x = max(
                    0,
                    min(1.0, (selection_x - (x_offset + track_x_start)) / track_width),
                )

                # Calculate position on the track
                handle_x = int(track_x_start + rel_x * track_width)
                handle_y = track_y

                # Calculate size based on selected position
                if self.eraser_mode:
                    # 10-50 for eraser
                    selected_size = int(10 + rel_x * 40)
                else:
                    # 1-20 for pen
                    selected_size = max(1, int(1 + rel_x * 19))

                # Draw handle with glow effect
                # Outer glow
                cv2.circle(
                    adjusted_template, (handle_x, handle_y), 12, (100, 200, 255, 80), -1
                )
                cv2.circle(
                    adjusted_template, (handle_x, handle_y), 8, (150, 220, 255, 120), -1
                )
                cv2.circle(
                    adjusted_template, (handle_x, handle_y), 5, (255, 255, 255, 200), -1
                )

                # Show selected size value
                size_text = f"{selected_size}"
                text_y = h - 10
                text_size = cv2.getTextSize(
                    size_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )[0]
                text_x = w // 2 - text_size[0] // 2

                # Add text background
                cv2.rectangle(
                    adjusted_template,
                    (text_x - 5, text_y - 15),
                    (text_x + text_size[0] + 5, text_y + 5),
                    (0, 0, 0, 100),
                    -1,
                )

                cv2.putText(
                    adjusted_template,
                    size_text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255, 200),
                    1,
                )

                # Add tool type indicator
                tool_text = "ERASER" if self.eraser_mode else "PEN"
                tool_text_size = cv2.getTextSize(
                    tool_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
                )[0]
                tool_x = w // 2 - tool_text_size[0] // 2
                tool_y = text_y - 20

                cv2.putText(
                    adjusted_template,
                    tool_text,
                    (tool_x, tool_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (200, 200, 255, 180),
                    1,
                )

        # Overlay the template onto the frame
        # Calculate region to overlay
        y_end = min(y_offset + h, self.height)
        x_end = min(x_offset + w, self.width)
        y_start = max(0, y_offset)
        x_start = max(0, x_offset)

        if y_end > y_start and x_end > x_start:
            # Get the visible region of the template
            template_y_start = y_start - y_offset if y_start > y_offset else 0
            template_x_start = x_start - x_offset if x_start > x_offset else 0
            template_y_end = template_y_start + (y_end - y_start)
            template_x_end = template_x_start + (x_end - x_start)

            template_region = adjusted_template[
                template_y_start:template_y_end, template_x_start:template_x_end
            ]
            frame_region = result[y_start:y_end, x_start:x_end]

            if template_region.shape[:2] == frame_region.shape[:2]:
                # Overlay using alpha blending
                alpha = template_region[:, :, 3] / 255.0
                for c in range(3):
                    frame_region[:, :, c] = (
                        frame_region[:, :, c] * (1 - alpha)
                        + template_region[:, :, c] * alpha
                    )

        return result

    def register_hands(self, frame):
        """Handle the hand registration process"""
        if self.registration_stage == 0:
            # Starting registration
            self.registration_stage = 1
            return frame

        # Get hand data
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        # Create overlay for instructions
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, self.height), (0, 0, 0), -1)
        alpha = 0.7
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        if self.registration_stage == 1:
            # Registering drawing hand
            cv2.putText(
                frame,
                "HAND REGISTRATION - STEP 1",
                (self.width // 4, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                "Raise your DRAWING hand and make a fist",
                (self.width // 4 - 50, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                "Press SPACE when ready",
                (self.width // 3, self.height - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            # Draw detected hands
            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style(),
                    )

                    # If we have handedness information
                    if results.multi_handedness and idx < len(results.multi_handedness):
                        handedness = (
                            results.multi_handedness[idx].classification[0].label
                        )
                        pos_x = int(hand_landmarks.landmark[0].x * self.width)
                        pos_y = int(hand_landmarks.landmark[0].y * self.height)
                        cv2.putText(
                            frame,
                            f"{handedness} Hand",
                            (pos_x - 50, pos_y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 0),
                            2,
                        )

            # Check for space key to select the drawing hand
            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # Space key
                if results.multi_handedness and len(results.multi_handedness) > 0:
                    self.drawing_hand = (
                        results.multi_handedness[0].classification[0].label.lower()
                    )
                    self.registration_stage = 2
                    print(f"Drawing hand registered as: {self.drawing_hand}")

        elif self.registration_stage == 2:
            # Registering control hand
            cv2.putText(
                frame,
                "HAND REGISTRATION - STEP 2",
                (self.width // 4, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                "Now raise your CONTROL hand and make a fist",
                (self.width // 4 - 50, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                "Press SPACE when ready",
                (self.width // 3, self.height - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            # Draw detected hands
            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style(),
                    )

                    # If we have handedness information
                    if results.multi_handedness and idx < len(results.multi_handedness):
                        handedness = (
                            results.multi_handedness[idx].classification[0].label
                        )
                        pos_x = int(hand_landmarks.landmark[0].x * self.width)
                        pos_y = int(hand_landmarks.landmark[0].y * self.height)
                        cv2.putText(
                            frame,
                            f"{handedness} Hand",
                            (pos_x - 50, pos_y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 0),
                            2,
                        )

            # Check for space key to select the control hand
            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # Space key
                if results.multi_handedness and len(results.multi_handedness) > 0:
                    candidate_control_hand = (
                        results.multi_handedness[0].classification[0].label.lower()
                    )
                    # Ensure control hand is different from drawing hand
                    if candidate_control_hand != self.drawing_hand:
                        self.control_hand = candidate_control_hand
                        self.hands_registered = True
                        self.registration_stage = 3
                        print(f"Control hand registered as: {self.control_hand}")
                        print("Registration complete. You can now use the whiteboard.")
                    else:
                        # Error message if same hand detected
                        cv2.putText(
                            frame,
                            "Please use your other hand for control.",
                            (self.width // 4, 200),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2,
                        )

        elif self.registration_stage == 3:
            # Show completion screen
            cv2.putText(
                frame,
                "HAND REGISTRATION COMPLETE",
                (self.width // 4, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Drawing Hand: {self.drawing_hand.upper()}",
                (self.width // 3, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Control Hand: {self.control_hand.upper()}",
                (self.width // 3, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                "Press SPACE to begin using the whiteboard",
                (self.width // 3 - 50, self.height - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            # Check for space key to continue
            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # Space key
                # Registration completed, move to main operation
                self.registration_stage = 4

        return frame

    def detect_hands_and_gestures(self, frame):
        """Detect both hands and analyze their gestures"""
        # Reset hand tracking variables
        self.finger_position = None
        self.object_position = None

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe
        results = self.hands.process(rgb_frame)

        # Check if any hands are detected
        if results.multi_hand_landmarks:
            drawing_landmarks = None
            control_landmarks = None

            # Identify drawing and control hands based on registration
            for idx, (hand_landmarks, handedness) in enumerate(
                zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness if results.multi_handedness else [],
                )
            ):
                hand_type = handedness.classification[0].label.lower()

                if hand_type == self.drawing_hand:
                    drawing_landmarks = hand_landmarks
                elif hand_type == self.control_hand:
                    control_landmarks = hand_landmarks

            # Process drawing hand if detected
            if drawing_landmarks is not None:
                # Draw hand landmarks for visual feedback
                if not self.whiteboard_mode or self.debug_mode:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        drawing_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style(),
                    )

                # Get index finger position for drawing
                self.finger_position = (
                    int(drawing_landmarks.landmark[8].x * self.width),
                    int(drawing_landmarks.landmark[8].y * self.height),
                )

                # Detect the object position if calibrated
                if self.calibrated:
                    self.object_position = self.detect_object_position(frame)

            # Process control hand if detected
            if control_landmarks is not None:
                # Draw hand landmarks for visual feedback
                if not self.whiteboard_mode or self.debug_mode:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        control_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style(),
                    )

                # Get key control hand landmark positions
                wrist = (
                    int(control_landmarks.landmark[0].x * self.width),
                    int(control_landmarks.landmark[0].y * self.height),
                )

                index_tip = (
                    int(control_landmarks.landmark[8].x * self.width),
                    int(control_landmarks.landmark[8].y * self.height),
                )

                index_pip = (
                    int(control_landmarks.landmark[6].x * self.width),
                    int(control_landmarks.landmark[6].y * self.height),
                )

                index_mcp = (
                    int(control_landmarks.landmark[5].x * self.width),
                    int(control_landmarks.landmark[5].y * self.height),
                )

                thumb_tip = (
                    int(control_landmarks.landmark[4].x * self.width),
                    int(control_landmarks.landmark[4].y * self.height),
                )

                thumb_ip = (
                    int(control_landmarks.landmark[3].x * self.width),
                    int(control_landmarks.landmark[3].y * self.height),
                )

                thumb_mcp = (
                    int(control_landmarks.landmark[2].x * self.width),
                    int(control_landmarks.landmark[2].y * self.height),
                )

                middle_tip = (
                    int(control_landmarks.landmark[12].x * self.width),
                    int(control_landmarks.landmark[12].y * self.height),
                )

                ring_tip = (
                    int(control_landmarks.landmark[16].x * self.width),
                    int(control_landmarks.landmark[16].y * self.height),
                )

                pinky_tip = (
                    int(control_landmarks.landmark[20].x * self.width),
                    int(control_landmarks.landmark[20].y * self.height),
                )

                # Check for index finger raised (for color selection)
                index_raised = (
                    # Index finger is extended
                    index_tip[1] < index_pip[1]
                    and index_pip[1] < index_mcp[1]
                    and
                    # Other fingers are curled
                    middle_tip[1] > index_mcp[1]
                    and ring_tip[1] > index_mcp[1]
                    and pinky_tip[1] > index_mcp[1]
                )

                # Check for thumb raised (for size selection)
                thumb_raised = (
                    # Thumb is extended to the side
                    (
                        thumb_tip[0] < thumb_ip[0]
                        if wrist[0] < thumb_mcp[0]
                        else thumb_tip[0] > thumb_ip[0]
                    )
                    and
                    # Other fingers are curled
                    index_tip[1] > thumb_mcp[1]
                    and middle_tip[1] > thumb_mcp[1]
                    and ring_tip[1] > thumb_mcp[1]
                    and pinky_tip[1] > thumb_mcp[1]
                )

                # Track gesture stability
                if index_raised:
                    self.color_gesture_frames += 1
                    self.size_gesture_frames = 0
                elif thumb_raised:
                    self.size_gesture_frames += 1
                    self.color_gesture_frames = 0
                else:
                    self.color_gesture_frames = 0
                    self.size_gesture_frames = 0

                # Only activate selectors after stability threshold
                color_selector_now_active = (
                    self.color_gesture_frames >= self.gesture_stability_threshold
                )
                size_selector_now_active = (
                    self.size_gesture_frames >= self.gesture_stability_threshold
                )

                # Set color selector state
                if color_selector_now_active:
                    self.color_selector_active = True
                    self.color_selector_position = wrist
                    self.color_selector_target_alpha = 1.0  # Target for animation

                    # Store horizontal position for selection
                    self.color_selection_x = index_tip[0]

                    # Process color selection
                    # Map the horizontal position to a color index
                    selector_width = self.selector_width
                    selector_x_offset = max(
                        10,
                        min(
                            self.width - selector_width - 10,
                            int(wrist[0] - selector_width / 2),
                        ),
                    )
                    rel_x = max(
                        0,
                        min(
                            1.0,
                            (index_tip[0] - (selector_x_offset + 10))
                            / (selector_width - 20),
                        ),
                    )
                    selected_index = min(
                        len(self.colors) - 1, int(rel_x * len(self.colors))
                    )

                    # Update color if needed
                    if selected_index != self.current_color_index:
                        self.current_color_index = selected_index
                        self.drawing_color = self.colors[selected_index]["bgr"]
                        if (
                            self.eraser_mode and selected_index < len(self.colors) - 1
                        ):  # Not eraser
                            self.eraser_mode = False
                else:
                    # Start fade-out if wheel was previously active
                    if self.color_selector_active:
                        self.color_selector_target_alpha = 0.0

                    # Only set to inactive after animation completes
                    if self.color_selector_alpha < 0.1:
                        self.color_selector_active = False

                # Set size selector state
                if size_selector_now_active:
                    self.size_selector_active = True
                    self.size_selector_position = wrist
                    self.size_selector_target_alpha = 1.0  # Target for animation

                    # Store horizontal position for selection
                    self.size_selection_x = thumb_tip[0]

                    # Process size selection
                    # Map the horizontal position to a size value
                    selector_width = self.selector_width
                    selector_x_offset = max(
                        10,
                        min(
                            self.width - selector_width - 10,
                            int(wrist[0] - selector_width / 2),
                        ),
                    )
                    track_width = selector_width - 40
                    track_x_start = selector_x_offset + 20

                    rel_x = max(
                        0, min(1.0, (thumb_tip[0] - track_x_start) / track_width)
                    )

                    # Update size based on selected position
                    if self.eraser_mode:
                        # 10-50 for eraser
                        new_size = int(10 + rel_x * 40)
                        self.eraser_size = new_size
                    else:
                        # 1-20 for pen
                        new_size = max(1, int(1 + rel_x * 19))
                        self.pen_thickness = new_size
                else:
                    # Start fade-out if selector was previously active
                    if self.size_selector_active:
                        self.size_selector_target_alpha = 0.0

                    # Only set to inactive after animation completes
                    if self.size_selector_alpha < 0.1:
                        self.size_selector_active = False
        else:
            # No hands detected, fade out selectors
            self.color_selector_target_alpha = 0.0
            self.size_selector_target_alpha = 0.0

            if self.color_selector_alpha < 0.1:
                self.color_selector_active = False

            if self.size_selector_alpha < 0.1:
                self.size_selector_active = False

            # Reset gesture counters
            self.color_gesture_frames = 0
            self.size_gesture_frames = 0

        # Animate selectors (smooth transitions)
        if self.color_selector_alpha < self.color_selector_target_alpha:
            self.color_selector_alpha = min(
                1.0, self.color_selector_alpha + self.animation_speed
            )
        elif self.color_selector_alpha > self.color_selector_target_alpha:
            self.color_selector_alpha = max(
                0.0, self.color_selector_alpha - self.animation_speed
            )

        if self.size_selector_alpha < self.size_selector_target_alpha:
            self.size_selector_alpha = min(
                1.0, self.size_selector_alpha + self.animation_speed
            )
        elif self.size_selector_alpha > self.size_selector_target_alpha:
            self.size_selector_alpha = max(
                0.0, self.size_selector_alpha - self.animation_speed
            )

        # Check for touch between finger and object
        if self.finger_position is not None and self.object_position is not None:
            self.detect_touch(frame)

    def detect_object_position(self, frame):
        """Detect the object position using color thresholding"""
        if not self.calibrated:
            return None

        # Convert frame to HSV color space
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create a mask using the defined color range
        mask = cv2.inRange(hsv_frame, self.lower_color, self.upper_color)

        # Apply morphological operations to remove noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Return None if no contours found
        if not contours:
            return None

        # Find the largest contour (assuming it's the object)
        largest_contour = max(contours, key=cv2.contourArea)

        # Ignore small contours that might be noise
        if cv2.contourArea(largest_contour) < 100:
            return None

        # Find the topmost point of the contour (object tip)
        topmost = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])

        # Visual feedback for object detection (if not in whiteboard mode)
        if not self.whiteboard_mode or self.debug_mode:
            # Draw a circle at the detected object tip
            cv2.circle(frame, topmost, 8, self.drawing_color, -1)

            # Draw the contour outline
            cv2.drawContours(frame, [largest_contour], 0, self.drawing_color, 2)

        return topmost

    def detect_touch(self, frame):
        """Detect if the finger is touching the object"""
        # Both finger and object must be detected
        if self.finger_position is None or self.object_position is None:
            self.is_touching = False
            self.touch_frames = 0
            self.was_touching = False
            self.new_stroke = True
            return False

        # Calculate distance between finger and object
        distance = np.sqrt(
            (self.finger_position[0] - self.object_position[0]) ** 2
            + (self.finger_position[1] - self.object_position[1]) ** 2
        )

        # Determine if touching based on distance threshold
        current_touch = distance < self.touch_distance_threshold

        # Store previous touch state
        prev_is_touching = self.is_touching

        # For stability, require multiple consecutive frames with touch detected
        if current_touch:
            self.touch_frames += 1
            if self.touch_frames >= self.touch_stability_threshold:
                self.was_touching = self.is_touching  # Store previous state
                self.is_touching = True
        else:
            self.touch_frames = 0
            self.was_touching = self.is_touching  # Store previous state
            self.is_touching = False

        # Set new_stroke flag when touch is first detected
        if self.is_touching and not prev_is_touching:
            self.new_stroke = True
        else:
            self.new_stroke = False

        # Visual feedback for touch detection (if debug mode is on)
        if self.debug_mode:
            # Draw a line between finger and object
            cv2.line(
                frame, self.finger_position, self.object_position, (255, 0, 255), 2
            )

            # Show distance
            midpoint = (
                (self.finger_position[0] + self.object_position[0]) // 2,
                (self.finger_position[1] + self.object_position[1]) // 2,
            )
            cv2.putText(
                frame,
                f"Dist: {distance:.1f}",
                midpoint,
                self.font,
                0.6,
                (255, 0, 255),
                2,
            )

            # Show touch status
            status_color = (0, 255, 0) if self.is_touching else (0, 0, 255)
            cv2.putText(
                frame,
                f"Touch: {'YES' if self.is_touching else 'NO'}",
                (50, 50),
                self.font,
                1,
                status_color,
                2,
            )

            # Show new stroke indicator
            cv2.putText(
                frame,
                f"New Stroke: {'YES' if self.new_stroke else 'NO'}",
                (50, 80),
                self.font,
                1,
                (255, 255, 0) if self.new_stroke else (0, 0, 255),
                2,
            )

            # Draw touch threshold circle around finger
            cv2.circle(
                frame,
                self.finger_position,
                self.touch_distance_threshold,
                status_color,
                1,
            )

        return self.is_touching

    def smooth_point(self, point):
        """Apply smoothing to reduce jitter"""
        if point is None:
            return None

        # Add point to buffer
        self.points_buffer.append(point)

        # Don't smooth if we don't have enough points
        if len(self.points_buffer) < 2:
            return point

        # Calculate average point (weighted more towards recent points)
        weights = np.linspace(0.6, 1.0, len(self.points_buffer))
        weights = weights / weights.sum()

        x = int(sum(p[0] * w for p, w in zip(self.points_buffer, weights)))
        y = int(sum(p[1] * w for p, w in zip(self.points_buffer, weights)))

        return (x, y)

    def draw_line(self, start_point, end_point):
        """Draw a line between two points on the canvas"""
        if start_point is None or end_point is None:
            return

        if self.eraser_mode:
            # Draw with white color (eraser)
            cv2.line(
                self.canvas, start_point, end_point, (255, 255, 255), self.eraser_size
            )
        else:
            # Draw with selected color and thickness
            cv2.line(
                self.canvas,
                start_point,
                end_point,
                self.drawing_color,
                self.pen_thickness,
            )

        # Add points to current stroke for writing support
        if self.writing_support_enabled:
            self.current_stroke.append(end_point)
            self.last_draw_time = time.time()

    def check_for_completed_stroke(self):
        """Check if a stroke has been completed and process it"""
        if (
            not self.writing_support_enabled
            or not self.current_stroke
            or self.last_draw_time is None
        ):
            return

        # Check if enough time has passed since last drawing
        if (
            time.time() - self.last_draw_time > self.stroke_timeout
            and len(self.current_stroke) > 10
        ):
            # Process the completed stroke
            self.stroke_history.append(self.current_stroke.copy())

            # Try to recognize the character or shape
            recognized = self.recognize_character(self.current_stroke)

            # If recognized, replace with clean version
            if recognized:
                print(f"Recognized: {recognized}")
                # Clear the stroke area and draw clean version
                self.redraw_clean_character(recognized, self.current_stroke)

            # Reset current stroke
            self.current_stroke = []

    def recognize_character(self, stroke):
        """Simplified character recognition based on stroke geometry"""
        if len(stroke) < 10:
            return None

        # Find bounding box
        min_x = min(p[0] for p in stroke)
        max_x = max(p[0] for p in stroke)
        min_y = min(p[1] for p in stroke)
        max_y = max(p[1] for p in stroke)

        width = max_x - min_x
        height = max_y - min_y

        if width < 20 or height < 20:
            return None  # Too small to be a character

        # Normalize stroke to 0-1 range
        normalized_stroke = [
            ((p[0] - min_x) / width, (p[1] - min_y) / height) for p in stroke
        ]

        # Compare with templates
        best_match = None
        best_score = float("inf")

        for char, template_strokes in self.character_templates.items():
            # Calculate distance between normalized stroke and template
            score = self.calculate_template_distance(
                normalized_stroke, template_strokes
            )

            if score < best_score and score < 0.4:  # Threshold for accepting a match
                best_score = score
                best_match = char

        return best_match

    def calculate_template_distance(self, stroke, template_strokes):
        """Calculate distance between a stroke and a template"""
        # Simplify to a basic shape comparison
        # For each point in the stroke, find minimum distance to any template point
        total_distance = 0

        # Flatten template strokes into a single list of points
        flat_template = []
        for template_stroke in template_strokes:
            flat_template.extend(template_stroke)

        # Sample points from the stroke (reduce computation)
        sampled_stroke = stroke[:: max(1, len(stroke) // 20)]

        for point in sampled_stroke:
            min_dist = min(
                (point[0] - tp[0]) ** 2 + (point[1] - tp[1]) ** 2
                for tp in flat_template
            )
            total_distance += min_dist

        # Normalize by number of points
        return total_distance / len(sampled_stroke) if sampled_stroke else float("inf")

    def redraw_clean_character(self, char, stroke):
        """Redraw a recognized character with clean lines"""
        # Find bounding box of the original stroke
        min_x = min(p[0] for p in stroke)
        max_x = max(p[0] for p in stroke)
        min_y = min(p[1] for p in stroke)
        max_y = max(p[1] for p in stroke)

        width = max_x - min_x
        height = max_y - min_y

        # Clear the area with a white rectangle (with some padding)
        padding = 5
        cv2.rectangle(
            self.canvas,
            (min_x - padding, min_y - padding),
            (max_x + padding, max_y + padding),
            (255, 255, 255),
            -1,
        )

        # Draw the clean character
        for template_stroke in self.character_templates[char]:
            points = []
            for i in range(len(template_stroke)):
                x = int(min_x + template_stroke[i][0] * width)
                y = int(min_y + template_stroke[i][1] * height)
                points.append((x, y))

            # Draw lines between points
            for i in range(len(points) - 1):
                cv2.line(
                    self.canvas,
                    points[i],
                    points[i + 1],
                    self.drawing_color,
                    self.pen_thickness,
                )

        # Display a small confirmation animation
        temp_canvas = self.canvas.copy()
        cv2.rectangle(
            temp_canvas,
            (min_x - padding, min_y - padding),
            (max_x + padding, max_y + padding),
            (0, 255, 0),
            2,
        )
        return temp_canvas

    def display_ui(self, frame):
        """Display user interface elements"""
        # Calculate FPS
        current_time = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (current_time - self.prev_frame_time)
        self.prev_frame_time = current_time

        if self.show_help:
            # Background for text panel
            alpha = 0.7
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (350, 270), (255, 255, 255), -1)
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            cv2.rectangle(frame, (10, 10), (350, 270), (0, 0, 0), 1)

            # Display instructions
            instructions = [
                f"Drawing Mode: {'OFF' if not self.drawing_enabled else 'ON'}",
                f"Whiteboard Mode: {'ON' if self.whiteboard_mode else 'OFF'}",
                f"Writing Support: {'ON' if self.writing_support_enabled else 'OFF'}",
                f"Touch Detection: {'YES' if self.is_touching else 'NO'}",
                f"Current Tool: {'Eraser' if self.eraser_mode else 'Pen'}",
                f"Current Color: {self.colors[self.current_color_index]['name']}",
                f"Current Size: {self.eraser_size if self.eraser_mode else self.pen_thickness}",
                "Gestures:",
                "- Raise index finger on control hand to select color",
                "- Raise thumb on control hand to adjust size",
                "- Move hand left/right to change values",
                "Commands:",
                "c: Clear | s: Save | d: Toggle drawing",
                "e: Toggle eraser | w: Whiteboard mode",
                "r: Toggle writing support | b: Debug mode",
                "t: Touch threshold | h: Toggle help | q: Quit",
                f"FPS: {int(fps)}",
            ]

            y_pos = 30
            for instruction in instructions:
                cv2.putText(
                    frame,
                    instruction,
                    (20, y_pos),
                    self.font,
                    self.font_scale,
                    self.font_color,
                    self.font_thickness,
                )
                y_pos += 15
        else:
            # Just show minimal info when help is hidden
            cv2.putText(
                frame,
                f"FPS: {int(fps)} | Press 'h' for help",
                (20, 30),
                self.font,
                self.font_scale,
                self.font_color,
                self.font_thickness,
            )

        # Always show current color indicator
        color_indicator_size = 30
        cv2.rectangle(
            frame,
            (self.width - color_indicator_size - 20, 20),
            (self.width - 20, 20 + color_indicator_size),
            self.drawing_color,
            -1,
        )
        cv2.rectangle(
            frame,
            (self.width - color_indicator_size - 20, 20),
            (self.width - 20, 20 + color_indicator_size),
            (0, 0, 0),
            1,
        )

        # Show size indicator
        size_text = f"{self.eraser_size if self.eraser_mode else self.pen_thickness}"
        cv2.putText(
            frame,
            size_text,
            (self.width - color_indicator_size - 10, 20 + color_indicator_size + 20),
            self.font,
            0.7,
            (0, 0, 0),
            2,
        )

        return frame

    def save_drawing(self):
        """Save the current canvas as an image"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"whiteboard_{timestamp}.png")
        cv2.imwrite(filename, self.canvas)
        print(f"Drawing saved as {filename}")

        # Create a copy of the canvas with a saved message
        temp_canvas = self.canvas.copy()
        cv2.putText(
            temp_canvas,
            "Drawing Saved!",
            (self.width // 4, self.height // 2),
            self.font,
            2,
            (0, 200, 0),
            3,
        )
        return temp_canvas, filename

    def calibrate_color(self):
        """Interactive calibration for pen object detection"""
        print("Pen Object Calibration Mode")
        print("Hold your pen/object in the frame and press 'c' to capture its color")
        print("Press 'q' to exit calibration mode")

        # Create a flag for successful calibration
        calibration_successful = False
        sampled_bgr_color = None

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            # Draw a target region in the center
            center_x, center_y = self.width // 2, self.height // 2
            target_size = 40  # Larger target area
            cv2.rectangle(
                frame,
                (center_x - target_size, center_y - target_size),
                (center_x + target_size, center_y + target_size),
                (0, 255, 0),
                2,
            )

            # Display the current HSV values in the target region
            roi = frame[
                center_y - target_size : center_y + target_size,
                center_x - target_size : center_x + target_size,
            ]

            if roi.size > 0:  # Make sure ROI is not empty
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                avg_h = np.mean(hsv_roi[:, :, 0])
                avg_s = np.mean(hsv_roi[:, :, 1])
                avg_v = np.mean(hsv_roi[:, :, 2])

                # Calculate the average BGR color for the pen color
                avg_b = np.mean(roi[:, :, 0])
                avg_g = np.mean(roi[:, :, 1])
                avg_r = np.mean(roi[:, :, 2])
                sampled_bgr_color = (int(avg_b), int(avg_g), int(avg_r))
            else:
                avg_h, avg_s, avg_v = 0, 0, 0
                sampled_bgr_color = (0, 0, 255)  # Default to red

            # Display instructions
            instructions = [
                "PEN OBJECT CALIBRATION",
                "1. Position your pen/object in the green box",
                "2. Press 'c' to capture its color",
                "3. Press 't' to test detection",
                "4. Press 'a' to accept or 'r' to retry",
                "Press 'q' to cancel calibration",
                f"Current HSV: H:{avg_h:.1f} S:{avg_s:.1f} V:{avg_v:.1f}",
            ]

            # Draw semi-transparent background for text
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (400, 170), (255, 255, 255), -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

            # Display instructions
            y_pos = 30
            for instruction in instructions:
                cv2.putText(
                    frame, instruction, (20, y_pos), self.font, 0.6, (0, 0, 0), 2
                )
                y_pos += 20

            if hasattr(self, "temp_lower") and hasattr(self, "temp_upper"):
                cv2.putText(
                    frame,
                    f"Color Range: {self.temp_lower} to {self.temp_upper}",
                    (20, y_pos),
                    self.font,
                    0.6,
                    (0, 0, 0),
                    2,
                )

                # Show a color swatch of the sampled color
                cv2.rectangle(frame, (300, 140), (350, 190), sampled_bgr_color, -1)
                cv2.putText(frame, "Color", (305, 185), self.font, 0.5, (0, 0, 0), 1)

            cv2.imshow("Pen Calibration", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                # Exit without saving
                break

            elif key == ord("c"):
                # Sample the color in the target region
                if roi.size > 0:  # Ensure ROI is not empty
                    # Calculate average HSV values (already done above)

                    # Set color range with some margin
                    h_margin, s_margin, v_margin = (
                        15,
                        80,
                        80,
                    )  # Wider margins for better detection
                    self.temp_lower = np.array(
                        [
                            max(0, int(avg_h - h_margin)),
                            max(0, int(avg_s - s_margin)),
                            max(0, int(avg_v - v_margin)),
                        ]
                    )
                    self.temp_upper = np.array(
                        [
                            min(179, int(avg_h + h_margin)),
                            min(255, int(avg_s + s_margin)),
                            min(255, int(avg_v + v_margin)),
                        ]
                    )

                    print(
                        f"Color sampled. HSV range: {self.temp_lower} to {self.temp_upper}"
                    )
                    print("Press 't' to test the calibration")

            elif (
                key == ord("t")
                and hasattr(self, "temp_lower")
                and hasattr(self, "temp_upper")
            ):
                # Test the calibration
                hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv_frame, self.temp_lower, self.temp_upper)

                # Apply morphological operations to remove noise
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.erode(mask, kernel, iterations=1)
                mask = cv2.dilate(mask, kernel, iterations=2)

                # Show the mask
                cv2.imshow("Calibration Test", mask)

                # Find contours in the mask for visualization
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                # Create a visualization frame
                viz_frame = frame.copy()
                if contours:
                    # Draw all contours
                    cv2.drawContours(viz_frame, contours, -1, (0, 255, 0), 2)

                    # Find the largest contour
                    largest_contour = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(largest_contour) > 100:
                        # Find the topmost point
                        topmost = tuple(
                            largest_contour[largest_contour[:, :, 1].argmin()][0]
                        )
                        cv2.circle(viz_frame, topmost, 10, (0, 0, 255), -1)

                        # Show text indicating successful detection
                        cv2.putText(
                            viz_frame,
                            "Object detected!",
                            (50, 50),
                            self.font,
                            1,
                            (0, 255, 0),
                            2,
                        )

                cv2.imshow("Detection Test", viz_frame)

                # Wait for keypress
                print("Press 'a' to accept this calibration or 'r' to retry")
                while True:
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord("a"):
                        self.lower_color = self.temp_lower
                        self.upper_color = self.temp_upper

                        # Set the custom color to match the pen color
                        self.colors[-1][
                            "bgr"
                        ] = sampled_bgr_color  # Update the Custom color
                        self.current_color_index = (
                            len(self.colors) - 1
                        )  # Set to Custom color
                        self.drawing_color = sampled_bgr_color

                        calibration_successful = True
                        print(
                            f"Calibration accepted. HSV range: {self.lower_color} to {self.upper_color}"
                        )
                        print(f"Drawing color set to match pen: {sampled_bgr_color}")
                        break
                    elif key == ord("r"):
                        # Retry calibration
                        break

                # Close test windows
                cv2.destroyWindow("Calibration Test")
                cv2.destroyWindow("Detection Test")

                if calibration_successful:
                    break

        cv2.destroyWindow("Pen Calibration")

        # Set calibration flag
        self.calibrated = calibration_successful

        return calibration_successful

    def adjust_touch_threshold(self):
        """Interactive adjustment of touch distance threshold"""
        print("Touch Threshold Adjustment Mode")
        print("Use + and - keys to adjust the threshold")
        print("Press Enter to accept, Esc to cancel")

        temp_threshold = self.touch_distance_threshold

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            # Detect hands
            self.detect_hands_and_gestures(frame)

            # Use the temporary threshold for detection visualization
            old_threshold = self.touch_distance_threshold
            self.touch_distance_threshold = temp_threshold
            self.detect_touch(frame)
            self.touch_distance_threshold = old_threshold

            # Add UI for threshold adjustment
            cv2.rectangle(frame, (10, 10), (400, 120), (255, 255, 255), -1)

            instructions = [
                "TOUCH THRESHOLD ADJUSTMENT",
                f"Current threshold: {temp_threshold} pixels",
                "Use + and - keys to adjust",
                "Press Enter to accept, Esc to cancel",
            ]

            y_pos = 30
            for instruction in instructions:
                cv2.putText(
                    frame, instruction, (20, y_pos), self.font, 0.6, (0, 0, 0), 2
                )
                y_pos += 25

            # Draw threshold visualization
            if self.finger_position is not None:
                cv2.circle(
                    frame, self.finger_position, temp_threshold, (0, 255, 255), 2
                )

            cv2.imshow("Adjust Touch Threshold", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # Esc key
                # Cancel without saving
                break

            elif key == 13:  # Enter key
                # Accept the new threshold
                self.touch_distance_threshold = temp_threshold
                print(f"Touch threshold set to {self.touch_distance_threshold} pixels")
                break

            elif key == ord("+") or key == ord("="):
                # Increase threshold
                temp_threshold = min(100, temp_threshold + 5)

            elif key == ord("-") or key == ord("_"):
                # Decrease threshold
                temp_threshold = max(5, temp_threshold - 5)

        cv2.destroyWindow("Adjust Touch Threshold")

    def run(self):
        """Main loop for the virtual whiteboard"""
        prev_point = None
        current_point = None
        saved_notification = None
        saved_notification_time = 0

        # Start with hand registration
        while not self.hands_registered and self.registration_stage < 4:
            # Capture frame from webcam
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break

            # Flip frame horizontally for a more natural interaction
            frame = cv2.flip(frame, 1)

            # Process registration
            frame = self.register_hands(frame)

            # Show the frame
            cv2.imshow("Hand Registration", frame)

            # Exit registration if Esc is pressed
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # Esc key
                print("Registration aborted.")
                # Use defaults if registration was aborted
                if not self.hands_registered:
                    self.drawing_hand = "right"
                    self.control_hand = "left"
                    self.hands_registered = True
                break

        # Check if we need to calibrate first
        if not self.calibrated:
            print("Initial pen calibration required.")
            self.calibrated = self.calibrate_color()

        # Main operation loop
        while True:
            # Capture frame from webcam
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break

            # Flip frame horizontally for a more natural interaction
            frame = cv2.flip(frame, 1)

            # Make a copy of the frame for UI elements
            display_frame = frame.copy()

            # Detect hands, gestures, and objects
            self.detect_hands_and_gestures(display_frame)

            # Check if finger is touching the object
            is_touching = self.is_touching

            # Get drawing point if touching
            pen_tip = self.object_position if is_touching else None

            # Apply smoothing to reduce jitter
            smooth_tip = self.smooth_point(pen_tip)

            # Handle drawing logic
            if is_touching:  # Only update when touching
                # If this is a new touch (finger just started touching the pen)
                if self.new_stroke:
                    # Initialize with the current point but don't draw yet
                    prev_point = smooth_tip

                    # Start a new stroke for writing support
                    if self.writing_support_enabled:
                        # Check if previous stroke is complete
                        self.check_for_completed_stroke()
                        # Start a new stroke
                        self.current_stroke = []
                        if smooth_tip is not None:
                            self.current_stroke.append(smooth_tip)
                else:
                    # Continue existing stroke
                    current_point = smooth_tip

                    # Draw line if drawing is enabled and points are valid
                    if (
                        self.drawing_enabled
                        and prev_point is not None
                        and current_point is not None
                    ):
                        self.draw_line(prev_point, current_point)
                        prev_point = current_point
            else:
                # Check for completed stroke if we just stopped touching
                if self.was_touching and self.writing_support_enabled:
                    # Don't immediately check - let the timeout handle it
                    pass

            # Check for completed strokes after timeout
            if self.writing_support_enabled:
                self.check_for_completed_stroke()

            # Prepare the display based on whiteboard mode
            if self.whiteboard_mode:
                # In whiteboard mode, just show the canvas with minimal UI
                combined_view = self.canvas.copy()

                # Add a small camera preview in the corner
                preview_width = self.width // 5
                preview_height = self.height // 5
                preview = cv2.resize(display_frame, (preview_width, preview_height))

                # Place the preview in the bottom-right corner
                combined_view[
                    self.height - preview_height : self.height,
                    self.width - preview_width : self.width,
                ] = preview

                # If the color selector is active, add it
                if (
                    self.color_selector_alpha > 0.05
                    and self.color_selector_position is not None
                ):
                    combined_view = self.overlay_color_selector(
                        combined_view,
                        self.color_selector_position,
                        self.color_selection_x,
                    )

                # If the size selector is active, add it
                if (
                    self.size_selector_alpha > 0.05
                    and self.size_selector_position is not None
                ):
                    combined_view = self.overlay_size_selector(
                        combined_view,
                        self.size_selector_position,
                        self.size_selection_x,
                    )
            else:
                # In camera mode, combine canvas with camera feed
                combined_view = cv2.addWeighted(display_frame, 0.7, self.canvas, 0.7, 0)

                # If the color selector is active, add it
                if (
                    self.color_selector_alpha > 0.05
                    and self.color_selector_position is not None
                ):
                    combined_view = self.overlay_color_selector(
                        combined_view,
                        self.color_selector_position,
                        self.color_selection_x,
                    )

                # If the size selector is active, add it
                if (
                    self.size_selector_alpha > 0.05
                    and self.size_selector_position is not None
                ):
                    combined_view = self.overlay_size_selector(
                        combined_view,
                        self.size_selector_position,
                        self.size_selection_x,
                    )

            # Display UI elements
            combined_view = self.display_ui(combined_view)

            # Show the "Saved" notification if it exists
            current_time = cv2.getTickCount()
            if saved_notification is not None:
                if (
                    current_time - saved_notification_time
                ) / cv2.getTickFrequency() < 2:  # Show for 2 seconds
                    combined_view = cv2.addWeighted(
                        combined_view, 0.7, saved_notification, 0.3, 0
                    )
                else:
                    saved_notification = None

            # Show calibration reminder if needed
            if not self.calibrated:
                cv2.putText(
                    combined_view,
                    "Object not calibrated. Press 'p' to calibrate.",
                    (self.width // 4, 50),
                    self.font,
                    1,
                    (0, 0, 255),
                    2,
                )

            # Show the result
            cv2.imshow("Virtual Whiteboard", combined_view)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                # Quit
                break

            elif key == ord("c"):
                # Clear canvas
                self.canvas.fill(255)
                prev_point = None
                current_point = None
                self.current_stroke = []
                self.stroke_history = []

            elif key == ord("s"):
                # Save drawing
                saved_notification, _ = self.save_drawing()
                saved_notification_time = current_time

            elif key == ord("d"):
                # Toggle drawing mode
                self.drawing_enabled = not self.drawing_enabled
                prev_point = None  # Reset points to prevent unwanted lines
                current_point = None

            elif key == ord("e"):
                # Toggle eraser mode
                self.eraser_mode = not self.eraser_mode
                print(f"Eraser mode: {'ON' if self.eraser_mode else 'OFF'}")

            elif key == ord("h"):
                # Toggle help display
                self.show_help = not self.show_help

            elif key == ord("b"):
                # Toggle debug mode
                self.debug_mode = not self.debug_mode
                print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")

            elif key == ord("p"):
                # Calibrate pen
                self.calibrate_color()

            elif key == ord("t"):
                # Adjust touch threshold
                self.adjust_touch_threshold()

            elif key == ord("w"):
                # Toggle whiteboard mode
                self.whiteboard_mode = not self.whiteboard_mode
                print(f"Whiteboard mode: {'ON' if self.whiteboard_mode else 'OFF'}")

            elif key == ord("r"):
                # Toggle writing support
                self.writing_support_enabled = not self.writing_support_enabled
                print(
                    f"Writing support: {'ON' if self.writing_support_enabled else 'OFF'}"
                )

            elif key == ord("+") or key == ord("="):
                # Increase pen/eraser size
                if self.eraser_mode:
                    self.eraser_size = min(50, self.eraser_size + 5)
                else:
                    self.pen_thickness = min(20, self.pen_thickness + 1)

            elif key == ord("-") or key == ord("_"):
                # Decrease pen/eraser size
                if self.eraser_mode:
                    self.eraser_size = max(10, self.eraser_size - 5)
                else:
                    self.pen_thickness = max(1, self.pen_thickness - 1)

            # Color selection with number keys
            if key >= ord("1") and key <= ord("8"):
                index = key - ord("1")
                if index < len(self.colors):
                    self.current_color_index = index
                    self.drawing_color = self.colors[index]["bgr"]
                    self.eraser_mode = False  # Switch back to pen mode
            elif key == ord("0"):
                # Custom color (from pen calibration)
                self.current_color_index = len(self.colors) - 1
                self.drawing_color = self.colors[-1]["bgr"]
                self.eraser_mode = False

        # Clean up
        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()


def main():
    print("Starting Virtual Whiteboard...")

    try:
        whiteboard = VirtualWhiteboard()

        print("\nHand registration will begin first.")
        print("You'll be asked to identify your drawing hand and control hand.")

        # Run the main application
        whiteboard.run()

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        print("\nTroubleshooting:")
        print("1. Make sure your webcam is connected and working")
        print("2. Ensure you have the following libraries installed:")
        print("   - opencv-python")
        print("   - numpy")
        print("   - mediapipe")
        print("\nInstall with: pip install opencv-python numpy mediapipe")


if __name__ == "__main__":
    main()
