import cv2
import numpy as np
import os
import mediapipe as mp
from datetime import datetime
from collections import deque


class VirtualWhiteboardWithTouchDetection:
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

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Initialize canvas
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.canvas.fill(255)  # White background

        # Drawing settings and state
        self.drawing_enabled = True
        self.colors = {
            "red": (0, 0, 255),
            "green": (0, 255, 0),
            "blue": (255, 0, 0),
            "black": (0, 0, 0),
            "yellow": (0, 255, 255),
            "purple": (255, 0, 255),
            "custom": (0, 0, 255),  # Default, will be updated during calibration
        }
        self.current_color_name = "custom"
        self.drawing_color = self.colors[self.current_color_name]
        self.pen_thickness = 5
        self.eraser_mode = False
        self.eraser_size = 20

        # Whiteboard mode
        self.whiteboard_mode = False  # Start with camera feed mode

        # For smoothing movement
        self.points_buffer = deque(maxlen=8)

        # UI settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_color = (0, 0, 0)  # Black
        self.font_thickness = 2

        # Create output directory
        self.output_dir = "whiteboard_captures"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # For tracking FPS
        self.prev_frame_time = 0

        # For color-based tracking (will be updated during calibration)
        self.lower_color = np.array([100, 100, 100])  # HSV lower bound
        self.upper_color = np.array([140, 255, 255])  # HSV upper bound

        # Flag to indicate successful calibration
        self.calibrated = False

        # UI state
        self.show_help = True

        # Touch detection settings
        self.touch_distance_threshold = (
            30  # Maximum distance in pixels to be considered touching
        )
        self.is_touching = False
        self.finger_position = None
        self.object_position = None
        self.touch_frames = 0  # Count consecutive frames with touch detected
        self.touch_stability_threshold = 3  # Number of frames needed to confirm touch

        # Debug mode
        self.debug_mode = False  # Toggle with 'b' key to show touch detection metrics

    def detect_hand_index_finger(self, frame):
        """Detect the index fingertip using MediaPipe Hands"""
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe
        results = self.hands.process(rgb_frame)

        # Check if any hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks for visual feedback (if not in whiteboard mode)
                if not self.whiteboard_mode or self.debug_mode:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )

                # Get the coordinates of the index fingertip (landmark 8)
                index_finger_tip = hand_landmarks.landmark[8]
                x = int(index_finger_tip.x * self.width)
                y = int(index_finger_tip.y * self.height)

                # Store the finger position for touch detection
                self.finger_position = (x, y)

                # Draw a circle at the index fingertip for visual feedback
                if not self.whiteboard_mode or self.debug_mode:
                    cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)

                return (x, y)

        self.finger_position = None
        return None

    def detect_color_object(self, frame):
        """Detect object using color thresholding"""
        if not self.calibrated:
            # If not calibrated, we cannot detect the object properly
            text = "Object not calibrated. Press 'p' to calibrate."
            cv2.putText(
                frame,
                text,
                (self.width // 4, self.height // 2),
                self.font,
                1,
                (0, 0, 255),
                2,
            )
            self.object_position = None
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
            self.object_position = None
            return None

        # Find the largest contour (assuming it's the object)
        largest_contour = max(contours, key=cv2.contourArea)

        # Ignore small contours that might be noise
        if cv2.contourArea(largest_contour) < 100:
            self.object_position = None
            return None

        # Find the topmost point of the contour (object tip)
        topmost = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])

        # Store the object position for touch detection
        self.object_position = topmost

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
            return False

        # Calculate distance between finger and object
        distance = np.sqrt(
            (self.finger_position[0] - self.object_position[0]) ** 2
            + (self.finger_position[1] - self.object_position[1]) ** 2
        )

        # Determine if touching based on distance threshold
        current_touch = distance < self.touch_distance_threshold

        # For stability, require multiple consecutive frames with touch detected
        if current_touch:
            self.touch_frames += 1
            if self.touch_frames >= self.touch_stability_threshold:
                self.is_touching = True
        else:
            self.touch_frames = 0
            self.is_touching = False

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

            # Draw touch threshold circle around finger
            cv2.circle(
                frame,
                self.finger_position,
                self.touch_distance_threshold,
                status_color,
                1,
            )

        return self.is_touching

    def detect_pen_and_finger(self, frame):
        """Detect both the pen object and finger simultaneously"""
        # Always detect both finger and object
        finger_tip = self.detect_hand_index_finger(frame)
        object_tip = self.detect_color_object(frame)

        # Check if finger is touching the object
        is_touching = self.detect_touch(frame)

        # For drawing, use the object position if touch is detected
        if is_touching and object_tip is not None:
            return object_tip

        return None  # No drawing if not touching

    def smooth_point(self, point):
        """Apply smoothing to reduce jitter"""
        if point is None:
            return None

        # Add point to buffer
        self.points_buffer.append(point)

        # Don't smooth if we don't have enough points
        if len(self.points_buffer) < 3:
            return point

        # Calculate average point (weighted more towards recent points)
        weights = np.linspace(0.5, 1.0, len(self.points_buffer))
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
            cv2.rectangle(overlay, (10, 10), (350, 230), (255, 255, 255), -1)
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            cv2.rectangle(frame, (10, 10), (350, 230), (0, 0, 0), 1)

            # Display instructions
            instructions = [
                f"Drawing Mode: {'OFF' if not self.drawing_enabled else 'ON'}",
                f"Whiteboard Mode: {'ON' if self.whiteboard_mode else 'OFF'}",
                f"Touch Detection: {'YES' if self.is_touching else 'NO'}",
                f"Current Tool: {'Eraser' if self.eraser_mode else 'Pen'}",
                f"Current Color: {self.current_color_name.capitalize()}",
                "Commands:",
                "c: Clear | s: Save | d: Toggle drawing",
                "e: Toggle eraser | +/-: Size | w: Whiteboard mode",
                "1-6: Change colors | p: Calibrate | b: Debug mode",
                "t: Adjust touch threshold",
                "h: Toggle help | q: Quit",
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
                y_pos += 20
        else:
            # Just show minimal info when help is hidden
            cv2.putText(
                frame,
                f"FPS: {int(fps)} | Touch: {'YES' if self.is_touching else 'NO'} | Press 'h' for help",
                (20, 30),
                self.font,
                self.font_scale,
                self.font_color,
                self.font_thickness,
            )

        # Always show color palette
        palette_y = self.height - 50
        swatch_size = 30
        spacing = 5

        # Draw semi-transparent background for color palette
        palette_width = (swatch_size + spacing) * (len(self.colors)) + spacing
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (10, palette_y - 10),
            (10 + palette_width, palette_y + swatch_size + 10),
            (255, 255, 255),
            -1,
        )
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        # Draw color swatches
        for i, (color_name, color_bgr) in enumerate(self.colors.items()):
            # Position
            x = 10 + spacing + (swatch_size + spacing) * i

            # Draw color swatch
            cv2.rectangle(
                frame,
                (x, palette_y),
                (x + swatch_size, palette_y + swatch_size),
                color_bgr,
                -1,
            )

            # Highlight selected color
            if color_name == self.current_color_name:
                cv2.rectangle(
                    frame,
                    (x - 2, palette_y - 2),
                    (x + swatch_size + 2, palette_y + swatch_size + 2),
                    (0, 0, 0),
                    2,
                )

        # Add tool indicator
        tool_x = 10 + palette_width + 20
        cv2.putText(
            frame,
            f"Tool: {'Eraser' if self.eraser_mode else 'Pen'}",
            (tool_x, palette_y + 20),
            self.font,
            self.font_scale,
            self.font_color,
            self.font_thickness,
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
                        # Set the drawing color to match the pen color
                        self.colors["custom"] = sampled_bgr_color
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

            # Detect finger and object
            _ = self.detect_hand_index_finger(frame)
            _ = self.detect_color_object(frame)

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

        # Check if we need to calibrate first
        if not self.calibrated:
            print("Initial pen calibration required.")
            self.calibrated = self.calibrate_color()

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

            # Detect both finger and object, and check if they're touching
            pen_tip = self.detect_pen_and_finger(display_frame)

            # Apply smoothing to reduce jitter
            smooth_tip = self.smooth_point(pen_tip)

            # Update points for drawing
            prev_point = current_point
            current_point = smooth_tip

            # Draw line if drawing is enabled and points are valid
            if (
                self.drawing_enabled
                and prev_point is not None
                and current_point is not None
            ):
                self.draw_line(prev_point, current_point)

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
            else:
                # In camera mode, combine canvas with camera feed
                combined_view = cv2.addWeighted(display_frame, 0.7, self.canvas, 0.7, 0)

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
            cv2.imshow("Virtual Whiteboard with Touch Detection", combined_view)

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
            color_keys = {
                ord("1"): "red",
                ord("2"): "green",
                ord("3"): "blue",
                ord("4"): "black",
                ord("5"): "yellow",
                ord("6"): "purple",
                ord("0"): "custom",  # Key 0 to select the custom pen color
            }

            if key in color_keys:
                self.current_color_name = color_keys[key]
                self.drawing_color = self.colors[self.current_color_name]
                self.eraser_mode = False  # Switch back to pen mode

        # Clean up
        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()


def main():
    print("Starting Virtual Whiteboard with Touch Detection...")

    try:
        whiteboard = VirtualWhiteboardWithTouchDetection()

        print("\nPlease calibrate your drawing object first.")
        if whiteboard.calibrate_color():
            print("Calibration successful! Your object has been detected.")
        else:
            print(
                "Calibration skipped or failed. Please press 'p' during operation to calibrate."
            )

        print("\nVirtual Whiteboard is running!")
        print("Instructions:")
        print(
            "- Drawing only happens when your index finger is touching the calibrated object"
        )
        print("- Calibrate your object with the 'p' key")
        print("- Press 't' to adjust the touch sensitivity")
        print("- Press 'b' to toggle debug mode (shows touch detection details)")
        print("- Press 'w' to toggle whiteboard mode (full white screen)")
        print("- Press 'c' to clear the canvas")
        print("- Press 's' to save the current drawing")
        print("- Press 'e' to toggle eraser mode")
        print("- Press '0' to use your object's original color")
        print("- Press '1'-'6' to change colors")
        print("- Press 'h' to toggle help display")
        print("- Press 'q' to quit")

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
