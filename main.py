import cv2
import numpy as np
import os
import mediapipe as mp
from datetime import datetime
from collections import deque


class VirtualWhiteboardML:
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
        }
        self.current_color_name = "red"
        self.drawing_color = self.colors[self.current_color_name]
        self.pen_thickness = 5
        self.eraser_mode = False
        self.eraser_size = 20

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

        # Detection modes
        self.detection_modes = ["hand_index", "color_object"]
        self.current_mode = "hand_index"  # Default to hand tracking

        # For color-based tracking (as fallback)
        self.lower_color = np.array([100, 100, 100])  # HSV lower bound
        self.upper_color = np.array([140, 255, 255])  # HSV upper bound

        # UI state
        self.show_help = True

        # For gesture detection
        self.prev_gesture = None
        self.gesture_frames = 0
        self.gesture_threshold = 10  # Frames to confirm gesture

        # Track hand presence
        self.hand_present = False

    def detect_hand_index_finger(self, frame):
        """Detect the index fingertip using MediaPipe Hands"""
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe
        results = self.hands.process(rgb_frame)

        # Check if any hands are detected
        if results.multi_hand_landmarks:
            self.hand_present = True
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks for visual feedback
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )

                # Get the coordinates of the index fingertip (landmark 8)
                index_finger_tip = hand_landmarks.landmark[8]
                x = int(index_finger_tip.x * self.width)
                y = int(index_finger_tip.y * self.height)

                # Get coordinates of other landmarks for gesture detection
                thumb_tip = hand_landmarks.landmark[4]
                middle_finger_tip = hand_landmarks.landmark[12]

                # Detect clear gesture (thumb and middle finger touching)
                thumb_middle_dist = np.sqrt(
                    (thumb_tip.x - middle_finger_tip.x) ** 2
                    + (thumb_tip.y - middle_finger_tip.y) ** 2
                )

                # If thumb and middle finger are close, trigger clear canvas
                if thumb_middle_dist < 0.05:  # Threshold to detect pinch
                    if self.prev_gesture != "clear":
                        self.gesture_frames = 0
                        self.prev_gesture = "clear"
                    else:
                        self.gesture_frames += 1

                    if self.gesture_frames >= self.gesture_threshold:
                        self.canvas.fill(255)  # Clear canvas
                        self.prev_gesture = None
                        self.gesture_frames = 0
                        print("Canvas cleared by gesture")
                else:
                    self.prev_gesture = None
                    self.gesture_frames = 0

                # Draw a circle at the index fingertip
                cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)

                return (x, y)

        self.hand_present = False
        return None

    def detect_color_object(self, frame):
        """Detect pen using color thresholding (fallback method)"""
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

        # Find the largest contour (assuming it's the pen)
        largest_contour = max(contours, key=cv2.contourArea)

        # Ignore small contours that might be noise
        if cv2.contourArea(largest_contour) < 100:
            return None

        # Find the topmost point of the contour (pen tip)
        topmost = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])

        # Draw a circle at the detected pen tip and contour for visual feedback
        cv2.circle(frame, topmost, 5, (0, 255, 0), -1)
        cv2.drawContours(frame, [largest_contour], 0, (255, 0, 0), 2)

        return topmost

    def detect_pen(self, frame):
        """Detect pen using the current detection mode"""
        if self.current_mode == "hand_index":
            return self.detect_hand_index_finger(frame)
        else:
            return self.detect_color_object(frame)

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
            cv2.rectangle(overlay, (10, 10), (300, 200), (255, 255, 255), -1)
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            cv2.rectangle(frame, (10, 10), (300, 200), (0, 0, 0), 1)

            # Display instructions
            instructions = [
                f"Drawing Mode: {'OFF' if not self.drawing_enabled else 'ON'}",
                f"Detection: {self.current_mode.replace('_', ' ').title()}",
                f"Current Tool: {'Eraser' if self.eraser_mode else 'Pen'}",
                f"Current Color: {self.current_color_name.capitalize()}",
                "Commands:",
                "c: Clear | s: Save | d: Toggle drawing",
                "e: Toggle eraser | +/-: Size",
                "1-6: Change colors | m: Switch mode",
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
                f"FPS: {int(fps)} | Press 'h' for help",
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
        palette_width = (swatch_size + spacing) * len(self.colors) + spacing
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
        """Interactive calibration for color object detection"""
        print("Color Calibration Mode")
        print("Hold your object in the frame and press 'c' to capture color")
        print("Press 'q' to exit calibration mode")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            # Draw a target region in the center
            center_x, center_y = self.width // 2, self.height // 2
            cv2.rectangle(
                frame,
                (center_x - 20, center_y - 20),
                (center_x + 20, center_y + 20),
                (0, 255, 0),
                2,
            )

            # Display the current HSV values in the target region
            roi = frame[center_y - 20 : center_y + 20, center_x - 20 : center_x + 20]
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            avg_h = np.mean(hsv_roi[:, :, 0])
            avg_s = np.mean(hsv_roi[:, :, 1])
            avg_v = np.mean(hsv_roi[:, :, 2])

            cv2.putText(
                frame,
                "Position object in the green box",
                (20, 30),
                self.font,
                self.font_scale,
                (0, 255, 0),
                self.font_thickness,
            )
            cv2.putText(
                frame,
                "Press 'c' to capture color, 't' to test, 'q' to exit",
                (20, 60),
                self.font,
                self.font_scale,
                (0, 255, 0),
                self.font_thickness,
            )
            cv2.putText(
                frame,
                f"Current HSV: H:{avg_h:.1f} S:{avg_s:.1f} V:{avg_v:.1f}",
                (20, 90),
                self.font,
                self.font_scale,
                (0, 255, 0),
                self.font_thickness,
            )

            if hasattr(self, "temp_lower") and hasattr(self, "temp_upper"):
                cv2.putText(
                    frame,
                    f"Range: {self.temp_lower} to {self.temp_upper}",
                    (20, 120),
                    self.font,
                    self.font_scale,
                    (0, 255, 0),
                    self.font_thickness,
                )

            cv2.imshow("Color Calibration", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            elif key == ord("c"):
                # Sample the color in the target region
                # Calculate average HSV values (already done above)

                # Set color range with some margin
                h_margin, s_margin, v_margin = 10, 50, 50
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

                # Wait for keypress
                print(
                    "Press any key to continue testing or 'a' to accept this calibration"
                )
                if cv2.waitKey(0) & 0xFF == ord("a"):
                    self.lower_color = self.temp_lower
                    self.upper_color = self.temp_upper
                    print(
                        f"Calibration accepted. HSV range: {self.lower_color} to {self.upper_color}"
                    )
                    cv2.destroyWindow("Calibration Test")

        cv2.destroyWindow("Color Calibration")
        if (
            hasattr(self, "temp_lower")
            and hasattr(self, "temp_upper")
            and cv2.getWindowProperty("Calibration Test", cv2.WND_PROP_VISIBLE) > 0
        ):
            cv2.destroyWindow("Calibration Test")

    def run(self):
        """Main loop for the virtual whiteboard"""
        prev_point = None
        current_point = None
        saved_notification = None
        saved_notification_time = 0
        consecutive_hand_missing = 0
        max_hand_missing = 10  # Number of frames to wait before resetting drawing state

        while True:
            # Capture frame from webcam
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break

            # Flip frame horizontally for a more natural interaction
            frame = cv2.flip(frame, 1)

            # Detect pen tip
            pen_tip = self.detect_pen(frame)

            # Apply smoothing
            smooth_tip = self.smooth_point(pen_tip)

            # Handle case when hand disappears
            if self.current_mode == "hand_index" and not self.hand_present:
                consecutive_hand_missing += 1
                if consecutive_hand_missing > max_hand_missing:
                    # Reset drawing state if hand is missing for too long
                    prev_point = None
                    current_point = None
            else:
                consecutive_hand_missing = 0

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

            # Combine canvas with camera frame with transparency
            alpha = 0.7  # Canvas opacity
            combined_view = cv2.addWeighted(frame, 1.0, self.canvas, alpha, 0)

            # Display UI
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

            # Show the result
            cv2.imshow("Virtual Whiteboard (ML Edition)", combined_view)

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

            elif key == ord("m"):
                # Switch detection mode
                current_idx = self.detection_modes.index(self.current_mode)
                next_idx = (current_idx + 1) % len(self.detection_modes)
                self.current_mode = self.detection_modes[next_idx]
                print(f"Switched to {self.current_mode} detection mode")

                # Reset points when switching modes
                prev_point = None
                current_point = None

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
    print("Starting Advanced Virtual Whiteboard (ML Edition)...")
    print("This version uses MediaPipe for hand tracking!")

    try:
        whiteboard = VirtualWhiteboardML()

        # Ask if color calibration is needed
        response = input(
            "Do you want to calibrate color detection for object tracking? (y/n): "
        )
        if response.lower() == "y":
            whiteboard.calibrate_color()

        print("\nVirtual Whiteboard is running!")
        print("Instructions:")
        print("- Use your index finger or a colored object as your pen")
        print("- Gesture: Pinch thumb and middle finger to clear canvas")
        print("- Press 'c' to clear the canvas")
        print("- Press 's' to save the current drawing")
        print("- Press 'd' to toggle drawing mode")
        print("- Press 'e' to toggle eraser mode")
        print("- Press 'm' to switch between hand and color detection")
        print("- Press '+'/'-' to adjust pen/eraser size")
        print("- Press '1'-'6' to change colors")
        print("- Press 'h' to toggle help display")
        print("- Press 'q' to quit")

        # Run the main application
        whiteboard.run()

    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure your webcam is connected and working")
        print("2. Ensure you have the following libraries installed:")
        print("   - opencv-python")
        print("   - numpy")
        print("   - mediapipe")
        print("\nInstall with: pip install opencv-python numpy mediapipe")


if __name__ == "__main__":
    main()
