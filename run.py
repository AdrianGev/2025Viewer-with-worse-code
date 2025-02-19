import cv2
import numpy as np
import pyrealsense2 as rs

# Configure RealSense pipeline
w, h, fps = 640, 480, 30
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
pipeline.start(config)

try:
    while True:
        # Wait for a coherent pair of frames: depth
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue

        # Convert depth frame to a numpy array (in meters if desired, here we'll keep as raw units)
        depth_array = np.asanyarray(depth_frame.get_data(), dtype=np.float32)

        # Normalize the depth image for visualization.
        # (This normalization is solely for display; actual measurements use depth_array directly.)
        max_val = np.max(depth_array)
        if max_val == 0:
            max_val = 1
        depth_normalized = depth_array / max_val
        depth_8bit = cv2.convertScaleAbs(depth_normalized * 255)

        # Apply a Gaussian blur to reduce noise before edge detection
        blurred = cv2.GaussianBlur(depth_8bit, (5, 5), 0)

        # Perform Canny edge detection on the blurred image
        edges = cv2.Canny(blurred, 50, 150)

        # Use morphological closing to fill small gaps in edges
        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Find contours from the closed edge image
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Apply a color map to the 8-bit depth image to create a colorized representation.
        depth_color = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)

        # Process each contour
        for contour in contours:
            # Get bounding box for each contour
            x, y, w_rect, h_rect = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            aspect_ratio = w_rect / float(h_rect) if h_rect != 0 else 0

            # Filter contours based on size and shape: adjust these thresholds as needed.
            if h_rect > 60 and w_rect < 40 and area > 300 and aspect_ratio < 0.8:
                # Draw a green bounding box on the color-mapped depth image
                cv2.rectangle(depth_color, (x, y), (x + w_rect, y + h_rect), (0, 255, 0), 2)

                # Extract the ROI from the original depth_array
                roi = depth_array[y:y + h_rect, x:x + w_rect]
                min_depth = np.min(roi)
                print(f"Detected coral post at ({x},{y}) with bounding box {w_rect}x{h_rect} and min depth {min_depth:.2f}")

        # Show the final image with the color map and drawn contours
        cv2.imshow("Depth Color Map with Contours", depth_color)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop the RealSense pipeline and close windows
    pipeline.stop()
    cv2.destroyAllWindows()
