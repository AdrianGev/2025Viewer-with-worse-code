import sys
import time
import tkinter as tk

from PIL import Image, ImageTk
from cscore import CameraServer, VideoMode
import numpy as np
import pyrealsense2 as rs
import cv2  # for edge detection and contour detection

w, h, fps = 640, 480, 30
pipe = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
config.enable_stream(rs.stream.color, w, h, rs.format.rgb8, fps)
pipe.start(config)

# align depth to color for better accuracy bc that seems like a smart person thing to do
align_to = rs.stream.color
align = rs.align(align_to)

root = tk.Tk()
txt = tk.Label(root, text="Color + Depth")
txt.pack()
c_label = tk.Label(root)
c_label.pack()
d_label = tk.Label(root)
d_label.pack()

cs_video = CameraServer.putVideo("RealSense", w, h)

while True:
    frames = pipe.wait_for_frames()
    
    # align depth to color for consistency
    aligned_frames = align.process(frames)
    depth = aligned_frames.get_depth_frame()
    color = aligned_frames.get_color_frame()

    if not depth or not color:
        continue  # skip if frames ain't valid

    # convert color and depth frames to numpy arrays
    color_array = np.asanyarray(color.get_data())
    depth_array = np.asanyarray(depth.get_data(), dtype=np.float64)

    # normalize depth for visualization!!!!!
    depth_array /= depth_array.max()
    depth_8bit = cv2.convertScaleAbs(depth_array * 255)  # convert to 8-bit for openCV bc its annoying

    # detect vertical posts using edge detection (often reffered to as edging)
    edges = cv2.Canny(depth_8bit, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    post_x_coords = []  # store x-cords of detected posts

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # summon wizard box

        # look for posty stuff
        if h > 50 and w < 50:
            post_x_coords.append(x + w // 2)  # store x-center of the detected post
            cv2.rectangle(color_array, (x, y), (x + w, y + h), (0, 255, 0), 2)  # draw rectangle around post

    if post_x_coords:
        avg_x_pixel = int(np.mean(post_x_coords))  # average post position in pixels
        depth_value = depth.get_distance(avg_x_pixel, h // 2)  # depth at post center

        # convert pixel position cords
        intrinsics = depth.profile.as_video_stream_profile().get_intrinsics()
        post_coords = rs.rs2_deproject_pixel_to_point(intrinsics, [avg_x_pixel, h // 2], depth_value)

        x_world, y_world, z_world = post_coords
        left_right_offset = x_world  # X represents horizontal shift

        # determine post position relative to center
        direction = "Center"
        if x_world < -0.05:
            direction = "Left"
        elif x_world > 0.05:
            direction = "Right"

        # update gooey with post position
        txt.config(text=f"Post Position: {direction} ({x_world:.2f}m left/right)")

    # convert images for gooey display
    c_img = Image.fromarray(color_array)
    d_img = Image.fromarray(depth_8bit)

    # send color image to camera server
    cs_video.putFrame(np.ascontiguousarray(color_array[..., ::-1]))

    # update gooey with new images
    c_imgtk = ImageTk.PhotoImage(c_img)
    d_imgtk = ImageTk.PhotoImage(d_img)
    c_label.config(image=c_imgtk)
    d_label.config(image=d_imgtk)

    # show frame with contours in opencv window
    cv2.imshow("Detected Posts", color_array)
    cv2.imshow("Edges", edges)

    # exit key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    root.update()
    
pipe.stop()
cv2.destroyAllWindows()