## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
import statistics
import csv
from sklearn import linear_model

row_vals = []

def save_value(row):
        with open(r'values_depth.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(row)

def get_frame(pipeline):
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    return depth_frame, color_frame

def get_image(depth_frame, color_frame):
    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    return depth_image, color_image

def draw_rectangle_hand(results, images, w, h):
    """"The following code to get the rectangle around the hand is inspired on code found on: https://stackoverflow.com/questions/66876906/create-a-rectangle-around-all-the-points-returned-from-mediapipe-hand-landmark-d"""
    for hand_landmarks in results.multi_hand_landmarks:
        x_max = 0
        y_max = 0
        x_min = w
        y_min = h
        for lm in hand_landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            if x > x_max:
                x_max = x
            if x < x_min:
                x_min = x
            if y > y_max:
                y_max = y
            if y < y_min:
                y_min = y
        cv2.rectangle(images, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        return hand_landmarks

def apply_ransac_to_matrix(depth):
    depth_list = np.array(depth).reshape(-1,).tolist()
    X = np.arange(len(depth_list))
    X = X[:, np.newaxis]
    ransac = linear_model.RANSACRegressor()
    ransac.fit(X, depth_list)
    average_from_ransac = statistics.mean(ransac.predict(X))
    return average_from_ransac


def depth_wrist(images, depth_image, hand_landmarks, mp_hands, row_vals):
    x_wrist = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * w)
    y_wrist = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * h)

    # Draw rectangle around wrist
    cv2.rectangle(images, (x_wrist - 5, y_wrist + 5), (x_wrist + 5, y_wrist - 5), (0,255,0), 3)

    depth = depth_image

    # Pass all pixels in rectangle around wrist
    depth = depth[x_wrist-5:x_wrist+5, y_wrist-5:y_wrist+5]

    # Apply ransac to fill zero's in depth matrix with average value and retrieve the average distance
    average_val = apply_ransac_to_matrix(depth)

    # Print depth value
    print("Detected a hand {:.3} meters away.".format(average_val/1000))

    # Save depth value
    row_vals.append(average_val/1000)

if __name__ == '__main__':
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    # Check if right camera is being used
    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    # Enable stream
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            depth_frame, color_frame = get_frame(pipeline)

            if not depth_frame or not color_frame:
                continue

            depth_image, color_image = get_image(depth_frame, color_frame)

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape

            # If depth and color resolutions are different, resize color image to match depth image for display
            if depth_colormap_dim != color_colormap_dim:
                resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
                images = np.hstack((resized_color_image, depth_colormap))
            else:
                images = np.hstack((color_image, depth_colormap))

            h, w, c = images.shape

            with mp_hands.Hands(
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:

                    # To improve performance, optionally mark the image as not writeable to
                    # pass by reference.
                    images.flags.writeable = False
                    images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
                    results = hands.process(images)

                    # Draw the hand annotations on the image.
                    images.flags.writeable = True
                    images = cv2.cvtColor(images, cv2.COLOR_RGB2BGR)
                    if results.multi_hand_landmarks:
                        hand_landmarks = draw_rectangle_hand(results, images, w, h)
                        mp_drawing.draw_landmarks(
                            images,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())

                        depth_wrist(images, depth_image, hand_landmarks, mp_hands, row_vals)
                            
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            key = cv2.waitKey(1)

            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                save_value(row_vals) #distance is saved in meters
                cv2.destroyAllWindows()
                break

    finally:
        # Stop streaming
        pipeline.stop()