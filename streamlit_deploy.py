%%writefile hello_world.py

import streamlit as st
import cv2
import numpy as np
import pyautogui
import time
import dlib
from imutils import face_utils

# Initialize variables to store previous pupil positions and wink detection time
prev_left_pupil_center = None
prev_right_pupil_center = None
wink_detected_time = 0
wink_duration = 0.5  # Duration in seconds for detecting a wink (adjusted)
scroll_active = False  # Flag to indicate if scrolling is active

# Initialize variables for eye tracking data smoothing
smoothing_window_size = 5
left_eye_positions = []
right_eye_positions = []

# Function to center the mouse cursor on the screen
def center_cursor():
    screen_width, screen_height = pyautogui.size()
    pyautogui.moveTo(screen_width // 2, screen_height // 2)

# Function to smooth eye tracking data using a moving average filter
def smooth_eye_positions(left_eye_center, right_eye_center):
    global left_eye_positions, right_eye_positions

    # Add current eye positions to the lists
    left_eye_positions.append(left_eye_center)
    right_eye_positions.append(right_eye_center)

    # Truncate the lists if they exceed the smoothing window size
    if len(left_eye_positions) > smoothing_window_size:
        left_eye_positions = left_eye_positions[-smoothing_window_size:]
    if len(right_eye_positions) > smoothing_window_size:
        right_eye_positions = right_eye_positions[-smoothing_window_size:]

    # Calculate the average eye positions
    smoothed_left_eye_center = np.mean(left_eye_positions, axis=0)
    smoothed_right_eye_center = np.mean(right_eye_positions, axis=0)

    return smoothed_left_eye_center, smoothed_right_eye_center

# Function to calculate vertical eye movement
def calculate_vertical_movement(left_eye_center):
    global prev_left_pupil_center

    # Calculate the vertical distance between current and previous eye positions
    if prev_left_pupil_center is not None:
        vertical_movement = left_eye_center[1] - prev_left_pupil_center[1]
        return vertical_movement
    else:
        return 0

# Function to move cursor based on eye movement
def move_cursor(left_pupil_center):
    global prev_left_pupil_center

    # Calculate the direction of eye movement based on pupil positions
    if prev_left_pupil_center is not None:
        eye_direction_x = left_pupil_center[0] - prev_left_pupil_center[0]
        eye_direction_y = left_pupil_center[1] - prev_left_pupil_center[1]

        # Move the cursor
        move_x, move_y = -eye_direction_x, eye_direction_y
        pyautogui.moveRel(move_x * 5, move_y * 5, duration=0.01)

    # Update previous pupil position
    prev_left_pupil_center = left_pupil_center

# Function to check for wink detection and perform scroll action
def check_wink_detection(shape):
    global wink_detected_time, scroll_active

    # Find the center of the right eye
    right_eye_pts = shape[42:48]
    right_eye_center = right_eye_pts.mean(axis=0).astype(int)

    # Find the center of the right pupil
    right_pupil_center = shape[43].astype(int)

    # Detect wink based on eye aspect ratio
    right_ear = eye_aspect_ratio(right_eye_pts)
    if right_ear < 0.2 and wink_detected_time == 0:
        wink_detected_time = time.time()
        st.write("Wink detected!")

    # Check if wink duration exceeds threshold
    if wink_detected_time != 0 and time.time() - wink_detected_time >= wink_duration:
        if right_ear < 0.2:
            scroll_active = not scroll_active
            wink_detected_time = 0
            st.write("Scroll activated!" if scroll_active else "Scroll deactivated!")

    # Perform scroll action if scrolling is active
    if scroll_active:
        # Calculate vertical eye movement for scrolling
        vertical_movement = calculate_vertical_movement(right_pupil_center)

        # Determine scrolling direction
        if vertical_movement > 0:
            pyautogui.scroll(2)  # Scroll up
            st.write("Scrolling up...")
        elif vertical_movement < 0:
            pyautogui.scroll(-2)  # Scroll down
            st.write("Scrolling down...")

# Function to check for blink detection and perform click action
def check_blink_detection(left_eye_pts, right_eye_pts):
    left_ear = eye_aspect_ratio(left_eye_pts)
    right_ear = eye_aspect_ratio(right_eye_pts)

    # Detect blink based on eye aspect ratio
    if left_ear < 0.2 and right_ear < 0.2:
        time.sleep(0.2)
        # Perform click action for blink
        pyautogui.click()
        st.write("Blink detected and click performed!")
        return True
    else:
        return False

# Function to calculate eye aspect ratio
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

import io

# Function to run eye tracking and control
def run_eye_tracking():
    # Initialize dlib's face detector (HOG-based) and create the facial landmark predictor
    p = "/Users/nandinidhiran/Downloads/shape_predictor_68_face_landmarks.dat"  
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    # Open the camera capture
    cap = cv2.VideoCapture(0)

    # Center the mouse cursor on the screen
    center_cursor()

    # Create a Streamlit placeholder for displaying the image
    image_placeholder = st.empty()

    while True:
        # Read frame from camera
        _, image = cap.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image
        rects = detector(gray, 0)

        # Loop over the face detections
        for rect in rects:
            # Determine the facial landmarks for the face region
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # Draw facial landmarks on the image
            for (x, y) in shape:
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

            # Find the center of the left eye
            left_eye_pts = shape[36:42]
            left_eye_center = left_eye_pts.mean(axis=0).astype(int)

            # Find the center of the left pupil
            left_pupil_center = shape[37].astype(int)

            # Find the center of the right eye
            right_eye_pts = shape[42:48]
            right_eye_center = right_eye_pts.mean(axis=0).astype(int)

            # Find the center of the right pupil
            right_pupil_center = shape[43].astype(int)

            # Smooth eye positions
            smoothed_left_eye_center, smoothed_right_eye_center = smooth_eye_positions(left_eye_center, right_eye_center)

            # Move cursor based on smoothed eye movement
            move_cursor(smoothed_left_eye_center)

            # Check for blink detection and perform click action
            if check_blink_detection(left_eye_pts, right_eye_pts):
                continue  # Skip wink detection if blink is detected

            # Check for wink detection and perform scroll action
            check_wink_detection(shape)

        # Display the image in the Streamlit app
        image_placeholder.image(image, channels="BGR", use_column_width=True)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera capture
    cap.release()
    cv2.destroyAllWindows()

# Streamlit app
st.title("Eye Tracking App")

# Run eye tracking
if st.button("Start Eye Tracking"):
    run_eye_tracking()

!streamlit run hello_world.py
#%run hello_world.py