

import cv2
import mediapipe as mp
import numpy as np
import os
import absl.logging
import math

# Suppress logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
absl.logging.set_verbosity(absl.logging.ERROR)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=1, static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Set up video capture
cap = cv2.VideoCapture('karlaapose.mp4')

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    norm_ba, norm_bc = np.linalg.norm(ba), np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 0
    cosine_angle = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))

# Function to draw angle arcs
def draw_angle_arc(frame, a, b, c, angle):
    center = tuple(b)
    radius = 30
    start_angle = int(math.degrees(math.atan2(a[1] - b[1], a[0] - b[0])))
    end_angle = start_angle + int(angle)
    cv2.ellipse(frame, center, (radius, radius), 0, start_angle, end_angle, (0, 255, 255), 2)

# Set resolution for vertical short video (9:16 aspect ratio, 640x420 resolution)
screen_width, screen_height = 420, 640

# Force calculation
weight = 24  # kg
g = 9.81  # Gravity (m/s^2)
force_per_leg = (weight * g) / 2  # N per leg

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (screen_width, screen_height))
    h, w, _ = frame.shape
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        keypoints = {landmark.name: (int(lm[landmark].x * w), int(lm[landmark].y * h)) for landmark in mp_pose.PoseLandmark}
        
        # Calculate joint angles
        joint_angles = {
            'Left Knee': (calculate_angle(keypoints['LEFT_HIP'], keypoints['LEFT_KNEE'], keypoints['LEFT_ANKLE']), 'LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE'),
            'Right Knee': (calculate_angle(keypoints['RIGHT_HIP'], keypoints['RIGHT_KNEE'], keypoints['RIGHT_ANKLE']), 'RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE'),
            'Left Elbow': (calculate_angle(keypoints['LEFT_SHOULDER'], keypoints['LEFT_ELBOW'], keypoints['LEFT_WRIST']), 'LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST'),
            'Right Elbow': (calculate_angle(keypoints['RIGHT_SHOULDER'], keypoints['RIGHT_ELBOW'], keypoints['RIGHT_WRIST']), 'RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST')
        }
        
        # Draw joint arcs
        for joint, (angle, a, b, c) in joint_angles.items():
            draw_angle_arc(frame, keypoints[a], keypoints[b], keypoints[c], angle)
        
        # Draw force resultant vector for legs
        for foot in ['LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']:
            foot_x, foot_y = keypoints[foot]
            resultant_force = int(force_per_leg * (joint_angles['Left Knee'][0] + joint_angles['Right Knee'][0]) / 180)
            cv2.arrowedLine(frame, (foot_x, foot_y), (foot_x, foot_y - resultant_force), (0, 0, 255), 3)
        
        # Hand force vectors (X, Y directions)
        for wrist in ['LEFT_WRIST', 'RIGHT_WRIST']:
            wrist_x, wrist_y = keypoints[wrist]
            cv2.arrowedLine(frame, (wrist_x, wrist_y), (wrist_x + 30, wrist_y), (255, 0, 0), 3)  # X-direction
            cv2.arrowedLine(frame, (wrist_x, wrist_y), (wrist_x, wrist_y - 30), (0, 255, 0), 3)  # Y-direction
        
        # Kinetic linking: Power transfer visualization as moving rings
        for joint, (_, a, b, c) in joint_angles.items():
            mid_x = (keypoints[a][0] + keypoints[c][0]) // 2
            mid_y = (keypoints[a][1] + keypoints[c][1]) // 2
            cv2.circle(frame, (mid_x, mid_y), 10, (0, 255, 0), -1)
        
        # Display joint angles dynamically
        y_offset = 50
        for joint, (angle, _, _, _) in joint_angles.items():
            cv2.putText(frame, f"{joint}: {int(angle)} deg", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 40
        
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    cv2.imshow('Biomechanics Visualization', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
