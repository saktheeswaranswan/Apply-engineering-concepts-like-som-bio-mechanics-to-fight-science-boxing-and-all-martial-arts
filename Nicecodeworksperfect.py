# ---------------------------
# INSTALL (Colab)
# ---------------------------
!pip install mediapipe opencv-python numpy

import os
import cv2
import json
import urllib.request
import numpy as np
import mediapipe as mp
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------------------
# DOWNLOAD MODEL
# ---------------------------
MODEL_PATH = "pose_landmarker_full.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"

if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# ---------------------------
# CONFIG
# ---------------------------
VIDEO_PATH = "video.mp4"
OUTPUT_VIDEO = "output.mp4"
OUTPUT_JSON = "biomech_advanced.json"

# ---------------------------
# TOGGLES
# ---------------------------
DRAW_POINTS = True
DRAW_GRF = True
DRAW_KINETIC_CHAIN = True
DRAW_BALANCE = True
DRAW_TRAJECTORY = True
DRAW_ANGLES = True

# ---------------------------
# LANDMARKS
# ---------------------------
LS, RS = 11, 12
LE, RE = 13, 14
LW, RW = 15, 16
LH, RH = 23, 24
LK, RK = 25, 26
LA, RA = 27, 28
LHEEL, RHEEL = 29, 30
LFOOT, RFOOT = 31, 32

# ---------------------------
# HELPERS
# ---------------------------
def pt(lm, i, w, h):
    return np.array([lm[i].x * w, lm[i].y * h], dtype=np.float32)

def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else np.zeros_like(v)

def smooth(prev, curr, alpha=0.7):
    return alpha * prev + (1 - alpha) * curr

# 🔥 BOOSTED VECTOR DRAW
def draw_vec(img, start, v, color, scale=1):
    v = v * 150
    end = (start + v * scale).astype(int)
    cv2.arrowedLine(img, tuple(start.astype(int)), tuple(end), color, 3, tipLength=0.3)

# ---------------------------
# ANGLES + FORCES
# ---------------------------
def joint_angle(a, b, c):
    ab = normalize(a - b)
    cb = normalize(c - b)
    dot = np.clip(np.dot(ab, cb), -1.0, 1.0)
    return np.degrees(np.arccos(dot))

def angle_to_force_dir(angle):
    rad = np.radians(angle)
    return normalize(np.array([np.cos(rad), np.sin(rad)]))

def ground_reaction_force(mag):
    return np.array([0, -mag])

def hand_force_from_angle(angle, mag=1.5):
    rad = np.radians(angle)
    return np.array([mag * np.cos(rad), mag * np.sin(rad)])

# ---------------------------
# DYNAMICS
# ---------------------------
def compute_velocity(prev, curr, dt):
    return (curr - prev) / (dt + 1e-6)

def compute_acceleration(prev_v, curr_v, dt):
    return (curr_v - prev_v) / (dt + 1e-6)

def stability_score(com, cop):
    return np.linalg.norm(com - cop)

# ---------------------------
# FOOT FORCE
# ---------------------------
def foot_force(heel, toe):
    return min(np.linalg.norm(toe - heel) / 120, 2.5)

# ---------------------------
# COM / COP
# ---------------------------
def compute_com(pts):
    torso = (pts[LS] + pts[RS] + pts[LH] + pts[RH]) / 4
    return (
        torso * 0.5
        + (pts[LH] + pts[RH]) * 0.2
        + (pts[LK] + pts[RK]) * 0.1
        + (pts[LA] + pts[RA]) * 0.05
        + (pts[LS] + pts[RS]) * 0.15
    )

def compute_cop(pts, lf, rf):
    left = (pts[LHEEL] + pts[LFOOT]) / 2
    right = (pts[RHEEL] + pts[RFOOT]) / 2
    total = lf + rf + 1e-6
    return (left * lf + right * rf) / total

# ---------------------------
# MODEL
# ---------------------------
options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO
)

pose = vision.PoseLandmarker.create_from_options(options)

# ---------------------------
# VIDEO
# ---------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
w, h = int(cap.get(3)), int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS) or 30

out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

timeline = []
fid = 0

com_track = deque(maxlen=30)
cop_track = deque(maxlen=30)

prev_com = None
prev_cop = None
prev_vel = None

DT = 1.0 / fps

# ---------------------------
# LOOP
# ---------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    res = pose.detect_for_video(mp_img, timestamp_ms)

    data = {"frame": fid}

    if not res.pose_landmarks:
        cv2.putText(frame, "NO POSE", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if res.pose_landmarks:
        lm = res.pose_landmarks[0]
        pts = [pt(lm, i, w, h) for i in range(len(lm))]

        if DRAW_POINTS:
            for p in pts:
                cv2.circle(frame, tuple(p.astype(int)), 6, (0, 255, 0), -1)

        connections = [
            (LS, RS), (LS, LE), (LE, LW),
            (RS, RE), (RE, RW),
            (LS, LH), (RS, RH),
            (LH, RH), (LH, LK), (LK, LA),
            (RH, RK), (RK, RA)
        ]

        for a, b in connections:
            cv2.line(frame,
                     tuple(pts[a].astype(int)),
                     tuple(pts[b].astype(int)),
                     (100, 255, 100), 2)

        lf = foot_force(pts[LHEEL], pts[LFOOT])
        rf = foot_force(pts[RHEEL], pts[RFOOT])

        com = compute_com(pts)
        cop = compute_cop(pts, lf, rf)

        if prev_com is not None:
            com = smooth(prev_com, com)
            cop = smooth(prev_cop, cop)

        if prev_com is not None:
            vel = compute_velocity(prev_com, com, DT)
            acc = compute_acceleration(prev_vel, vel, DT) if prev_vel is not None else np.zeros(2)
        else:
            vel = np.zeros(2)
            acc = np.zeros(2)

        prev_vel = vel
        prev_com, prev_cop = com, cop

        com_track.append(com)
        cop_track.append(cop)

        stab = stability_score(com, cop)

        knee_L = joint_angle(pts[LH], pts[LK], pts[LA])
        elbow_L = joint_angle(pts[LS], pts[LE], pts[LW])

        if DRAW_GRF:
            draw_vec(frame, pts[LFOOT], ground_reaction_force(lf), (255, 0, 0))
            draw_vec(frame, pts[RFOOT], ground_reaction_force(rf), (255, 0, 0))

        if DRAW_KINETIC_CHAIN:
            draw_vec(frame, pts[LK], angle_to_force_dir(knee_L), (0, 255, 255))
            draw_vec(frame, pts[LW], hand_force_from_angle(elbow_L), (255, 0, 255))

        if DRAW_BALANCE:
            cv2.circle(frame, tuple(com.astype(int)), 10, (0, 0, 255), -1)
            cv2.circle(frame, tuple(cop.astype(int)), 8, (255, 255, 0), -1)
            cv2.line(frame, tuple(com.astype(int)), tuple(cop.astype(int)), (0, 255, 255), 2)

        if DRAW_TRAJECTORY:
            for i in range(1, len(com_track)):
                cv2.line(frame,
                         tuple(com_track[i - 1].astype(int)),
                         tuple(com_track[i].astype(int)),
                         (0, 0, 255), 2)

        if DRAW_ANGLES:
            cv2.putText(frame, f"K:{int(knee_L)}",
                        tuple(pts[LK].astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    timeline.append(data)
    out.write(frame)
    fid += 1

# ---------------------------
# SAVE
# ---------------------------
cap.release()
out.release()
pose.close()

with open(OUTPUT_JSON, "w") as f:
    json.dump(timeline, f, indent=2)

print("🔥 DONE — FULL VISIBLE BIOMECHANICS")
