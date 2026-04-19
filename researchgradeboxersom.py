# ---------------------------
# INSTALL (Colab)
# ---------------------------
# !pip install mediapipe opencv-python numpy

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
DRAW_SKELETON = True
DRAW_ARCS = True
DRAW_GRF = True
DRAW_KINETIC_CHAIN = True
DRAW_BALANCE = True
DRAW_TRAJECTORY = True

MAX_VECTOR_UNITS = 2.0

# ---------------------------
# LANDMARKS
# ---------------------------
LS, RS = 11,12
LE, RE = 13,14
LW, RW = 15,16
LH, RH = 23,24
LK, RK = 25,26
LA, RA = 27,28
LHEEL, RHEEL = 29,30
LFOOT, RFOOT = 31,32

# ---------------------------
# HELPERS
# ---------------------------
def pt(lm,i,w,h):
    return np.array([lm[i].x*w, lm[i].y*h], dtype=np.float32)

def normalize(v):
    n = np.linalg.norm(v)
    return v/n if n>1e-6 else np.zeros_like(v)

def smooth(prev, curr, alpha=0.7):
    return alpha*prev + (1-alpha)*curr

# ---------------------------
# 🔥 DYNAMICS (ADDED - NEXT LEVEL)
# ---------------------------
def compute_velocity(prev, curr, dt):
    return (curr - prev) / (dt + 1e-6)

def compute_acceleration(prev_v, curr_v, dt):
    return (curr_v - prev_v) / (dt + 1e-6)

def stability_score(com, cop):
    return np.linalg.norm(com - cop)

def detect_fall(score, threshold=80):
    return score > threshold

# ---------------------------
# FOOT FORCE
# ---------------------------
def foot_force(heel, toe):
    length = np.linalg.norm(toe - heel)
    return min(length/120, 2.5)

# ---------------------------
# ADVANCED COM
# ---------------------------
def compute_com(pts):
    torso = (pts[LS] + pts[RS] + pts[LH] + pts[RH]) / 4

    com = (
        torso * 0.5 +
        (pts[LH] + pts[RH]) * 0.2 +
        (pts[LK] + pts[RK]) * 0.1 +
        (pts[LA] + pts[RA]) * 0.05 +
        (pts[LS] + pts[RS]) * 0.15
    )
    return com

# ---------------------------
# COP
# ---------------------------
def compute_cop(pts, lf, rf):
    left = (pts[LHEEL] + pts[LFOOT]) / 2
    right = (pts[RHEEL] + pts[RFOOT]) / 2

    total = lf + rf + 1e-6
    return (left*lf + right*rf)/total

# ---------------------------
# DRAW VECTOR
# ---------------------------
def draw_vec(img,start,v,color):
    end = (start + v*60).astype(int)
    cv2.arrowedLine(img, tuple(start.astype(int)), tuple(end), color, 2)

# ---------------------------
# BALANCE DRAW
# ---------------------------
def draw_balance(frame, com, cop, w, h):
    cv2.circle(frame, tuple(com.astype(int)), 7, (0,0,255), -1)
    cv2.putText(frame,"COM",tuple(com.astype(int)+np.array([5,-5])),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)

    cv2.circle(frame, tuple(cop.astype(int)), 7, (255,255,0), -1)
    cv2.putText(frame,"COP",tuple(cop.astype(int)+np.array([5,-5])),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1)

    proj = np.array([com[0], h])
    cv2.line(frame, tuple(com.astype(int)), tuple(proj.astype(int)), (0,255,0),1)

    cv2.line(frame, tuple(com.astype(int)), tuple(cop.astype(int)), (0,255,255),2)

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
w,h = int(cap.get(3)), int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS) or 30

out = cv2.VideoWriter(OUTPUT_VIDEO,
                      cv2.VideoWriter_fourcc(*"mp4v"),
                      fps,(w,h))

timeline=[]
fid=0

# Trajectory buffers
com_track = deque(maxlen=30)
cop_track = deque(maxlen=30)

prev_com = None
prev_cop = None

# 🔥 NEW STATE VARIABLES
prev_vel = None
DT = 1.0 / fps

# ---------------------------
# LOOP
# ---------------------------
while cap.isOpened():
    ret,frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)

    res = pose.detect_for_video(mp_img, int(fid*1000/fps))

    data={"frame":fid}

    if res.pose_landmarks:
        lm = res.pose_landmarks[0]
        pts=[pt(lm,i,w,h) for i in range(len(lm))]

        # DRAW POINTS
        if DRAW_POINTS:
            for p in pts:
                cv2.circle(frame, tuple(p.astype(int)),3,(0,255,0),-1)

        # FOOT FORCE
        lf = foot_force(pts[LHEEL], pts[LFOOT])
        rf = foot_force(pts[RHEEL], pts[RFOOT])

        # COM + COP
        com = compute_com(pts)
        cop = compute_cop(pts, lf, rf)

        # SMOOTH
        if prev_com is not None:
            com = smooth(prev_com, com)
            cop = smooth(prev_cop, cop)

        prev_com, prev_cop = com, cop

        # TRACK
        com_track.append(com)
        cop_track.append(cop)

        # ---------------------------
        # 🔥 DYNAMICS
        # ---------------------------
        if prev_com is not None:
            vel = compute_velocity(prev_com, com, DT)
        else:
            vel = np.zeros(2)

        if prev_vel is not None:
            acc = compute_acceleration(prev_vel, vel, DT)
        else:
            acc = np.zeros(2)

        prev_vel = vel

        stab = stability_score(com, cop)
        falling = detect_fall(stab)

        # FORCE VECTOR (COP → COM)
        force_vec = com - cop
        force_dir = normalize(force_vec)
        draw_vec(frame, cop, force_dir, (0,128,255))

        # FALL ALERT
        if falling:
            cv2.putText(frame, "UNSTABLE / FALL RISK",
                        (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,(0,0,255),2)

        # DRAW BALANCE
        if DRAW_BALANCE:
            draw_balance(frame, com, cop, w, h)

        # TRAJECTORY
        if DRAW_TRAJECTORY:
            for i in range(1,len(com_track)):
                cv2.line(frame,
                         tuple(com_track[i-1].astype(int)),
                         tuple(com_track[i].astype(int)),
                         (0,0,255),1)

            for i in range(1,len(cop_track)):
                cv2.line(frame,
                         tuple(cop_track[i-1].astype(int)),
                         tuple(cop_track[i].astype(int)),
                         (255,255,0),1)

        # STORE
        data["COM"] = com.tolist()
        data["COP"] = cop.tolist()
        data["velocity"] = vel.tolist()
        data["acceleration"] = acc.tolist()
        data["stability"] = float(stab)
        data["fall_risk"] = bool(falling)
        data["force"] = {"L":lf,"R":rf}

    timeline.append(data)
    out.write(frame)
    fid+=1

# ---------------------------
# SAVE
# ---------------------------
cap.release()
out.release()
pose.close()

with open(OUTPUT_JSON,"w") as f:
    json.dump(timeline,f,indent=2)

print("🔥 DONE — FULL NEXT LEVEL BIOMECHANICS ACTIVE")
