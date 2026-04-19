# ---------------------------
# INSTALL (Colab)
# ---------------------------
# !pip install mediapipe opencv-python numpy

import os, cv2, json, math, urllib.request
import numpy as np
import mediapipe as mp
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
OUTPUT_JSON = "biomech.json"

# ---------------------------
# TOGGLES
# ---------------------------
DRAW_POINTS = True
DRAW_SKELETON = True
DRAW_ARCS = True
DRAW_ANGLES = True
DRAW_FOOT_STRETCH = True
DRAW_GRF = True
DRAW_HAND_BASIS = True
DRAW_KINETIC_CHAIN = True

MAX_VECTOR_UNITS = 2.0

# ---------------------------
# LANDMARKS
# ---------------------------
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_HEEL, RIGHT_HEEL = 29, 30
LEFT_FOOT, RIGHT_FOOT = 31, 32

POSE_EDGES = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(27,29),(29,31),
    (24,26),(26,28),(28,30),(30,32)
]

# ---------------------------
# HELPERS
# ---------------------------
def pt(lm, i, w, h):
    return np.array([lm[i].x*w, lm[i].y*h], dtype=np.float32)

def safe_norm(v):
    n = np.linalg.norm(v)
    return v/n if n>1e-6 else np.zeros_like(v)

def normalize(v):
    mag = np.linalg.norm(v)
    if mag < 1e-6:
        return v, 0.0
    scale = min(MAX_VECTOR_UNITS/mag,1.0)
    return v*scale, mag

def angle(a,b,c):
    ba, bc = a-b, c-b
    cosv = np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6)
    return np.degrees(np.arccos(np.clip(cosv,-1,1)))

# ---------------------------
# DRAW
# ---------------------------
def draw_arc(img,a,b,c):
    ang1 = np.degrees(np.arctan2(a[1]-b[1],a[0]-b[0]))
    ang2 = np.degrees(np.arctan2(c[1]-b[1],c[0]-b[0]))
    cv2.ellipse(img,tuple(b.astype(int)),(30,30),0,ang1,ang2,(0,255,255),2)

def draw_vec(img,start,v,color=(0,0,255)):
    end = (start+v*60).astype(int)
    cv2.arrowedLine(img,tuple(start.astype(int)),tuple(end),color,2,tipLength=0.2)

# ---------------------------
# FOOT + GRF
# ---------------------------
def foot_grf(img,heel,toe,ankle):
    stretch = toe-heel
    stretch_len = np.linalg.norm(stretch)

    norm = min(stretch_len/100,2.0)

    if DRAW_FOOT_STRETCH:
        cv2.line(img,tuple(heel.astype(int)),tuple(toe.astype(int)),(255,255,0),2)

    if DRAW_GRF:
        grf = np.array([0,-1])*norm
        draw_vec(img,ankle,grf,(0,255,255))

    return norm

# ---------------------------
# HAND BASIS
# ---------------------------
def hand_basis(img,elbow,wrist):
    axis_y = safe_norm(wrist-elbow)
    axis_x = np.array([-axis_y[1],axis_y[0]])

    axis_y,_ = normalize(axis_y)
    axis_x,_ = normalize(axis_x)

    if DRAW_HAND_BASIS:
        draw_vec(img,wrist,axis_y,(0,255,0))
        draw_vec(img,wrist,axis_x,(255,0,255))

    return axis_x,axis_y

# ---------------------------
# KINETIC CHAIN
# ---------------------------
def kinetic_chain(img,chain,force):
    for i in range(len(chain)-1):
        start = chain[i]
        draw_vec(img,start,force*(1-0.15*i),(255,128,0))

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
w,h = int(cap.get(3)),int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS) or 30

out = cv2.VideoWriter(OUTPUT_VIDEO,cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h))

timeline=[]
fid=0

# ---------------------------
# LOOP
# ---------------------------
while cap.isOpened():
    ret,frame = cap.read()
    if not ret: break

    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)

    res = pose.detect_for_video(mp_img,int(fid*1000/fps))

    data={"frame":fid,"angles":{},"forces":{}}

    if res.pose_landmarks:
        lm = res.pose_landmarks[0]
        pts=[pt(lm,i,w,h) for i in range(len(lm))]

        # ---------------------------
        # DRAW SKELETON
        # ---------------------------
        if DRAW_POINTS:
            for p in pts:
                cv2.circle(frame,tuple(p.astype(int)),3,(0,255,0),-1)

        if DRAW_SKELETON:
            for a,b in POSE_EDGES:
                cv2.line(frame,tuple(pts[a].astype(int)),tuple(pts[b].astype(int)),(255,0,0),2)

        # ---------------------------
        # ANGLES
        # ---------------------------
        joints=[
            (LEFT_HIP,LEFT_KNEE,LEFT_ANKLE,"L_knee"),
            (RIGHT_HIP,RIGHT_KNEE,RIGHT_ANKLE,"R_knee"),
            (LEFT_SHOULDER,LEFT_ELBOW,LEFT_WRIST,"L_elbow"),
            (RIGHT_SHOULDER,RIGHT_ELBOW,RIGHT_WRIST,"R_elbow")
        ]

        for a,b,c,name in joints:
            ang=angle(pts[a],pts[b],pts[c])
            data["angles"][name]=float(ang)

            if DRAW_ARCS:
                draw_arc(frame,pts[a],pts[b],pts[c])

        # ---------------------------
        # FOOT + GRF
        # ---------------------------
        lf = foot_grf(frame,pts[LEFT_HEEL],pts[LEFT_FOOT],pts[LEFT_ANKLE])
        rf = foot_grf(frame,pts[RIGHT_HEEL],pts[RIGHT_FOOT],pts[RIGHT_ANKLE])

        # ---------------------------
        # HAND BASIS
        # ---------------------------
        hand_basis(frame,pts[LEFT_ELBOW],pts[LEFT_WRIST])
        hand_basis(frame,pts[RIGHT_ELBOW],pts[RIGHT_WRIST])

        # ---------------------------
        # KINETIC CHAIN
        # ---------------------------
        if DRAW_KINETIC_CHAIN:
            chainL=[pts[LEFT_ANKLE],pts[LEFT_KNEE],pts[LEFT_HIP],pts[LEFT_SHOULDER]]
            chainR=[pts[RIGHT_ANKLE],pts[RIGHT_KNEE],pts[RIGHT_HIP],pts[RIGHT_SHOULDER]]

            kinetic_chain(frame,chainL,np.array([0,-lf]))
            kinetic_chain(frame,chainR,np.array([0,-rf]))

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

print("✅ DONE — Full Biomechanics System Active")
