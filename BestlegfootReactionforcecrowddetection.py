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
OUTPUT_JSON = "timeline.json"

# ---------------------------
# DRAW TOGGLES
# ---------------------------
DRAW_KEYPOINTS = True
DRAW_GRF = True
DRAW_ANGLES = True
DRAW_TEXT = True
DRAW_ARM_VECTORS = True
SHOW_VIDEO = True

# ---------------------------
# LANDMARK INDEX
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

def draw_vec(img, start, v, color):
    end = (start + v * 120).astype(int)
    cv2.arrowedLine(img, tuple(start.astype(int)), tuple(end), color, 3, tipLength=0.3)

# ---------------------------
# BIOMECH
# ---------------------------
def joint_angle(a, b, c):
    ab = normalize(a - b)
    cb = normalize(c - b)
    return np.degrees(np.arccos(np.clip(np.dot(ab, cb), -1.0, 1.0)))

def draw_angle_arc(img, a, b, c, color=(0,255,255)):
    ba = normalize(a - b)
    bc = normalize(c - b)

    center = tuple(b.astype(int))
    r = 40

    ang1 = np.degrees(np.arctan2(ba[1], ba[0]))
    ang2 = np.degrees(np.arctan2(bc[1], bc[0]))

    cv2.ellipse(img, center, (r, r), 0, ang1, ang2, color, 2)

# ---------------------------
# GRF
# ---------------------------
def ground_reaction_force(heel, toe, mag):
    foot_vec = toe - heel
    foot_dir = normalize(foot_vec)

    normal = np.array([-foot_dir[1], foot_dir[0]])

    if normal[1] > 0:
        normal = -normal

    return normal * mag

# ---------------------------
# FORCE + STATE
# ---------------------------
def foot_force(heel, toe):
    return min(np.linalg.norm(toe - heel) / 120, 2.5)

def is_airborne(lf, rf):
    return (lf + rf) < 0.2

# ---------------------------
# COM
# ---------------------------
def compute_com(pts):
    torso = (pts[LS] + pts[RS] + pts[LH] + pts[RH]) / 4
    return (
        torso * 0.5 +
        (pts[LH] + pts[RH]) * 0.2 +
        (pts[LK] + pts[RK]) * 0.1 +
        (pts[LA] + pts[RA]) * 0.05 +
        (pts[LS] + pts[RS]) * 0.15
    )

# ---------------------------
# MODEL
# ---------------------------
options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO
)

pose = vision.PoseLandmarker.create_from_options(options)

# ---------------------------
# VIDEO IO
# ---------------------------
cap = cv2.VideoCapture(VIDEO_PATH)

w, h = int(cap.get(3)), int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS) or 30

out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# ---------------------------
# TIMELINE STORAGE
# ---------------------------
timeline = []

prev_com = None
prev_vel = None
DT = 1.0 / fps
frame_id = 0

# ---------------------------
# LOOP
# ---------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    res = pose.detect_for_video(mp_img, int(cap.get(cv2.CAP_PROP_POS_MSEC)))

    frame_data = {
        "frame": frame_id,
        "time_sec": round(frame_id / fps, 4),
        "state": "",
        "impact": False,
        "center_of_mass": None,
        "keypoints": [],
        "grf": {
            "left": {"vector": [0,0], "magnitude": 0},
            "right": {"vector": [0,0], "magnitude": 0}
        },
        "angles": {
            "left_knee": None,
            "left_elbow": None,
            "right_elbow": None
        },
        "arm_vectors": {
            "left": [0,0],
            "right": [0,0]
        }
    }

    if res.pose_landmarks:
        lm = res.pose_landmarks[0]
        pts = [pt(lm, i, w, h) for i in range(len(lm))]

        # KEYPOINTS
        if DRAW_KEYPOINTS:
            for p in pts:
                cv2.circle(frame, tuple(p.astype(int)), 4, (0,255,0), -1)

        frame_data["keypoints"] = [p.tolist() for p in pts]

        lf = foot_force(pts[LHEEL], pts[LFOOT])
        rf = foot_force(pts[RHEEL], pts[RFOOT])

        airborne = is_airborne(lf, rf)
        com = compute_com(pts)
        frame_data["center_of_mass"] = com.tolist()

        # ACCELERATION
        if prev_com is not None:
            vel = (com - prev_com) / (DT + 1e-6)
            acc = (vel - prev_vel) / (DT + 1e-6) if prev_vel is not None else np.zeros(2)
        else:
            vel = np.zeros(2)
            acc = np.zeros(2)

        prev_com = com
        prev_vel = vel

        # GRF
        if not airborne and DRAW_GRF:
            lf_vec = ground_reaction_force(pts[LHEEL], pts[LFOOT], lf)
            rf_vec = ground_reaction_force(pts[RHEEL], pts[RFOOT], rf)

            draw_vec(frame, pts[LFOOT], lf_vec, (0,255,255))
            draw_vec(frame, pts[RFOOT], rf_vec, (0,255,255))

            frame_data["grf"]["left"]["vector"] = lf_vec.tolist()
            frame_data["grf"]["left"]["magnitude"] = float(lf)

            frame_data["grf"]["right"]["vector"] = rf_vec.tolist()
            frame_data["grf"]["right"]["magnitude"] = float(rf)

        # ARM VECTORS
        l_arm_vec = normalize(pts[LW] - pts[LE])
        r_arm_vec = normalize(pts[RW] - pts[RE])

        frame_data["arm_vectors"]["left"] = l_arm_vec.tolist()
        frame_data["arm_vectors"]["right"] = r_arm_vec.tolist()

        if DRAW_ARM_VECTORS:
            draw_vec(frame, pts[LE], l_arm_vec, (255,0,0))
            draw_vec(frame, pts[RE], r_arm_vec, (255,0,0))

        # ANGLES
        lk = joint_angle(pts[LH], pts[LK], pts[LA])
        le_ang = joint_angle(pts[LS], pts[LE], pts[LW])
        re_ang = joint_angle(pts[RS], pts[RE], pts[RW])

        frame_data["angles"]["left_knee"] = float(lk)
        frame_data["angles"]["left_elbow"] = float(le_ang)
        frame_data["angles"]["right_elbow"] = float(re_ang)

        if DRAW_ANGLES:
            draw_angle_arc(frame, pts[LH], pts[LK], pts[LA])
            draw_angle_arc(frame, pts[LS], pts[LE], pts[LW])
            draw_angle_arc(frame, pts[RS], pts[RE], pts[RW])

        # IMPACT
        if np.linalg.norm(acc) > 50:
            frame_data["impact"] = True
            if DRAW_TEXT:
                cv2.putText(frame, "IMPACT!", (50,100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

        # STATE
        state = "AIR" if airborne else "GROUND"
        frame_data["state"] = state

        if DRAW_TEXT:
            cv2.putText(frame, state, (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255,255,0) if airborne else (0,255,0), 2)

    timeline.append(frame_data)
    out.write(frame)

    if SHOW_VIDEO:
        cv2.imshow("Biomechanics", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    frame_id += 1

# SAVE JSON
with open(OUTPUT_JSON, "w") as f:
    json.dump(timeline, f, indent=2)

cap.release()
out.release()
pose.close()
cv2.destroyAllWindows()

print("🔥 DONE: Full Body Biomechanics + Timeline JSON saved")
