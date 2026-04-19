# ---------------------------
# INSTALL (Colab)
# ---------------------------
# !pip install ultralytics opencv-python numpy -q

import cv2
import json
import numpy as np
from ultralytics import YOLO
from collections import deque

class BiomechYOLOv11:

    def __init__(self):
        # ---------------------------
        # CONFIG
        # ---------------------------
        self.INPUT_VIDEO = "/content/drive/MyDrive/vis.mp4"
        self.OUTPUT_VIDEO = "/content/drive/MyDrive/output.mp4"
        self.OUTPUT_JSON  = "/content/drive/MyDrive/timeline.json"

        self.MODEL_PATH = "yolo11n-pose.pt"
        self.CONF = 0.5

        self.MASS = 70.0
        self.DT = 1 / 30.0

        self.FORCE_SCALE = 0.04
        self.ARC_RADIUS = 28

        self.DRAW = {
            "beam": True,
            "foot_axis": True,
            "normal": True,
            "reaction": True,
            "com": True,
            "cop": True,
            "angles": True,
            "vectors": True,
            "forces": True,
            "arcs": True,
            "labels": True,
        }

        # ---------------------------
        # STATE
        # ---------------------------
        self.com_hist = deque(maxlen=5)

        # ---------------------------
        # MODEL
        # ---------------------------
        self.model = YOLO(self.MODEL_PATH)

    # ---------------------------
    # UTILS
    # ---------------------------
    def vec(self, a, b): return b - a
    def norm(self, v): return float(np.linalg.norm(v))
    def unit(self, v): return v / (self.norm(v) + 1e-6)
    def perp(self, v): return np.array([-v[1], v[0]], dtype=np.float32)
    def clamp(self, x, a, b): return max(a, min(b, x))
    def orient_up(self, v): return -v if v[1] > 0 else v

    def angle(self, a, b, c):
        v1 = a - b
        v2 = c - b
        cosv = np.dot(v1, v2) / (self.norm(v1)*self.norm(v2) + 1e-6)
        return float(np.degrees(np.arccos(np.clip(cosv, -1, 1))))

    def draw_arrow(self, img, p, v, color, t=2):
        p1 = tuple(np.round(p).astype(int))
        p2 = tuple(np.round(p + v).astype(int))
        cv2.arrowedLine(img, p1, p2, color, t, tipLength=0.2)

    def draw_text(self, img, txt, pos, c=(255,255,255)):
        cv2.putText(img, txt, (int(pos[0]), int(pos[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)

    def draw_arc(self, img, center, a, b, r=30, col=(0,255,255)):
        ang1 = np.degrees(np.arctan2(a[1], a[0]))
        ang2 = np.degrees(np.arctan2(b[1], b[0]))

        if ang2 - ang1 > 180: ang2 -= 360
        elif ang2 - ang1 < -180: ang2 += 360

        pts = []
        for t in np.linspace(0, 1, 20):
            ang = np.radians(ang1 + t*(ang2-ang1))
            p = center + np.array([np.cos(ang), np.sin(ang)]) * r
            pts.append(np.round(p).astype(int))

        cv2.polylines(img, [np.array(pts)], False, col, 2)

    # ---------------------------
    # PROCESS FRAME
    # ---------------------------
    def process_frame(self, frame, frame_idx):

        results = self.model(frame, conf=self.CONF, verbose=False)
        data = {"frame": frame_idx}

        if results and results[0].keypoints is not None:

            pts = results[0].keypoints.xy[0].cpu().numpy()

            l_an, r_an = pts[15], pts[16]
            l_hp, r_hp = pts[11], pts[12]
            l_sh, r_sh = pts[5], pts[6]

            # COM
            com = np.mean([l_sh, r_sh, l_hp, r_hp], axis=0)
            self.com_hist.append(com)

            vel = np.zeros(2)
            acc = np.zeros(2)

            if len(self.com_hist) >= 2:
                vel = (self.com_hist[-1] - self.com_hist[-2]) / self.DT

            if len(self.com_hist) >= 3:
                acc = (self.com_hist[-1] - 2*self.com_hist[-2] + self.com_hist[-3]) / (self.DT**2)

            # SUPPORT
            support_vec = self.vec(l_an, r_an)
            tan = self.unit(support_vec)
            norm_v = self.unit(self.orient_up(self.perp(tan)))

            foot_center = (l_an + r_an) / 2.0

            # FORCES
            gravity = np.array([0, self.MASS * 9.81])
            inertial = -self.MASS * acc
            reaction = gravity + inertial

            # DRAW
            if self.DRAW["foot_axis"]:
                self.draw_arrow(frame, foot_center, tan*80, (0,255,255))

            if self.DRAW["normal"]:
                self.draw_arrow(frame, foot_center, norm_v*80, (255,0,0))

            if self.DRAW["com"]:
                cv2.circle(frame, tuple(com.astype(int)), 6, (255,0,255), -1)

            if self.DRAW["vectors"]:
                self.draw_arrow(frame, com, vel*0.1, (0,255,0))

            if self.DRAW["forces"]:
                self.draw_arrow(frame, com, reaction*self.FORCE_SCALE, (0,0,255))

            if self.DRAW["reaction"]:
                self.draw_arrow(frame, foot_center, reaction*self.FORCE_SCALE, (255,255,0))

            if self.DRAW["angles"]:
                ang = self.angle(l_an, foot_center, com)
                self.draw_text(frame, f"{ang:.1f}", foot_center + np.array([10,-10]))

            if self.DRAW["arcs"]:
                self.draw_arc(frame, foot_center, tan*40, norm_v*40, self.ARC_RADIUS)

            if self.DRAW["labels"]:
                self.draw_text(frame, "COM", com + np.array([5,-5]))

            data["com"] = com.tolist()
            data["vel"] = vel.tolist()
            data["acc"] = acc.tolist()

        return frame, data

    # ---------------------------
    # RUN
    # ---------------------------
    def run(self):

        cap = cv2.VideoCapture(self.INPUT_VIDEO)
        if not cap.isOpened():
            raise Exception("❌ Cannot open video")

        w = int(cap.get(3))
        h = int(cap.get(4))
        fps = cap.get(5) or 30.0

        out = cv2.VideoWriter(
            self.OUTPUT_VIDEO,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (w, h)
        )

        timeline = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame, data = self.process_frame(frame, frame_idx)

            timeline.append(data)
            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()

        with open(self.OUTPUT_JSON, "w") as f:
            json.dump(timeline, f, indent=2)

        print("✅ DONE — YOLOv11 Biomechanics Engine (OOP)")

# ---------------------------
# RUN
# ---------------------------
engine = BiomechYOLOv11()
engine.run()
