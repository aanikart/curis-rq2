# src/gait.py — RGB-vs-depth knee-angle pipeline
import os, glob
import numpy as np
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
HIP, KNEE, ANKLE = 23, 25, 27

# --- constants ---
C = 7000
fx = fy = 580.0
cx, cy = 320.0, 240.0
BASE = "https://fenix.ur.edu.pl/mkepski/ds/data"

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

def joint_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    u, v = a - b, c - b
    cos = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))

def depth_at_gated(depth_map, u, v, k=3, min_valid_frac=0.5):
    patch = depth_map[max(0,v-k):v+k+1, max(0,u-k):u+k+1]
    valid = patch[patch > 0]
    if valid.size < min_valid_frac * patch.size:
        return np.nan
    return np.median(valid)

def backproject(u, v, d):
    return np.array([(u - cx) * d / fx, (v - cy) * d / fy, d])

def download_sequence(seq):
    """Download RGB + depth for a sequence if not already present."""
    for kind in ["rgb", "d"]:
        out = os.path.join(DATA_DIR, f"{seq}_{'rgb' if kind=='rgb' else 'depth'}")
        if not glob.glob(f"{out}/**/*.png", recursive=True):
            tmp = os.path.join(DATA_DIR, "_tmp.zip")
            os.system(f"curl -sL {BASE}/{seq}-cam0-{kind}.zip -o {tmp}")
            os.system(f"unzip -q -o {tmp} -d {out}")
    rgb   = sorted(glob.glob(os.path.join(DATA_DIR, f"{seq}_rgb",   "**", "*.png"), recursive=True))
    depth = sorted(glob.glob(os.path.join(DATA_DIR, f"{seq}_depth", "**", "*.png"), recursive=True))
    return rgb, depth

def process_sequence(seq):
    """Run the full pipeline. Returns dict of 2D and 3D knee angles per frame."""
    rgb_paths, depth_paths = download_sequence(seq)
    n = min(len(rgb_paths), len(depth_paths))
    a2d = np.full(n, np.nan)
    a3d = np.full(n, np.nan)

    pose = mp_pose.Pose(static_image_mode=False, model_complexity=2)
    for i in range(n):
        rgb = cv2.cvtColor(cv2.imread(rgb_paths[i]), cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if not res.pose_landmarks:
            continue
        lm = res.pose_landmarks.landmark
        H, W = 480, 640

        # 2D (pixels)
        px = lambda idx: np.array([lm[idx].x * W, lm[idx].y * H])
        a2d[i] = joint_angle(px(HIP), px(KNEE), px(ANKLE))

        # 3D (depth, gated)
        dmm = C * cv2.imread(depth_paths[i], cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0
        pts, ok = {}, True
        for name, idx in [("hip", HIP), ("knee", KNEE), ("ankle", ANKLE)]:
            u, v = int(round(lm[idx].x * W)), int(round(lm[idx].y * H))
            d = depth_at_gated(dmm, u, v)
            if np.isnan(d): ok = False; break
            pts[name] = backproject(u, v, d)
        if ok:
            a3d[i] = joint_angle(pts["hip"], pts["knee"], pts["ankle"])
    pose.close()
    return {"seq": seq, "a2d": a2d, "a3d": a3d, "n": n}
