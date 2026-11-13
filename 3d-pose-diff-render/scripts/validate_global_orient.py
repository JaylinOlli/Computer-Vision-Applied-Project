"""
Real-image validation of the MediaPipe -> SMPL global_orient pipeline.

    photo -> MediaPipe PoseLandmarker (Tasks API) -> world landmarks (camera-aligned)
          -> torso frame R_mp -> global_orient_from_torso(R_smpl_rest, R_mp)
          -> SMPL forward (body_pose = betas = transl = 0) -> posed SMPL torso frame

Outputs (in --out-dir, which should be OUTSIDE the repository):
    metrics.json          all measurements below
    landmarks_2d.png      photo with the 33 landmarks and labelled torso points
    orientation_3d.html   interactive Plotly view (open in a browser)
With --mirror, the same files with a "_mirrored" suffix for a horizontally
flipped copy of the photo (a robustness check only).

Measurements:
    A  orientation error: posed SMPL torso frame vs MediaPipe torso frame
       (near zero by construction; confirms the pipeline on real data)
    B  pixel/world correlations (diagnostic, not pass/fail)
    C  left/right label order in the image vs. estimated facing direction
    D  anatomical forward checks: nose and toes should lie on the +forward side
    E  global_orient and torso facing angles (descriptive)
    F  per-bone residuals: shoulder line, hip line, spine (informational)

Display only: each skeleton is shifted so its own hip midpoint is at the
origin. This is a translation for drawing; no camera translation is estimated,
and no extra rotation or scaling is applied.

Usage (from 3d-pose-diff-render/):
    .venv/bin/python scripts/validate_global_orient.py --image PHOTO.jpg \
        --smpl-model /path/SMPL_NEUTRAL.pkl --out-dir ~/cv_outputs/run1 [--mirror]
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.smpl_utils import (  # noqa: E402
    MEDIAPIPE_LANDMARK_NAMES,
    MEDIAPIPE_TORSO_LANDMARKS,
    SMPL_JOINT_NAMES,
    SMPL_TORSO_JOINTS,
    global_orient_from_torso,
    load_smpl,
    mediapipe_to_arrays,
    mediapipe_torso_orientation,
    mediapipe_world_to_camera_aligned,
    rotation_angle_deg,
    smpl_forward,
    smpl_rest_torso_orientation,
    smpl_torso_orientation_from_joints,
)

DEFAULT_POSE_MODEL = os.path.expanduser("~/models/mediapipe/pose_landmarker_heavy.task")
MIN_TORSO_VISIBILITY = 0.5
BONE_FLAG_DEG = 20.0

_MP = {n: i for i, n in enumerate(MEDIAPIPE_LANDMARK_NAMES)}
MP_CONNECTIONS = [(_MP[a], _MP[b]) for a, b in [
    ("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("left_ankle", "left_heel"), ("left_heel", "left_foot_index"), ("left_ankle", "left_foot_index"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("right_ankle", "right_heel"), ("right_heel", "right_foot_index"), ("right_ankle", "right_foot_index"),
    ("nose", "left_eye"), ("nose", "right_eye"), ("left_eye", "left_ear"), ("right_eye", "right_ear"),
    ("mouth_left", "mouth_right"),
]]


# --------------------------------------------------------------------------- #
# MediaPipe detection
# --------------------------------------------------------------------------- #
def read_rgb(path):
    import cv2

    bgr = cv2.imread(path, cv2.IMREAD_COLOR)  # applies EXIF orientation
    if bgr is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def detect_pose(rgb, pose_model_path):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    if not os.path.isfile(pose_model_path):
        raise FileNotFoundError(f"MediaPipe pose model not found: {pose_model_path}")
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=pose_model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        output_segmentation_masks=False,
    )
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        return landmarker.detect(image)


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def _unit(v):
    return v / np.linalg.norm(v)


def _angle_deg(u, v):
    u, v = _unit(np.asarray(u, float)), _unit(np.asarray(v, float))
    return float(np.degrees(np.arctan2(np.linalg.norm(np.cross(u, v)), u @ v)))


def _torso_vectors(ls, rs, lh, rh):
    return {"shoulder_line": ls - rs, "hip_line": lh - rh,
            "spine": 0.5 * (ls + rs) - 0.5 * (lh + rh)}


def analyze(result, image_hw, model, R_rest):
    """Run the pipeline on one MediaPipe result and return (metrics, arrays)."""
    import torch

    world_raw, pixels, vis = mediapipe_to_arrays(result, image_hw)
    world = mediapipe_world_to_camera_aligned(world_raw).astype(np.float64)
    t_mp, t_sm = MEDIAPIPE_TORSO_LANDMARKS, SMPL_TORSO_JOINTS

    torso_vis = {n: float(vis[i]) for n, i in t_mp.items()}
    R_mp = mediapipe_torso_orientation(world)
    r = global_orient_from_torso(R_rest, R_mp)
    with torch.no_grad():
        joints = smpl_forward(model, global_orient=r)["joints"][0].double().cpu().numpy()
    R_smpl = smpl_torso_orientation_from_joints(joints)

    # A. orientation error (by construction ~0)
    err_A = rotation_angle_deg(R_smpl, R_mp)

    # B. pixel/world correlations (diagnostic)
    corr_x = float(np.corrcoef(world[:, 0], pixels[:, 0])[0, 1])
    corr_y = float(np.corrcoef(world[:, 1], pixels[:, 1])[0, 1])

    # C. left/right order in the image vs. facing direction
    forward = R_mp[:, 2]
    facing_camera = bool(forward[2] < 0)  # +Z is away from the camera
    sh_px = float(pixels[t_mp["left_shoulder"], 0] - pixels[t_mp["right_shoulder"], 0])
    hip_px = float(pixels[t_mp["left_hip"], 0] - pixels[t_mp["right_hip"], 0])
    expected_sign = 1.0 if facing_camera else -1.0  # facing camera: person's left on image right
    lr_consistent = bool(np.sign(sh_px) == expected_sign and np.sign(hip_px) == expected_sign)

    # D. anatomical forward checks (independent of where the camera is)
    torso_center = world[[t_mp[k] for k in t_mp]].mean(axis=0)
    nose_fwd = float((world[_MP["nose"]] - torso_center) @ forward)
    toe_fwd = {s: float((world[_MP[f"{s}_foot_index"]] - world[_MP[f"{s}_heel"]]) @ forward)
               for s in ("left", "right")}

    # E. descriptive orientation
    theta = float(np.degrees(np.linalg.norm(r)))
    yaw = float(np.degrees(np.arctan2(forward[0], -forward[2])))  # 0 = facing camera, + = turned to image right
    spine_tilt = _angle_deg(R_mp[:, 1], [0.0, -1.0, 0.0])          # 0 = upright in the camera frame

    # F. per-bone residuals: posed SMPL vs MediaPipe torso vectors
    mp_vec = _torso_vectors(*(world[t_mp[k]] for k in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")))
    sm_vec = _torso_vectors(*(joints[t_sm[k]] for k in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")))
    residuals = {k: _angle_deg(sm_vec[k], mp_vec[k]) for k in mp_vec}

    metrics = {
        "A_orientation_error_deg": err_A,
        "B_correlation": {"world_x_vs_pixel_x": corr_x, "world_y_vs_pixel_y": corr_y,
                          "note": "diagnostic only, all 33 landmarks"},
        "C_left_right": {"left_minus_right_shoulder_px": sh_px, "left_minus_right_hip_px": hip_px,
                         "facing_camera_from_world": facing_camera,
                         "consistent_with_facing": lr_consistent},
        "D_anatomical_forward": {"nose_ahead_m": nose_fwd, "left_toe_ahead_m": toe_fwd["left"],
                                 "right_toe_ahead_m": toe_fwd["right"],
                                 "all_positive": bool(nose_fwd > 0 and min(toe_fwd.values()) > 0)},
        "E_orientation": {"global_orient_axis_angle": r.tolist(), "global_orient_deg": theta,
                          "global_orient_axis": (r / np.linalg.norm(r)).tolist() if theta > 0 else None,
                          "torso_yaw_deg": yaw, "spine_tilt_from_image_up_deg": spine_tilt,
                          "R_mediapipe": R_mp.tolist(), "R_smpl_rest": R_rest.tolist()},
        "F_bone_residuals_deg": {**residuals, "flag_threshold_deg": BONE_FLAG_DEG,
                                 "flagged": [k for k, v in residuals.items() if v > BONE_FLAG_DEG],
                                 "note": "spine residual is ~0 by construction (spine is the primary axis)"},
        "torso_visibility": torso_vis,
        "low_visibility_torso_landmarks": [k for k, v in torso_vis.items() if v < MIN_TORSO_VISIBILITY],
        "image_hw": list(image_hw),
    }
    arrays = {"world": world, "pixels": pixels, "vis": vis, "joints": joints,
              "R_mp": R_mp, "R_smpl": R_smpl, "parents": model.parents.cpu().numpy()}
    return metrics, arrays


# --------------------------------------------------------------------------- #
# Visualisation
# --------------------------------------------------------------------------- #
def save_overlay(rgb, arrays, path):
    import cv2

    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    px, vis = arrays["pixels"], arrays["vis"]
    s = max(1, round(max(img.shape[:2]) / 600))
    for a, b in MP_CONNECTIONS:
        cv2.line(img, tuple(int(v) for v in px[a]), tuple(int(v) for v in px[b]), (200, 200, 200), s, cv2.LINE_AA)
    for i, (x, y) in enumerate(px):
        color = (0, 200, 0) if vis[i] >= MIN_TORSO_VISIBILITY else (0, 0, 255)
        cv2.circle(img, (int(x), int(y)), 3 * s, color, -1, cv2.LINE_AA)
    for name, i in MEDIAPIPE_TORSO_LANDMARKS.items():
        label = name.replace("left_", "L ").replace("right_", "R ")
        org = (int(px[i, 0]) + 5 * s, int(px[i, 1]) - 5 * s)
        cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, (0, 0, 0), 3 * s, cv2.LINE_AA)
        cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, (255, 255, 0), s, cv2.LINE_AA)
    cv2.imwrite(path, img)


def save_3d(arrays, metrics, path, title):
    import plotly.graph_objects as go

    world, joints = arrays["world"], arrays["joints"]
    t_mp, t_sm = MEDIAPIPE_TORSO_LANDMARKS, SMPL_TORSO_JOINTS
    # Display-only translation: each skeleton centred on its own hip midpoint.
    mp_pts = world - 0.5 * (world[t_mp["left_hip"]] + world[t_mp["right_hip"]])
    sm_pts = joints - 0.5 * (joints[t_sm["left_hip"]] + joints[t_sm["right_hip"]])

    def lines(pts, pairs, name, color, width):
        xs, ys, zs = [], [], []
        for a, b in pairs:
            xs += [pts[a, 0], pts[b, 0], None]
            ys += [pts[a, 1], pts[b, 1], None]
            zs += [pts[a, 2], pts[b, 2], None]
        return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name=name,
                            line=dict(color=color, width=width), hoverinfo="skip")

    def points(pts, idx, names, name, color, size, text=False):
        return go.Scatter3d(x=pts[idx, 0], y=pts[idx, 1], z=pts[idx, 2], name=name,
                            mode="markers+text" if text else "markers",
                            marker=dict(color=color, size=size),
                            text=[names[i] for i in idx] if text else None,
                            hovertext=[f"{names[i]}<br>({pts[i, 0]:.3f}, {pts[i, 1]:.3f}, {pts[i, 2]:.3f}) m"
                                       for i in idx], hoverinfo="text", textposition="top center")

    def triad(R, name, dash, length=0.3):
        out = []
        for k, (axis, color) in enumerate(zip(("left", "up", "forward"), ("#d62728", "#2ca02c", "#1f77b4"))):
            e = R[:, k] * length
            out.append(go.Scatter3d(x=[0, e[0]], y=[0, e[1]], z=[0, e[2]], mode="lines+text",
                                    name=f"{name} {axis}", text=["", f"{name}: {axis}"],
                                    line=dict(color=color, width=7, dash=dash), hoverinfo="name"))
        return out

    parents = arrays["parents"]
    sm_pairs = [(int(parents[j]), j) for j in range(1, len(parents))]
    torso_mp_idx = list(t_mp.values())
    torso_sm_idx = list(t_sm.values())
    traces = [
        lines(mp_pts, MP_CONNECTIONS, "MediaPipe skeleton", "#7f7f7f", 4),
        points(mp_pts, list(range(33)), MEDIAPIPE_LANDMARK_NAMES, "MediaPipe landmarks", "#7f7f7f", 3),
        points(mp_pts, torso_mp_idx, MEDIAPIPE_LANDMARK_NAMES, "MediaPipe torso", "#ff7f0e", 7, text=True),
        lines(sm_pts, sm_pairs, "SMPL skeleton (body_pose = 0)", "#9467bd", 6),
        points(sm_pts, list(range(24)), SMPL_JOINT_NAMES, "SMPL joints", "#9467bd", 3),
        points(sm_pts, torso_sm_idx, SMPL_JOINT_NAMES, "SMPL torso", "#9467bd", 7, text=True),
        *triad(arrays["R_mp"], "MP", "solid"),
        *triad(arrays["R_smpl"], "SMPL", "dash"),
    ]
    A = metrics["A_orientation_error_deg"]
    E = metrics["E_orientation"]
    subtitle = (f"torso frame error {A:.2e} deg | global_orient {E['global_orient_deg']:.1f} deg | "
                f"yaw {E['torso_yaw_deg']:.1f} deg. Display only: both skeletons centred on their own "
                "hip midpoint (no camera translation estimated). SMPL arms in T-pose (body_pose = 0).")
    fig = go.Figure(traces)
    fig.update_layout(
        title=dict(text=f"{title}<br><sup>{subtitle}</sup>"),
        scene=dict(xaxis_title="X (right)", yaxis_title="Y (down)", zaxis_title="Z (away from camera)",
                   aspectmode="data",
                   camera=dict(up=dict(x=0, y=-1, z=0), eye=dict(x=0.0, y=-0.3, z=-2.2))),
        legend=dict(itemsizing="constant"), margin=dict(l=0, r=0, t=80, b=0),
    )
    fig.write_html(path, include_plotlyjs=True)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run_one(rgb, label, args, model, R_rest):
    result = detect_pose(rgb, args.pose_model)
    metrics, arrays = analyze(result, rgb.shape[:2], model, R_rest)
    suffix = "" if label == "original" else f"_{label}"
    paths = {k: os.path.join(args.out_dir, f"{k}{suffix}.{ext}")
             for k, ext in (("metrics", "json"), ("landmarks_2d", "png"), ("orientation_3d", "html"))}
    metrics["input"] = {"image": os.path.abspath(args.image), "variant": label}
    with open(paths["metrics"], "w") as fh:
        json.dump(metrics, fh, indent=2)
    save_overlay(rgb, arrays, paths["landmarks_2d"])
    save_3d(arrays, metrics, paths["orientation_3d"], f"global_orient validation ({label})")
    return metrics, paths


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--image", required=True)
    p.add_argument("--smpl-model", default=os.environ.get("SMPL_MODEL_PATH"))
    p.add_argument("--pose-model", default=DEFAULT_POSE_MODEL)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--mirror", action="store_true", help="also run on a horizontally flipped copy")
    args = p.parse_args(argv)
    if not args.smpl_model:
        p.error("--smpl-model is required (or set SMPL_MODEL_PATH)")
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    args.out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
    if os.path.commonpath([args.out_dir, repo]) == repo:
        p.error(f"--out-dir must be outside the repository ({repo})")
    os.makedirs(args.out_dir, exist_ok=True)

    model = load_smpl(args.smpl_model)
    R_rest = smpl_rest_torso_orientation(model)
    rgb = read_rgb(args.image)
    variants = [("original", rgb)]
    if args.mirror:
        variants.append(("mirrored", np.ascontiguousarray(rgb[:, ::-1])))

    for label, img in variants:
        metrics, paths = run_one(img, label, args, model, R_rest)
        print(f"\n=== {label} ===")
        print(json.dumps({k: v for k, v in metrics.items() if k != "E_orientation"}, indent=2))
        E = metrics["E_orientation"]
        print(f"global_orient = {np.round(E['global_orient_axis_angle'], 4).tolist()} "
              f"({E['global_orient_deg']:.2f} deg), torso yaw {E['torso_yaw_deg']:.1f} deg, "
              f"spine tilt {E['spine_tilt_from_image_up_deg']:.1f} deg")
        for k, v in paths.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
