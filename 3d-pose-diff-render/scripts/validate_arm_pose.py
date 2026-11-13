"""
Real-image validation of limb initialisation: swing-only (Increment A) vs
twist-corrected with a pure-hinge elbow / knee (Increment B), for arms, legs or all.

    photo -> MediaPipe -> global_orient (torso)
          -> init_{arm,leg}_pose (A) / refine_{arm,leg}_twist (B) -> SMPL

betas = transl = 0; only the selected limbs' proximal/middle joints of
body_pose are set (all other joints stay at zero). For legs, the foot
(heel -> toe) is NOT used by the solver and serves as an independent check of
the estimated hip twist (ankle rotation stays zero).

Outputs (in --out-dir, outside the repository; prefix arm / leg / limbs):
    <prefix>_metrics_<side>.json   residuals and diagnostics for A and B
    <prefix>_pose_3d_<side>.html   interactive Plotly view: dropdown A / B / both,
                                   toggleable SMPL meshes, hinge-axis arrows

Display only: each skeleton/mesh is shifted so its own hip midpoint is at the
origin; no camera translation, scaling or extra rotation is applied.

Usage (from 3d-pose-diff-render/):
    .venv/bin/python scripts/validate_arm_pose.py --image PHOTO.jpg \
        --smpl-model /path/SMPL_NEUTRAL.pkl --out-dir ~/cv_outputs/arm_pose [--side both] [--limbs all]
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.smpl_utils import (  # noqa: E402
    MEDIAPIPE_LANDMARK_NAMES,
    SMPL_JOINT_NAMES,
    SMPL_PARENTS,
    elbow_hinge_axis,
    global_joint_rotations,
    global_orient_from_torso,
    init_arm_pose,
    init_leg_pose,
    knee_hinge_axis,
    load_smpl,
    mediapipe_to_arrays,
    mediapipe_torso_orientation,
    mediapipe_world_to_camera_aligned,
    refine_arm_twist,
    refine_leg_twist,
    smpl_forward,
    smpl_rest_torso_orientation,
)
from validate_global_orient import DEFAULT_POSE_MODEL, MP_CONNECTIONS, detect_pose, read_rgb  # noqa: E402

MIN_VISIBILITY = 0.5
STRAIGHT_ARM_DEG = 15.0
STRAIGHT_LEG_DEG = 20.0
MAX_HIP_TWIST_DEG = 60.0
INDICES = {
    ("arm", "left"): ((11, 13, 15), (16, 18, 20)), ("arm", "right"): ((12, 14, 16), (17, 19, 21)),
    ("leg", "left"): ((23, 25, 27), (1, 4, 7)), ("leg", "right"): ((24, 26, 28), (2, 5, 8)),
}
FOOT = {"left": ((29, 31), (7, 10)), "right": ((30, 32), (8, 11))}  # MediaPipe heel->toe, SMPL ankle->foot
SOLVERS = {
    "arm": (init_arm_pose, lambda w, J, go, bp, side: refine_arm_twist(
        w, J, go, body_pose=bp, side=side, straight_arm_deg=STRAIGHT_ARM_DEG), elbow_hinge_axis),
    "leg": (init_leg_pose, lambda w, J, go, bp, side: refine_leg_twist(
        w, J, go, body_pose=bp, side=side, straight_leg_deg=STRAIGHT_LEG_DEG,
        max_hip_twist_deg=MAX_HIP_TWIST_DEG), knee_hinge_axis),
}


def _angle(u, v):
    u, v = np.asarray(u, float), np.asarray(v, float)
    return float(np.degrees(np.arctan2(np.linalg.norm(np.cross(u, v)), u @ v)))


def _hip_mid(P, left, right):
    return 0.5 * (P[left] + P[right])


def analyze(world, vis, model, limbs):
    """limbs: list of (limb, side) with limb in {"arm", "leg"}."""
    import torch

    R_rest = smpl_rest_torso_orientation(model)
    R_mp = mediapipe_torso_orientation(world)
    go = global_orient_from_torso(R_rest, R_mp)

    def foot_yaw_pitch(v):
        """Foot direction in the body frame: yaw (+ toward the person's left) and pitch (+ toes up)."""
        v = v / np.linalg.norm(v)
        hz = v - (v @ R_mp[:, 1]) * R_mp[:, 1]
        return (float(np.degrees(np.arctan2(hz @ R_mp[:, 0], hz @ R_mp[:, 2]))),
                float(np.degrees(np.arcsin(np.clip(v @ R_mp[:, 1], -1.0, 1.0)))))
    with torch.no_grad():
        rest = smpl_forward(model)["joints"][0].double().numpy()
    bp = {"A": np.zeros(69), "B": np.zeros(69)}
    diag = {}
    for limb, side in limbs:
        init_fn, refine_fn, _ = SOLVERS[limb]
        bp["A"], dA = init_fn(world, rest, go, side=side, body_pose=bp["A"])
        bp["B"], dB = refine_fn(world, rest, go, bp["B"], side)
        diag[(limb, side)] = (dA, dB)

    J, V = {}, {}
    with torch.no_grad():
        for k in ("A", "B"):
            o = smpl_forward(model, global_orient=go, body_pose=bp[k])
            J[k] = o["joints"][0].double().numpy()
            V[k] = o["vertices"][0].double().numpy()
        faces = o["faces"].numpy()

    metrics = {"global_orient": go.tolist(), "straight_arm_deg": STRAIGHT_ARM_DEG,
               "straight_leg_deg": STRAIGHT_LEG_DEG, "max_hip_twist_deg": MAX_HIP_TWIST_DEG, "limbs": {}}
    hinge_axes = {}
    for limb, side in limbs:
        (m_s, m_e, m_w), (s_s, s_e, s_w) = INDICES[(limb, side)]
        dA, dB = diag[(limb, side)]
        d_up, d_fo = world[m_e] - world[m_s], world[m_w] - world[m_e]
        h = SOLVERS[limb][2](rest, side)
        per = {}
        for k in ("A", "B"):
            Jk = J[k]
            Q = global_joint_rotations(go, bp[k])
            hinge_axes[(limb, side, k)] = Q[s_s] @ h  # hinge axis in camera-aligned coordinates
            per[k] = {
                "direction_residual_deg": {"proximal_bone": _angle(Jk[s_e] - Jk[s_s], d_up),
                                           "distal_bone": _angle(Jk[s_w] - Jk[s_e], d_fo)},
                "bend_smpl_deg": _angle(Jk[s_e] - Jk[s_s], Jk[s_w] - Jk[s_e]),
                "proximal_body_pose": bp[k][3 * (s_s - 1): 3 * s_s].tolist(),
                "middle_body_pose": bp[k][3 * (s_e - 1): 3 * s_e].tolist(),
                "hinge_axis_vs_bend_normal_deg": _angle(hinge_axes[(limb, side, k)], np.cross(d_up, d_fo)),
            }
            if limb == "leg":
                # Independent check (foot is not used by the solver; ankle rotation is zero).
                # Pitch differs by definition (SMPL ankle->ball joint vs MediaPipe heel->toe),
                # so only the yaw difference is informative about hip twist.
                (mh, mt), (sa, sf) = FOOT[side]
                yaw_s, pitch_s = foot_yaw_pitch(Jk[sf] - Jk[sa])
                yaw_m, pitch_m = foot_yaw_pitch(world[mt] - world[mh])
                per[k]["foot"] = {"yaw_smpl_deg": yaw_s, "yaw_mediapipe_deg": yaw_m,
                                  "yaw_diff_deg": abs((yaw_s - yaw_m + 180.0) % 360.0 - 180.0),
                                  "pitch_smpl_deg": pitch_s, "pitch_mediapipe_deg": pitch_m}
        per["A"]["off_hinge_deg"] = dB["off_hinge_swing_only_deg"]
        keys = ["off_hinge_deg", "twist_resolved", "reason", "psi_deg", "phi_deg", "phi_plausible"]
        if limb == "leg":
            keys += ["twist_deg", "bend_direction_deg"]
        per["B"].update({key: dB[key] for key in keys})
        lower = "forearm" if limb == "arm" else "shin"
        metrics["limbs"][f"{limb}_{side}"] = {
            "visibility": {MEDIAPIPE_LANDMARK_NAMES[i]: float(vis[i]) for i in (m_s, m_e, m_w)},
            "low_visibility": [MEDIAPIPE_LANDMARK_NAMES[i] for i in (m_s, m_e, m_w) if vis[i] < MIN_VISIBILITY],
            "bend_mediapipe_deg": dB["bend_deg"],
            "rest": {"phi0_rest_deg": dB["phi0_rest_deg"],
                     "distal_out_of_hinge_plane_deg": dB[f"{lower}_out_of_hinge_plane_deg"]},
            "A_swing_only": per["A"],
            "B_twist_corrected": per["B"],
        }
    arrays = {"world": world, "J": J, "V": V, "faces": faces, "hinge_axes": hinge_axes}
    return metrics, arrays


def build_figure(arrays, metrics, limbs, show_mesh=None):
    """
    Interactive Plotly figure (dropdown: both / swing-only A / twist-corrected B).
    show_mesh: None (meshes hidden, toggle via legend), "A" or "B" (that mesh shown initially).
    """
    import plotly.graph_objects as go

    if show_mesh not in (None, "A", "B"):
        raise ValueError(f"show_mesh must be None, 'A' or 'B', got {show_mesh!r}")

    world, J, V, F = arrays["world"], arrays["J"], arrays["V"], arrays["faces"]
    mp = world - _hip_mid(world, 23, 24)
    off = {k: _hip_mid(J[k], 1, 2) for k in J}
    jc = {k: J[k] - off[k] for k in J}
    vc = {k: V[k] - off[k] for k in V}
    color = {"A": "#bcbd22", "B": "#9467bd"}
    label = {"A": "swing-only (A)", "B": "twist-corrected (B)"}

    def lines(P, pairs, name, col, width, dash=None):
        xs, ys, zs = [], [], []
        for a, b in pairs:
            xs += [P[a, 0], P[b, 0], None]
            ys += [P[a, 1], P[b, 1], None]
            zs += [P[a, 2], P[b, 2], None]
        return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name=name,
                            line=dict(color=col, width=width, dash=dash), hoverinfo="skip")

    def arrow(start, d, length, name, col, width=8, dash=None):
        e = start + np.asarray(d) / np.linalg.norm(d) * length
        return go.Scatter3d(x=[start[0], e[0]], y=[start[1], e[1]], z=[start[2], e[2]], mode="lines",
                            name=name, line=dict(color=col, width=width, dash=dash), hoverinfo="name")

    sm_pairs = [(SMPL_PARENTS[j], j) for j in range(1, 24)]
    traces, groups = [], []  # groups: "common", "A", "B", "meshA", "meshB"

    def add(trace, group):
        traces.append(trace)
        groups.append(group)

    add(lines(mp, MP_CONNECTIONS, "MediaPipe skeleton", "#7f7f7f", 4), "common")
    for limb, side in limbs:
        (m_s, m_e, m_w), (s_s, s_e, s_w) = INDICES[(limb, side)]
        mid = "elbow" if limb == "arm" else "knee"
        d1, d2 = mp[m_e] - mp[m_s], mp[m_w] - mp[m_e]
        n = np.cross(d1, d2)
        add(go.Scatter3d(x=mp[[m_s, m_e, m_w], 0], y=mp[[m_s, m_e, m_w], 1], z=mp[[m_s, m_e, m_w], 2],
                         mode="markers", marker=dict(color="#ff7f0e", size=6), name=f"MediaPipe {side} {limb}"),
            "common")
        if np.linalg.norm(n) > 1e-9:
            add(arrow(mp[m_e], n, 0.15, f"observed bend normal n ({side} {limb})", "#ff7f0e", 10), "common")
        if limb == "leg":
            (mh, mt), (sa, sf) = FOOT[side]
            add(arrow(mp[mh], mp[mt] - mp[mh], 0.2, f"MediaPipe foot direction ({side})", "#8c564b", 8), "common")
        for k in ("A", "B"):
            add(arrow(jc[k][s_s], d1, np.linalg.norm(jc[k][s_e] - jc[k][s_s]),
                      f"target proximal bone ({side} {limb}, {k})", "#17becf", 6, "dot"), k)
            add(arrow(jc[k][s_e], d2, np.linalg.norm(jc[k][s_w] - jc[k][s_e]),
                      f"target distal bone ({side} {limb}, {k})", "#17becf", 6, "dot"), k)
            add(arrow(jc[k][s_e], arrays["hinge_axes"][(limb, side, k)], 0.15,
                      f"{mid} hinge axis {label[k]} ({side})", color[k], 10), k)
            if limb == "leg":
                add(arrow(jc[k][sa], jc[k][sf] - jc[k][sa], 0.2, f"SMPL foot direction {label[k]} ({side})",
                          color[k], 8, "dash"), k)
    for k in ("A", "B"):
        add(lines(jc[k], sm_pairs, f"SMPL {label[k]}", color[k], 5), k)
        add(go.Mesh3d(x=vc[k][:, 0], y=vc[k][:, 1], z=vc[k][:, 2], i=F[:, 0], j=F[:, 1], k=F[:, 2],
                      name=f"SMPL mesh {label[k]} (toggle)", color=color[k], opacity=0.35, flatshading=True,
                      showlegend=True, hoverinfo="skip"), f"mesh{k}")

    def vis_for(mode):
        out = []
        for g in groups:
            if g == "common":
                out.append(True)
            elif g in ("A", "B"):
                out.append(mode in (g, "both"))
            else:  # meshes: toggleable via legend when their pose is shown
                out.append("legendonly" if mode in (g[-1], "both") else False)
        return out

    for t, v, g in zip(traces, vis_for("both"), groups):
        t.visible = True if (show_mesh is not None and g == f"mesh{show_mesh}") else v
    buttons = [dict(label=lbl, method="update", args=[{"visible": vis_for(mode)}])
               for lbl, mode in (("Both", "both"), ("Swing-only (A)", "A"), ("Twist-corrected (B)", "B"))]
    parts = []
    for limb, side in limbs:
        e = metrics["limbs"][f"{limb}_{side}"]
        b = e["B_twist_corrected"]
        tail = (f"B {b['off_hinge_deg']:.1e} deg" if b["twist_resolved"] else "B = A (fallback)")
        parts.append(f"{side} {limb}: off-hinge A {e['A_swing_only']['off_hinge_deg']:.1f} deg -> {tail}")
    sub = ("; ".join(parts) + ". Display only: centred on each body's hip midpoint. "
           "Meshes: click legend entries to show.")
    fig = go.Figure(traces)
    fig.update_layout(
        title=dict(text=f"Limb initialisation: swing-only vs twist-corrected<br><sup>{sub}</sup>"),
        updatemenus=[dict(buttons=buttons, direction="down", x=0.0, xanchor="left", y=1.0, yanchor="top")],
        scene=dict(xaxis_title="X (right)", yaxis_title="Y (down)", zaxis_title="Z (away from camera)",
                   aspectmode="data", camera=dict(up=dict(x=0, y=-1, z=0), eye=dict(x=0.0, y=-0.3, z=-2.2))),
        legend=dict(itemsizing="constant"), margin=dict(l=0, r=0, t=90, b=0),
    )
    return fig


def save_3d(arrays, metrics, limbs, path):
    build_figure(arrays, metrics, limbs).write_html(path, include_plotlyjs=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--image", required=True)
    p.add_argument("--smpl-model", default=os.environ.get("SMPL_MODEL_PATH"))
    p.add_argument("--pose-model", default=DEFAULT_POSE_MODEL)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--side", choices=("left", "right", "both"), default="both")
    p.add_argument("--limbs", choices=("arms", "legs", "all"), default="arms")
    args = p.parse_args(argv)
    if not args.smpl_model:
        p.error("--smpl-model is required (or set SMPL_MODEL_PATH)")
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.abspath(os.path.expanduser(args.out_dir))
    if os.path.commonpath([out, repo]) == repo:
        p.error(f"--out-dir must be outside the repository ({repo})")
    os.makedirs(out, exist_ok=True)
    sides = ("left", "right") if args.side == "both" else (args.side,)
    kinds = {"arms": ("arm",), "legs": ("leg",), "all": ("arm", "leg")}[args.limbs]
    limbs = [(limb, side) for limb in kinds for side in sides]
    prefix = {"arms": "arm", "legs": "leg", "all": "limbs"}[args.limbs]

    rgb = read_rgb(args.image)
    world_raw, _, vis = mediapipe_to_arrays(detect_pose(rgb, args.pose_model), rgb.shape[:2])
    world = mediapipe_world_to_camera_aligned(world_raw).astype(np.float64)
    model = load_smpl(args.smpl_model)
    metrics, arrays = analyze(world, vis, model, limbs)
    metrics["input"] = {"image": os.path.abspath(args.image)}

    mpath = os.path.join(out, f"{prefix}_metrics_{args.side}.json")
    hpath = os.path.join(out, f"{prefix}_pose_3d_{args.side}.html")
    with open(mpath, "w") as fh:
        json.dump(metrics, fh, indent=2)
    save_3d(arrays, metrics, limbs, hpath)
    print(json.dumps(metrics, indent=2))
    print(f"metrics: {mpath}\nplot:    {hpath}")


if __name__ == "__main__":
    main()
