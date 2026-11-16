"""
Tests for the SMPL loader / forward wrappers in src/smpl_utils.py.

Needs torch + smplx (NOT the official SMPL model):
  - Synthetic tests build a fake SMPL-format pickle (random numbers, valid
    kinematic tree, skinning weights and faces) in a temp dir. Nothing licensed
    is used and nothing is written inside the repository.
  - Real-model integration tests run only if SMPL_MODEL_PATH points to the
    Chumpy-free official model, e.g. in Colab:
        SMPL_MODEL_PATH=/content/drive/MyDrive/smpl_models_clean/SMPL_NEUTRAL.pkl \
            python tests/test_smpl_model.py
    Otherwise they are reported as [skip].

Run from 3d-pose-diff-render/:  python tests/test_smpl_model.py   (pytest also works)
"""

import atexit
import os
import pickle
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

try:
    import torch
    import smplx  # noqa: F401
    _DEPS_ERROR = None
except ImportError as e:  # pragma: no cover
    _DEPS_ERROR = str(e)

from src.smpl_utils import (  # noqa: E402
    NUM_SMPL_JOINTS,
    SMPL_JOINT_NAMES,
    SMPL_PARENTS as SRC_SMPL_PARENTS,
    SMPL_TORSO_JOINTS,
    global_joint_rotations,
    axis_angle_to_rotation_matrix,
    elbow_hinge_axis,
    global_orient_from_torso,
    init_arm_pose,
    init_leg_pose,
    initialize_smpl_from_mediapipe,
    knee_hinge_axis,
    load_smpl,
    refine_arm_twist,
    refine_leg_twist,
    rotation_between_vectors,
    rotation_matrix_to_axis_angle,
    smpl_forward,
    smpl_rest_torso_orientation,
    torso_orientation_matrix,
)

# --------------------------------------------------------------------------- #
# Synthetic SMPL fixture
# --------------------------------------------------------------------------- #
NV, NJ, NF, NB = 6890, 24, 13776, 10
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
PELVIS = np.array([0.01, -0.22, 0.03])  # non-zero, to test rotation about the pelvis
# Rough SMPL-like rest joints relative to the pelvis: +x person's left, +y up, +z forward.
REST_REL = np.array([
    [0, 0, 0], [0.06, -0.09, 0], [-0.06, -0.09, 0], [0, 0.10, 0],           # pelvis, hips, spine1
    [0.10, -0.45, 0], [-0.10, -0.45, 0], [0, 0.23, 0],                       # knees, spine2
    [0.10, -0.85, -0.03], [-0.10, -0.85, -0.03], [0, 0.28, 0],               # ankles, spine3
    [0.12, -0.90, 0.10], [-0.12, -0.90, 0.10], [0, 0.50, 0],                 # feet, neck
    [0.08, 0.40, 0], [-0.08, 0.40, 0], [0, 0.60, 0.05],                      # collars, head
    [0.18, 0.42, 0], [-0.18, 0.42, 0], [0.45, 0.42, 0], [-0.45, 0.42, 0],    # shoulders, elbows
    [0.70, 0.42, 0], [-0.70, 0.42, 0], [0.78, 0.42, 0], [-0.78, 0.42, 0],    # wrists, hands
])
REST_JOINTS = REST_REL + PELVIS

_FIXTURE = {}


def _fixture():
    """Create (once) a fake SMPL-format, Chumpy-free pickle and return its data."""
    if _FIXTURE:
        return _FIXTURE
    rng = np.random.default_rng(0)
    # Vertices 0..23 sit exactly on the joints; J_regressor picks them one-hot.
    v_template = np.concatenate([REST_JOINTS, PELVIS + rng.normal(scale=0.3, size=(NV - NJ, 3))])
    J_regressor = np.zeros((NJ, NV))
    J_regressor[np.arange(NJ), np.arange(NJ)] = 1.0
    # Skinning weights: non-negative, rows sum to 1; joint vertices bound to their own joint.
    weights = rng.random((NV, NJ)) ** 4
    weights[:NJ] = np.eye(NJ)
    weights /= weights.sum(axis=1, keepdims=True)
    # Faces: valid indices, three distinct vertices per face.
    f = np.stack([rng.choice(NV, size=3, replace=False) for _ in range(NF)]).astype(np.uint32)
    kintree = np.array([[4294967295] + SMPL_PARENTS[1:], list(range(NJ))], dtype=np.int64)
    data = {
        "v_template": v_template,
        "shapedirs": rng.normal(scale=0.01, size=(NV, 3, NB)),
        "posedirs": np.zeros((NV, 3, 207), dtype=np.float32),
        "J_regressor": J_regressor,
        "weights": weights,
        "kintree_table": kintree,
        "f": f,
    }
    tmp = tempfile.mkdtemp(prefix="fake_smpl_")
    atexit.register(shutil.rmtree, tmp, ignore_errors=True)
    path = os.path.join(tmp, "SMPL_NEUTRAL.pkl")
    with open(path, "wb") as fh:
        pickle.dump(data, fh, protocol=2)
    chumpy_path = os.path.join(tmp, "chumpy_model.pkl")
    with open(chumpy_path, "wb") as fh:
        fh.write(b"cchumpy.ch\nCh\n.")  # protocol-0 GLOBAL chumpy.ch.Ch, then STOP
    _FIXTURE.update(data=data, dir=tmp, path=path, chumpy_path=chumpy_path,
                    model=load_smpl(path))
    return _FIXTURE


def _need_deps():
    if _DEPS_ERROR:
        raise unittest.SkipTest(f"torch/smplx unavailable: {_DEPS_ERROR}")


def _expect_raises(exc_type, fn, match=""):
    try:
        fn()
    except exc_type as e:
        assert match in str(e), f"expected '{match}' in error message, got: {e}"
        return
    raise AssertionError(f"expected {exc_type.__name__} (match '{match}')")


def _rodrigues(r):
    r = np.asarray(r, dtype=np.float64)
    th = np.linalg.norm(r)
    if th == 0:
        return np.eye(3)
    a = r / th
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def _random_rotation(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _torso_R(joints):
    J = np.asarray(joints, dtype=np.float64)
    t = SMPL_TORSO_JOINTS
    return torso_orientation_matrix(J[t["left_shoulder"]], J[t["right_shoulder"]],
                                    J[t["left_hip"]], J[t["right_hip"]])


# --------------------------------------------------------------------------- #
# Synthetic-model tests
# --------------------------------------------------------------------------- #
def test_torso_joint_indices():
    assert SMPL_TORSO_JOINTS == {"left_shoulder": 16, "right_shoulder": 17, "left_hip": 1, "right_hip": 2}


def test_load_paths_and_errors():
    _need_deps()
    fx = _fixture()
    m = fx["model"]
    assert not m.training, "model not in eval mode"
    assert list(m.parameters()) == [], "loader must not create trainable parameters"
    m_dir = load_smpl(fx["dir"])  # directory -> SMPL_NEUTRAL.pkl
    assert torch.equal(m_dir.v_template, m.v_template)
    _expect_raises(FileNotFoundError, lambda: load_smpl(os.path.join(fx["dir"], "nope.pkl")), "not found")
    _expect_raises(FileNotFoundError, lambda: load_smpl(fx["dir"], gender="female"), "SMPL_FEMALE.pkl")
    _expect_raises(ValueError, lambda: load_smpl(fx["path"], gender="other"), "gender")
    _expect_raises(ValueError, lambda: load_smpl(fx["path"], device="not_a_device"), "device")


def test_load_chumpy_pickle_clear_error():
    _need_deps()
    _expect_raises(ImportError, lambda: load_smpl(_fixture()["chumpy_path"]), "Chumpy-free")


def test_forward_shapes_and_dtypes():
    _need_deps()
    fx = _fixture()
    out = smpl_forward(fx["model"])
    assert out["vertices"].shape == (1, NV, 3) and out["vertices"].dtype == torch.float32
    assert out["joints"].shape == (1, NUM_SMPL_JOINTS, 3) and out["joints"].dtype == torch.float32
    assert out["faces"].shape == (NF, 3) and out["faces"].dtype == torch.long
    assert np.array_equal(out["faces"].numpy(), fx["data"]["f"].astype(np.int64))


def test_forward_rest_pose_and_joint_slice():
    _need_deps()
    fx = _fixture()
    out = smpl_forward(fx["model"])
    assert np.allclose(out["joints"][0].numpy(), REST_JOINTS, atol=1e-6)
    assert np.allclose(out["vertices"][0].numpy(), fx["data"]["v_template"], atol=1e-6)
    # Raw smplx output has extra vertex landmarks after the 24 kinematic joints.
    z = lambda w: torch.zeros(1, w)  # noqa: E731
    raw = fx["model"](global_orient=z(3), body_pose=z(69), betas=z(NB), transl=z(3)).joints
    assert raw.shape[1] > NUM_SMPL_JOINTS, raw.shape
    assert torch.equal(raw[:, :NUM_SMPL_JOINTS], out["joints"])


def test_forward_betas():
    _need_deps()
    fx = _fixture()
    d = fx["data"]
    betas = np.linspace(-2, 2, NB)
    out = smpl_forward(fx["model"], betas=betas)
    v_shaped = d["v_template"] + d["shapedirs"] @ betas
    assert np.allclose(out["joints"][0].numpy(), d["J_regressor"] @ v_shaped, atol=1e-5)


def test_forward_input_forms_and_batching():
    _need_deps()
    m = _fixture()["model"]
    rng = np.random.default_rng(1)
    go, bp, tr = rng.normal(size=(3, 3)) * 0.5, rng.normal(size=(3, 69)) * 0.2, rng.normal(size=(3, 3))
    batched = smpl_forward(m, global_orient=torch.tensor(go, dtype=torch.float64),
                           body_pose=torch.tensor(bp, dtype=torch.float64), transl=tr)
    assert batched["joints"].shape == (3, NUM_SMPL_JOINTS, 3)
    for i in range(3):
        single = smpl_forward(m, global_orient=go[i], body_pose=bp[i].tolist(), transl=tr[i])
        assert torch.allclose(single["joints"][0], batched["joints"][i], atol=1e-5)
        assert torch.allclose(single["vertices"][0], batched["vertices"][i], atol=1e-5)
    # transl is a pure shift.
    base = smpl_forward(m, global_orient=go[0])
    shifted = smpl_forward(m, global_orient=go[0], transl=tr[0])
    assert torch.allclose(shifted["joints"] - base["joints"],
                          torch.tensor(tr[0], dtype=torch.float32).expand(1, NUM_SMPL_JOINTS, 3), atol=1e-5)


def test_forward_kinematics():
    _need_deps()
    m = _fixture()["model"]
    # global_orient rotates every joint about the rest pelvis.
    r = np.array([0.3, -1.1, 0.7])
    J = smpl_forward(m, global_orient=r)["joints"][0].double().numpy()
    expected = (REST_JOINTS - PELVIS) @ _rodrigues(r).T + PELVIS
    assert np.allclose(J, expected, atol=1e-5)
    # Bending the left elbow moves only its descendants (left wrist, left hand).
    bp = np.zeros(69)
    elbow = SMPL_JOINT_NAMES.index("left_elbow")
    bp[3 * (elbow - 1): 3 * elbow] = [0.0, 0.0, 1.2]
    J = smpl_forward(m, body_pose=bp)["joints"][0].numpy()
    moved = set(np.flatnonzero(np.abs(J - REST_JOINTS).max(axis=1) > 1e-6))
    assert moved == {SMPL_JOINT_NAMES.index("left_wrist"), SMPL_JOINT_NAMES.index("left_hand")}, moved


def test_forward_gradients():
    _need_deps()
    m = _fixture()["model"]
    go = torch.tensor([[0.2, 0.1, -0.3], [0.0, 0.5, 0.1]], dtype=torch.float64, requires_grad=True)
    bp = torch.full((2, 69), 0.05, requires_grad=True)
    betas = torch.zeros(2, NB, requires_grad=True)
    out = smpl_forward(m, global_orient=go, body_pose=bp, betas=betas)
    loss = out["joints"].square().sum() + out["vertices"].square().mean()
    loss.backward()
    for name, t in (("global_orient", go), ("body_pose", bp), ("betas", betas)):
        assert t.grad is not None and torch.isfinite(t.grad).all(), name
        assert t.grad.abs().sum() > 0, f"{name}: zero gradient"
    assert go.grad.dtype == torch.float64  # gradient flows back through the dtype cast


def test_forward_rejects_bad_inputs():
    _need_deps()
    m = _fixture()["model"]
    cases = {
        "body_pose (68,)": (dict(body_pose=np.zeros(68)), "shape"),
        "global_orient 3-D": (dict(global_orient=np.zeros((1, 1, 3))), "shape"),
        "empty batch": (dict(transl=np.zeros((0, 3))), "shape"),
        "betas width": (dict(betas=np.zeros(NB + 1)), "shape"),
        "batch mismatch": (dict(global_orient=np.zeros((2, 3)), body_pose=np.zeros((3, 69))), "batch"),
        "int tensor": (dict(global_orient=torch.zeros(1, 3, dtype=torch.long)), "floating"),
        "bool array": (dict(transl=np.zeros(3, dtype=bool)), "real numbers"),
        "string": (dict(transl=["a", "b", "c"]), "real numbers"),
        "wrong device": (dict(global_orient=torch.zeros(1, 3, device="meta")), "device"),
    }
    for name, (kwargs, match) in cases.items():
        try:
            _expect_raises(ValueError, lambda k=kwargs: smpl_forward(m, **k), match)
        except AssertionError as e:
            raise AssertionError(f"{name}: {e}") from None


def test_rest_torso_orientation():
    _need_deps()
    m = _fixture()["model"]
    R = smpl_rest_torso_orientation(m)
    assert R.shape == (3, 3) and R.dtype == np.float64
    assert np.allclose(R, _torso_R(REST_JOINTS), atol=1e-6)
    assert np.allclose(R, np.eye(3), atol=1e-6)  # fixture is built in SMPL rest convention
    betas = torch.linspace(-1, 1, NB, requires_grad=True)
    R_b = smpl_rest_torso_orientation(m, betas=betas)  # no_grad: must not fail or build a graph
    assert np.allclose(R_b, _torso_R(smpl_forward(m, betas=betas.detach())["joints"][0].numpy()), atol=1e-6)
    assert betas.grad is None
    _expect_raises(ValueError, lambda: smpl_rest_torso_orientation(m, betas=np.zeros((2, NB))), "one subject")


def test_global_orient_pipeline_synthetic():
    _need_deps()
    m = _fixture()["model"]
    R_rest = smpl_rest_torso_orientation(m)
    rng = np.random.default_rng(2)
    for _ in range(20):
        R_target = _random_rotation(rng)
        r = global_orient_from_torso(R_rest, R_target)
        posed = smpl_forward(m, global_orient=r)["joints"][0].numpy()
        assert np.allclose(_torso_R(posed), R_target, atol=1e-5)


def _arm_world(P, rng):
    """MediaPipe-like world landmarks with both arms on posed SMPL joints."""
    w = rng.normal(scale=0.3, size=(33, 3))
    for m, j in zip((11, 13, 15, 12, 14, 16), (16, 18, 20, 17, 19, 21)):
        w[m] = P[j]
    return w


def _vec_angle_deg(u, v):
    u, v = np.asarray(u, float), np.asarray(v, float)
    return float(np.degrees(np.arctan2(np.linalg.norm(np.cross(u, v)), u @ v)))


def _check_arm_round_trip(m, rest, n_poses, seed, atol_deg):
    rng = np.random.default_rng(seed)
    worst = 0.0
    for side, (s, e, w) in (("left", (16, 18, 20)), ("right", (17, 19, 21))):
        for _ in range(n_poses):
            go = rng.normal(size=3)
            bp_true = rng.normal(scale=0.2, size=69)
            bp_true[3 * (s - 1): 3 * s] = rng.normal(size=3)
            bp_true[3 * (e - 1): 3 * e] = rng.normal(size=3)
            P_true = smpl_forward(m, global_orient=go, body_pose=bp_true)["joints"][0].double().numpy()
            start = bp_true.copy()
            start[3 * (s - 1): 3 * s] = 0.0
            start[3 * (e - 1): 3 * e] = 0.0
            bp, _ = init_arm_pose(_arm_world(P_true, rng), rest, go, side=side, body_pose=start)
            P = smpl_forward(m, global_orient=go, body_pose=bp)["joints"][0].double().numpy()
            for a, b in ((s, e), (e, w)):
                worst = max(worst, _vec_angle_deg(P[b] - P[a], P_true[b] - P_true[a]))
    assert worst < atol_deg, f"max bone-direction error {worst:.2e} deg"
    return worst


def test_parents_match_model():
    _need_deps()
    assert tuple(_fixture()["model"].parents.tolist()) == SRC_SMPL_PARENTS


def test_global_rotations_match_smplx():
    # ORIENTATIONS: Q[j] from global_joint_rotations rotates every rest bone j->child exactly as smplx does.
    _need_deps()
    m = _fixture()["model"]
    rng = np.random.default_rng(20)
    go, bp = rng.normal(size=3), rng.normal(scale=0.5, size=69)
    P = smpl_forward(m, global_orient=go, body_pose=bp)["joints"][0].double().numpy()
    Q = global_joint_rotations(go, bp)
    for c in range(1, NUM_SMPL_JOINTS):
        p = SRC_SMPL_PARENTS[c]
        assert np.allclose(P[c] - P[p], Q[p] @ (REST_JOINTS[c] - REST_JOINTS[p]), atol=1e-5), (p, c)


def test_smplx_local_rotation_moves_descendants_not_itself():
    # POSITIONS through smplx: a joint's local rotation moves its descendants, not the joint.
    _need_deps()
    m = _fixture()["model"]
    n = SMPL_JOINT_NAMES.index
    base = smpl_forward(m)["joints"][0].numpy()
    for joint, expected in (("left_elbow", {"left_wrist", "left_hand"}),
                            ("left_shoulder", {"left_elbow", "left_wrist", "left_hand"})):
        bp = np.zeros(69)
        j = n(joint)
        bp[3 * (j - 1): 3 * j] = [0.4, -0.3, 0.8]
        P = smpl_forward(m, body_pose=bp)["joints"][0].numpy()
        moved = {SMPL_JOINT_NAMES[k] for k in range(NUM_SMPL_JOINTS) if np.abs(P[k] - base[k]).max() > 1e-6}
        assert moved == expected, (joint, moved)


def test_init_arm_pose_smplx_round_trip():
    _need_deps()
    m = _fixture()["model"]
    rest = smpl_forward(m)["joints"][0].double().numpy()
    _check_arm_round_trip(m, rest, n_poses=15, seed=21, atol_deg=1e-3)


def _check_hinge_round_trip(m, rest, n_poses, seed):
    """Hinge-elbow poses through smplx: Increment B recovers the full shoulder rotation."""
    rng = np.random.default_rng(seed)
    worst_dir = worst_sh = 0.0
    for side, (s, e, w) in (("left", (16, 18, 20)), ("right", (17, 19, 21))):
        h = elbow_hinge_axis(rest, side)
        for _ in range(n_poses):
            go = rng.normal(size=3)
            bp_true = rng.normal(scale=0.2, size=69)
            bp_true[3 * (s - 1): 3 * s] = rng.normal(size=3)
            bp_true[3 * (e - 1): 3 * e] = np.radians(rng.uniform(20, 140)) * h
            P_true = smpl_forward(m, global_orient=go, body_pose=bp_true)["joints"][0].double().numpy()
            start = bp_true.copy()
            start[3 * (s - 1): 3 * s] = 0.0
            start[3 * (e - 1): 3 * e] = 0.0
            bp, diag = refine_arm_twist(_arm_world(P_true, rng), rest, go, start, side)
            assert diag["twist_resolved"] and diag["off_hinge_deg"] < 1e-9
            P = smpl_forward(m, global_orient=go, body_pose=bp)["joints"][0].double().numpy()
            for a, b in ((s, e), (e, w)):
                worst_dir = max(worst_dir, _vec_angle_deg(P[b] - P[a], P_true[b] - P_true[a]))
            Rd = axis_angle_to_rotation_matrix(bp[3 * (s - 1): 3 * s]).T @ axis_angle_to_rotation_matrix(
                bp_true[3 * (s - 1): 3 * s])
            worst_sh = max(worst_sh, float(np.degrees(np.linalg.norm(rotation_matrix_to_axis_angle(Rd)))))
    # float32 SMPL joints feed the landmarks, so tolerances are float32-level.
    assert worst_dir < 1e-3 and worst_sh < 1e-2, (worst_dir, worst_sh)
    return worst_dir, worst_sh


def test_refine_arm_twist_smplx_round_trip():
    _need_deps()
    m = _fixture()["model"]
    rest = smpl_forward(m)["joints"][0].double().numpy()
    _check_hinge_round_trip(m, rest, n_poses=15, seed=23)


LEG_SIDES = ("left", "right")
_LEG = {"left": ((23, 25, 27), (1, 4, 7)), "right": ((24, 26, 28), (2, 5, 8))}


def _leg_world_smpl(P, rng):
    w = _arm_world(P, rng)
    for mp_idx, sm_idx in _LEG.values():
        for mm, j in zip(mp_idx, sm_idx):
            w[mm] = P[j]
    return w


def _check_leg_round_trip(m, rest, n_poses, seed):
    """Plausible hip (thigh within 60 deg, twist within 50 deg) + knee hinge through smplx:
    Increment A matches directions; Increment B also recovers the full hip rotation."""
    rng = np.random.default_rng(seed)
    worst_dir = worst_hip = 0.0
    n_resolved = 0
    for side in LEG_SIDES:
        (_, (hp, kn, an)) = _LEG[side]
        b1 = (rest[kn] - rest[hp]) / np.linalg.norm(rest[kn] - rest[hp])
        h = knee_hinge_axis(rest, side)
        for _ in range(n_poses):
            go = rng.normal(size=3)
            bp_true = rng.normal(scale=0.2, size=69)
            ax = np.cross(b1, rng.normal(size=3))
            target = axis_angle_to_rotation_matrix(np.radians(rng.uniform(0, 60)) * ax / np.linalg.norm(ax)) @ b1
            R_hip = rotation_between_vectors(b1, target) @ axis_angle_to_rotation_matrix(
                np.radians(rng.uniform(-50, 50)) * b1)
            bp_true[3 * (hp - 1): 3 * hp] = rotation_matrix_to_axis_angle(R_hip)
            bp_true[3 * (kn - 1): 3 * kn] = np.radians(rng.uniform(30, 120)) * h
            P_true = smpl_forward(m, global_orient=go, body_pose=bp_true)["joints"][0].double().numpy()
            start = bp_true.copy()
            start[3 * (hp - 1): 3 * hp] = 0.0
            start[3 * (kn - 1): 3 * kn] = 0.0
            w = _leg_world_smpl(P_true, rng)
            for fn in (init_leg_pose, refine_leg_twist):
                bp, diag = fn(w, rest, go, side=side, body_pose=start)
                P = smpl_forward(m, global_orient=go, body_pose=bp)["joints"][0].double().numpy()
                for a, b in ((hp, kn), (kn, an)):
                    worst_dir = max(worst_dir, _vec_angle_deg(P[b] - P[a], P_true[b] - P_true[a]))
            if diag["twist_resolved"]:
                n_resolved += 1
                Rd = axis_angle_to_rotation_matrix(bp[3 * (hp - 1): 3 * hp]).T @ R_hip
                worst_hip = max(worst_hip, float(np.degrees(np.linalg.norm(rotation_matrix_to_axis_angle(Rd)))))
    assert worst_dir < 1e-3 and worst_hip < 1e-2 and n_resolved >= 0.8 * n_poses * len(LEG_SIDES), \
        (worst_dir, worst_hip, n_resolved)
    return worst_dir, worst_hip


def test_leg_round_trip_smplx():
    _need_deps()
    m = _fixture()["model"]
    rest = smpl_forward(m)["joints"][0].double().numpy()
    _check_leg_round_trip(m, rest, n_poses=15, seed=25)


def _check_full_body_smplx(m, rest, n_poses, seed):
    """Full-body init through smplx: all 8 limb bones match; integrated == independent limb calls."""
    rng = np.random.default_rng(seed)
    bones = [(16, 18), (18, 20), (17, 19), (19, 21), (1, 4), (4, 7), (2, 5), (5, 8)]
    worst = 0.0
    for _ in range(n_poses):
        go = rng.normal(size=3)
        bp_true = np.zeros(69)  # spine/collars zero so the torso is rigid
        for side, (s_, e_) in (("left", (16, 18)), ("right", (17, 19))):
            bp_true[3 * (s_ - 1): 3 * s_] = rng.normal(size=3)
            bp_true[3 * (e_ - 1): 3 * e_] = np.radians(rng.uniform(25, 120)) * elbow_hinge_axis(rest, side)
        for side, (h_, k_) in (("left", (1, 4)), ("right", (2, 5))):
            bp_true[3 * (h_ - 1): 3 * h_] = rng.normal(scale=0.4, size=3)
            bp_true[3 * (k_ - 1): 3 * k_] = np.radians(rng.uniform(25, 120)) * knee_hinge_axis(rest, side)
        P_true = smpl_forward(m, global_orient=go, body_pose=bp_true)["joints"][0].double().numpy()
        w = _leg_world_smpl(P_true, rng)
        out = initialize_smpl_from_mediapipe(w, rest)
        P = smpl_forward(m, global_orient=out["global_orient"], body_pose=out["body_pose"])["joints"][0].double().numpy()
        for a, b in bones:
            worst = max(worst, _vec_angle_deg(P[b] - P[a], P_true[b] - P_true[a]))
        # Same pose as running the limb initializers one by one.
        bp = np.zeros(69)
        bp, _ = refine_arm_twist(w, rest, out["global_orient"], body_pose=bp, side="left")
        bp, _ = refine_arm_twist(w, rest, out["global_orient"], body_pose=bp, side="right")
        bp, _ = refine_leg_twist(w, rest, out["global_orient"], body_pose=bp, side="left")
        bp, _ = refine_leg_twist(w, rest, out["global_orient"], body_pose=bp, side="right")
        assert np.array_equal(bp, out["body_pose"])
    assert worst < 1e-3, f"worst bone-direction error {worst:.2e} deg"
    return worst


def test_full_body_smplx_round_trip():
    _need_deps()
    m = _fixture()["model"]
    rest = smpl_forward(m)["joints"][0].double().numpy()
    _check_full_body_smplx(m, rest, n_poses=10, seed=27)


# --------------------------------------------------------------------------- #
# Real-model integration tests (optional)
# --------------------------------------------------------------------------- #
_REAL = {}


def _real_model():
    _need_deps()
    path = os.environ.get("SMPL_MODEL_PATH")
    if not path:
        raise unittest.SkipTest("SMPL_MODEL_PATH not set")
    if "model" not in _REAL:
        _REAL["model"] = load_smpl(path)
    return _REAL["model"]


def test_real_model_shapes():
    m = _real_model()
    out = smpl_forward(m)
    assert out["vertices"].shape == (1, 6890, 3)
    assert out["joints"].shape == (1, 24, 3)
    assert out["faces"].shape == (13776, 3)
    z = lambda w: torch.zeros(1, w)  # noqa: E731
    raw = m(global_orient=z(3), body_pose=z(69), betas=z(m.num_betas), transl=z(3)).joints
    print(f"       (real model: raw smplx joints = {raw.shape[1]}, num_betas = {m.num_betas})")


def test_real_rest_convention():
    m = _real_model()
    J = smpl_forward(m)["joints"][0].numpy()
    i = SMPL_JOINT_NAMES.index
    assert J[i("left_hip"), 0] > J[i("right_hip"), 0], "+x is not the person's left"
    assert J[i("head"), 1] > J[i("pelvis"), 1], "+y is not up"
    assert J[i("left_foot"), 2] > J[i("left_ankle"), 2], "+z is not forward"
    R = smpl_rest_torso_orientation(m)
    angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    print(f"       (real model: rest torso frame is {angle:.2f} deg from identity)")
    assert angle < 10.0, angle


def test_real_front_facing_pipeline():
    m = _real_model()
    R_rest = smpl_rest_torso_orientation(m)
    # Synthetic front-facing MediaPipe torso, camera-aligned (+x right, +y down, +z away).
    ls, rs, lh, rh = (np.array([0.18, -0.45, 2.0]), np.array([-0.18, -0.45, 2.0]),
                      np.array([0.10, 0.0, 2.0]), np.array([-0.10, 0.0, 2.0]))
    R_mp = torso_orientation_matrix(ls, rs, lh, rh)
    r = global_orient_from_torso(R_rest, R_mp)
    posed = smpl_forward(m, global_orient=r)["joints"][0].numpy()
    assert np.allclose(_torso_R(posed), R_mp, atol=1e-5)


def test_real_random_round_trip():
    m = _real_model()
    R_rest = smpl_rest_torso_orientation(m)
    rng = np.random.default_rng(3)
    for _ in range(20):
        R_target = _random_rotation(rng)
        posed = smpl_forward(m, global_orient=global_orient_from_torso(R_rest, R_target))["joints"][0]
        assert np.allclose(_torso_R(posed.numpy()), R_target, atol=1e-5)


def test_real_parents_and_arm_round_trip():
    m = _real_model()
    assert tuple(m.parents.tolist()) == SRC_SMPL_PARENTS
    rest = smpl_forward(m)["joints"][0].double().numpy()
    worst = _check_arm_round_trip(m, rest, n_poses=20, seed=22, atol_deg=1e-3)
    print(f"       (real model: worst arm bone-direction error {worst:.2e} deg)")


def test_real_hinge_convention_and_round_trip():
    m = _real_model()
    rest = smpl_forward(m)["joints"][0].double().numpy()
    assert elbow_hinge_axis(rest, "left")[1] < -0.95 and elbow_hinge_axis(rest, "right")[1] > 0.95
    # Convention check: SMPL rest pose is palms down / thumbs forward (thumb tip ahead of pinky tip).
    from smplx.vertex_ids import vertex_ids
    V = smpl_forward(m)["vertices"][0].double().numpy()
    fwd = smpl_rest_torso_orientation(m)[:, 2]
    for c in ("l", "r"):
        ahead = (V[vertex_ids["smplh"][c + "thumb"]] - V[vertex_ids["smplh"][c + "pinky"]]) @ fwd
        assert ahead > 0.10, f"{c}: thumb only {ahead:.3f} m ahead of pinky"
    worst_dir, worst_sh = _check_hinge_round_trip(m, rest, n_poses=20, seed=24)
    print(f"       (real model: worst direction error {worst_dir:.1e} deg, worst shoulder error {worst_sh:.1e} deg)")


def test_real_leg_convention_and_round_trip():
    m = _real_model()
    rest = smpl_forward(m)["joints"][0].double().numpy()
    fwd = smpl_rest_torso_orientation(m)[:, 2]
    for side in LEG_SIDES:
        (_, (hp, kn, an)) = _LEG[side]
        assert knee_hinge_axis(rest, side)[0] > 0.95, f"{side}: knee hinge should be ~+x"
        foot = SMPL_JOINT_NAMES.index(f"{side}_foot")
        ahead = (rest[foot] - rest[an]) / np.linalg.norm(rest[foot] - rest[an]) @ fwd
        assert ahead > 0.8, f"{side}: rest foot not pointing forward ({ahead:.2f})"
    worst_dir, worst_hip = _check_leg_round_trip(m, rest, n_poses=20, seed=26)
    print(f"       (real model legs {LEG_SIDES}: worst direction error {worst_dir:.1e} deg, "
          f"worst hip error {worst_hip:.1e} deg)")


def test_real_full_body_round_trip():
    m = _real_model()
    rest = smpl_forward(m)["joints"][0].double().numpy()
    worst = _check_full_body_smplx(m, rest, n_poses=15, seed=28)
    print(f"       (real model full body: worst limb bone-direction error {worst:.1e} deg)")


if __name__ == "__main__":
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    failed = skipped = 0
    for name, fn in tests:
        try:
            fn()
            print(f"[ok]   {name}")
        except unittest.SkipTest as e:
            skipped += 1
            print(f"[skip] {name}: {e}")
        except Exception as e:
            failed += 1
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed - skipped}/{len(tests)} passed, {skipped} skipped, {failed} failed")
    sys.exit(1 if failed else 0)
