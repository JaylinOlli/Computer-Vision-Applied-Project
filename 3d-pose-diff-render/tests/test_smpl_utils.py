"""
Tests for the MediaPipe -> SMPL correspondence table in src/smpl_utils.py.

Pure Python, no dependencies. Run from 3d-pose-diff-render/:
    python tests/test_smpl_utils.py
(also discoverable by pytest if it is installed)
"""

import os
import sys
from types import SimpleNamespace as NS

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.smpl_utils import (  # noqa: E402
    APPROXIMATE_CORRESPONDENCES,
    MEDIAPIPE_LANDMARK_NAMES,
    NUM_MEDIAPIPE_LANDMARKS,
    NUM_SMPL_JOINTS,
    SMPL_JOINT_NAMES,
    get_mediapipe_to_smpl_correspondence,
    mediapipe_to_arrays,
    mediapipe_world_to_camera_aligned,
    MEDIAPIPE_TORSO_LANDMARKS,
    SMPL_PARENTS,
    axis_angle_to_rotation_matrix,
    elbow_hinge_axis,
    global_joint_rotations,
    global_orient_from_torso,
    FULL_BODY_LIMBS,
    init_arm_pose,
    init_leg_pose,
    initialize_smpl_from_mediapipe,
    knee_hinge_axis,
    refine_arm_twist,
    refine_leg_twist,
    rotation_between_vectors,
    mediapipe_torso_orientation,
    rotation_angle_deg,
    rotation_matrix_to_axis_angle,
    smpl_torso_orientation_from_joints,
    torso_orientation_matrix,
)


def _side(name):
    """'left' / 'right' / None, accepting both 'left_x' and 'mouth_left' styles."""
    for side in ("left", "right"):
        if name.startswith(side + "_") or name.endswith("_" + side):
            return side
    return None


def _mirror(name):
    """Swap left/right in a joint name."""
    side = _side(name)
    other = {"left": "right", "right": "left"}.get(side)
    return name.replace(side, other) if side else name


def test_counts():
    assert NUM_MEDIAPIPE_LANDMARKS == len(MEDIAPIPE_LANDMARK_NAMES) == 33
    assert NUM_SMPL_JOINTS == len(SMPL_JOINT_NAMES) == 24
    assert len(set(MEDIAPIPE_LANDMARK_NAMES)) == 33, "duplicate MediaPipe names"
    assert len(set(SMPL_JOINT_NAMES)) == 24, "duplicate SMPL names"


def test_known_name_indices():
    # Spot-check the official orderings.
    mp = {0: "nose", 11: "left_shoulder", 16: "right_wrist", 23: "left_hip", 32: "right_foot_index"}
    smpl = {0: "pelvis", 1: "left_hip", 12: "neck", 15: "head", 16: "left_shoulder", 23: "right_hand"}
    for i, n in mp.items():
        assert MEDIAPIPE_LANDMARK_NAMES[i] == n, (i, MEDIAPIPE_LANDMARK_NAMES[i], n)
    for i, n in smpl.items():
        assert SMPL_JOINT_NAMES[i] == n, (i, SMPL_JOINT_NAMES[i], n)


def test_aligned_and_in_range():
    mp_idx, smpl_idx = get_mediapipe_to_smpl_correspondence()
    assert len(mp_idx) == len(smpl_idx) > 0
    assert all(isinstance(i, int) for i in mp_idx + smpl_idx)
    assert all(0 <= i < NUM_MEDIAPIPE_LANDMARKS for i in mp_idx)
    assert all(0 <= i < NUM_SMPL_JOINTS for i in smpl_idx)


def test_unique():
    mp_idx, smpl_idx = get_mediapipe_to_smpl_correspondence()
    assert len(set(mp_idx)) == len(mp_idx), "a MediaPipe landmark is used twice"
    assert len(set(smpl_idx)) == len(smpl_idx), "an SMPL joint is used twice"


def test_required_mappings():
    pairs = set(zip(*get_mediapipe_to_smpl_correspondence()))
    for pair in [(11, 16), (12, 17), (13, 18), (14, 19), (15, 20), (16, 21)]:
        assert pair in pairs, f"missing required mapping {pair}"


def test_names_match():
    # Every direct correspondence maps a joint to the same-named joint.
    for m, s in zip(*get_mediapipe_to_smpl_correspondence()):
        assert MEDIAPIPE_LANDMARK_NAMES[m] == SMPL_JOINT_NAMES[s], (m, s)


def test_left_right_consistency():
    mp_idx, smpl_idx = get_mediapipe_to_smpl_correspondence()
    pairs = set(zip(mp_idx, smpl_idx))
    for m, s in pairs:
        mp_name, smpl_name = MEDIAPIPE_LANDMARK_NAMES[m], SMPL_JOINT_NAMES[s]
        # Same side on both skeletons.
        assert _side(mp_name) == _side(smpl_name), (mp_name, smpl_name)
        # Mirrored pair is present too (table is left/right symmetric).
        mirrored = (MEDIAPIPE_LANDMARK_NAMES.index(_mirror(mp_name)),
                    SMPL_JOINT_NAMES.index(_mirror(smpl_name)))
        assert mirrored in pairs, f"missing mirror of {mp_name}->{smpl_name}"


def test_approximate_kept_separate():
    _, smpl_idx = get_mediapipe_to_smpl_correspondence()
    direct = {SMPL_JOINT_NAMES[s] for s in smpl_idx}
    for smpl_name, (mp_names, _) in APPROXIMATE_CORRESPONDENCES.items():
        assert smpl_name in SMPL_JOINT_NAMES, smpl_name
        assert smpl_name not in direct, f"{smpl_name} is both direct and approximate"
        for n in mp_names:
            assert n in MEDIAPIPE_LANDMARK_NAMES, n


# --------------------------------------------------------------------------- #
# Step 2: mediapipe_to_arrays (synthetic MediaPipe-like results)
# --------------------------------------------------------------------------- #
H, W = 480, 640


def _image_lm(i):
    # Distinct per index; z is image-space relative depth and must be ignored.
    return NS(x=i / 40.0, y=1.0 - i / 40.0, z=-999.0 - i, visibility=i / 32.0)


def _world_lm(i):
    # Values chosen to be clearly distinct from the image landmarks.
    return NS(x=0.01 * i, y=-0.02 * i, z=0.5 + 0.03 * i, visibility=0.0)


def _legacy_result(n=33):
    return NS(
        pose_landmarks=NS(landmark=[_image_lm(i) for i in range(n)]),
        pose_world_landmarks=NS(landmark=[_world_lm(i) for i in range(n)]),
    )


def _tasks_result(n=33, people=1):
    return NS(
        pose_landmarks=[[_image_lm(i) for i in range(n)] for _ in range(people)],
        pose_world_landmarks=[[_world_lm(i) for i in range(n)] for _ in range(people)],
    )


def _expect_raises(exc_type, fn, match=""):
    try:
        fn()
    except exc_type as e:
        assert match in str(e), f"expected '{match}' in error message, got: {e}"
        return
    raise AssertionError(f"expected {exc_type.__name__} (match '{match}')")


def _expected():
    idx = np.arange(33, dtype=np.float64)
    pixels = np.stack([idx / 40.0 * W, (1.0 - idx / 40.0) * H], axis=1)
    world = np.stack([0.01 * idx, -0.02 * idx, 0.5 + 0.03 * idx], axis=1)
    return world, pixels, idx / 32.0


def test_extract_shapes_and_dtype():
    for result in (_legacy_result(), _tasks_result()):
        world, pixels, vis = mediapipe_to_arrays(result, (H, W))
        assert world.shape == (33, 3), world.shape
        assert pixels.shape == (33, 2), pixels.shape
        assert vis.shape == (33,), vis.shape
        assert world.dtype == pixels.dtype == vis.dtype == np.float32


def test_extract_values_both_formats():
    exp_world, exp_pixels, exp_vis = _expected()
    for result in (_legacy_result(), _tasks_result()):
        world, pixels, vis = mediapipe_to_arrays(result, (H, W))
        assert np.allclose(pixels, exp_pixels, atol=1e-3), "pixel conversion wrong"
        assert np.allclose(world, exp_world, atol=1e-6), "world coords wrong"
        assert np.allclose(vis, exp_vis, atol=1e-6), "visibility wrong"
        # Image z (-999 - i) must never leak into the world coordinates.
        assert (world > -900).all(), "image-landmark z leaked into world"


def test_extract_pixel_convention():
    # x scales by width, y by height (not swapped), even for non-square images.
    lm = [NS(x=0.0, y=0.0, z=0.0, visibility=1.0) for _ in range(33)]
    lm[0] = NS(x=1.0, y=1.0, z=0.0, visibility=1.0)
    lm[1] = NS(x=0.25, y=0.5, z=0.0, visibility=1.0)
    lm[2] = NS(x=-0.1, y=1.2, z=0.0, visibility=1.0)  # off-screen: not clipped
    result = NS(pose_landmarks=NS(landmark=lm),
                pose_world_landmarks=NS(landmark=[_world_lm(i) for i in range(33)]))
    _, pixels, _ = mediapipe_to_arrays(result, (100, 200))
    assert np.allclose(pixels[0], [200, 100])
    assert np.allclose(pixels[1], [50, 50])
    assert np.allclose(pixels[2], [-20, 120])


def test_extract_ordering():
    world, pixels, vis = mediapipe_to_arrays(_tasks_result(), (H, W))
    for name in ("nose", "left_shoulder", "right_wrist", "left_hip", "right_foot_index"):
        i = MEDIAPIPE_LANDMARK_NAMES.index(name)
        assert np.isclose(pixels[i, 0], i / 40.0 * W, atol=1e-3), name
        assert np.isclose(world[i, 2], 0.5 + 0.03 * i, atol=1e-6), name
        assert np.isclose(vis[i], i / 32.0, atol=1e-6), name


def test_extract_person_index():
    result = _tasks_result(people=2)
    result.pose_landmarks[1][0] = NS(x=0.5, y=0.5, z=0.0, visibility=0.7)
    result.pose_world_landmarks[1][0] = NS(x=9.0, y=9.0, z=9.0, visibility=0.0)
    world, pixels, vis = mediapipe_to_arrays(result, (H, W), person_index=1)
    assert np.allclose(pixels[0], [W / 2, H / 2]) and np.allclose(world[0], 9.0)
    assert np.isclose(vis[0], 0.7)
    _expect_raises(ValueError, lambda: mediapipe_to_arrays(result, (H, W), person_index=2),
                   "out of range")
    _expect_raises(ValueError, lambda: mediapipe_to_arrays(_legacy_result(), (H, W), person_index=1),
                   "person_index")


def test_extract_no_detection():
    legacy_none = NS(pose_landmarks=None, pose_world_landmarks=None)
    tasks_empty = NS(pose_landmarks=[], pose_world_landmarks=[])
    _expect_raises(ValueError, lambda: mediapipe_to_arrays(legacy_none, (H, W)), "no pose")
    _expect_raises(ValueError, lambda: mediapipe_to_arrays(tasks_empty, (H, W)), "no pose")


def test_extract_malformed():
    # Wrong landmark count.
    _expect_raises(ValueError, lambda: mediapipe_to_arrays(_legacy_result(n=32), (H, W)), "expected 33")
    _expect_raises(ValueError, lambda: mediapipe_to_arrays(_tasks_result(n=34), (H, W)), "expected 33")
    # Image and world present, but world missing.
    r = _legacy_result()
    r.pose_world_landmarks = None
    _expect_raises(ValueError, lambda: mediapipe_to_arrays(r, (H, W)), "pose_world_landmarks")
    # Not a MediaPipe result / unsupported container.
    _expect_raises(TypeError, lambda: mediapipe_to_arrays(NS(), (H, W)), "pose_landmarks")
    bad = NS(pose_landmarks="oops", pose_world_landmarks="oops")
    _expect_raises(TypeError, lambda: mediapipe_to_arrays(bad, (H, W)), "unsupported type")
    # Missing field on a landmark (names the landmark).
    r = _tasks_result()
    r.pose_landmarks[0][11] = NS(x=0.1, y=0.2, z=0.0)
    _expect_raises(ValueError, lambda: mediapipe_to_arrays(r, (H, W)), "left_shoulder")
    # Non-finite value.
    r = _tasks_result()
    r.pose_world_landmarks[0][5] = NS(x=float("nan"), y=0.0, z=0.0)
    _expect_raises(ValueError, lambda: mediapipe_to_arrays(r, (H, W)), "non-finite")


def test_extract_bad_image_hw():
    r = _legacy_result()
    for bad in [(0, 640), (480, -1), (480,), (480, 640, 3), None, (480.5, 640), ("480", 640),
                (float("inf"), 640), (True, 640)]:
        _expect_raises(ValueError, lambda b=bad: mediapipe_to_arrays(r, b), "")
    # numpy ints / int-valued floats are accepted.
    mediapipe_to_arrays(r, (np.int64(480), 640.0))


# --------------------------------------------------------------------------- #
# Step 3A: mediapipe_world_to_camera_aligned (identity, validated)
# --------------------------------------------------------------------------- #
def _signed_world():
    # Mixed signs on every axis, so any flip or permutation would show up.
    idx = np.arange(33, dtype=np.float64)
    return np.stack([idx - 16.0, 0.5 - 0.03 * idx, -0.2 + 0.013 * idx], axis=1) * 0.1


def test_camera_aligned_preserves_values():
    world = _signed_world()
    assert (world < 0).any(axis=0).all() and (world > 0).any(axis=0).all()
    out = mediapipe_world_to_camera_aligned(world)
    assert out.shape == (33, 3)
    assert out.dtype == np.float64
    assert np.array_equal(out, world), "values changed (flip/permutation applied?)"


def test_camera_aligned_from_step2_output():
    world, _, _ = mediapipe_to_arrays(_tasks_result(), (H, W))
    out = mediapipe_world_to_camera_aligned(world)
    assert out.dtype == np.float32 and np.array_equal(out, world)


def test_camera_aligned_returns_copy():
    world = _signed_world()
    out = mediapipe_world_to_camera_aligned(world)
    out[0, 0] = 123.0
    assert world[0, 0] != 123.0, "output aliases the input"


def test_camera_aligned_accepts_lists_and_ints():
    world = _signed_world()
    assert np.array_equal(mediapipe_world_to_camera_aligned(world.tolist()), world)
    ints = np.arange(99).reshape(33, 3) - 50
    out = mediapipe_world_to_camera_aligned(ints)
    assert out.dtype == np.float32 and np.array_equal(out, ints)


def test_camera_aligned_rejects_malformed():
    good = _signed_world()
    for bad_shape in [good[:32], good[:, :2], good.reshape(1, 33, 3), good.ravel(), np.zeros((33, 4))]:
        _expect_raises(ValueError, lambda b=bad_shape: mediapipe_world_to_camera_aligned(b), "shape")
    for bad_value in (np.nan, np.inf, -np.inf):
        b = good.copy()
        b[7, 2] = bad_value
        _expect_raises(ValueError, lambda b=b: mediapipe_world_to_camera_aligned(b), "non-finite")
    for bad_type in [np.full((33, 3), "x"), np.ones((33, 3), dtype=bool),
                     np.ones((33, 3), dtype=complex), None]:
        _expect_raises(TypeError, lambda b=bad_type: mediapipe_world_to_camera_aligned(b), "real numbers")


# --------------------------------------------------------------------------- #
# Step 3B: torso_orientation_matrix
# --------------------------------------------------------------------------- #
# Canonical torso in SMPL rest convention: person's left = +x, up = +y.
CANON = (np.array([0.2, 0.5, 0.0]), np.array([-0.2, 0.5, 0.0]),   # ls, rs
         np.array([0.1, 0.0, 0.0]), np.array([-0.1, 0.0, 0.0]))   # lh, rh


def _assert_rotation(R, atol=1e-9):
    assert R.shape == (3, 3) and R.dtype == np.float64
    assert np.allclose(R.T @ R, np.eye(3), atol=atol), "not orthonormal"
    assert np.isclose(np.linalg.det(R), 1.0, atol=atol), f"det = {np.linalg.det(R)}"


def _random_rotation(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _torso_at_angle(theta_deg):
    """Torso whose lateral direction is theta_deg away from the spine direction."""
    t = np.radians(theta_deg)
    a = 0.15 * np.array([np.sin(t), np.cos(t), 0.0])
    s = np.array([0.0, 0.5, 0.0])
    return s + a, s - a, a, -a


def test_torso_canonical_is_identity():
    R = torso_orientation_matrix(*CANON)
    _assert_rotation(R)
    assert np.allclose(R, np.eye(3))


def test_torso_front_facing_camera():
    # Camera-aligned (+x right, +y down, +z away). Person faces the camera:
    # their left is image right (+x), shoulders above hips (-y), all at depth 2 m.
    ls, rs, lh, rh = [np.array([p[0], -p[1], 2.0]) for p in CANON]
    R = torso_orientation_matrix(ls, rs, lh, rh)
    _assert_rotation(R)
    assert np.allclose(R, np.diag([1.0, -1.0, -1.0])), R  # 180 deg about X, not hard-coded


def test_torso_back_facing_camera():
    # Person faces away: their left is image left (-x).
    ls, rs, lh, rh = [np.array([-p[0], -p[1], 2.0]) for p in CANON]
    R = torso_orientation_matrix(ls, rs, lh, rh)
    _assert_rotation(R)
    assert np.allclose(R, np.diag([-1.0, -1.0, 1.0])), R


def test_torso_recovers_known_rotations():
    rng = np.random.default_rng(0)
    for _ in range(20):
        Q = _random_rotation(rng)
        scale = rng.uniform(0.5, 3.0)
        shift = rng.normal(size=3) * 5.0
        pts = [scale * (Q @ p) + shift for p in CANON]
        R = torso_orientation_matrix(*pts)
        _assert_rotation(R)
        assert np.allclose(R, Q, atol=1e-9), "rotation not recovered"


def test_torso_messy_inputs_orthonormal():
    rng = np.random.default_rng(1)
    for _ in range(50):
        Q = _random_rotation(rng)
        pts = [Q @ (p + rng.normal(scale=0.05, size=3)) + rng.normal(size=3) for p in CANON]
        pts[0] = pts[0] + Q @ np.array([0.0, 0.08, 0.03])   # shrugged / twisted shoulder
        ls, rs, lh, rh = pts
        R = torso_orientation_matrix(ls, rs, lh, rh)
        _assert_rotation(R)
        up = 0.5 * (ls + rs) - 0.5 * (lh + rh)
        lateral = (ls - rs) + (lh - rh)
        assert np.allclose(np.cross(R[:, 1], up), 0.0, atol=1e-9), "spine axis not kept exactly"
        assert R[:, 1] @ up > 0, "up axis points down the spine"
        assert R[:, 0] @ lateral > 0, "x axis does not point to the person's left"


def test_torso_label_swap_is_rotation_not_reflection():
    ls, rs, lh, rh = CANON
    R = torso_orientation_matrix(ls, rs, lh, rh)
    R_swap = torso_orientation_matrix(rs, ls, rh, lh)
    _assert_rotation(R_swap)
    assert np.allclose(R_swap[:, 0], -R[:, 0]) and np.allclose(R_swap[:, 2], -R[:, 2])
    assert np.allclose(R_swap[:, 1], R[:, 1])


def test_torso_degenerate_rejected():
    ls, rs, lh, rh = CANON
    mid_s, mid_h = np.array([0.0, 0.5, 0.0]), np.zeros(3)
    cases = {
        "zero lateral": ((mid_s, mid_s, mid_h, mid_h), "lateral"),
        "zero spine": ((ls - mid_s, rs - mid_s, lh, rh), "spine"),
        "parallel": (_torso_at_angle(0.0), "parallel"),
        "4 deg": (_torso_at_angle(4.0), "parallel"),
        "176 deg": (_torso_at_angle(176.0), "parallel"),
        "nan": ((np.array([np.nan, 0.5, 0.0]), rs, lh, rh), "non-finite"),
        "inf": ((ls, rs, np.array([0.1, np.inf, 0.0]), rh), "non-finite"),
        "shape (2,)": ((ls[:2], rs, lh, rh), "shape"),
        "shape (1,3)": ((ls, rs, lh, rh[None]), "shape"),
        "string": ((np.array(["a", "b", "c"]), rs, lh, rh), "real numbers"),
    }
    for name, (pts, match) in cases.items():
        try:
            _expect_raises(ValueError, lambda p=pts: torso_orientation_matrix(*p), match)
        except AssertionError as e:
            raise AssertionError(f"{name}: {e}") from None
    for bad in ({"min_length": 0.0}, {"min_angle_deg": 0.0}, {"min_angle_deg": 90.0}):
        _expect_raises(ValueError, lambda b=bad: torso_orientation_matrix(*CANON, **b), "must be")


def test_torso_just_above_threshold():
    for theta in (6.0, 174.0):
        R = torso_orientation_matrix(*_torso_at_angle(theta))
        _assert_rotation(R)
        assert np.allclose(R[:, 1], [0.0, 1.0, 0.0]), "spine axis changed"


# --------------------------------------------------------------------------- #
# Step 3C: rotation_matrix_to_axis_angle, global_orient_from_torso
# --------------------------------------------------------------------------- #
def _rodrigues(r):
    """Independent axis-angle -> matrix (exact formula, same form as smplx)."""
    r = np.asarray(r, dtype=np.float64)
    theta = np.linalg.norm(r)
    if theta == 0.0:
        return np.eye(3)
    a = r / theta
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


def _rodrigues_smplx(r):
    """Literal port of smplx.lbs.batch_rodrigues, including its +1e-8 in the norm."""
    r = np.asarray(r, dtype=np.float64)
    angle = np.linalg.norm(r + 1e-8)
    rx, ry, rz = r / angle
    K = np.array([[0, -rz, ry], [rz, 0, -rx], [-ry, rx, 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


AXES = {
    "x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]), "z": np.array([0, 0, 1.0]),
    "(1,1,0)": np.array([1.0, 1, 0]) / np.sqrt(2), "(1,2,3)": np.array([1.0, 2, 3]) / np.sqrt(14),
    "(-1,2,-3)": np.array([-1.0, 2, -3]) / np.sqrt(14),
}


def _unit(rng):
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def test_axis_angle_identity():
    assert np.array_equal(rotation_matrix_to_axis_angle(np.eye(3)), np.zeros(3))
    rng = np.random.default_rng(2)
    for _ in range(10):
        R = _random_rotation(rng)
        r = global_orient_from_torso(R, R)
        assert np.abs(r).max() < 1e-12, r


def test_axis_angle_90_degrees():
    for name, a in AXES.items():
        r = rotation_matrix_to_axis_angle(_rodrigues(np.pi / 2 * a))
        assert np.allclose(r, np.pi / 2 * a, atol=1e-12), (name, r)


def test_axis_angle_180_degrees():
    for name, a in AXES.items():
        R = _rodrigues(np.pi * a)
        r = rotation_matrix_to_axis_angle(R)
        assert np.isclose(np.linalg.norm(r), np.pi, atol=1e-12), (name, r)
        assert np.allclose(np.cross(r, a), 0, atol=1e-12), f"{name}: axis not parallel"
        assert np.allclose(_rodrigues(r), R, atol=1e-12), f"{name}: reconstruction"
        first = r[np.flatnonzero(np.abs(r) > 1e-9)[0]]
        assert first > 0, f"{name}: canonical sign not applied: {r}"
    # Exact axis-aligned half turns.
    for diag, exp in [((1, -1, -1), (np.pi, 0, 0)), ((-1, 1, -1), (0, np.pi, 0)),
                      ((-1, -1, 1), (0, 0, np.pi))]:
        r = rotation_matrix_to_axis_angle(np.diag(np.array(diag, dtype=float)))
        assert np.allclose(r, exp, atol=1e-15), (diag, r)


def test_global_orient_front_facing_end_to_end():
    R_smpl = torso_orientation_matrix(*CANON)
    ls, rs, lh, rh = [np.array([p[0], -p[1], 2.0]) for p in CANON]
    R_mp = torso_orientation_matrix(ls, rs, lh, rh)
    r = global_orient_from_torso(R_smpl, R_mp)
    assert np.allclose(r, [np.pi, 0, 0], atol=1e-12), r


def test_axis_angle_near_zero():
    rng = np.random.default_rng(3)
    for theta in (1e-12, 1e-8, 1e-4, 1e-3 * (1 - 1e-9), 1e-3 * (1 + 1e-9), 1e-2):
        for _ in range(5):
            a = _unit(rng)
            R = _rodrigues(theta * a)
            r = rotation_matrix_to_axis_angle(R)
            assert np.allclose(r, theta * a, rtol=0, atol=theta * 1e-9 + 1e-15), (theta, r)
            assert np.allclose(_rodrigues(r), R, atol=1e-15), (theta, "reconstruction")


def test_axis_angle_near_pi():
    rng = np.random.default_rng(4)
    for eps in (1e-3, 1e-6, 1e-9):
        theta = np.pi - eps
        for _ in range(5):
            a = _unit(rng)
            R = _rodrigues(theta * a)
            r = rotation_matrix_to_axis_angle(R)
            assert np.allclose(_rodrigues(r), R, atol=1e-12), (eps, "reconstruction")
            assert np.allclose(r, theta * a, atol=1e-7), (eps, r, theta * a)
            assert np.linalg.norm(r) <= np.pi + 1e-12


def test_global_orient_random_round_trip():
    rng = np.random.default_rng(5)
    for _ in range(200):
        Rs, Rm = _random_rotation(rng), _random_rotation(rng)
        r = global_orient_from_torso(Rs, Rm)
        assert np.linalg.norm(r) <= np.pi + 1e-12
        # rodrigues(r) @ R_smpl_rest == R_mediapipe (also catches reversed order).
        assert np.allclose(_rodrigues(r) @ Rs, Rm, atol=1e-10)
        # Same result through smplx's own Rodrigues form (has a 1e-8 bias).
        assert np.allclose(_rodrigues_smplx(r) @ Rs, Rm, atol=1e-7)


def test_rotation_inputs_rejected():
    good = np.eye(3)
    bad = {
        "scaled": 1.01 * np.eye(3),
        "reflection": np.diag([1.0, 1.0, -1.0]),
        "shape (3,)": np.ones(3),
        "shape (4,4)": np.eye(4),
        "nan": np.where(np.eye(3) == 1, np.nan, 0.0),
        "string": np.full((3, 3), "a"),
        "shear": np.array([[1.0, 0.1, 0], [0, 1, 0], [0, 0, 1]]),
    }
    for name, B in bad.items():
        for fn in (lambda b=B: rotation_matrix_to_axis_angle(b),
                   lambda b=B: global_orient_from_torso(b, good),
                   lambda b=B: global_orient_from_torso(good, b)):
            try:
                _expect_raises(ValueError, fn)
            except AssertionError as e:
                raise AssertionError(f"{name}: {e}") from None


# --------------------------------------------------------------------------- #
# Step 5: torso-orientation conveniences, rotation_angle_deg
# --------------------------------------------------------------------------- #
def test_rotation_angle_known_values():
    rng = np.random.default_rng(6)
    R = _random_rotation(rng)
    assert rotation_angle_deg(R, R) < 1e-9
    for deg in (1e-6, 0.5, 90.0, 179.0, 180.0):
        for a in AXES.values():
            Rb = R @ _rodrigues(np.radians(deg) * a)
            got = rotation_angle_deg(R, Rb)
            assert abs(got - deg) < 1e-9, (deg, got)
            assert abs(rotation_angle_deg(Rb, R) - got) < 1e-9, "not symmetric"


def test_rotation_angle_rejects_invalid():
    for bad in (np.diag([1.0, 1.0, -1.0]), 2 * np.eye(3), np.eye(2), np.full((3, 3), np.nan)):
        _expect_raises(ValueError, lambda b=bad: rotation_angle_deg(b, np.eye(3)))
        _expect_raises(ValueError, lambda b=bad: rotation_angle_deg(np.eye(3), b))


def test_mediapipe_torso_orientation():
    assert MEDIAPIPE_TORSO_LANDMARKS == {"left_shoulder": 11, "right_shoulder": 12, "left_hip": 23, "right_hip": 24}
    rng = np.random.default_rng(7)
    world = rng.normal(scale=0.3, size=(33, 3))
    ls, rs, lh, rh = [np.array([p[0], -p[1], 0.1]) for p in CANON]  # front-facing
    world[11], world[12], world[23], world[24] = ls, rs, lh, rh
    R = mediapipe_torso_orientation(world)
    assert np.allclose(R, torso_orientation_matrix(ls, rs, lh, rh))
    assert np.allclose(R, np.diag([1.0, -1.0, -1.0]))
    _expect_raises(ValueError, lambda: mediapipe_torso_orientation(world[:32]), "shape")
    _expect_raises(ValueError, lambda: mediapipe_torso_orientation(world, min_angle_deg=0.0), "min_angle_deg")


def test_smpl_torso_orientation_from_joints():
    rng = np.random.default_rng(8)
    J = rng.normal(scale=0.3, size=(24, 3))
    J[16], J[17], J[1], J[2] = CANON
    assert np.allclose(smpl_torso_orientation_from_joints(J), np.eye(3))
    Q = _random_rotation(rng)
    assert np.allclose(smpl_torso_orientation_from_joints(J @ Q.T + 1.0), Q)
    for bad, match in ((J[:23], "shape"), (np.zeros((45, 3)), "shape"),
                       (J.astype(str), "real numbers")):
        _expect_raises(ValueError, lambda b=bad: smpl_torso_orientation_from_joints(b), match)


# --------------------------------------------------------------------------- #
# Step 6A: rotations between vectors, kinematic chain, swing-only arm init
# --------------------------------------------------------------------------- #
# Plausible SMPL-like rest skeleton (+x person's left, +y up, +z forward), with
# arms slightly off-axis like the official model.
REST = np.array([
    [0, 0, 0], [0.06, -0.09, 0], [-0.06, -0.09, 0], [0, 0.10, 0], [0.10, -0.45, 0], [-0.10, -0.45, 0],
    [0, 0.23, 0], [0.10, -0.85, -0.03], [-0.10, -0.85, -0.03], [0, 0.28, 0], [0.12, -0.90, 0.10],
    [-0.12, -0.90, 0.10], [0, 0.50, 0], [0.08, 0.40, 0], [-0.08, 0.40, 0], [0, 0.60, 0.05],
    [0.18, 0.42, 0], [-0.18, 0.42, 0], [0.45, 0.405, -0.03], [-0.45, 0.405, -0.025],
    [0.70, 0.415, -0.03], [-0.70, 0.41, -0.03], [0.78, 0.415, -0.03], [-0.78, 0.41, -0.03],
], dtype=float) + np.array([0.0, -0.2, 0.01])
ARM = {"left": ((11, 13, 15), (16, 18, 20)), "right": ((12, 14, 16), (17, 19, 21))}


def _fk_positions(J, Q):
    """Posed joint positions from rest joints and global rotations (smplx kinematics, no transl)."""
    P = np.empty_like(J)
    P[0] = J[0]
    for j in range(1, 24):
        p = SMPL_PARENTS[j]
        P[j] = P[p] + Q[p] @ (J[j] - J[p])
    return P


def _dir(v):
    return v / np.linalg.norm(v)


def _vec_angle_deg(u, v):
    u, v = _dir(u), _dir(v)
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(u, v)), u @ v))


def _world_from_joints(P, rng):
    """MediaPipe-like world landmarks whose arm landmarks sit on posed SMPL joints."""
    w = rng.normal(scale=0.3, size=(33, 3))
    for side, (mp_idx, sm_idx) in ARM.items():
        for m, s in zip(mp_idx, sm_idx):
            w[m] = P[s]
    return w


def test_smpl_parents_structure():
    assert len(SMPL_PARENTS) == 24 and SMPL_PARENTS[0] == -1
    assert all(0 <= SMPL_PARENTS[j] < j for j in range(1, 24))
    n = SMPL_JOINT_NAMES.index
    assert SMPL_PARENTS[n("left_shoulder")] == n("left_collar")
    assert SMPL_PARENTS[n("left_elbow")] == n("left_shoulder")
    assert SMPL_PARENTS[n("left_wrist")] == n("left_elbow")
    assert SMPL_PARENTS[n("right_elbow")] == n("right_shoulder")


def test_axis_angle_to_rotation_matrix():
    assert np.array_equal(axis_angle_to_rotation_matrix(np.zeros(3)), np.eye(3))
    rng = np.random.default_rng(10)
    for theta in (1e-10, 1e-5, 1e-4 * (1 - 1e-9), 1e-4 * (1 + 1e-9), 0.3, np.pi / 2, 3.0, np.pi):
        for _ in range(5):
            r = theta * _unit(rng)
            R = axis_angle_to_rotation_matrix(r)
            assert np.allclose(R, _rodrigues(r), atol=1e-14), theta
            _assert_rotation(R)
            assert np.allclose(axis_angle_to_rotation_matrix(rotation_matrix_to_axis_angle(R)), R, atol=1e-12)
    for bad in (np.zeros(2), [np.nan, 0, 0], np.array(["a", "b", "c"])):
        _expect_raises(ValueError, lambda b=bad: axis_angle_to_rotation_matrix(b))


def test_rotation_between_vectors_random():
    rng = np.random.default_rng(11)
    for _ in range(200):
        a, b = rng.normal(size=3) * rng.uniform(0.1, 10), rng.normal(size=3) * rng.uniform(0.1, 10)
        R = rotation_between_vectors(a, b)
        _assert_rotation(R, atol=1e-12)
        assert np.allclose(R @ _dir(a), _dir(b), atol=1e-12)
        r = rotation_matrix_to_axis_angle(R)
        assert abs(np.degrees(np.linalg.norm(r)) - _vec_angle_deg(a, b)) < 1e-9, "not the minimal angle"
        assert abs(r @ _dir(a)) < 1e-9 and abs(r @ _dir(b)) < 1e-9, "axis not perpendicular (twist added)"


def test_rotation_between_parallel_and_antiparallel():
    a = np.array([0.3, -0.5, 0.8])
    assert np.array_equal(rotation_between_vectors(a, 2.5 * a), np.eye(3))
    cases = {
        "perpendicular fallback": np.cross(a, [0, 0, 1.0]),
        "tilted fallback": np.array([0.0, 0.0, 1.0]),
        "fallback parallel to a": 3 * a,
        "no fallback": None,
    }
    for name, fb in cases.items():
        R = rotation_between_vectors(a, -a, fallback_axis=fb)
        _assert_rotation(R, atol=1e-12)
        assert np.allclose(R @ _dir(a), -_dir(a), atol=1e-12), name
        r = rotation_matrix_to_axis_angle(R)
        assert np.isclose(np.linalg.norm(r), np.pi), name
        assert abs(_dir(r) @ _dir(a)) < 1e-12, f"{name}: axis not perpendicular to a"
        if name in ("perpendicular fallback", "tilted fallback"):
            proj = fb - (fb @ _dir(a)) * _dir(a)
            assert np.allclose(np.cross(_dir(r), _dir(proj)), 0, atol=1e-12), f"{name}: not the projected fallback"
    # Nearly antiparallel (not exactly) still maps accurately.
    b = -a + np.array([1e-9, 0, 0])
    assert np.allclose(rotation_between_vectors(a, b) @ _dir(a), _dir(b), atol=1e-8)
    for bad in [(np.zeros(3), a), (a, np.zeros(3)), (a[:2], a), ([np.nan, 0, 1], a)]:
        _expect_raises(ValueError, lambda bb=bad: rotation_between_vectors(*bb))


def test_global_joint_rotations_chain():
    Q0 = global_joint_rotations(np.zeros(3), np.zeros(69))
    assert np.allclose(Q0, np.eye(3))
    rng = np.random.default_rng(12)
    go, bp = rng.normal(size=3), rng.normal(size=69) * 0.5
    Q = global_joint_rotations(go, bp)
    n = SMPL_JOINT_NAMES.index
    expected = axis_angle_to_rotation_matrix(go)
    for j in (3, 6, 9, 13, 16, 18):  # pelvis -> ... -> left elbow
        expected = expected @ axis_angle_to_rotation_matrix(bp[3 * (j - 1): 3 * j])
    assert np.allclose(Q[n("left_elbow")], expected, atol=1e-12)
    # A local elbow rotation changes the ORIENTATION of the elbow and its descendants only.
    bp2 = bp.copy()
    bp2[51:54] += [0.3, -0.2, 0.4]
    changed = {j for j in range(24) if not np.allclose(global_joint_rotations(go, bp2)[j], Q[j])}
    assert changed == {n("left_elbow"), n("left_wrist"), n("left_hand")}, changed
    _expect_raises(ValueError, lambda: global_joint_rotations(go, bp[:68]), "shape")


def test_fk_elbow_rotation_moves_descendants_not_itself():
    # POSITIONS: rotating the left elbow moves the left wrist and hand, not the elbow.
    rng = np.random.default_rng(13)
    go, bp = rng.normal(size=3), rng.normal(size=69) * 0.3
    P = _fk_positions(REST, global_joint_rotations(go, bp))
    bp2 = bp.copy()
    bp2[51:54] += [0.0, 0.5, 0.3]
    P2 = _fk_positions(REST, global_joint_rotations(go, bp2))
    moved = {j for j in range(24) if not np.allclose(P2[j], P[j], atol=1e-12)}
    n = SMPL_JOINT_NAMES.index
    assert moved == {n("left_wrist"), n("left_hand")}, moved


def test_init_arm_pose_round_trip():
    rng = np.random.default_rng(14)
    for side, (_, (s_s, s_e, s_w)) in ARM.items():
        for _ in range(50):
            go = _unit(rng) * rng.uniform(0, np.pi)
            bp_true = np.zeros(69)
            bp_true[[6, 7, 8, 15, 16, 17, 24, 25, 26]] = rng.normal(scale=0.2, size=9)  # spine1-3
            c = SMPL_PARENTS[s_s]                                                            # collar
            bp_true[3 * (c - 1): 3 * c] = rng.normal(scale=0.2, size=3)
            bp_true[3 * (s_s - 1): 3 * s_s] = _unit(rng) * rng.uniform(0, 2.5)              # shoulder
            bp_true[3 * (s_e - 1): 3 * s_e] = _unit(rng) * rng.uniform(0, 2.5)              # elbow
            P_true = _fk_positions(REST, global_joint_rotations(go, bp_true))
            world = _world_from_joints(P_true, rng)
            start = bp_true.copy()
            start[3 * (s_s - 1): 3 * s_s] = 0.0
            start[3 * (s_e - 1): 3 * s_e] = 0.0
            bp_est, diag = init_arm_pose(world, REST, go, side=side, body_pose=start)
            P = _fk_positions(REST, global_joint_rotations(go, bp_est))
            assert _vec_angle_deg(P[s_e] - P[s_s], P_true[s_e] - P_true[s_s]) < 1e-7, "upper arm"
            assert _vec_angle_deg(P[s_w] - P[s_e], P_true[s_w] - P_true[s_e]) < 1e-7, "forearm"
            # Twist ambiguity: estimated and true shoulder differ only by a twist about the rest bone.
            R_est = axis_angle_to_rotation_matrix(bp_est[3 * (s_s - 1): 3 * s_s])
            R_true = axis_angle_to_rotation_matrix(bp_true[3 * (s_s - 1): 3 * s_s])
            twist = rotation_matrix_to_axis_angle(R_est.T @ R_true)
            if np.linalg.norm(twist) > 1e-9:
                assert np.allclose(np.cross(_dir(twist), _dir(REST[s_e] - REST[s_s])), 0, atol=1e-7)
            assert np.isclose(diag["elbow_bend_mediapipe_deg"],
                              _vec_angle_deg(P_true[s_e] - P_true[s_s], P_true[s_w] - P_true[s_e]))


def test_init_arm_pose_needs_parent_frame():
    # A naive version that ignores the parent rotation gives a wrong arm direction.
    go = np.array([-2.9454, -0.1423, 0.2281])  # value estimated for the real test photo
    bp_true = np.zeros(69)
    bp_true[45:48] = [0.2, 0.3, -1.0]
    P_true = _fk_positions(REST, global_joint_rotations(go, bp_true))
    world = _world_from_joints(P_true, np.random.default_rng(15))
    d = P_true[18] - P_true[16]
    naive = np.zeros(69)
    naive[45:48] = rotation_matrix_to_axis_angle(rotation_between_vectors(REST[18] - REST[16], d))
    P_naive = _fk_positions(REST, global_joint_rotations(go, naive))
    assert _vec_angle_deg(P_naive[18] - P_naive[16], d) > 30.0, "naive version should fail"
    bp_est, _ = init_arm_pose(world, REST, go, "left")
    P = _fk_positions(REST, global_joint_rotations(go, bp_est))
    assert _vec_angle_deg(P[18] - P[16], d) < 1e-7


def test_init_arm_pose_preserves_other_entries():
    rng = np.random.default_rng(16)
    world = _world_from_joints(_fk_positions(REST, global_joint_rotations(np.zeros(3), np.zeros(69))), rng)
    world[13] += [0.0, 0.2, -0.1]  # bend the left arm
    world[14] += [0.1, 0.2, 0.0]   # bend the right arm
    start = rng.normal(size=69)
    start_copy = start.copy()
    for side, sl in (("left", [45, 46, 47, 51, 52, 53]), ("right", [48, 49, 50, 54, 55, 56])):
        bp, diag = init_arm_pose(world, REST, [0.1, 0.2, 0.3], side=side, body_pose=start)
        assert np.array_equal(start, start_copy), "input body_pose modified in place"
        others = np.setdiff1d(np.arange(69), sl)
        assert np.array_equal(bp[others], start[others]), f"{side}: unrelated entries changed"
        assert not np.allclose(bp[sl], start[sl]), f"{side}: arm entries not set"
        assert diag["shoulder_body_pose_slice"] == (sl[0], sl[2] + 1)
        assert diag["elbow_body_pose_slice"] == (sl[3], sl[5] + 1)


def test_init_arm_pose_straight_rest_and_degenerate():
    rng = np.random.default_rng(17)
    world = _world_from_joints(REST, rng)  # arms exactly in rest directions
    bp, diag = init_arm_pose(world, REST, np.zeros(3), "left")
    assert np.allclose(bp, 0, atol=1e-12) and diag["elbow_axis_vs_flexion_axis_deg"] is None
    bad = world.copy()
    bad[13] = bad[11]  # zero-length upper arm
    _expect_raises(ValueError, lambda: init_arm_pose(bad, REST, np.zeros(3)), "upper arm")
    _expect_raises(ValueError, lambda: init_arm_pose(world, REST, np.zeros(3), side="middle"), "side")
    _expect_raises(ValueError, lambda: init_arm_pose(world, REST[:23], np.zeros(3)), "rest_joints")
    _expect_raises(ValueError, lambda: init_arm_pose(world, REST, np.zeros(3), body_pose=np.zeros(68)), "shape")
    # Antiparallel target (arm pointing straight back across the body) is still matched.
    w2 = world.copy()
    w2[13] = w2[11] - (REST[18] - REST[16])
    bp2, _ = init_arm_pose(w2, REST, np.zeros(3), "left")
    P = _fk_positions(REST, global_joint_rotations(np.zeros(3), bp2))
    assert _vec_angle_deg(P[18] - P[16], w2[13] - w2[11]) < 1e-7


# --------------------------------------------------------------------------- #
# Step 6B: shoulder twist from the bend plane, elbow as a pure hinge
# --------------------------------------------------------------------------- #
def _rot_deg(R_a, R_b):
    return np.degrees(np.linalg.norm(rotation_matrix_to_axis_angle(R_a.T @ R_b)))


def _hinge_pose(rng, side, phi_deg_range=(20.0, 140.0), rest=REST):
    """Random body pose whose `side` elbow is a pure hinge about elbow_hinge_axis."""
    (_, (s_s, s_e, _)) = ARM[side]
    go = _unit(rng) * rng.uniform(0, np.pi)
    bp = np.zeros(69)
    bp[[6, 7, 8, 15, 16, 17, 24, 25, 26]] = rng.normal(scale=0.2, size=9)
    c = SMPL_PARENTS[s_s]
    bp[3 * (c - 1): 3 * c] = rng.normal(scale=0.2, size=3)
    bp[3 * (s_s - 1): 3 * s_s] = _unit(rng) * rng.uniform(0, 2.5)
    bp[3 * (s_e - 1): 3 * s_e] = np.radians(rng.uniform(*phi_deg_range)) * elbow_hinge_axis(rest, side)
    return go, bp


def _strip_arm(bp, side):
    (_, (s_s, s_e, _)) = ARM[side]
    out = bp.copy()
    out[3 * (s_s - 1): 3 * s_s] = 0.0
    out[3 * (s_e - 1): 3 * s_e] = 0.0
    return out


def test_elbow_hinge_axis_convention():
    for side, sign in (("left", -1.0), ("right", 1.0)):
        (_, (s_s, s_e, _)) = ARM[side]
        h = elbow_hinge_axis(REST, side)
        b1 = _dir(REST[s_e] - REST[s_s])
        assert np.isclose(np.linalg.norm(h), 1.0) and abs(h @ b1) < 1e-12
        assert sign * h[1] > 0.95, f"{side}: hinge should point {'up' if sign > 0 else 'down'}: {h}"
        # Positive rotation about h carries the upper-arm direction FORWARD (+z).
        assert (axis_angle_to_rotation_matrix(0.3 * h) @ b1)[2] > b1[2] + 0.1
    _expect_raises(ValueError, lambda: elbow_hinge_axis(REST, "middle"), "side")
    _expect_raises(ValueError, lambda: elbow_hinge_axis(REST[:23], "left"), "rest_joints")


def test_twist_preserves_upper_arm_direction():
    rng = np.random.default_rng(30)
    for side in ARM:
        (_, (s_s, s_e, _)) = ARM[side]
        go, bp_true = _hinge_pose(rng, side)
        P = _fk_positions(REST, global_joint_rotations(go, bp_true))
        bp_A, _ = init_arm_pose(_world_from_joints(P, rng), REST, go, side, _strip_arm(bp_true, side))
        d1 = P[s_e] - P[s_s]
        b1 = _dir(REST[s_e] - REST[s_s])
        for psi in np.linspace(-np.pi, np.pi, 13):
            bp = bp_A.copy()
            bp[3 * (s_s - 1): 3 * s_s] = rotation_matrix_to_axis_angle(
                axis_angle_to_rotation_matrix(bp_A[3 * (s_s - 1): 3 * s_s]) @ axis_angle_to_rotation_matrix(psi * b1))
            Pt = _fk_positions(REST, global_joint_rotations(go, bp))
            assert _vec_angle_deg(Pt[s_e] - Pt[s_s], d1) < 1e-9, psi


def test_refine_recovers_hinge_poses():
    rng = np.random.default_rng(31)
    for side in ARM:
        (_, (s_s, s_e, s_w)) = ARM[side]
        for _ in range(100):
            go, bp_true = _hinge_pose(rng, side)
            P_true = _fk_positions(REST, global_joint_rotations(go, bp_true))
            bp, diag = refine_arm_twist(_world_from_joints(P_true, rng), REST, go, _strip_arm(bp_true, side), side)
            assert diag["twist_resolved"], diag["reason"]
            P = _fk_positions(REST, global_joint_rotations(go, bp))
            assert _vec_angle_deg(P[s_e] - P[s_s], P_true[s_e] - P_true[s_s]) < 1e-9
            assert _vec_angle_deg(P[s_w] - P[s_e], P_true[s_w] - P_true[s_e]) < 1e-9
            # Full shoulder rotation INCLUDING twist is recovered (Increment A could not).
            sl = slice(3 * (s_s - 1), 3 * s_s)
            assert _rot_deg(axis_angle_to_rotation_matrix(bp[sl]), axis_angle_to_rotation_matrix(bp_true[sl])) < 1e-7
            assert np.allclose(bp[3 * (s_e - 1): 3 * s_e], bp_true[3 * (s_e - 1): 3 * s_e], atol=1e-9)


def test_refine_arbitrary_elbows_off_hinge_reduced():
    rng = np.random.default_rng(32)
    for side in ARM:
        (_, (s_s, s_e, s_w)) = ARM[side]
        h = elbow_hinge_axis(REST, side)
        off_A, off_B = [], []
        while len(off_B) < 100:
            go = _unit(rng) * rng.uniform(0, np.pi)
            bp_true = np.zeros(69)
            bp_true[3 * (s_s - 1): 3 * s_s] = _unit(rng) * rng.uniform(0, 2.5)
            bp_true[3 * (s_e - 1): 3 * s_e] = _unit(rng) * rng.uniform(0.4, 2.5)  # NOT a hinge
            P_true = _fk_positions(REST, global_joint_rotations(go, bp_true))
            world = _world_from_joints(P_true, rng)
            bp, diag = refine_arm_twist(world, REST, go, None, side)
            if not diag["twist_resolved"]:
                continue
            P = _fk_positions(REST, global_joint_rotations(go, bp))
            assert _vec_angle_deg(P[s_e] - P[s_s], P_true[s_e] - P_true[s_s]) < 1e-9
            assert _vec_angle_deg(P[s_w] - P[s_e], P_true[s_w] - P_true[s_e]) < 1e-9
            r_e = bp[3 * (s_e - 1): 3 * s_e]
            assert np.linalg.norm(np.cross(r_e, h)) < 1e-12, "elbow is not a pure hinge"
            off_A.append(diag["off_hinge_swing_only_deg"])
            off_B.append(diag["off_hinge_deg"])
        assert max(off_B) < 1e-9 and np.median(off_A) > 10.0, (np.median(off_A), max(off_B))


def _bent_world(rng, beta_deg, side="left"):
    """World landmarks with a given observed bend angle for one arm."""
    (mp_idx, _) = ARM[side]
    d1 = _unit(rng)
    perp = _dir(np.cross(d1, _unit(rng)))
    d2 = axis_angle_to_rotation_matrix(np.radians(beta_deg) * perp) @ d1
    w = rng.normal(scale=0.3, size=(33, 3))
    w[mp_idx[1]] = w[mp_idx[0]] + 0.25 * d1
    w[mp_idx[2]] = w[mp_idx[1]] + 0.25 * d2
    return w


def test_refine_straight_arm_threshold():
    rng = np.random.default_rng(33)
    go = np.array([0.2, -0.4, 0.1])
    for beta, resolved in ((14.9, False), (15.1, True), (100.0, True), (165.1, False), (170.0, False)):
        w = _bent_world(rng, beta)
        bp, diag = refine_arm_twist(w, REST, go, None, "left")
        assert diag["twist_resolved"] is resolved, (beta, diag["reason"])
        assert abs(diag["bend_deg"] - beta) < 1e-9
        if not resolved:
            bp_A, _ = init_arm_pose(w, REST, go, "left")
            assert np.array_equal(bp, bp_A), "fallback must equal Increment A exactly"
            assert diag["psi_deg"] is None and "ill-defined" in diag["reason"]
    _expect_raises(ValueError, lambda: refine_arm_twist(w, REST, go, straight_arm_deg=0.0), "straight_arm_deg")


def test_refine_mirror_symmetry():
    # Mirror x -> -x with left/right swapped: same hinge angle, mirrored shoulder.
    M = np.diag([-1.0, 1.0, 1.0])
    swap = {i: i for i in range(24)}
    for a, b in [(1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23)]:
        swap[a], swap[b] = b, a
    rest_sym = REST.copy()
    for j in (2, 5, 8, 11, 14, 17, 19, 21, 23):  # right side := mirror of left side
        rest_sym[j] = M @ REST[swap[j]]
    for j in (0, 3, 6, 9, 12, 15):
        rest_sym[j, 0] = 0.0
    rng = np.random.default_rng(34)
    for _ in range(20):
        go, bp_true = _hinge_pose(rng, "left", rest=rest_sym)
        P = _fk_positions(rest_sym, global_joint_rotations(go, bp_true))
        w_left = _world_from_joints(P, rng)
        w_right = w_left @ M.T
        w_right[[11, 13, 15, 12, 14, 16]] = (w_left @ M.T)[[12, 14, 16, 11, 13, 15]]
        go_m = go * np.array([1.0, -1.0, -1.0])  # M R M for axis-angle
        _, dl = refine_arm_twist(w_left, rest_sym, go, None, "left")
        _, dr = refine_arm_twist(w_right, rest_sym, go_m, None, "right")
        assert abs(dl["phi_deg"] - dr["phi_deg"]) < 1e-7, (dl["phi_deg"], dr["phi_deg"])
        assert abs(dl["psi_deg"] + dr["psi_deg"]) < 1e-7, (dl["psi_deg"], dr["psi_deg"])


def test_refine_phi_matches_brute_force():
    rng = np.random.default_rng(35)
    h = elbow_hinge_axis(REST, "left")
    b1, b2 = _dir(REST[18] - REST[16]), _dir(REST[20] - REST[18])
    phi0 = np.arctan2(h @ np.cross(b1, b2 - (b2 @ h) * h), b1 @ (b2 - (b2 @ h) * h))
    # Flexion branch only (phi + phi0 > 0); the other root is the hyperextension mirror.
    grid = np.radians(np.arange(-60.0, 180.0, 0.001))
    grid = grid[grid + phi0 > 0]
    for beta in (20.0, 45.0, 90.0, 140.0):
        _, diag = refine_arm_twist(_bent_world(rng, beta), REST, np.zeros(3), None, "left")
        # Brute force: hinge angles whose forearm makes the observed bend with b1.
        cos_b = np.array([b1 @ (axis_angle_to_rotation_matrix(p * h) @ b2) for p in grid[::100]])
        coarse = grid[::100][np.argmin(np.abs(cos_b - np.cos(np.radians(beta))))]
        fine = grid[np.abs(grid - coarse) < np.radians(0.2)]
        err = [abs(b1 @ (axis_angle_to_rotation_matrix(p * h) @ b2) - np.cos(np.radians(beta))) for p in fine]
        assert abs(np.degrees(fine[int(np.argmin(err))]) - diag["phi_deg"]) < 0.002, (beta, diag["phi_deg"])


def test_refine_preserves_other_entries():
    rng = np.random.default_rng(36)
    for side, sl in (("left", [45, 46, 47, 51, 52, 53]), ("right", [48, 49, 50, 54, 55, 56])):
        go, bp_true = _hinge_pose(rng, side)
        world = _world_from_joints(_fk_positions(REST, global_joint_rotations(go, bp_true)), rng)
        start = rng.normal(scale=0.3, size=69)
        start[sl] = 0.0
        start_copy = start.copy()
        bp, diag = refine_arm_twist(world, REST, go, start, side)
        assert np.array_equal(start, start_copy), "input modified in place"
        others = np.setdiff1d(np.arange(69), sl)
        assert np.array_equal(bp[others], start[others])
        assert diag["twist_resolved"] and diag["phi_plausible"] is not None


# --------------------------------------------------------------------------- #
# Full-body integration (Increment 1)
# --------------------------------------------------------------------------- #
LIMB_SLOTS = {  # body_pose slots written per limb: (proximal joint, middle joint)
    ("arm", "left"): (16, 18), ("arm", "right"): (17, 19), ("leg", "left"): (1, 4), ("leg", "right"): (2, 5)}


def _full_body_truth(rng):
    """Random full-body pose: spine/collars zero (torso rigid), hinge elbows/knees, plausible hips."""
    go = _unit(rng) * rng.uniform(0, np.pi)
    bp = np.zeros(69)
    for side in ("left", "right"):
        _, (s_s, s_e, _) = ARM[side]
        bp[3 * (s_s - 1): 3 * s_s] = _unit(rng) * rng.uniform(0, 2.0)
        bp[3 * (s_e - 1): 3 * s_e] = np.radians(rng.uniform(25, 120)) * elbow_hinge_axis(REST, side)
        _, leg_bp, _ = _plausible_leg_pose(rng, side)
        _, (h, k, _) = LEG[side]
        for j in (h, k):
            bp[3 * (j - 1): 3 * j] = leg_bp[3 * (j - 1): 3 * j]
    return go, bp


def _slots(limb_side):
    return [i for j in LIMB_SLOTS[limb_side] for i in range(3 * (j - 1), 3 * j)]


def test_full_body_matches_independent_initializers():
    rng = np.random.default_rng(50)
    for _ in range(30):
        go_true, bp_true = _full_body_truth(rng)
        w = _leg_world(_fk_positions(REST, global_joint_rotations(go_true, bp_true)), rng)
        go = global_orient_from_torso(smpl_torso_orientation_from_joints(REST), mediapipe_torso_orientation(w))
        for variant in ("twist", "swing"):
            out = initialize_smpl_from_mediapipe(w, REST, variant=variant)
            assert np.array_equal(out["global_orient"], go)
            for limb, side in FULL_BODY_LIMBS:
                if variant == "swing":
                    fn = init_arm_pose if limb == "arm" else init_leg_pose
                    bp_ind, d_ind = fn(w, REST, go, side=side)
                else:
                    fn = refine_arm_twist if limb == "arm" else refine_leg_twist
                    bp_ind, d_ind = fn(w, REST, go, body_pose=None, side=side)
                sl = _slots((limb, side))
                assert np.array_equal(out["body_pose"][sl], bp_ind[sl]), (variant, limb, side)
                assert out["limbs"][f"{limb}_{side}"] == d_ind, (variant, limb, side, "diagnostics")


def test_full_body_slots_and_unimplemented_zero():
    expected = {0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56}
    n = SMPL_JOINT_NAMES.index
    named = {i for name in ("left_hip", "right_hip", "left_knee", "right_knee", "left_shoulder",
                            "right_shoulder", "left_elbow", "right_elbow")
             for i in range(3 * (n(name) - 1), 3 * n(name))}
    assert named == expected, "slot table disagrees with SMPL joint names"
    rng = np.random.default_rng(51)
    for _ in range(20):
        go_true, bp_true = _full_body_truth(rng)
        w = _leg_world(_fk_positions(REST, global_joint_rotations(go_true, bp_true)), rng)
        for variant in ("twist", "swing"):
            out = initialize_smpl_from_mediapipe(w, REST, variant=variant)
            bp = out["body_pose"]
            assert bp.shape == (69,) and bp.dtype == np.float64 and out["global_orient"].shape == (3,)
            others = sorted(set(range(69)) - expected)
            assert np.all(bp[others] == 0.0), "unimplemented joints must be exactly zero"
            for limb_side in FULL_BODY_LIMBS:  # every written slot holds a real rotation
                assert np.linalg.norm(bp[_slots(limb_side)]) > 0
            diag = out["limbs"]
            for limb, side in FULL_BODY_LIMBS:
                prox, mid = ("shoulder", "elbow") if limb == "arm" else ("hip", "knee")
                a, b = diag[f"{limb}_{side}"][f"{prox}_body_pose_slice"]
                c, d = diag[f"{limb}_{side}"][f"{mid}_body_pose_slice"]
                assert list(range(a, b)) + list(range(c, d)) == _slots((limb, side))


def test_full_body_order_independent():
    import itertools

    rng = np.random.default_rng(52)
    go_true, bp_true = _full_body_truth(rng)
    w = _leg_world(_fk_positions(REST, global_joint_rotations(go_true, bp_true)), rng)
    for variant in ("twist", "swing"):
        ref = initialize_smpl_from_mediapipe(w, REST, variant=variant)
        for order in itertools.permutations(FULL_BODY_LIMBS):
            out = initialize_smpl_from_mediapipe(w, REST, limbs=order, variant=variant)
            assert np.array_equal(out["body_pose"], ref["body_pose"]), (variant, order)
            assert out["limbs"] == ref["limbs"]


def test_full_body_round_trip():
    rng = np.random.default_rng(53)
    n_exact = 0
    for _ in range(60):
        go_true, bp_true = _full_body_truth(rng)
        P_true = _fk_positions(REST, global_joint_rotations(go_true, bp_true))
        out = initialize_smpl_from_mediapipe(_leg_world(P_true, rng), REST)
        assert _rot_deg(axis_angle_to_rotation_matrix(out["global_orient"]),
                        axis_angle_to_rotation_matrix(go_true)) < 1e-7, "global_orient"
        P = _fk_positions(REST, global_joint_rotations(out["global_orient"], out["body_pose"]))
        for (_, (a, b, c)) in list(ARM.values()) + list(LEG.values()):
            assert _vec_angle_deg(P[b] - P[a], P_true[b] - P_true[a]) < 1e-7
            assert _vec_angle_deg(P[c] - P[b], P_true[c] - P_true[b]) < 1e-7
        if all(d["twist_resolved"] for d in out["limbs"].values()):
            n_exact += 1  # hinge truth + resolved twist -> the whole pose is recovered
            assert np.allclose(out["body_pose"], bp_true, atol=1e-7)
    assert n_exact > 30, n_exact


def test_full_body_subsets_and_validation():
    rng = np.random.default_rng(54)
    go_true, bp_true = _full_body_truth(rng)
    w = _leg_world(_fk_positions(REST, global_joint_rotations(go_true, bp_true)), rng)
    out = initialize_smpl_from_mediapipe(w, REST, limbs=[("leg", "left")])
    assert set(np.flatnonzero(out["body_pose"])) <= set(_slots(("leg", "left")))
    assert list(out["limbs"]) == ["leg_left"]
    assert not np.any(initialize_smpl_from_mediapipe(w, REST, limbs=())["body_pose"])
    # Thresholds are passed through: a huge straight-leg threshold forces the swing-only fallback.
    forced = initialize_smpl_from_mediapipe(w, REST, straight_leg_deg=89.0)
    assert not forced["limbs"]["leg_left"]["twist_resolved"]
    for bad, match in (({"variant": "hinge"}, "variant"), ({"limbs": [("arm", "middle")]}, "limb"),
                       ({"limbs": [("tail", "left")]}, "limb"), ({"limbs": [("arm", "left")] * 2}, "duplicate")):
        _expect_raises(ValueError, lambda b=bad: initialize_smpl_from_mediapipe(w, REST, **b), match)


# --------------------------------------------------------------------------- #
# Limb refactor: arm behaviour must stay bit-identical
# --------------------------------------------------------------------------- #
def test_arm_golden_regression():
    """Arm outputs recorded BEFORE the shared-limb refactor (tests/golden/arm_golden.json)."""
    import json

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden", "arm_golden.json")
    with open(path) as fh:
        cases = json.load(fh)["cases"]
    clean = lambda d: {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}  # noqa: E731
    assert len(cases) == 24
    for i, c in enumerate(cases):
        args = (np.array(c["world"]), np.array(c["rest"]), np.array(c["global_orient"]))
        bpA, dA = init_arm_pose(*args, c["side"], np.array(c["body_pose"]))
        bpB, dB = refine_arm_twist(*args, np.array(c["body_pose"]), c["side"])
        assert np.array_equal(bpA, np.array(c["A"])), f"case {i}: Increment A body_pose changed"
        assert np.array_equal(bpB, np.array(c["B"])), f"case {i}: Increment B body_pose changed"
        assert clean(dA) == c["A_diag"], f"case {i}: Increment A diagnostics changed"
        assert clean(dB) == c["B_diag"], f"case {i}: Increment B diagnostics changed"


# --------------------------------------------------------------------------- #
# Legs: swing-only (A) and hip twist + knee hinge (B)
# --------------------------------------------------------------------------- #
LEG = {"left": ((23, 25, 27), (1, 4, 7)), "right": ((24, 26, 28), (2, 5, 8))}
LEG_SIDES = ("left", "right")


def _leg_world(P, rng):
    """MediaPipe-like world landmarks with both legs (and arms) on posed SMPL joints."""
    w = _world_from_joints(P, rng)
    for mp_idx, sm_idx in LEG.values():
        for m, j in zip(mp_idx, sm_idx):
            w[m] = P[j]
    return w


def _strip_leg(bp, side):
    (_, (s_h, s_k, _)) = LEG[side]
    out = bp.copy()
    out[3 * (s_h - 1): 3 * s_h] = 0.0
    out[3 * (s_k - 1): 3 * s_k] = 0.0
    return out


def _plausible_leg_pose(rng, side, phi_deg=None, twist_deg=None, rest=REST):
    """Random pose: hip = swing (thigh within 60 deg of the rest thigh) @ twist about the rest thigh; knee hinge."""
    (_, (s_h, s_k, _)) = LEG[side]
    go = _unit(rng) * rng.uniform(0, np.pi)
    bp = rng.normal(scale=0.2, size=69)
    for k in (s_h, s_k):
        bp[3 * (k - 1): 3 * k] = 0.0
    b1 = _dir(rest[s_k] - rest[s_h])
    swing_axis = _dir(np.cross(b1, rng.normal(size=3)))
    target = axis_angle_to_rotation_matrix(np.radians(rng.uniform(0, 60)) * swing_axis) @ b1
    tau = rng.uniform(-50, 50) if twist_deg is None else twist_deg
    R_hip = rotation_between_vectors(b1, target) @ axis_angle_to_rotation_matrix(np.radians(tau) * b1)
    bp[3 * (s_h - 1): 3 * s_h] = rotation_matrix_to_axis_angle(R_hip)
    phi = rng.uniform(25, 120) if phi_deg is None else phi_deg
    bp[3 * (s_k - 1): 3 * s_k] = np.radians(phi) * knee_hinge_axis(rest, side)
    return go, bp, tau


def _leg_bent_world(rng, side, beta_deg, go, sign=1.0, rest=REST):
    """World whose thigh follows the rest thigh under global_orient and whose shin bends by
    beta about the posed knee hinge axis (sign=+1 flexion, -1 hyperextension)."""
    (mp_idx, (s_h, s_k, _)) = LEG[side]
    Rg = axis_angle_to_rotation_matrix(go)
    d1 = Rg @ _dir(rest[s_k] - rest[s_h])
    d2 = axis_angle_to_rotation_matrix(sign * np.radians(beta_deg) * (Rg @ knee_hinge_axis(rest, side))) @ d1
    w = rng.normal(scale=0.3, size=(33, 3))
    w[mp_idx[1]] = w[mp_idx[0]] + 0.4 * d1
    w[mp_idx[2]] = w[mp_idx[1]] + 0.4 * d2
    return w


def test_knee_hinge_axis_convention():
    for side in LEG_SIDES:
        (_, (s_h, s_k, _)) = LEG[side]
        h = knee_hinge_axis(REST, side)
        b1 = _dir(REST[s_k] - REST[s_h])
        assert np.isclose(np.linalg.norm(h), 1.0) and abs(h @ b1) < 1e-12
        assert h[0] > 0.95, f"{side}: knee hinge should point to +x for both legs: {h}"
        # Positive rotation about h carries the thigh direction BACKWARD (-z).
        assert (axis_angle_to_rotation_matrix(0.3 * h) @ b1)[2] < b1[2] - 0.1
    _expect_raises(ValueError, lambda: knee_hinge_axis(REST, "middle"), "side")


def test_leg_swing_round_trip_and_parent_frame():
    rng = np.random.default_rng(40)
    for side in LEG_SIDES:
        (_, (s_h, s_k, s_a)) = LEG[side]
        for _ in range(50):
            go = _unit(rng) * rng.uniform(0, np.pi)
            bp_true = rng.normal(scale=0.2, size=69)
            bp_true[3 * (s_h - 1): 3 * s_h] = _unit(rng) * rng.uniform(0, 2.0)
            bp_true[3 * (s_k - 1): 3 * s_k] = _unit(rng) * rng.uniform(0, 2.0)
            P_true = _fk_positions(REST, global_joint_rotations(go, bp_true))
            bp, diag = init_leg_pose(_leg_world(P_true, rng), REST, go, side, _strip_leg(bp_true, side))
            P = _fk_positions(REST, global_joint_rotations(go, bp))
            assert _vec_angle_deg(P[s_k] - P[s_h], P_true[s_k] - P_true[s_h]) < 1e-7, "thigh"
            assert _vec_angle_deg(P[s_a] - P[s_k], P_true[s_a] - P_true[s_k]) < 1e-7, "shin"
            sl = slice(3 * (s_h - 1), 3 * s_h)
            twist = rotation_matrix_to_axis_angle(
                axis_angle_to_rotation_matrix(bp[sl]).T @ axis_angle_to_rotation_matrix(bp_true[sl]))
            if np.linalg.norm(twist) > 1e-9:
                assert np.allclose(np.cross(_dir(twist), _dir(REST[s_k] - REST[s_h])), 0, atol=1e-7)
            assert diag["hip_body_pose_slice"] == (3 * (s_h - 1), 3 * s_h)
            assert diag["knee_body_pose_slice"] == (3 * (s_k - 1), 3 * s_k)
        # The hip's parent is the pelvis: ignoring global_orient gives the wrong thigh.
        go = np.array([-2.9454, -0.1423, 0.2281])
        bp_true = np.zeros(69)
        bp_true[3 * (s_h - 1): 3 * s_h] = [0.4, 0.1, -0.3]
        P_true = _fk_positions(REST, global_joint_rotations(go, bp_true))
        d = P_true[s_k] - P_true[s_h]
        naive = np.zeros(69)
        naive[3 * (s_h - 1): 3 * s_h] = rotation_matrix_to_axis_angle(
            rotation_between_vectors(REST[s_k] - REST[s_h], d))
        Pn = _fk_positions(REST, global_joint_rotations(go, naive))
        assert _vec_angle_deg(Pn[s_k] - Pn[s_h], d) > 30.0


def test_leg_refine_recovers_hinge_poses():
    rng = np.random.default_rng(41)
    for side in LEG_SIDES:
        (_, (s_h, s_k, s_a)) = LEG[side]
        n_ok = 0
        for _ in range(150):
            go, bp_true, tau = _plausible_leg_pose(rng, side)
            P_true = _fk_positions(REST, global_joint_rotations(go, bp_true))
            bp, diag = refine_leg_twist(_leg_world(P_true, rng), REST, go, _strip_leg(bp_true, side), side)
            P = _fk_positions(REST, global_joint_rotations(go, bp))
            assert _vec_angle_deg(P[s_k] - P[s_h], P_true[s_k] - P_true[s_h]) < 1e-7
            assert _vec_angle_deg(P[s_a] - P[s_k], P_true[s_a] - P_true[s_k]) < 1e-7
            if not diag["twist_resolved"]:
                assert "straight" in diag["reason"], diag["reason"]  # only small bends may fall back
                continue
            n_ok += 1
            sl = slice(3 * (s_h - 1), 3 * s_h)
            assert _rot_deg(axis_angle_to_rotation_matrix(bp[sl]), axis_angle_to_rotation_matrix(bp_true[sl])) < 1e-6
            assert np.allclose(bp[3 * (s_k - 1): 3 * s_k], bp_true[3 * (s_k - 1): 3 * s_k], atol=1e-9)
            assert abs(diag["twist_deg"] - diag["psi_deg"]) < 1e-6, "swing-twist twist should equal psi"
            assert abs(diag["twist_deg"] - tau) < 1e-6, "recovered hip twist differs from the truth"
            assert diag["bend_direction_deg"] > 0, "a flexed knee should bend backward"
        assert n_ok > 100, n_ok


def test_leg_refine_arbitrary_knees_off_hinge_reduced():
    rng = np.random.default_rng(42)
    for side in LEG_SIDES:
        (_, (s_h, s_k, s_a)) = LEG[side]
        h = knee_hinge_axis(REST, side)
        off_A, off_B = [], []
        while len(off_B) < 60:
            go, bp_true, _ = _plausible_leg_pose(rng, side)
            bp_true[3 * (s_k - 1): 3 * s_k] = rotation_matrix_to_axis_angle(
                axis_angle_to_rotation_matrix(bp_true[3 * (s_k - 1): 3 * s_k])
                @ axis_angle_to_rotation_matrix(rng.normal(scale=0.3, size=3)))  # perturbed: NOT a hinge
            P_true = _fk_positions(REST, global_joint_rotations(go, bp_true))
            bp, diag = refine_leg_twist(_leg_world(P_true, rng), REST, go, None, side)
            P = _fk_positions(REST, global_joint_rotations(go, bp))
            assert _vec_angle_deg(P[s_k] - P[s_h], P_true[s_k] - P_true[s_h]) < 1e-7
            assert _vec_angle_deg(P[s_a] - P[s_k], P_true[s_a] - P_true[s_k]) < 1e-7
            if diag["twist_resolved"]:
                assert np.linalg.norm(np.cross(bp[3 * (s_k - 1): 3 * s_k], h)) < 1e-12
                off_A.append(diag["off_hinge_swing_only_deg"])
                off_B.append(diag["off_hinge_deg"])
        assert max(off_B) < 1e-9 and np.median(off_A) > 10.0, (np.median(off_A), max(off_B))


def test_leg_thresholds_and_hyperextension():
    rng = np.random.default_rng(43)
    for side in LEG_SIDES:
        go = np.array([0.1, -0.2, 0.05])
        for beta, resolved in ((19.9, False), (20.1, True), (60.0, True)):
            w = _leg_bent_world(rng, side, beta, go)
            bp, diag = refine_leg_twist(w, REST, go, None, side)
            assert diag["twist_resolved"] is resolved, (beta, diag["reason"])
            assert abs(diag["bend_deg"] - beta) < 1e-9 and diag["bend_direction_deg"] > 0
            if not resolved:
                assert np.array_equal(bp, init_leg_pose(w, REST, go, side)[0]) and "straight" in diag["reason"]
            else:
                assert abs(diag["twist_deg"]) < 60.0
        # Hyperextension (shin bowed forward): forcing flexion needs a ~180 deg hip twist -> guard.
        w = _leg_bent_world(rng, side, 25.0, go, sign=-1.0)
        bp, diag = refine_leg_twist(w, REST, go, None, side)
        assert not diag["twist_resolved"] and "implausible hip twist" in diag["reason"], diag["reason"]
        assert abs(diag["twist_deg"]) > 150.0 and diag["bend_direction_deg"] < 0
        assert np.array_equal(bp, init_leg_pose(w, REST, go, side)[0])
        # The guard is configurable.
        _, d2 = refine_leg_twist(w, REST, go, None, side, max_hip_twist_deg=180.0)
        assert d2["twist_resolved"]
        _expect_raises(ValueError, lambda: refine_leg_twist(w, REST, go, straight_leg_deg=95.0), "straight_leg_deg")
        _expect_raises(ValueError, lambda: refine_leg_twist(w, REST, go, max_hip_twist_deg=0.0), "max twist")


def test_leg_mirror_symmetry():
    # Mirror x -> -x with left/right swapped: same knee flexion, opposite hip twist.
    M = np.diag([-1.0, 1.0, 1.0])
    pairs = [(1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23)]
    swap = {i: i for i in range(24)}
    for a, b in pairs:
        swap[a], swap[b] = b, a
    rest_sym = REST.copy()
    for j in (2, 5, 8, 11, 14, 17, 19, 21, 23):
        rest_sym[j] = M @ REST[swap[j]]
    for j in (0, 3, 6, 9, 12, 15):
        rest_sym[j, 0] = 0.0
    assert np.allclose(knee_hinge_axis(rest_sym, "left"), knee_hinge_axis(rest_sym, "right") * [1, -1, -1])
    rng = np.random.default_rng(45)
    mp_swap = [23, 25, 27, 24, 26, 28]
    for _ in range(20):
        go, bp_true, _ = _plausible_leg_pose(rng, "left", rest=rest_sym)
        P = _fk_positions(rest_sym, global_joint_rotations(go, bp_true))
        w_left = _leg_world(P, rng)
        w_right = w_left @ M.T
        w_right[mp_swap] = (w_left @ M.T)[[24, 26, 28, 23, 25, 27]]
        _, dl = refine_leg_twist(w_left, rest_sym, go, None, "left")
        _, dr = refine_leg_twist(w_right, rest_sym, go * np.array([1.0, -1.0, -1.0]), None, "right")
        assert dl["twist_resolved"] == dr["twist_resolved"]
        if dl["twist_resolved"]:
            assert abs(dl["phi_deg"] - dr["phi_deg"]) < 1e-7, (dl["phi_deg"], dr["phi_deg"])
            assert abs(dl["twist_deg"] + dr["twist_deg"]) < 1e-7, (dl["twist_deg"], dr["twist_deg"])
            assert abs(dl["bend_direction_deg"] - dr["bend_direction_deg"]) < 1e-7


def test_leg_and_arm_independence():
    rng = np.random.default_rng(44)
    for side in LEG_SIDES:
        (_, (s_h, s_k, _)) = LEG[side]
        go, bp_true, _ = _plausible_leg_pose(rng, side)
        bp_true[45:48] = [0.2, -0.1, -1.0]
        bp_true[51:54] = 0.6 * elbow_hinge_axis(REST, "left")
        P_true = _fk_positions(REST, global_joint_rotations(go, bp_true))
        w = _leg_world(P_true, rng)
        start = bp_true.copy()
        start[[45, 46, 47, 51, 52, 53]] = 0.0
        start = _strip_leg(start, side)
        start_copy = start.copy()
        arm_then_leg = refine_leg_twist(w, REST, go, refine_arm_twist(w, REST, go, start, "left")[0], side)[0]
        leg_then_arm = refine_arm_twist(w, REST, go, refine_leg_twist(w, REST, go, start, side)[0], "left")[0]
        assert np.array_equal(start, start_copy), "input modified in place"
        assert np.allclose(arm_then_leg, leg_then_arm, atol=1e-12), "order dependence between arm and leg"
        leg_sl = list(range(3 * (s_h - 1), 3 * s_h)) + list(range(3 * (s_k - 1), 3 * s_k))
        only_leg = refine_leg_twist(w, REST, go, start, side)[0]
        others = np.setdiff1d(np.arange(69), leg_sl)
        assert np.array_equal(only_leg[others], start[others]), "leg init changed unrelated entries"
        # Arm joint POSITIONS do not move when the leg is initialised (separate chains).
        Pa = _fk_positions(REST, global_joint_rotations(go, start))
        Pl = _fk_positions(REST, global_joint_rotations(go, only_leg))
        n = SMPL_JOINT_NAMES.index
        for j in ("left_shoulder", "left_elbow", "left_wrist", "right_wrist", "head"):
            assert np.allclose(Pa[n(j)], Pl[n(j)], atol=1e-12), j


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"[ok]   {name}")
        except Exception as e:
            failed += 1
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
