"""
MediaPipe-to-SMPL mapping utilities.

Step 1: joint correspondence table (pure Python, no dependencies).

Conventions:
  - "left" / "right" always refer to the *person's* left / right, in both
    MediaPipe Pose and SMPL.
  - MediaPipe Pose: 33 landmarks, official BlazePose index order.
  - SMPL: 24 kinematic joints, standard SMPL / smplx `SMPL` output order
    (the same order 3DPW uses for its 72-D `poses` vector).

Coordinate frames (see mediapipe_world_to_camera_aligned for details):
  - Camera frame: OpenCV, as in src/rendering.py -> +X right, +Y down,
    +Z forward (away from the camera), right-handed, meters;
    x_cam = R @ x_world + t, pixels = K @ (x/z, y/z, 1).
  - MediaPipe world landmarks: meters, origin at the hip midpoint (documented).
    Axis directions are NOT stated in MediaPipe's docs; this project ASSUMES
    they are aligned with the camera frame above (+X right, +Y down,
    +Z away from the camera). To be confirmed on a real image.
  - MediaPipe image landmarks: x, y normalised by image width / height
    (x right, y down); their z is a relative depth, not meters, and unused.
  - SMPL: its canonical (rest) orientation differs from the camera frame.
    That difference is a rotation and belongs in `global_orient`. It is never
    applied as a fixed flip of MediaPipe coordinates.
"""

import math
import os

import numpy as np

# --------------------------------------------------------------------------- #
# Landmark / joint names (index == position in the tuple)
# --------------------------------------------------------------------------- #
MEDIAPIPE_LANDMARK_NAMES = (
    "nose",              # 0
    "left_eye_inner",    # 1
    "left_eye",          # 2
    "left_eye_outer",    # 3
    "right_eye_inner",   # 4
    "right_eye",         # 5
    "right_eye_outer",   # 6
    "left_ear",          # 7
    "right_ear",         # 8
    "mouth_left",        # 9
    "mouth_right",       # 10
    "left_shoulder",     # 11
    "right_shoulder",    # 12
    "left_elbow",        # 13
    "right_elbow",       # 14
    "left_wrist",        # 15
    "right_wrist",       # 16
    "left_pinky",        # 17
    "right_pinky",       # 18
    "left_index",        # 19
    "right_index",       # 20
    "left_thumb",        # 21
    "right_thumb",       # 22
    "left_hip",          # 23
    "right_hip",         # 24
    "left_knee",         # 25
    "right_knee",        # 26
    "left_ankle",        # 27
    "right_ankle",       # 28
    "left_heel",         # 29
    "right_heel",        # 30
    "left_foot_index",   # 31
    "right_foot_index",  # 32
)

SMPL_JOINT_NAMES = (
    "pelvis",            # 0
    "left_hip",          # 1
    "right_hip",         # 2
    "spine1",            # 3
    "left_knee",         # 4
    "right_knee",        # 5
    "spine2",            # 6
    "left_ankle",        # 7
    "right_ankle",       # 8
    "spine3",            # 9
    "left_foot",         # 10
    "right_foot",        # 11
    "neck",              # 12
    "left_collar",       # 13
    "right_collar",      # 14
    "head",              # 15
    "left_shoulder",     # 16
    "right_shoulder",    # 17
    "left_elbow",        # 18
    "right_elbow",       # 19
    "left_wrist",        # 20
    "right_wrist",       # 21
    "left_hand",         # 22
    "right_hand",        # 23
)

NUM_MEDIAPIPE_LANDMARKS = len(MEDIAPIPE_LANDMARK_NAMES)  # 33
NUM_SMPL_JOINTS = len(SMPL_JOINT_NAMES)                  # 24

# --------------------------------------------------------------------------- #
# Correspondences
# --------------------------------------------------------------------------- #
# Direct anatomical correspondences: (MediaPipe index, SMPL index).
_DIRECT_CORRESPONDENCES = (
    (11, 16),  # left_shoulder  -> left_shoulder
    (12, 17),  # right_shoulder -> right_shoulder
    (13, 18),  # left_elbow     -> left_elbow
    (14, 19),  # right_elbow    -> right_elbow
    (15, 20),  # left_wrist     -> left_wrist
    (16, 21),  # right_wrist    -> right_wrist
    (23, 1),   # left_hip       -> left_hip
    (24, 2),   # right_hip      -> right_hip
    (25, 4),   # left_knee      -> left_knee
    (26, 5),   # right_knee     -> right_knee
    (27, 7),   # left_ankle     -> left_ankle
    (28, 8),   # right_ankle    -> right_ankle
)

# NOT used by get_mediapipe_to_smpl_correspondence(). Recorded here so later
# steps can implement them explicitly (they need averaging or offsets).
# Format: SMPL joint name -> (MediaPipe landmark names, note)
APPROXIMATE_CORRESPONDENCES = {
    "pelvis": (("left_hip", "right_hip"), "virtual: hip midpoint; SMPL pelvis sits higher"),
    "neck": (("left_shoulder", "right_shoulder"), "virtual: shoulder midpoint"),
    "head": (("left_ear", "right_ear"), "virtual: ear midpoint, rough head centre"),
    "left_foot": (("left_foot_index",), "approximate: toe landmark vs. SMPL ball-of-foot joint"),
    "right_foot": (("right_foot_index",), "approximate: toe landmark vs. SMPL ball-of-foot joint"),
}


def get_mediapipe_to_smpl_correspondence():
    """
    Direct MediaPipe -> SMPL joint correspondences.

    Returns:
        (mp_indices, smpl_indices): two aligned tuples of ints of equal length,
        so that MediaPipe landmark mp_indices[k] corresponds to SMPL joint
        smpl_indices[k]. Usage with arrays/tensors:
            mp_pts[..., mp_indices, :]  <->  smpl_joints[..., smpl_indices, :]
    """
    mp_indices = tuple(mp for mp, _ in _DIRECT_CORRESPONDENCES)
    smpl_indices = tuple(smpl for _, smpl in _DIRECT_CORRESPONDENCES)
    return mp_indices, smpl_indices


# --------------------------------------------------------------------------- #
# MediaPipe result -> arrays
# --------------------------------------------------------------------------- #
def _validate_image_hw(image_hw):
    """Return (H, W) as positive ints, or raise ValueError."""
    try:
        h, w = image_hw
    except (TypeError, ValueError):
        raise ValueError(f"image_hw must be a (height, width) pair, got {image_hw!r}") from None
    for name, v in (("height", h), ("width", w)):
        if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)):
            raise ValueError(f"image {name} must be a number, got {v!r}")
        if not math.isfinite(v) or v <= 0 or v != int(v):
            raise ValueError(f"image {name} must be a positive integer, got {v!r}")
    return int(h), int(w)


def _select_pose(container, field, person_index):
    """
    Return the list of landmarks for one person from either result format.

    Legacy Solutions: container is a LandmarkList-like object with `.landmark`
                      (None when no person was detected).
    Tasks API:        container is a list with one landmark list per person
                      (empty when no person was detected).
    """
    if container is None:
        raise ValueError(f"result.{field} is None: no pose detected")
    if hasattr(container, "landmark"):  # legacy Solutions
        if person_index != 0:
            raise ValueError("legacy Solutions results contain one person; person_index must be 0")
        return list(container.landmark)
    if isinstance(container, (list, tuple)):  # Tasks API
        if len(container) == 0:
            raise ValueError(f"result.{field} is empty: no pose detected")
        if not 0 <= person_index < len(container):
            raise ValueError(
                f"person_index {person_index} out of range: {len(container)} pose(s) detected"
            )
        return list(container[person_index])
    raise TypeError(
        f"result.{field} has unsupported type {type(container).__name__}; expected a "
        "legacy LandmarkList (with .landmark) or a Tasks-API list of landmark lists"
    )


def _landmarks_to_array(landmarks, fields, field_name):
    """Stack `fields` of each landmark into a (33, len(fields)) float32 array."""
    if len(landmarks) != NUM_MEDIAPIPE_LANDMARKS:
        raise ValueError(
            f"{field_name}: expected {NUM_MEDIAPIPE_LANDMARKS} landmarks, got {len(landmarks)}"
        )
    rows = []
    for i, lm in enumerate(landmarks):
        row = []
        for f in fields:
            v = getattr(lm, f, None)
            if v is None:
                raise ValueError(
                    f"{field_name}[{i}] ({MEDIAPIPE_LANDMARK_NAMES[i]}) is missing '{f}'"
                )
            row.append(v)
        rows.append(row)
    arr = np.asarray(rows, dtype=np.float32)
    if not np.isfinite(arr).all():
        raise ValueError(f"{field_name} contains non-finite values")
    return arr


def mediapipe_to_arrays(result, image_hw, person_index=0):
    """
    Extract world landmarks, pixel landmarks and visibility from a MediaPipe Pose result.

    Accepted result formats (detected automatically):
      - Legacy Solutions API (`mp.solutions.pose.Pose().process(rgb)`):
            result.pose_landmarks.landmark        -> 33 NormalizedLandmark
            result.pose_world_landmarks.landmark  -> 33 Landmark
      - Tasks API (`mp.tasks.vision.PoseLandmarker.detect(...)`, PoseLandmarkerResult):
            result.pose_landmarks[person]         -> 33 NormalizedLandmark
            result.pose_world_landmarks[person]   -> 33 Landmark
        `person_index` selects the person (Tasks API only; must be 0 for legacy).

    Coordinates are returned in MediaPipe's own conventions; no axis flips:
      - world:  meters, origin at the hip midpoint, as MediaPipe defines it.
      - pixels: x = x_norm * W, y = y_norm * H (x right, y down). Not clipped:
                off-screen landmarks may fall outside [0, W) x [0, H).
      - The image landmarks' `z` (relative depth, not meters) is ignored.

    Args:
        result:       a MediaPipe Pose result (either format above).
        image_hw:     (height, width) in pixels of the image passed to MediaPipe.
        person_index: which detected person to use (Tasks API).

    Returns:
        world:      (33, 3) float32, meters.
        pixels:     (33, 2) float32, pixel coordinates.
        visibility: (33,)   float32, MediaPipe visibility scores of the image landmarks.
        Rows follow MEDIAPIPE_LANDMARK_NAMES.

    Raises:
        ValueError: invalid image_hw, no pose detected, wrong landmark count,
                    missing fields, non-finite values, bad person_index.
        TypeError:  result is not in a recognised MediaPipe format.
    """
    h, w = _validate_image_hw(image_hw)
    for field in ("pose_landmarks", "pose_world_landmarks"):
        if not hasattr(result, field):
            raise TypeError(f"result has no '{field}' attribute; is it a MediaPipe Pose result?")

    image_lms = _select_pose(result.pose_landmarks, "pose_landmarks", person_index)
    world_lms = _select_pose(result.pose_world_landmarks, "pose_world_landmarks", person_index)

    image = _landmarks_to_array(image_lms, ("x", "y", "visibility"), "pose_landmarks")
    world = _landmarks_to_array(world_lms, ("x", "y", "z"), "pose_world_landmarks")

    pixels = image[:, :2] * np.array([w, h], dtype=np.float32)
    visibility = image[:, 2].copy()
    return world, pixels, visibility


# --------------------------------------------------------------------------- #
# Coordinate frames
# --------------------------------------------------------------------------- #
def mediapipe_world_to_camera_aligned(world):
    """
    Return MediaPipe world landmarks as camera-aligned coordinates (no change of values).

    This is an explicit identity mapping. It exists to state, in one place, the
    project's convention that MediaPipe world axes already match the OpenCV
    camera frame used by src/rendering.py:
        +X right, +Y down, +Z away from the camera, meters.
    No axis permutation, sign flip or rotation is applied. SMPL's different rest
    orientation is handled later through `global_orient`.

    "Camera-aligned" means the AXES match the camera; the ORIGIN is still the
    MediaPipe hip midpoint, not the camera centre. Translation into the camera
    (so that z > 0) must be estimated separately before rendering/projecting.

    Args:
        world: (33, 3) array-like of MediaPipe world landmarks (meters), rows in
               MEDIAPIPE_LANDMARK_NAMES order, e.g. from mediapipe_to_arrays().

    Returns:
        (33, 3) numpy array: a copy of `world` with identical values. Floating
        dtypes are preserved; integer input is converted to float32.

    Raises:
        TypeError:  non-numeric input (strings, objects, booleans, complex).
        ValueError: wrong shape or non-finite values.
    """
    arr = np.asarray(world)
    if arr.dtype.kind not in "fiu":
        raise TypeError(f"world landmarks must be real numbers, got dtype {arr.dtype}")
    if arr.shape != (NUM_MEDIAPIPE_LANDMARKS, 3):
        raise ValueError(
            f"world landmarks must have shape ({NUM_MEDIAPIPE_LANDMARKS}, 3), got {arr.shape}"
        )
    if not np.isfinite(arr).all():
        raise ValueError("world landmarks contain non-finite values")
    out_dtype = arr.dtype if arr.dtype.kind == "f" else np.float32
    return np.array(arr, dtype=out_dtype, copy=True)


# --------------------------------------------------------------------------- #
# Torso orientation
# --------------------------------------------------------------------------- #
def _as_point3(p, name):
    """Validate one 3D point and return it as a float64 (3,) array."""
    arr = np.asarray(p)
    if arr.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real numbers, got dtype {arr.dtype}")
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {arr.shape}")
    arr = arr.astype(np.float64)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return arr


def torso_orientation_matrix(left_shoulder, right_shoulder, left_hip, right_hip,
                             min_length=1e-6, min_angle_deg=5.0):
    """
    Estimate a right-handed orthonormal torso frame from four 3D landmarks.

    The four points must be in the same coordinate frame (any frame: e.g.
    MediaPipe camera-aligned landmarks 11, 12, 23, 24, or SMPL joints
    16, 17, 1, 2). "left"/"right" are the person's own sides.

    Body axes (columns of the result, expressed in the input frame):
        y_body (up):      spine direction, from hip midpoint to shoulder midpoint,
                          up = (ls + rs)/2 - (lh + rh)/2. PRIMARY axis: kept exactly.
        z_body (forward): ESTIMATED torso-forward direction,
                          z = normalize(cross(lateral, y_body)),
                          lateral = (ls - rs) + (lh - rh)  (points to person's left).
                          It is derived from the cross product of the lateral and
                          spine directions, NOT a measurement of chest orientation.
        x_body (left):    x = cross(y_body, z_body), i.e. `lateral` with its
                          spine component removed (Gram-Schmidt), normalised.
    Cross-product order: z = lateral x up, then x = up x z, so x x y = z:
    the frame is right-handed and det(R) = +1 by construction.

    With (left, up, forward) body axes, a torso already in SMPL's rest convention
    (+Y up, +Z forward, hence +X = person's left) gives R = I.

    Args:
        left_shoulder, right_shoulder, left_hip, right_hip: (3,) array-likes.
        min_length:    minimum length of the spine and lateral vectors (input units).
        min_angle_deg: minimum angle (degrees) between the lateral and spine
                       directions; below this, "forward" is ill-defined.

    Returns:
        R: (3, 3) float64, columns [x_body, y_body, z_body]; v_input = R @ v_body.

    Raises:
        ValueError: bad shape / non-finite input, or near-degenerate torso
                    (collapsed spine, collapsed lateral, lateral ~ parallel to spine).
    """
    if not min_length > 0:
        raise ValueError(f"min_length must be > 0, got {min_length!r}")
    if not 0 < min_angle_deg < 90:
        raise ValueError(f"min_angle_deg must be in (0, 90), got {min_angle_deg!r}")

    ls = _as_point3(left_shoulder, "left_shoulder")
    rs = _as_point3(right_shoulder, "right_shoulder")
    lh = _as_point3(left_hip, "left_hip")
    rh = _as_point3(right_hip, "right_hip")

    up = 0.5 * (ls + rs) - 0.5 * (lh + rh)
    lateral = (ls - rs) + (lh - rh)

    up_len = np.linalg.norm(up)
    lat_len = np.linalg.norm(lateral)
    if up_len < min_length:
        raise ValueError(
            f"degenerate torso: spine vector length {up_len:.3g} < min_length {min_length:.3g}"
        )
    if lat_len < min_length:
        raise ValueError(
            f"degenerate torso: lateral vector length {lat_len:.3g} < min_length {min_length:.3g}"
        )

    y = up / up_len
    forward = np.cross(lateral / lat_len, y)  # |forward| = sin(angle(lateral, up))
    sin_angle = np.linalg.norm(forward)
    if sin_angle < math.sin(math.radians(min_angle_deg)):
        angle = math.degrees(math.asin(min(sin_angle, 1.0)))
        raise ValueError(
            f"degenerate torso: lateral and spine directions are {angle:.2f} deg from "
            f"parallel (min_angle_deg={min_angle_deg})"
        )
    z = forward / sin_angle
    x = np.cross(y, z)
    return np.stack([x, y, z], axis=1)


# --------------------------------------------------------------------------- #
# Rotations: matrix -> axis-angle, global_orient initialisation
# --------------------------------------------------------------------------- #
def _check_rotation(R, name, atol):
    """Validate a proper rotation matrix (3x3, finite, orthonormal, det +1)."""
    arr = np.asarray(R)
    if arr.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real numbers, got dtype {arr.dtype}")
    if arr.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3), got {arr.shape}")
    arr = arr.astype(np.float64)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    ortho_err = np.abs(arr.T @ arr - np.eye(3)).max()
    if ortho_err > atol:
        raise ValueError(f"{name} is not orthonormal (max |R^T R - I| = {ortho_err:.3g})")
    det = np.linalg.det(arr)
    if abs(det - 1.0) > atol:
        raise ValueError(f"{name} is not a proper rotation (det = {det:.6g}; reflection?)")
    return arr


# Below this angle (radians) theta / sin(theta/2) uses its Taylor series.
_SMALL_ANGLE = 1e-3
# If the quaternion's scalar part is below this, the rotation is treated as
# exactly 180 deg (theta within ~2e-12 rad of pi) and r's sign is canonicalised.
_HALF_TURN_W = 1e-12


def rotation_matrix_to_axis_angle(R, atol=1e-6):
    """
    Convert a rotation matrix to axis-angle r = theta * axis, theta in [0, pi].

    Inverse of the Rodrigues formula used by smplx (batch_rodrigues):
        R = I + sin(theta) K + (1 - cos(theta)) K^2,  K = skew(axis).

    Method: matrix -> unit quaternion (Shepperd's method, branching on the
    largest of trace(R), R00, R11, R22 so no division by a small number), sign
    fixed so w >= 0, then theta = 2 * atan2(|q_v|, w) (accurate near 0 and pi)
    and r = q_v * theta / sin(theta/2), with a Taylor series for small theta.

    At exactly 180 deg, r and -r are the same rotation. The sign is then chosen
    so the first non-zero component of r is positive (deterministic output).

    Args:
        R:    (3, 3) proper rotation matrix.
        atol: tolerance for the orthonormality / determinant checks.

    Returns:
        r: (3,) float64 axis-angle vector, |r| = theta in [0, pi].

    Raises:
        ValueError: bad shape / dtype, non-finite, not orthonormal, or det != +1.
    """
    R = _check_rotation(R, "R", atol)
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    k = int(np.argmax([tr, m[0, 0], m[1, 1], m[2, 2]]))
    if k == 0:
        w = 0.5 * math.sqrt(max(1.0 + tr, 0.0))
        s = 0.25 / w
        q = np.array([w, (m[2, 1] - m[1, 2]) * s, (m[0, 2] - m[2, 0]) * s, (m[1, 0] - m[0, 1]) * s])
    elif k == 1:
        x = 0.5 * math.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 0.0))
        s = 0.25 / x
        q = np.array([(m[2, 1] - m[1, 2]) * s, x, (m[0, 1] + m[1, 0]) * s, (m[0, 2] + m[2, 0]) * s])
    elif k == 2:
        y = 0.5 * math.sqrt(max(1.0 - m[0, 0] + m[1, 1] - m[2, 2], 0.0))
        s = 0.25 / y
        q = np.array([(m[0, 2] - m[2, 0]) * s, (m[0, 1] + m[1, 0]) * s, y, (m[1, 2] + m[2, 1]) * s])
    else:
        z = 0.5 * math.sqrt(max(1.0 - m[0, 0] - m[1, 1] + m[2, 2], 0.0))
        s = 0.25 / z
        q = np.array([(m[1, 0] - m[0, 1]) * s, (m[0, 2] + m[2, 0]) * s, (m[1, 2] + m[2, 1]) * s, z])

    q /= np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    w, qv = q[0], q[1:]
    n = np.linalg.norm(qv)
    theta = 2.0 * math.atan2(n, w)

    if theta < _SMALL_ANGLE:
        scale = 2.0 + theta ** 2 / 12.0 + 7.0 * theta ** 4 / 2880.0  # theta / sin(theta/2)
    else:
        scale = theta / n
    r = scale * qv

    if w < _HALF_TURN_W:  # 180 deg: canonical sign
        nz = np.flatnonzero(np.abs(r) > 1e-12)
        if nz.size and r[nz[0]] < 0:
            r = -r
    return r


def global_orient_from_torso(R_smpl_rest, R_mediapipe, atol=1e-6):
    """
    Initial SMPL `global_orient` (axis-angle) from two torso orientation matrices.

    Both inputs come from torso_orientation_matrix(), so their COLUMNS share the
    same body axes (left, up, forward), but each is expressed in its own frame:
        R_smpl_rest: SMPL model frame. Built from SMPL joints 16, 17, 1, 2 with
                     global_orient = 0 and body_pose = 0 (any betas).
        R_mediapipe: camera-aligned frame. Built from MediaPipe landmarks
                     11, 12, 23, 24.
    The rotation from the SMPL model frame to the camera frame is
        R_global = R_mediapipe @ R_smpl_rest.T     (so R_global @ R_smpl_rest = R_mediapipe)
    and is returned as SMPL's 3-component axis-angle. No orientation is hard-coded.

    Note: smplx applies global_orient about the rest pelvis joint, not the origin;
    translation is handled separately.

    Returns:
        (3,) float64 axis-angle, |r| in [0, pi].

    Raises:
        ValueError: either input is not a valid proper rotation matrix.
    """
    Rs = _check_rotation(R_smpl_rest, "R_smpl_rest", atol)
    Rm = _check_rotation(R_mediapipe, "R_mediapipe", atol)
    return rotation_matrix_to_axis_angle(Rm @ Rs.T, atol=atol)


# --------------------------------------------------------------------------- #
# SMPL model loading and forward pass (torch / smplx imported lazily, so the
# rest of this module stays NumPy-only)
# --------------------------------------------------------------------------- #
SMPL_TORSO_JOINTS = {
    name: SMPL_JOINT_NAMES.index(name)
    for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
}  # {"left_shoulder": 16, "right_shoulder": 17, "left_hip": 1, "right_hip": 2}

_SMPL_BODY_POSE_DIM = 3 * (NUM_SMPL_JOINTS - 1)  # 69
_SMPL_GENDERS = ("neutral", "male", "female")


def load_smpl(model_path, gender="neutral", device="cpu"):
    """
    Load a Chumpy-free SMPL model with smplx.SMPL.

    LICENSE: the SMPL model is under the MPI-IS non-commercial license and must
    not be redistributed. Never commit the model file (*.pkl / *.npz are
    gitignored); pass its location explicitly, e.g. in Colab
    "/content/drive/MyDrive/smpl_models_clean/SMPL_NEUTRAL.pkl".

    The official pickle must have been converted once to plain NumPy arrays
    (no `chumpy` objects); loading an unconverted file raises ImportError.

    Args:
        model_path: path to the .pkl file (recommended), or a directory that
                    contains SMPL_{GENDER}.pkl (smplx's naming convention).
        gender:     "neutral", "male" or "female".
        device:     torch device for the model buffers.

    Returns:
        smplx.SMPL module in eval mode with NO trainable parameters
        (all pose / shape / translation inputs go through smpl_forward()).

    Raises:
        ValueError:        unknown gender or invalid device.
        FileNotFoundError: model file not found.
        ImportError:       the pickle still references chumpy.
    """
    import torch
    import smplx

    if gender not in _SMPL_GENDERS:
        raise ValueError(f"gender must be one of {_SMPL_GENDERS}, got {gender!r}")
    try:
        device = torch.device(device)
    except (RuntimeError, TypeError) as e:
        raise ValueError(f"invalid device {device!r}: {e}") from None

    path = os.fspath(model_path)
    if os.path.isdir(path):
        path = os.path.join(path, f"SMPL_{gender.upper()}.pkl")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"SMPL model file not found: {path}")

    try:
        model = smplx.SMPL(
            model_path=path, gender=gender, batch_size=1,
            create_global_orient=False, create_body_pose=False,
            create_betas=False, create_transl=False,
        )
    except ModuleNotFoundError as e:
        if (e.name or "").split(".")[0] == "chumpy":
            raise ImportError(
                f"{path} still contains chumpy objects. Convert it once to a "
                "Chumpy-free pickle (plain NumPy arrays) before loading."
            ) from e
        raise
    return model.to(device).eval()


def _to_model_tensor(value, name, width, device, dtype):
    """Validate one SMPL input and return a (B, width) tensor on the model's device/dtype."""
    import torch

    if torch.is_tensor(value):
        if not value.is_floating_point():
            raise ValueError(f"{name} must be a floating-point tensor, got {value.dtype}")
        if value.device != device:
            raise ValueError(
                f"{name} is on device {value.device}, but the SMPL model is on device {device}"
            )
        t = value.to(dtype)  # differentiable cast; keeps the autograd graph
    else:
        arr = np.asarray(value)
        if arr.dtype.kind not in "fiu":
            raise ValueError(f"{name} must contain real numbers, got dtype {arr.dtype}")
        t = torch.as_tensor(arr, dtype=dtype, device=device)

    if t.dim() == 1:
        t = t.unsqueeze(0)
    if t.dim() != 2 or t.shape[1] != width or t.shape[0] < 1:
        raise ValueError(
            f"{name} must have shape ({width},) or (B, {width}), got {tuple(np.shape(value))}"
        )
    return t


def smpl_forward(model, global_orient=None, body_pose=None, betas=None, transl=None):
    """
    Run SMPL and return vertices, the 24 kinematic joints, and faces.

    Inputs (torch tensors or array-likes; 1-D inputs are treated as batch 1):
        global_orient: (B, 3)  axis-angle, rotation about the rest pelvis joint.
        body_pose:     (B, 69) axis-angle for joints 1..23.
        betas:         (B, model.num_betas) shape coefficients.
        transl:        (B, 3)  translation added after skinning.
    Missing inputs are zeros. All given inputs must share the same batch size B.
    Tensors must be floating point and on the model's device; they are cast to
    the model dtype differentiably. Gradients are preserved (no detach/no_grad).
    Values are not checked for finiteness (to avoid a GPU sync per call).

    Returns:
        dict with
          "vertices": (B, 6890, 3) tensor
          "joints":   (B, 24, 3) tensor, SMPL_JOINT_NAMES order. smplx appends
                      extra vertex-based landmarks after these; they are dropped.
          "faces":    (F, 3) long tensor
    """
    import torch

    device, dtype = model.shapedirs.device, model.shapedirs.dtype
    specs = (
        ("global_orient", global_orient, 3),
        ("body_pose", body_pose, _SMPL_BODY_POSE_DIM),
        ("betas", betas, model.num_betas),
        ("transl", transl, 3),
    )
    inputs, batch = {}, None
    for name, value, width in specs:
        if value is None:
            continue
        t = _to_model_tensor(value, name, width, device, dtype)
        if batch is None:
            batch = t.shape[0]
        elif t.shape[0] != batch:
            raise ValueError(f"batch size mismatch: {name} has {t.shape[0]}, expected {batch}")
        inputs[name] = t
    batch = 1 if batch is None else batch
    for name, _, width in specs:
        if name not in inputs:
            inputs[name] = torch.zeros(batch, width, device=device, dtype=dtype)

    out = model(**inputs, return_verts=True)
    return {
        "vertices": out.vertices,
        "joints": out.joints[:, :NUM_SMPL_JOINTS],
        "faces": model.faces_tensor,
    }


def smpl_rest_torso_orientation(model, betas=None):
    """
    Torso orientation of SMPL's rest pose, for global_orient_from_torso().

    Runs SMPL with global_orient = 0 and body_pose = 0 (and the given betas, for
    ONE subject) under torch.no_grad(), then applies torso_orientation_matrix to
    joints 16, 17, 1, 2 (left/right shoulder, left/right hip).

    Args:
        model: SMPL model from load_smpl().
        betas: None, (num_betas,) or (1, num_betas).

    Returns:
        (3, 3) float64 NumPy rotation matrix, in SMPL model coordinates.
    """
    import torch

    if betas is not None:
        shape = tuple(np.shape(betas)) if not torch.is_tensor(betas) else tuple(betas.shape)
        if len(shape) == 2 and shape[0] != 1:
            raise ValueError(f"betas must describe one subject, got batch size {shape[0]}")
    with torch.no_grad():
        joints = smpl_forward(model, betas=betas)["joints"][0]
    J = joints.detach().cpu().double().numpy()
    t = SMPL_TORSO_JOINTS
    return torso_orientation_matrix(J[t["left_shoulder"]], J[t["right_shoulder"]],
                                    J[t["left_hip"]], J[t["right_hip"]])


# --------------------------------------------------------------------------- #
# Torso-orientation conveniences and rotation distance (NumPy only)
# --------------------------------------------------------------------------- #
MEDIAPIPE_TORSO_LANDMARKS = {
    name: MEDIAPIPE_LANDMARK_NAMES.index(name)
    for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
}  # {"left_shoulder": 11, "right_shoulder": 12, "left_hip": 23, "right_hip": 24}


def mediapipe_torso_orientation(world, **kwargs):
    """
    Torso orientation (3, 3) from MediaPipe world landmarks (33, 3).

    Uses landmarks 11, 12, 23, 24 (left/right shoulder, left/right hip) with
    torso_orientation_matrix(); kwargs (min_length, min_angle_deg) are passed on.
    Input is validated by mediapipe_world_to_camera_aligned().
    """
    w = mediapipe_world_to_camera_aligned(world)
    t = MEDIAPIPE_TORSO_LANDMARKS
    return torso_orientation_matrix(w[t["left_shoulder"]], w[t["right_shoulder"]],
                                    w[t["left_hip"]], w[t["right_hip"]], **kwargs)


def smpl_torso_orientation_from_joints(joints, **kwargs):
    """
    Torso orientation (3, 3) from SMPL joints (24, 3), e.g. posed joints from smpl_forward().

    Uses joints 16, 17, 1, 2 (left/right shoulder, left/right hip) with
    torso_orientation_matrix(); kwargs (min_length, min_angle_deg) are passed on.
    """
    J = np.asarray(joints)
    if J.dtype.kind not in "fiu":
        raise ValueError(f"joints must contain real numbers, got dtype {J.dtype}")
    if J.shape != (NUM_SMPL_JOINTS, 3):
        raise ValueError(f"joints must have shape ({NUM_SMPL_JOINTS}, 3), got {J.shape}")
    t = SMPL_TORSO_JOINTS
    return torso_orientation_matrix(J[t["left_shoulder"]], J[t["right_shoulder"]],
                                    J[t["left_hip"]], J[t["right_hip"]], **kwargs)


def rotation_angle_deg(R_a, R_b, atol=1e-6):
    """
    Geodesic angle (degrees, in [0, 180]) between two rotation matrices.

    Computed as |log(R_a^T R_b)| with rotation_matrix_to_axis_angle(), which is
    accurate near 0 and 180 deg (unlike arccos((trace - 1) / 2)).
    Symmetric in its arguments. Raises ValueError for invalid rotations.
    """
    Ra = _check_rotation(R_a, "R_a", atol)
    Rb = _check_rotation(R_b, "R_b", atol)
    return math.degrees(float(np.linalg.norm(rotation_matrix_to_axis_angle(Ra.T @ Rb, atol=atol))))


# --------------------------------------------------------------------------- #
# Kinematic tree, rotations between vectors, arm initialisation (NumPy only)
# --------------------------------------------------------------------------- #
# SMPL kinematic tree (smplx `parents`; pelvis is the root). Checked against
# the official model in tests/test_smpl_model.py.
SMPL_PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21)

# Limb specifications. Bones point distally (proximal -> middle -> distal joint).
#   flexion_sign: +1 -> flexion carries the distal bone FORWARD (elbow);
#                 -1 -> flexion carries it BACKWARD (knee).
_LIMB_SPECS = {
    "arm": dict(joints=("shoulder", "elbow", "wrist"), upper="upper arm", lower="forearm",
                flexion_sign=1.0),
    "leg": dict(joints=("hip", "knee", "ankle"), upper="thigh", lower="shin",
                flexion_sign=-1.0),
}
# (limb, side) -> ((MediaPipe proximal, middle, distal), (SMPL proximal, middle, distal))
_LIMB_INDICES = {
    (limb, side): (tuple(MEDIAPIPE_LANDMARK_NAMES.index(f"{side}_{j}") for j in spec["joints"]),
                   tuple(SMPL_JOINT_NAMES.index(f"{side}_{j}") for j in spec["joints"]))
    for limb, spec in _LIMB_SPECS.items() for side in ("left", "right")
}
# arm left: (11, 13, 15), (16, 18, 20); arm right: (12, 14, 16), (17, 19, 21)
# leg left: (23, 25, 27), (1, 4, 7);    leg right: (24, 26, 28), (2, 5, 8)
_ARM_INDICES = {side: _LIMB_INDICES[("arm", side)] for side in ("left", "right")}


def _as_vec3(v, name, min_length=None):
    """Validate a (3,) real, finite vector; optionally require a minimum length."""
    arr = _as_point3(v, name)
    if min_length is not None:
        n = np.linalg.norm(arr)
        if n < min_length:
            raise ValueError(f"{name} has length {n:.3g} < {min_length:.3g}")
    return arr


def axis_angle_to_rotation_matrix(r):
    """
    Axis-angle (3,) -> rotation matrix (3, 3), float64.

    Rodrigues' formula in the same convention as smplx's batch_rodrigues, but
    without its +1e-8 offset:
        R = I + (sin t / t) [r]x + ((1 - cos t) / t^2) [r]x^2,  t = |r|,
    with Taylor series for the two coefficients when t < 1e-4 (exact at r = 0).
    Inverse of rotation_matrix_to_axis_angle().
    """
    r = _as_vec3(r, "r")
    t2 = float(r @ r)
    t = math.sqrt(t2)
    if t < 1e-4:
        a = 1.0 - t2 / 6.0 + t2 * t2 / 120.0
        b = 0.5 - t2 / 24.0 + t2 * t2 / 720.0
    else:
        a = math.sin(t) / t
        b = (1.0 - math.cos(t)) / t2
    K = np.array([[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]])
    return np.eye(3) + a * K + b * (K @ K)


def rotation_between_vectors(a, b, fallback_axis=None, min_length=1e-12):
    """
    Minimal rotation R (3, 3) with R @ unit(a) = unit(b).

    The rotation axis is perpendicular to both vectors and the angle is the
    angle between them (the "swing"); no twist about `a` is added.

    Special cases:
      - parallel (|a x b| ~ 0, a.b > 0): identity.
      - antiparallel: the rotation is 180 deg about an axis perpendicular to `a`.
        That axis is `fallback_axis` projected onto the plane perpendicular to
        `a`. If no fallback is given, or its projection is (nearly) zero, the
        axis is built from the coordinate axis least aligned with `a`.

    Raises ValueError for invalid / (near-)zero-length inputs.
    """
    a = _as_vec3(a, "a", min_length)
    b = _as_vec3(b, "b", min_length)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    c = np.cross(a, b)
    s, cos_ab = float(np.linalg.norm(c)), float(a @ b)
    if s > 1e-12:
        return axis_angle_to_rotation_matrix(math.atan2(s, cos_ab) * c / s)
    if cos_ab > 0:
        return np.eye(3)

    # Antiparallel: 180 deg about an axis perpendicular to a.
    axis = None
    if fallback_axis is not None:
        p = _as_vec3(fallback_axis, "fallback_axis")
        p = p - (p @ a) * a
        if np.linalg.norm(p) > 1e-6 * max(1.0, float(np.linalg.norm(fallback_axis))):
            axis = p / np.linalg.norm(p)
    if axis is None:
        e = np.zeros(3)
        e[int(np.argmin(np.abs(a)))] = 1.0
        axis = np.cross(a, e)
        axis /= np.linalg.norm(axis)
    return axis_angle_to_rotation_matrix(math.pi * axis)


def _check_body_pose(body_pose):
    bp = np.asarray(body_pose)
    if bp.dtype.kind not in "fiu":
        raise ValueError(f"body_pose must contain real numbers, got dtype {bp.dtype}")
    if bp.shape != (3 * (NUM_SMPL_JOINTS - 1),):
        raise ValueError(f"body_pose must have shape ({3 * (NUM_SMPL_JOINTS - 1)},), got {bp.shape}")
    bp = bp.astype(np.float64)
    if not np.isfinite(bp).all():
        raise ValueError("body_pose contains non-finite values")
    return bp


def global_joint_rotations(global_orient, body_pose):
    """
    Global (model-to-world) rotation of every SMPL joint, (24, 3, 3).

    Q[0] = rodrigues(global_orient); Q[j] = Q[parent(j)] @ rodrigues(body_pose[3(j-1):3j]).
    Q[j] rotates the bones from joint j to its children:
        p_child - p_j = Q[j] @ (J_child_rest - J_j_rest).
    (A joint's own local rotation moves its descendants, not itself.)
    """
    go = _as_vec3(global_orient, "global_orient")
    bp = _check_body_pose(body_pose)
    Q = np.empty((NUM_SMPL_JOINTS, 3, 3))
    Q[0] = axis_angle_to_rotation_matrix(go)
    for j in range(1, NUM_SMPL_JOINTS):
        Q[j] = Q[SMPL_PARENTS[j]] @ axis_angle_to_rotation_matrix(bp[3 * (j - 1): 3 * j])
    return Q


def _check_rest_joints(rest_joints):
    J = np.asarray(rest_joints)
    if J.dtype.kind not in "fiu" or J.shape != (NUM_SMPL_JOINTS, 3) or not np.isfinite(J).all():
        raise ValueError(f"rest_joints must be a finite real ({NUM_SMPL_JOINTS}, 3) array")
    return J.astype(np.float64)


def _init_limb_swing(limb, world, rest_joints, global_orient, side, body_pose, min_bone_length):
    """Shared Increment A (swing-only) for one limb. See init_arm_pose / init_leg_pose."""
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    spec = _LIMB_SPECS[limb]
    prox, mid, _ = spec["joints"]
    w = mediapipe_world_to_camera_aligned(world).astype(np.float64)
    J = _check_rest_joints(rest_joints)
    go = _as_vec3(global_orient, "global_orient")
    bp = np.zeros(3 * (NUM_SMPL_JOINTS - 1)) if body_pose is None else _check_body_pose(body_pose).copy()

    (m_s, m_e, m_w), (s_s, s_e, s_w) = _LIMB_INDICES[(limb, side)]

    def unit(v, name):
        return _as_vec3(v, name, min_bone_length) / np.linalg.norm(v)

    d_upper = unit(w[m_e] - w[m_s], f"MediaPipe {side} {spec['upper']}")
    d_fore = unit(w[m_w] - w[m_e], f"MediaPipe {side} {spec['lower']}")
    b_upper = unit(J[s_e] - J[s_s], f"SMPL rest {side} {spec['upper']}")
    b_fore = unit(J[s_w] - J[s_e], f"SMPL rest {side} {spec['lower']}")
    fallback = smpl_torso_orientation_from_joints(J)[:, 2]  # rest torso forward

    Q_parent = global_joint_rotations(go, bp)[SMPL_PARENTS[s_s]]
    R_s = rotation_between_vectors(b_upper, Q_parent.T @ d_upper, fallback_axis=fallback)
    bp[3 * (s_s - 1): 3 * s_s] = rotation_matrix_to_axis_angle(R_s)

    Q_s = Q_parent @ axis_angle_to_rotation_matrix(bp[3 * (s_s - 1): 3 * s_s])
    R_e = rotation_between_vectors(b_fore, Q_s.T @ d_fore, fallback_axis=fallback)
    bp[3 * (s_e - 1): 3 * s_e] = rotation_matrix_to_axis_angle(R_e)

    r_e = bp[3 * (s_e - 1): 3 * s_e]
    hinge_rest = np.cross(b_upper, fallback if spec["flexion_sign"] > 0 else -fallback)
    hinge_rest /= np.linalg.norm(hinge_rest)
    e_angle = float(np.linalg.norm(r_e))
    diagnostics = {
        f"{prox}_rotation_deg": math.degrees(float(np.linalg.norm(bp[3 * (s_s - 1): 3 * s_s]))),
        f"{mid}_rotation_deg": math.degrees(e_angle),
        f"{mid}_bend_mediapipe_deg": math.degrees(math.atan2(np.linalg.norm(np.cross(d_upper, d_fore)),
                                                             d_upper @ d_fore)),
        # Diagnostic only (no constraint): angle between the middle joint's rotation
        # axis and the rest flexion axis unit(b_upper x (+/-)forward). ~0 = flexion,
        # ~180 = bend the other way, ~90 = off-hinge. None if the joint is ~straight.
        f"{mid}_axis_vs_flexion_axis_deg": (
            math.degrees(math.acos(float(np.clip((r_e / e_angle) @ hinge_rest, -1.0, 1.0))))
            if e_angle > 1e-6 else None),
        f"{prox}_body_pose_slice": (3 * (s_s - 1), 3 * s_s),
        f"{mid}_body_pose_slice": (3 * (s_e - 1), 3 * s_e),
    }
    return bp, diagnostics


def init_arm_pose(world, rest_joints, global_orient, side="left", body_pose=None,
                  min_bone_length=1e-6):
    """
    Swing-only initialisation of one arm's shoulder and elbow local rotations.

    Increment A: each bone direction is matched with the MINIMAL rotation.
    The twist of each bone about its own axis is NOT observable from its
    direction and is left at whatever the minimal rotation implies; no
    shoulder-twist correction or anatomical hinge constraint is applied.

    Order (root to leaves):
      1. Parent frame of the shoulder: Q_p = global_joint_rotations(...)[collar],
         using `global_orient` and the CURRENT body_pose (spine/collar included).
      2. Shoulder: R_s = rotation_between_vectors(b_upper_rest, Q_p^T d_upper).
      3. Elbow:    R_e = rotation_between_vectors(b_fore_rest, Q_s^T d_fore),
         with Q_s = Q_p @ rodrigues(stored shoulder axis-angle).
    Here d_* are unit bone directions from MediaPipe world landmarks
    (camera-aligned) and b_*_rest are unit bone directions of the SMPL rest
    joints (same betas, zero pose). Antiparallel cases rotate about the rest
    torso's forward axis (projected), see rotation_between_vectors().

    Args:
        world:       (33, 3) MediaPipe world landmarks (camera-aligned).
        rest_joints: (24, 3) SMPL joints at zero global_orient / body_pose.
        global_orient: (3,) axis-angle used for SMPL (e.g. from global_orient_from_torso).
        side:        "left" or "right" (the person's side).
        body_pose:   (69,) existing pose; all entries except this arm's shoulder
                     and elbow are preserved. Default zeros. Not modified in place.

    Returns:
        (body_pose (69,) float64, diagnostics dict)
    """
    return _init_limb_swing("arm", world, rest_joints, global_orient, side, body_pose, min_bone_length)


def init_leg_pose(world, rest_joints, global_orient, side="left", body_pose=None,
                  min_bone_length=1e-6):
    """
    Swing-only initialisation of one leg's hip and knee local rotations
    (same method as init_arm_pose; MediaPipe hip/knee/ankle 23/25/27 (left) or
    24/26/28 (right) <-> SMPL hip/knee/ankle 1/4/7 or 2/5/8). The hip's parent
    is the pelvis (root), so its parent frame is rodrigues(global_orient).
    Hip twist about the thigh is NOT estimated. Only this leg's hip and knee
    entries of body_pose change.
    """
    return _init_limb_swing("leg", world, rest_joints, global_orient, side, body_pose, min_bone_length)


# --------------------------------------------------------------------------- #
# Increment B: proximal twist from the bend plane, middle joint as a pure hinge
# --------------------------------------------------------------------------- #
def _signed_angle(a, b, axis):
    """Signed angle (radians) from a to b about `axis` (a, b roughly perpendicular to axis)."""
    return math.atan2(float(axis @ np.cross(a, b)), float(a @ b))


def _off_hinge_deg(r, h):
    """Angle between the rotation axis of axis-angle r and the LINE of h, in [0, 90]; 0 if r ~ 0."""
    n = float(np.linalg.norm(r))
    if n < 1e-12:
        return 0.0
    perp = r - (r @ h) * h
    return math.degrees(math.asin(min(1.0, float(np.linalg.norm(perp)) / n)))


def _twist_about_axis_deg(R, axis):
    """Twist angle (degrees, in (-180, 180]) of R about unit `axis` (swing-twist decomposition R = S @ T_axis)."""
    r = rotation_matrix_to_axis_angle(R)
    theta = float(np.linalg.norm(r))
    if theta < 1e-12:
        return 0.0
    tau = 2.0 * math.atan2(math.sin(theta / 2.0) * float((r / theta) @ axis), math.cos(theta / 2.0))
    tau = (tau + math.pi) % (2.0 * math.pi) - math.pi
    return math.degrees(tau)


def _limb_hinge_axis(limb, rest_joints, side):
    """h = unit(b1 x (flexion_sign * f_perp)); see elbow_hinge_axis / knee_hinge_axis."""
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    spec = _LIMB_SPECS[limb]
    J = _check_rest_joints(rest_joints)
    _, (s_s, s_e, _) = _LIMB_INDICES[(limb, side)]
    b1 = _as_vec3(J[s_e] - J[s_s], f"SMPL rest {side} {spec['upper']}", 1e-6)
    b1 = b1 / np.linalg.norm(b1)
    f = smpl_torso_orientation_from_joints(J)[:, 2]
    f_perp = f - (f @ b1) * b1
    if np.linalg.norm(f_perp) < 1e-6:
        raise ValueError(f"rest {side} {spec['upper']} is parallel to the torso forward axis")
    f_dir = f_perp / np.linalg.norm(f_perp)
    h = np.cross(b1, f_dir if spec["flexion_sign"] > 0 else -f_dir)
    return h / np.linalg.norm(h)


def elbow_hinge_axis(rest_joints, side):
    """
    Reference elbow flexion axis h (3,) in SMPL model coordinates, from the rest geometry.

        h = unit(b1 x f_perp)
        b1     = unit(J_elbow - J_shoulder)          (rest upper arm, pointing distally)
        f_perp = rest torso forward minus its b1 component

    Convention: elbow flexion carries the forearm FORWARD (anteriorly): a positive
    rotation about h moves b1 toward f_perp. This matches SMPL's rest pose, which
    has palms down / thumbs forward (verified on the official mesh), so the elbow
    crease faces forward. Because b1 points to +x (left) or -x (right), h points
    roughly down for the left arm and up for the right arm; the same formula
    serves both sides.
    """
    return _limb_hinge_axis("arm", rest_joints, side)


def knee_hinge_axis(rest_joints, side):
    """
    Reference knee flexion axis h (3,) in SMPL model coordinates, from the rest geometry.

        h = unit(b1 x (-f_perp)),  b1 = unit(J_knee - J_hip) (rest thigh, pointing down)

    Convention: knee flexion carries the shin BACKWARD (posteriorly): a positive
    rotation about h moves b1 toward -f_perp. Both thighs point down, so h is
    roughly +x (the person's left) for BOTH legs. On the official model the rest
    feet point forward (ankle->foot along the torso forward axis ~0.85-0.89),
    which confirms the leg's anterior direction.
    """
    return _limb_hinge_axis("leg", rest_joints, side)


def _refine_limb_twist(limb, world, rest_joints, global_orient, body_pose, side,
                       straight_deg, straight_name, max_twist_deg, min_bone_length):
    """Shared Increment B. See refine_arm_twist / refine_leg_twist."""
    if not 0.0 < straight_deg < 90.0:
        raise ValueError(f"{straight_name} must be in (0, 90), got {straight_deg!r}")
    if max_twist_deg is not None and not 0.0 < max_twist_deg <= 180.0:
        raise ValueError(f"max twist must be in (0, 180], got {max_twist_deg!r}")
    spec = _LIMB_SPECS[limb]
    prox, mid, _ = spec["joints"]
    bp_A, diag_A = _init_limb_swing(limb, world, rest_joints, global_orient, side, body_pose, min_bone_length)
    w = mediapipe_world_to_camera_aligned(world).astype(np.float64)
    J = np.asarray(rest_joints, dtype=np.float64)
    go = _as_vec3(global_orient, "global_orient")
    (m_s, m_e, m_w), (s_s, s_e, s_w) = _LIMB_INDICES[(limb, side)]
    sl_s, sl_e = slice(3 * (s_s - 1), 3 * s_s), slice(3 * (s_e - 1), 3 * s_e)

    d1 = w[m_e] - w[m_s]
    d1 = d1 / np.linalg.norm(d1)
    d2 = w[m_w] - w[m_e]
    d2 = d2 / np.linalg.norm(d2)
    b1 = (J[s_e] - J[s_s]) / np.linalg.norm(J[s_e] - J[s_s])
    b2 = (J[s_w] - J[s_e]) / np.linalg.norm(J[s_w] - J[s_e])
    h = _limb_hinge_axis(limb, J, side)

    cross12 = np.cross(d1, d2)
    beta = math.atan2(float(np.linalg.norm(cross12)), float(d1 @ d2))
    b2_perp = b2 - (b2 @ h) * h
    cos_alpha = float(np.linalg.norm(b2_perp))
    phi0 = _signed_angle(b1, b2_perp / cos_alpha, h)

    diag = {
        "bend_deg": math.degrees(beta),
        f"{spec['lower']}_out_of_hinge_plane_deg": math.degrees(math.acos(min(1.0, cos_alpha))),
        "phi0_rest_deg": math.degrees(phi0),
        "off_hinge_swing_only_deg": _off_hinge_deg(bp_A[sl_e], h),
        f"{prox}_body_pose_slice": diag_A[f"{prox}_body_pose_slice"],
        f"{mid}_body_pose_slice": diag_A[f"{mid}_body_pose_slice"],
    }
    if max_twist_deg is not None:
        # Leg-only diagnostics: twist limit, and whether the distal bone bends backward
        # (+, knee-flexion-like) or forward (-, hyperextension-like) w.r.t. the torso.
        f_cam = axis_angle_to_rotation_matrix(go) @ smpl_torso_orientation_from_joints(J)[:, 2]
        ax = np.cross(d1, spec["flexion_sign"] * f_cam)  # rotating d1 toward the flexion side is +
        diag["max_twist_deg"] = max_twist_deg
        diag["bend_direction_deg"] = (math.degrees(_signed_angle(d1, d2, ax / np.linalg.norm(ax)))
                                      if np.linalg.norm(ax) > 1e-6 else None)
        diag["twist_deg"] = None

    def unresolved(reason):
        diag.update(twist_resolved=False, reason=reason, psi_deg=None, phi_deg=None,
                    hinge_vs_bend_normal_deg=None, off_hinge_deg=diag["off_hinge_swing_only_deg"],
                    phi_plausible=None)
        return bp_A, diag

    if math.sin(beta) < math.sin(math.radians(straight_deg)):
        return unresolved(f"bend plane ill-defined ({limb} nearly straight or fully folded)")
    c = math.cos(beta) / cos_alpha
    if abs(c) > 1.0:
        return unresolved(f"bend smaller than the rest {spec['lower']}'s out-of-hinge-plane offset")

    phi = math.acos(c) - phi0
    Q_parent = global_joint_rotations(go, bp_A)[SMPL_PARENTS[s_s]]
    R_A = axis_angle_to_rotation_matrix(bp_A[sl_s])
    t = (Q_parent @ R_A).T @ d2
    v = axis_angle_to_rotation_matrix(phi * h) @ b2
    v_p, t_p = v - (v @ b1) * b1, t - (t @ b1) * b1
    psi = _signed_angle(v_p / np.linalg.norm(v_p), t_p / np.linalg.norm(t_p), b1)

    R_s = R_A @ axis_angle_to_rotation_matrix(psi * b1)
    if max_twist_deg is not None:
        tau = _twist_about_axis_deg(R_s, b1)
        diag["twist_deg"] = tau
        if abs(tau) > max_twist_deg:
            return unresolved(f"implausible {prox} twist {tau:.1f} deg (> {max_twist_deg} deg; "
                              "possible hyperextension or depth noise)")

    bp = bp_A.copy()
    bp[sl_s] = rotation_matrix_to_axis_angle(R_s)
    bp[sl_e] = phi * h
    Q_s = Q_parent @ axis_angle_to_rotation_matrix(bp[sl_s])
    hb = Q_s @ h
    n = cross12 / np.linalg.norm(cross12)
    diag.update(
        twist_resolved=True, reason=None,
        psi_deg=math.degrees(psi), phi_deg=math.degrees(phi),
        # Expected ~asin(sin(alpha)/sin(beta)) because the rest distal bone is not in the hinge plane.
        hinge_vs_bend_normal_deg=math.degrees(math.atan2(float(np.linalg.norm(np.cross(hb, n))), float(hb @ n))),
        off_hinge_deg=_off_hinge_deg(bp[sl_e], h),
        phi_plausible=bool(-10.0 <= math.degrees(phi) <= 150.0),
    )
    return bp, diag


def refine_arm_twist(world, rest_joints, global_orient, body_pose=None, side="left",
                     straight_arm_deg=15.0, min_bone_length=1e-6):
    """
    Increment B: arm initialisation with shoulder twist from the elbow bend plane
    and the elbow as a PURE hinge about elbow_hinge_axis().

    Steps (all in the shoulder's rest-local frame):
      1. Increment A (init_arm_pose): shoulder swing R_A aligns the rest upper
         arm b1 with the observed upper arm d1.
      2. Joint solve of the elbow hinge angle phi and the shoulder twist psi:
           cos(phi0 + phi) = cos(beta) / |b2_perp|   (flexion branch: phi = acos(.) - phi0)
         where beta = observed bend angle(d1, d2), b2_perp = rest forearm minus its
         component along h, and phi0 = rest bend of b2_perp from b1 about h.
         Then psi = signed angle about b1 from the hinged rest forearm to the
         target forearm t = (Q_parent R_A)^T d2.
      3. Shoulder = R_A @ rodrigues(psi * b1)   (twist about the upper arm: its
         direction is unchanged); elbow = phi * h (pure hinge).
    Upper-arm and forearm directions match the observation exactly.

    Nearly straight (or fully folded) arms: if sin(beta) < sin(straight_arm_deg),
    the bend plane is ill-defined; the Increment A result is returned unchanged
    with diagnostics["twist_resolved"] = False.

    Returns:
        (body_pose (69,), diagnostics dict). Only this arm's shoulder and elbow
        entries differ from the input body_pose; the input is not modified.
    """
    return _refine_limb_twist("arm", world, rest_joints, global_orient, body_pose, side,
                              straight_arm_deg, "straight_arm_deg", None, min_bone_length)


def refine_leg_twist(world, rest_joints, global_orient, body_pose=None, side="left",
                     straight_leg_deg=20.0, max_hip_twist_deg=60.0, min_bone_length=1e-6):
    """
    Increment B for one leg: hip twist from the knee bend plane, knee as a PURE
    hinge about knee_hinge_axis() (flexion carries the shin backward). Same joint
    solve as refine_arm_twist; thigh and shin directions match exactly.

    Falls back to the swing-only result (init_leg_pose), with
    diagnostics["twist_resolved"] = False, when
      - sin(bend) < sin(straight_leg_deg)  (default 20 deg; the knee's rest shin is
        ~7-8 deg out of the hinge plane, so small bends are noise-sensitive), or
      - |hip twist| > max_hip_twist_deg  (default 60 deg; swing-twist decomposition
        about the rest thigh). A forced flexion on a hyperextended or depth-noisy
        knee shows up as a large hip twist.
    Extra diagnostics: twist_deg (hip twist), bend_direction_deg (+: shin bends
    backward relative to the torso, flexion-like; -: forward, hyperextension-like).
    bend_direction_deg is only meaningful while the thigh points roughly down
    (it is measured against the torso's forward axis); it is not used by any guard.
    """
    return _refine_limb_twist("leg", world, rest_joints, global_orient, body_pose, side,
                              straight_leg_deg, "straight_leg_deg", max_hip_twist_deg, min_bone_length)
