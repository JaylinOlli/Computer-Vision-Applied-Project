
import os
import pickle
import numpy as np

# Joint numbers we want to compare
# Shoulders, elbows, wrists, knees, ankles
SMPL_JOINTS = [16, 17, 18, 19, 20, 21, 4, 5, 7, 8]
MEDIAPIPE_JOINTS = [11, 12, 13, 14, 15, 16, 25, 26, 27, 28]


def load_3dpw_sequence(dataset_path, split, sequence_name):

    # Find the dataset file
    file_path = os.path.join(dataset_path, split, sequence_name + ".pkl")

    # Open the file
    with open(file_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    # Get the 3D joints for each person
    all_joints = []

    for person in data["jointPositions"]:
        joints = np.array(person).reshape(-1, 24, 3)
        all_joints.append(joints)

    # Keep only the 10 joints we need
    selected_joints = []

    for person in all_joints:
        selected_joints.append(person[:, SMPL_JOINTS, :])

    # Return the data we need
    return {
        "sequence": data["sequence"],
        "joints_3d": selected_joints,
        "cam_poses": data["cam_poses"],
        "cam_intrinsics": data["cam_intrinsics"],
        "campose_valid": data["campose_valid"]
    }
