import argparse
import os
import pickle

import joblib
import mujoco as mj
import numpy as np

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.params import ROBOT_XML_DICT


# G1 body names whose world (pos, quat) become IK targets for p73
G1_LANDMARKS = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True, help="path to PHC-style G1 pkl")
    parser.add_argument("--clip", default=None, help="clip name; defaults to first")
    parser.add_argument("--quat_order", default="xyzw", choices=["xyzw", "wxyz"],
                        help="quaternion convention of root_rot in the pkl")
    parser.add_argument("--save_path", default=None, help="output p73 motion pkl")
    parser.add_argument("--visualize", action="store_true", help="play the result live")
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_path", default="videos/g1_to_p73.mp4")
    args = parser.parse_args()

    # ---- load G1 model and prepare landmark body ids ----
    g1_xml = str(ROBOT_XML_DICT["unitree_g1"])
    g1_model = mj.MjModel.from_xml_path(g1_xml)
    g1_data = mj.MjData(g1_model)
    landmark_ids = {}
    for name in G1_LANDMARKS:
        bid = mj.mj_name2id(g1_model, mj.mjtObj.mjOBJ_BODY, name)
        if bid == -1:
            raise RuntimeError(f"G1 body '{name}' not found in {g1_xml}")
        landmark_ids[name] = bid

    expected_nq = 7 + 29
    if g1_model.nq != expected_nq:
        raise RuntimeError(f"G1 model nq={g1_model.nq}, expected {expected_nq} "
                           "(7 free + 29 dof). Check the mjcf.")

    # ---- load pkl ----
    data = joblib.load(args.pkl)
    print(f"Available clips: {list(data.keys())}")
    clip_name = args.clip or next(iter(data.keys()))
    print(f"Retargeting clip: {clip_name}")
    clip = data[clip_name]

    root_pos = np.asarray(clip["root_trans_offset"], dtype=np.float64)
    root_rot = np.asarray(clip["root_rot"], dtype=np.float64)
    dof = np.asarray(clip["dof"], dtype=np.float64)
    fps = int(clip.get("fps", 30))
    T = len(root_pos)

    if args.quat_order == "xyzw":
        root_rot = root_rot[:, [3, 0, 1, 2]]  # -> wxyz

    # ---- build frame-0 G1 landmark dict for calibration ----
    # Matches the per-(source, robot) calibration pattern used by bvh_to_robot.py /
    # smplx_to_robot.py: pose the source at t=0, sample landmark world transforms,
    # and let GMR's _calibrate_scale_table LSQ-fit the human_scale_table.
    g1_data.qpos[:3] = root_pos[0]
    g1_data.qpos[3:7] = root_rot[0]
    g1_data.qpos[7:7 + 29] = dof[0]
    mj.mj_forward(g1_model, g1_data)
    calibration_frame = {
        name: (g1_data.xpos[bid].copy(), g1_data.xquat[bid].copy())
        for name, bid in landmark_ids.items()
    }

    # ---- init retargeter ----
    retarget = GMR(
        src_human="g1",
        tgt_robot="p73",
        actual_human_height=None,  # LSQ calibration fits scales directly; no ratio needed
        calibration_frame=calibration_frame,
    )
    # retarget.set_ground_offset(-0.06)

    # ---- per-frame: G1 FK -> landmarks dict -> IK on p73 ----
    qpos_list = []
    viewer = None
    if args.visualize or args.record_video:
        viewer = RobotMotionViewer(
            robot_type="p73",
            motion_fps=fps,
            transparent_robot=0,
            record_video=args.record_video,
            video_path=args.video_path,
        )

    for t in range(T):
        # set G1 qpos and run FK
        g1_data.qpos[:3] = root_pos[t]
        g1_data.qpos[3:7] = root_rot[t]
        g1_data.qpos[7:7 + 29] = dof[t]
        mj.mj_forward(g1_model, g1_data)

        # build landmark dict {name: (pos, quat_wxyz)}
        human_data = {}
        for name, bid in landmark_ids.items():
            human_data[name] = (
                g1_data.xpos[bid].copy(),
                g1_data.xquat[bid].copy(),
            )

        qpos = retarget.retarget(human_data)
        qpos_list.append(qpos)

        if viewer is not None:
            viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                human_motion_data=retarget.scaled_human_data,
                human_pos_offset=np.array([1.0, 0.0, 0.0]),  # show G1 keypoints next to p73
                show_human_body_name=False,
                rate_limit=True,
                follow_camera=False,
            )

    if viewer is not None:
        viewer.close()

    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        out_root_pos = np.array([q[:3] for q in qpos_list])
        out_root_rot = np.array([q[3:7][[1, 2, 3, 0]] for q in qpos_list])  # wxyz -> xyzw
        out_dof = np.array([q[7:] for q in qpos_list])
        motion_data = {
            "fps": fps,
            "root_pos": out_root_pos,
            "root_rot": out_root_rot,
            "dof_pos": out_dof,
            "local_body_pos": None,
            "link_body_list": None,
        }
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved p73 motion to {args.save_path}")


if __name__ == "__main__":
    main()
