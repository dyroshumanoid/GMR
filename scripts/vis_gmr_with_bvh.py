import argparse
import time
import numpy as np
import mujoco as mj
import mujoco.viewer as mjv
from scipy.spatial.transform import Rotation as R
from loop_rate_limiters import RateLimiter
from rich import print
from tqdm import tqdm

from general_motion_retargeting import (
    GeneralMotionRetargeting as GMR,
    ROBOT_XML_DICT,
    ROBOT_BASE_DICT,
    VIEWER_CAM_DISTANCE_DICT,
)
from general_motion_retargeting.utils.lafan1 import load_bvh_file
from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh


def draw_sphere(viewer, pos, radius=0.025, rgba=(1.0, 0.3, 0.3, 1.0)):
    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mj.mjv_initGeom(
        geom,
        type=mj.mjtGeom.mjGEOM_SPHERE,
        size=[radius, 0, 0],
        pos=np.asarray(pos, dtype=np.float64),
        mat=np.eye(3).flatten(),
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    viewer.user_scn.ngeom += 1


def draw_bone(viewer, from_pos, to_pos, radius=0.012, rgba=(0.9, 0.9, 0.2, 1.0)):
    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mj.mjv_initGeom(
        geom,
        type=mj.mjtGeom.mjGEOM_CAPSULE,
        size=np.zeros(3),
        pos=np.zeros(3),
        mat=np.eye(3).flatten(),
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    mj.mjv_connector(
        geom,
        type=mj.mjtGeom.mjGEOM_CAPSULE,
        width=radius,
        from_=np.asarray(from_pos, dtype=np.float64),
        to=np.asarray(to_pos, dtype=np.float64),
    )
    viewer.user_scn.ngeom += 1


def draw_frame_axes(viewer, pos, quat_wxyz, size=0.08):
    mat = R.from_quat(quat_wxyz, scalar_first=True).as_matrix()
    rgba_list = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
    for i in range(3):
        geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mj.mjv_initGeom(
            geom,
            type=mj.mjtGeom.mjGEOM_ARROW,
            size=np.zeros(3),
            pos=np.zeros(3),
            mat=np.eye(3).flatten(),
            rgba=rgba_list[i],
        )
        mj.mjv_connector(
            geom,
            type=mj.mjtGeom.mjGEOM_ARROW,
            width=0.005,
            from_=pos,
            to=pos + size * mat[:, i],
        )
        viewer.user_scn.ngeom += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Solve GMR retargeting for G1 and show BVH human skeleton in the same MuJoCo viewer."
    )
    parser.add_argument("--bvh_file", required=True, type=str, help="BVH motion file.")
    parser.add_argument("--format", choices=["lafan1", "nokov"], default="lafan1")
    parser.add_argument("--robot", default="unitree_g1",
                        choices=["unitree_g1", "unitree_g1_with_hands", "booster_t1",
                                 "stanford_toddy", "fourier_n1", "engineai_pm01"])
    parser.add_argument("--motion_fps", type=int, default=30)
    parser.add_argument("--loop", action="store_true", default=False)
    parser.add_argument("--rate_limit", action="store_true", default=True,
                        help="Cap playback at motion_fps (on by default).")
    parser.add_argument("--no_rate_limit", action="store_true", default=False,
                        help="Disable the FPS cap.")

    # BVH skeleton drawing options
    parser.add_argument("--human_offset", type=float, nargs=3, default=[1.0, 0.0, 0.0],
                        metavar=("X", "Y", "Z"),
                        help="World offset (m) applied to the BVH skeleton so it sits next to the robot.")
    parser.add_argument("--hide_axes", action="store_true", default=False,
                        help="Hide the per-joint coordinate frames on the BVH skeleton.")
    parser.add_argument("--axes_size", type=float, default=0.2,
                        help="Length of the per-joint coordinate frame axes (meters).")
    parser.add_argument("--no_follow_camera", action="store_true", default=False,
                        help="Disable camera following the robot base.")
    parser.add_argument("--show_robot", action="store_true", default=False,
                        help="Render the robot mesh too (off by default — only keyframe axes are shown).")
    args = parser.parse_args()

    # --- Load BVH motion ----------------------------------------------------------------
    frames, actual_human_height = load_bvh_file(args.bvh_file, format=args.format)
    raw = read_bvh(args.bvh_file)
    bones = list(raw.bones)
    parents = list(raw.parents)
    edges = [(bones[i], bones[p]) for i, p in enumerate(parents)
             if p >= 0 and bones[i] in frames[0]]
    print(f"Loaded {len(frames)} BVH frames "
          f"(human_height={actual_human_height:.2f} m, {len(bones)} bones)")

    # --- Initialize retargeter and robot MuJoCo model -----------------------------------
    retargeter = GMR(
        src_human=f"bvh_{args.format}",
        tgt_robot=args.robot,
        actual_human_height=actual_human_height,
    )

    xml_path = ROBOT_XML_DICT[args.robot]
    robot_base = ROBOT_BASE_DICT[args.robot]
    cam_distance = VIEWER_CAM_DISTANCE_DICT[args.robot]

    # Robot bodies that are IK match targets for the BVH joints — only these get
    # coordinate frames drawn on the robot.
    robot_keyframe_bodies = sorted(set(retargeter.ik_match_table1.keys())
                                   | set(retargeter.ik_match_table2.keys()))
    print(f"Robot keyframe bodies ({len(robot_keyframe_bodies)}): {robot_keyframe_bodies}")

    model = mj.MjModel.from_xml_path(str(xml_path))
    data = mj.MjData(model)
    mj.mj_step(model, data)

    # Hide the robot mesh — keep only the keyframe coordinate frames visible.
    # Leave plane geoms (e.g. the floor) visible.
    if not args.show_robot:
        for gid in range(model.ngeom):
            if model.geom_type[gid] != mj.mjtGeom.mjGEOM_PLANE:
                model.geom_rgba[gid, 3] = 0.0

    viewer = mjv.launch_passive(model=model, data=data,
                                show_left_ui=False, show_right_ui=False)
    viewer.opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = 0
    viewer.opt.geomgroup[2] = 0
    viewer.cam.distance = cam_distance
    viewer.cam.elevation = -10

    human_offset = np.array(args.human_offset, dtype=np.float64)
    rate_limited = args.rate_limit and not args.no_rate_limit
    rate_limiter = RateLimiter(frequency=args.motion_fps, warn=False) if rate_limited else None

    pbar = tqdm(total=len(frames), desc="GMR + BVH")
    i = 0

    try:
        while viewer.is_running():
            frame = frames[i]

            # 1) Retarget human → robot qpos and push to MuJoCo state
            qpos = retargeter.retarget(frame)
            data.qpos[:3] = qpos[:3]
            data.qpos[3:7] = qpos[3:7]
            data.qpos[7:] = qpos[7:]
            mj.mj_forward(model, data)

            # 2) Camera follow on the robot base
            if not args.no_follow_camera:
                viewer.cam.lookat = data.xpos[model.body(robot_base).id]

            # 3) Overlay BVH human skeleton, offset sideways so it does not overlap
            viewer.user_scn.ngeom = 0

            # 3a) Robot keyframe-body coordinate frames (only IK match targets)
            if not args.hide_axes:
                for body_name in robot_keyframe_bodies:
                    bid = model.body(body_name).id
                    draw_frame_axes(viewer, data.xpos[bid], data.xquat[bid],
                                    size=args.axes_size)

            for bone_name, (pos, quat) in frame.items():
                if bone_name.endswith("Mod"):
                    continue
                world_pos = pos + human_offset
                rgba = (1.0, 0.4, 0.2, 1.0) if bone_name == "Hips" else (0.2, 0.6, 1.0, 1.0)
                radius = 0.04 if bone_name == "Hips" else 0.025
                draw_sphere(viewer, world_pos, radius=radius, rgba=rgba)
                if not args.hide_axes:
                    draw_frame_axes(viewer, world_pos, quat, size=args.axes_size)

            for child_name, parent_name in edges:
                child_pos = frame[child_name][0] + human_offset
                parent_pos = frame[parent_name][0] + human_offset
                draw_bone(viewer, parent_pos, child_pos)

            viewer.sync()
            if rate_limiter is not None:
                rate_limiter.sleep()
            pbar.update(1)

            if args.loop:
                i = (i + 1) % len(frames)
                if i == 0:
                    pbar.reset()
            else:
                i += 1
                if i >= len(frames):
                    break
    finally:
        pbar.close()
        viewer.close()
        time.sleep(0.3)
