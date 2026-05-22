"""Visualize the BVH calibration frame and the G1 rest pose used by `calibrate_scale.py`.

This launches a MuJoCo viewer showing
    1. the robot in its rest pose (the same pose used to read link distances), and
    2. the BVH skeleton at the chosen calibration frame, drawn next to the robot.

The bone segments that enter the LSQ fit (parent -> child within the same scale
group, with both ends mapped to a robot body) are highlighted on BOTH sides with
a color per scale group, so you can see exactly which human segments are being
matched against which robot links.
"""
import argparse
import json
import pathlib
import time
from collections import defaultdict

import numpy as np
import mujoco as mj
import mujoco.viewer as mjv
from scipy.spatial.transform import Rotation as R
from rich import print

from general_motion_retargeting import (
    GeneralMotionRetargeting as GMR,
    ROBOT_XML_DICT,
    IK_CONFIG_DICT,
)
from general_motion_retargeting.utils.lafan1 import load_bvh_file
from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh


# ----------------------------------------------------------------------------
# Drawing helpers (same primitives as vis_bvh_motion.py / vis_gmr_with_bvh.py)
# ----------------------------------------------------------------------------
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


def draw_capsule(viewer, from_pos, to_pos, radius=0.012, rgba=(0.9, 0.9, 0.2, 1.0)):
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


# ----------------------------------------------------------------------------
# LSQ-matching logic (mirrors calibrate_scale.py)
#
# GMR's scale_human_data scales each body's position relative to the root by
# the body's own scale, independently of any chain parent. So the right thing
# to LSQ-fit is root-relative distances, NOT parent->child bone segments.
# ----------------------------------------------------------------------------
def build_human_to_robot_mapping(config):
    h_to_r = {}
    for table_key in ("ik_match_table1", "ik_match_table2"):
        for r_body, entry in config.get(table_key, {}).items():
            h_body = entry[0]
            h_to_r.setdefault(h_body, r_body)
    return h_to_r


def compute_lsq_distances(config, h_frame, model, data):
    """Root-relative distance pairs used for the per-group LSQ fit.

    Returns:
        measurements: {h_body: (d_h, d_r, r_body)}
        skipped:      [(h_body, reason), ...]
        h_to_r:       {h_body: r_body}
    """
    h_root_name = config["human_root_name"]
    r_root_name = config["robot_root_name"]
    scale_table = config["human_scale_table"]

    h_to_r = build_human_to_robot_mapping(config)
    h_root_pos = np.asarray(h_frame[h_root_name][0])
    r_root_pos = data.xpos[model.body(r_root_name).id].copy()

    measurements = {}
    skipped = []
    for h_body in scale_table:
        if h_body == h_root_name:
            continue
        if h_body not in h_to_r:
            skipped.append((h_body, "no IK mapping"))
            continue
        if h_body not in h_frame:
            skipped.append((h_body, "missing in BVH frame"))
            continue
        try:
            r_bid = model.body(h_to_r[h_body]).id
        except KeyError as e:
            skipped.append((h_body, f"robot body missing: {e}"))
            continue
        d_h = float(np.linalg.norm(np.asarray(h_frame[h_body][0]) - h_root_pos))
        d_r = float(np.linalg.norm(data.xpos[r_bid] - r_root_pos))
        if d_h < 1e-6:
            skipped.append((h_body, "d_h ≈ 0 (body coincides with root)"))
            continue
        measurements[h_body] = (d_h, d_r, h_to_r[h_body])
    return measurements, skipped, h_to_r


# Distinct, easy-to-read colors for scale groups.
GROUP_PALETTE = [
    (0.95, 0.30, 0.30, 1.0),   # red
    (0.30, 0.85, 0.40, 1.0),   # green
    (0.30, 0.55, 0.95, 1.0),   # blue
    (0.95, 0.75, 0.20, 1.0),   # amber
    (0.75, 0.40, 0.95, 1.0),   # purple
    (0.20, 0.85, 0.85, 1.0),   # cyan
    (0.95, 0.55, 0.30, 1.0),   # orange
    (0.50, 0.85, 0.95, 1.0),   # sky
    (0.85, 0.50, 0.70, 1.0),   # pink
]


def pair_key(body_name):
    """Logical scale-key that fuses 'LeftX' and 'RightX' under 'X'.

    Bodies that don't start with Left/Right are their own keys. This is the
    granularity at which we fit a single scale: per-body, with L/R symmetric
    pairs sharing one value.
    """
    for prefix in ("Left", "Right"):
        if body_name.startswith(prefix) and len(body_name) > len(prefix):
            return body_name[len(prefix):]
    return body_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bvh_file", required=True, type=str)
    parser.add_argument("--format", choices=["lafan1", "nokov", "xsens"], default="lafan1")
    parser.add_argument("--robot", required=True, type=str,
                        help="Target robot key (e.g. unitree_g1).")
    parser.add_argument("--frame", type=int, default=0,
                        help="BVH frame index used for calibration (default: 0).")
    parser.add_argument("--actual_human_height", type=float, default=None,
                        help="Override actual human height (m). Defaults to load_bvh_file's value.")
    parser.add_argument("--human_offset", type=float, nargs=3, default=[1.2, 0.0, 0.0],
                        metavar=("X", "Y", "Z"),
                        help="World offset (m) applied to the BVH skeleton.")
    parser.add_argument("--hide_axes", action="store_true", default=False,
                        help="Hide coordinate-frame axes on the robot keyframe bodies and BVH joints.")
    parser.add_argument("--axes_size", type=float, default=0.15,
                        help="Length of coordinate-frame axes (meters).")
    parser.add_argument("--hide_robot_mesh", action="store_true", default=False,
                        help="Hide the robot mesh; show only its keyframe bodies and link capsules.")
    parser.add_argument("--no_orient_robot", action="store_true", default=False,
                        help="Skip the orientation-only GMR pre-pass and read link distances "
                             "directly from the robot rest pose. By default we first pose the "
                             "robot to match the BVH calibration frame's joint orientations so "
                             "that the LSQ link-length comparison is pose-aligned.")
    parser.add_argument("--calib_src", type=str, default="bvh_calibration",
                        help="IK_CONFIG_DICT source key used for the orientation-only pre-pass.")
    parser.add_argument("--calib_iters", type=int, default=3,
                        help="Extra retarget() iterations after GMR's built-in warmup, used to "
                             "let the orientation IK fully settle on the static calibration frame.")
    args = parser.parse_args()

    # --- Load IK config ---------------------------------------------------------------
    src_key = f"bvh_{args.format}"
    if src_key not in IK_CONFIG_DICT or args.robot not in IK_CONFIG_DICT[src_key]:
        raise SystemExit(f"No IK config registered for src='{src_key}', robot='{args.robot}'")
    config_path = pathlib.Path(IK_CONFIG_DICT[src_key][args.robot])
    config = json.loads(config_path.read_text())
    print(f"[bold]IK config:[/bold] {config_path}")

    h_root_name = config["human_root_name"]
    r_root_name = config["robot_root_name"]
    scale_table = config["human_scale_table"]

    # --- Load BVH calibration frame --------------------------------------------------
    frames, default_human_height = load_bvh_file(args.bvh_file, format=args.format)
    actual_human_height = (args.actual_human_height
                           if args.actual_human_height is not None
                           else default_human_height)
    human_height_assumption = config["human_height_assumption"]
    ratio = actual_human_height / human_height_assumption
    print(f"actual_human_height={actual_human_height:.4f}  "
          f"human_height_assumption={human_height_assumption:.4f}  "
          f"ratio={ratio:.4f}")
    if not 0 <= args.frame < len(frames):
        raise SystemExit(f"--frame {args.frame} out of range [0, {len(frames)})")
    h_frame = frames[args.frame]

    raw = read_bvh(args.bvh_file)
    bones = list(raw.bones)
    parents_idx = list(raw.parents)
    edges = [(bones[i], bones[p]) for i, p in enumerate(parents_idx)
             if p >= 0 and bones[i] in h_frame]

    # --- Robot rest pose -------------------------------------------------------------
    model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[args.robot]))
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    r_root_pos = data.xpos[model.body(r_root_name).id].copy()
    print(f"Robot rest pose: root '{r_root_name}' at {r_root_pos.round(3).tolist()}")

    # --- Orient the robot to match the BVH calibration frame -------------------------
    # The LSQ link-length comparison is only meaningful when both sides are in similar
    # joint configurations. Robot qpos0 typically has arms forward while the BVH
    # calibration frame might have arms spread sideways, so we first run an
    # orientation-only GMR pass (position weights all 0) to pose the robot to match
    # the BVH joint orientations, then read link distances from that pose.
    pose_label = "rest pose"
    if not args.no_orient_robot:
        if (args.calib_src not in IK_CONFIG_DICT
                or args.robot not in IK_CONFIG_DICT[args.calib_src]):
            raise SystemExit(
                f"Calibration IK config not registered for src='{args.calib_src}', "
                f"robot='{args.robot}'. Create one (zero all pos_weight in ik_match_table*) "
                f"and add it to IK_CONFIG_DICT['{args.calib_src}']."
            )
        print(f"\n[bold]Orientation-only pre-pass[/bold] using "
              f"'{IK_CONFIG_DICT[args.calib_src][args.robot]}'")
        retargeter = GMR(
            src_human=args.calib_src,
            tgt_robot=args.robot,
            actual_human_height=actual_human_height,
            verbose=False,
        )
        # First call runs GMR's built-in warmup (100 iters). Extra calls let the
        # static-frame solution settle further.
        qpos = retargeter.retarget(h_frame)
        for _ in range(max(0, args.calib_iters)):
            qpos = retargeter.retarget(h_frame)
        data.qpos[:] = qpos
        mj.mj_forward(model, data)
        r_root_pos = data.xpos[model.body(r_root_name).id].copy()
        pose_label = "orientation-matched pose"
        print(f"Robot oriented: root '{r_root_name}' now at {r_root_pos.round(3).tolist()}")

    # --- LSQ measurements & per-group LSQ fit ----------------------------------------
    # We compare root-relative distances because that is what GMR's scale_human_data
    # actually applies: each body's scaled position is `scale[body] * (body - root)`,
    # so the per-body LSQ target is `scale[body] * d_h[body] ≈ d_r[body]`.
    print(f"\nReading robot link distances from [bold]{pose_label}[/bold] "
          f"(root-relative, root='{r_root_name}').")
    measurements, skipped, h_to_r = compute_lsq_distances(config, h_frame, model, data)
    if skipped:
        print("[yellow]Skipped bodies (not used in LSQ):[/yellow]")
        for body, reason in skipped:
            print(f"  - {body}: {reason}")

    # --- Root scale (Hips -> pelvis) -------------------------------------------------
    # The root is the only body GMR scales in absolute world coordinates:
    #   scaled_root_pos = s_root * h_root_pos
    # So fitting it is a direct height ratio, not a LSQ — single equation, single
    # unknown. We report the z-axis ratio because that's what defines standing
    # height; x/y are mostly root motion, not a scale issue.
    h_root_z = float(np.asarray(h_frame[h_root_name][0])[2])
    r_root_z = float(r_root_pos[2])
    if abs(h_root_z) > 1e-6:
        s_root_eff = r_root_z / h_root_z
        old_root_json = scale_table.get(h_root_name, float("nan"))
        print(f"\n[bold]Root scale[/bold] ({h_root_name} z={h_root_z:.4f}  ->  "
              f"{r_root_name} z={r_root_z:.4f}):  "
              f"s = {s_root_eff:.4f}  (old_json={old_root_json:.4f})")
    else:
        s_root_eff = float("nan")
        print(f"\n[yellow]Root scale skipped:[/yellow] human root z≈0")

    # --- Build logical groups (L/R symmetric pairs share one scale) -----------------
    # pair_key('LeftArm') == pair_key('RightArm') == 'Arm', so the two bodies enter
    # the same LSQ. Standalone bodies (Hips, Spine2, ...) form a single-element
    # "pair" — LSQ on 1 equation collapses to d_r/d_h.
    logical_groups = defaultdict(list)
    for body in scale_table:
        logical_groups[pair_key(body)].append(body)
    group_keys = sorted(logical_groups.keys())
    group_color = {gk: GROUP_PALETTE[i % len(GROUP_PALETTE)]
                   for i, gk in enumerate(group_keys)}

    # Per-pair LSQ: s = Σ(d_h·d_r) / Σ(d_h²). With one body it's just d_r/d_h;
    # with an L/R pair it averages the two near-symmetric measurements.
    print("\n[bold]Per-pair LSQ fit (L/R pairs share one scale, root-relative):[/bold]")
    new_scale_table = dict(scale_table)
    for gk in group_keys:
        bodies = logical_groups[gk]
        pair_data = [(b, *measurements[b]) for b in bodies if b in measurements]
        # Root is a special case — d_h=0 puts it outside the root-relative LSQ.
        if gk == h_root_name:
            if not np.isnan(s_root_eff):
                for b in bodies:
                    new_scale_table[b] = s_root_eff
                c = group_color[gk]
                tag = f"[rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})]"
                old_v = scale_table.get(h_root_name, float("nan"))
                print(f"  {tag}■[/] {gk:8s} [root height]                                      "
                      f"=>  s = {s_root_eff:.4f}  (old_json={old_v:.4f})")
            continue
        if not pair_data:
            print(f"  {gk:8s}: [yellow]no measurements — keeping old value[/yellow]")
            continue
        d_h_arr = np.array([p[1] for p in pair_data])
        d_r_arr = np.array([p[2] for p in pair_data])
        s_eff = float(np.sum(d_h_arr * d_r_arr) / np.sum(d_h_arr ** 2))
        for b in bodies:
            new_scale_table[b] = s_eff
        c = group_color[gk]
        tag = f"[rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})]"
        members = ", ".join(b for b, *_ in pair_data)
        old_vs = sorted({scale_table[b] for b in bodies})
        old_str = "/".join(f"{v:.4f}" for v in old_vs)
        print(f"  {tag}■[/] {gk:8s} [{members}]"
              f"  =>  s = {s_eff:.4f}  (old_json={old_str})")
        for h_body, dh, dr, r_body in pair_data:
            r_per_h = dr / dh if dh > 1e-6 else float("nan")
            print(f"      {h_root_name:10s} -> {h_body:14s}  d_h={dh:.4f}   |   "
                  f"{r_root_name:10s} -> {r_body:24s}  d_r={dr:.4f}  ratio={r_per_h:.3f}")

    # --- Emit the recommended JSON snippet ------------------------------------------
    # GMR multiplies the JSON scale by `actual/assumption` at runtime. To avoid that
    # second adjustment (since our s_eff already encodes the actual person's link
    # ratios), set `human_height_assumption` equal to `actual_human_height` so the
    # runtime ratio collapses to 1.0.
    print("\n[bold]Recommended JSON update[/bold] "
          f"(set [italic]human_height_assumption = {actual_human_height:.4f}[/italic] "
          "so runtime ratio = 1.0):")
    print(json.dumps({
        "human_height_assumption": actual_human_height,
        "human_scale_table": {b: round(v, 4) for b, v in new_scale_table.items()},
    }, indent=4))

    # --- Robot bodies that get coordinate-axis markers -------------------------------
    robot_keyframe_bodies = sorted({h_to_r[h] for h in scale_table if h in h_to_r})

    # --- Hide robot mesh if requested ------------------------------------------------
    if args.hide_robot_mesh:
        for gid in range(model.ngeom):
            if model.geom_type[gid] != mj.mjtGeom.mjGEOM_PLANE:
                model.geom_rgba[gid, 3] = 0.0

    # --- Launch viewer ---------------------------------------------------------------
    viewer = mjv.launch_passive(model=model, data=data,
                                show_left_ui=False, show_right_ui=False)
    viewer.opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = 0
    viewer.opt.geomgroup[2] = 0
    viewer.cam.distance = 3.5
    viewer.cam.elevation = -15
    viewer.cam.azimuth = 120
    viewer.cam.lookat[:] = r_root_pos + np.array([args.human_offset[0] * 0.5, 0.0, 0.0])

    human_offset = np.array(args.human_offset, dtype=np.float64)

    # Color each body by its logical (pair) group, so L/R symmetric bodies share.
    body_to_group = {b: pair_key(b) for b in scale_table}

    print("\n[green]Viewer is open. Close the window to exit.[/green]")
    try:
        while viewer.is_running():
            viewer.user_scn.ngeom = 0

            # ---- Robot: keyframe-body coordinate axes -------------------------------
            if not args.hide_axes:
                for body_name in robot_keyframe_bodies:
                    bid = model.body(body_name).id
                    draw_frame_axes(viewer, data.xpos[bid], data.xquat[bid],
                                    size=args.axes_size)

            # Root anchor points (recomputed each frame in case data.qpos changes).
            r_root_world = data.xpos[model.body(r_root_name).id].copy()
            h_root_world = np.asarray(h_frame[h_root_name][0]) + human_offset

            # ---- Robot: highlight root->body spokes (one per LSQ measurement) -------
            for h_body, (dh, dr, r_body) in measurements.items():
                color = group_color[body_to_group[h_body]]
                c_pos = data.xpos[model.body(r_body).id]
                draw_capsule(viewer, r_root_world, c_pos, radius=0.014, rgba=color)
                draw_sphere(viewer, c_pos, radius=0.025, rgba=color)

            # ---- Human: full skeleton (gray) ----------------------------------------
            for child_name, parent_name in edges:
                if child_name.endswith("Mod") or parent_name.endswith("Mod"):
                    continue
                cp = np.asarray(h_frame[child_name][0]) + human_offset
                pp = np.asarray(h_frame[parent_name][0]) + human_offset
                draw_capsule(viewer, pp, cp, radius=0.010, rgba=(0.55, 0.55, 0.60, 1.0))

            # ---- Human: joints as spheres + axes ------------------------------------
            for bone_name, (pos, quat) in h_frame.items():
                if bone_name.endswith("Mod"):
                    continue
                world_pos = np.asarray(pos) + human_offset
                if bone_name == h_root_name:
                    rgba = (1.0, 0.45, 0.20, 1.0)
                    radius = 0.04
                else:
                    rgba = (0.30, 0.55, 0.85, 1.0)
                    radius = 0.022
                draw_sphere(viewer, world_pos, radius=radius, rgba=rgba)
                if not args.hide_axes:
                    draw_frame_axes(viewer, world_pos, quat, size=args.axes_size)

            # ---- Human: highlight root->body spokes ---------------------------------
            for h_body, (dh, dr, r_body) in measurements.items():
                color = group_color[body_to_group[h_body]]
                cp = np.asarray(h_frame[h_body][0]) + human_offset
                draw_capsule(viewer, h_root_world, cp, radius=0.013, rgba=color)
                draw_sphere(viewer, cp, radius=0.028, rgba=color)

            # ---- Both sides: emphasise the root anchor ------------------------------
            draw_sphere(viewer, r_root_world, radius=0.045, rgba=(1.0, 0.45, 0.20, 1.0))
            draw_sphere(viewer, h_root_world, radius=0.045, rgba=(1.0, 0.45, 0.20, 1.0))

            viewer.sync()
            time.sleep(1.0 / 60.0)
    finally:
        viewer.close()
        time.sleep(0.3)
