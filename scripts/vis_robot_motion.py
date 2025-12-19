from general_motion_retargeting import RobotMotionViewer, load_robot_motion
import argparse
import os
from tqdm import tqdm
import mujoco as mj

def _get_mj_model_data(env):
    # RobotMotionViewer 내부 구현마다 속성명이 다를 수 있어서 후보를 여러 개 확인
    model = None
    data = None

    for name in ["model", "mj_model", "_model", "m", "_m"]:
        if hasattr(env, name):
            model = getattr(env, name)
            break

    for name in ["data", "mj_data", "_data", "d", "_d"]:
        if hasattr(env, name):
            data = getattr(env, name)
            break

    return model, data

def _contacts_info(model, data, only_penetration=True):
    """
    only_penetration=True: dist < 0 (관통)일 때만 '충돌'로 간주
    False면 접촉(contact)만 있어도 출력
    """
    if data is None:
        return []

    ncon = int(getattr(data, "ncon", 0))
    if ncon <= 0:
        return []

    infos = []
    for i in range(ncon):
        c = data.contact[i]
        dist = float(c.dist)

        if only_penetration and dist >= 0.0:
            continue

        g1, g2 = int(c.geom1), int(c.geom2)
        if model is not None:
            n1 = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g1) or f"geom#{g1}"
            n2 = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g2) or f"geom#{g2}"
        else:
            n1, n2 = f"geom#{g1}", f"geom#{g2}"

        infos.append((n1, n2, dist))

    return infos

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="unitree_g1")
    parser.add_argument("--robot_motion_path", type=str, required=True)
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_path", type=str, default="videos/example.mp4")
    parser.add_argument("--print_contact", action="store_true",
                        help="접촉(contact)만 있어도 출력(기본은 관통(dist<0)만 출력)")
    args = parser.parse_args()

    robot_type = args.robot
    robot_motion_path = args.robot_motion_path

    if not os.path.exists(robot_motion_path):
        raise FileNotFoundError(f"Motion file {robot_motion_path} not found")

    motion_data, motion_fps, motion_root_pos, motion_root_rot, motion_dof_pos, motion_local_body_pos, motion_link_body_list = load_robot_motion(robot_motion_path)

    env = RobotMotionViewer(
        robot_type=robot_type,
        motion_fps=motion_fps,
        camera_follow=False,
        record_video=args.record_video,
        video_path=args.video_path,
    )

    model, data = _get_mj_model_data(env)
    if data is None:
        print("[WARN] env에서 MuJoCo data를 찾지 못했습니다. (env.data / env.mj_data / env._data 등)")

    last_pairs = set()

    frame_idx = 0
    while True:
        env.step(
            motion_root_pos[frame_idx],
            motion_root_rot[frame_idx],
            motion_dof_pos[frame_idx],
            rate_limit=True,
        )

        infos = _contacts_info(model, data, only_penetration=(not args.print_contact))
        cur_pairs = set(tuple(sorted((a, b))) for a, b, _ in infos)

        # 스팸 방지: "새로운 접촉 쌍"이 생겼을 때만 출력
        new_pairs = cur_pairs - last_pairs
        if new_pairs:
            # 해당 new_pairs에 대한 dist만 모아서 출력
            lines = []
            for a, b, dist in infos:
                key = tuple(sorted((a, b)))
                if key in new_pairs:
                    lines.append(f"{a} <-> {b} (dist={dist:.6f})")
            if lines:
                tag = "CONTACT" if args.print_contact else "COLLISION(dist<0)"
                print(f"[{tag}] frame={frame_idx} n={len(lines)}")
                for s in lines:
                    print("  -", s)

        # 접촉이 완전히 사라지면 last_pairs 초기화(원하면 유지해도 됨)
        last_pairs = cur_pairs

        frame_idx += 1
        if frame_idx >= len(motion_root_pos):
            frame_idx = 0

    env.close()
