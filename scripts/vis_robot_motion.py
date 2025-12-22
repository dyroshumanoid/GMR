from general_motion_retargeting import RobotMotionViewer, load_robot_motion
import argparse
import os
import sys
import select
import termios
import tty
import mujoco as mj

def _get_mj_viewer(env):
    # RobotMotionViewer 구현에 따라 이름이 다를 수 있어서 후보를 여러 개 확인
    for name in ["viewer", "_viewer", "mj_viewer", "_mj_viewer", "v", "_v"]:
        if hasattr(env, name):
            return getattr(env, name)
    return None

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


class KeyPoller:
    """터미널에서 1글자 non-blocking으로 읽기 (Linux/Unix)."""
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)  # non-canonical
        return self

    def __exit__(self, exc_type, exc, tb):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def poll(self):
        dr, _, _ = select.select([sys.stdin], [], [], 0)
        if dr:
            return sys.stdin.read(1)
        return None

def _body_name(model, body_id: int) -> str:
    if model is None:
        return f"body#{body_id}"
    name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, int(body_id))
    if not name:
        return "world" if int(body_id) == 0 else f"body#{int(body_id)}"
    return name

def _contacts_info(model, data, only_penetration=True):
    """
    return: list of (link1, link2, dist, b1, b2)
      - link1/link2: geom이 속한 body(link) 이름
      - b1/b2: body id (worldbody면 0)
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

        # geom id가 비정상인 경우 방어
        if model is None:
            # model이 없으면 어쩔 수 없이 geom로 fallback
            n1, n2 = f"geom#{g1}", f"geom#{g2}"
            b1, b2 = -1, -1
            infos.append((n1, n2, dist, b1, b2))
            continue

        if g1 < 0 or g1 >= model.ngeom or g2 < 0 or g2 >= model.ngeom:
            # 범위 밖이면 스킵(혹은 geom#로 출력해도 됨)
            continue

        b1 = int(model.geom_bodyid[g1])
        b2 = int(model.geom_bodyid[g2])

        link1 = _body_name(model, b1)
        link2 = _body_name(model, b2)

        infos.append((link1, link2, dist, b1, b2))

    return infos



def _mode_str(mode: int) -> str:
    return {
        1: "OFF (체킹/출력 없음)",
        2: "SELF-COLLISION ONLY (robot-robot)",
        3: "GROUND CONTACT ONLY (world-robot)",
        4: "ALL",
    }.get(mode, "UNKNOWN")


def _filter_infos_by_mode(model, infos, mode: int):
    """
    mode:
      1: 체킹/출력 없음
      2: self-collision만 (robot-robot)
      3: 지면-로봇만 (world-robot)
      4: 전부

    분류 기준(가능하면 이걸 씀):
      - body id == 0  -> worldbody(지면 포함)
      - body id != 0  -> 로봇 바디
    """
    if mode == 1:
        return []
    if mode == 4:
        return infos

    # model이 없으면 이름 기반 fallback (완벽하진 않음)
    if model is None:
        def is_world_geom(name: str) -> bool:
            low = name.lower()
            return (name == "ground") or ("plane" in low) or ("floor" in low) or ("terrain" in low)

        out = []
        for n1, n2, dist, b1, b2 in infos:
            w1 = is_world_geom(n1)
            w2 = is_world_geom(n2)
            if mode == 2 and (not w1) and (not w2):
                out.append((n1, n2, dist, b1, b2))
            elif mode == 3 and (w1 ^ w2):
                out.append((n1, n2, dist, b1, b2))
        return out

    out = []
    for n1, n2, dist, b1, b2 in infos:
        w1 = (b1 == 0)
        w2 = (b2 == 0)

        if mode == 2:
            # 로봇-로봇
            if (not w1) and (not w2):
                out.append((n1, n2, dist, b1, b2))
        elif mode == 3:
            # world-robot (한쪽만 world)
            if w1 ^ w2:
                out.append((n1, n2, dist, b1, b2))

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="unitree_g1")
    parser.add_argument("--robot_motion_path", type=str, required=True)
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_path", type=str, default="videos/example.mp4")
    parser.add_argument(
        "--print_contact", action="store_true",
        help="접촉(contact)만 있어도 출력(기본은 관통(dist<0)만 출력)",
    )
    args = parser.parse_args()

    robot_type = args.robot
    robot_motion_path = args.robot_motion_path

    if not os.path.exists(robot_motion_path):
        raise FileNotFoundError(f"Motion file {robot_motion_path} not found")

    motion_data, motion_fps, motion_root_pos, motion_root_rot, motion_dof_pos, motion_local_body_pos, motion_link_body_list = load_robot_motion(
        robot_motion_path
    )

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

    # 1: off, 2: self, 3: ground-only, 4: all
    mode = 1

    print("=== Collision Output Mode (press key in terminal) ===")
    print("  [1] collision 체킹/출력 없음")
    print("  [2] self-collision만 출력 (robot-robot)")
    print("  [3] 지면-링크 접촉만 출력 (world-robot)")
    print("  [4] 전부 출력")
    print(f"Current mode: {mode} -> {_mode_str(mode)}")
    print("====================================================")

    last_pairs = set()
    frame_idx = 0
    collision_group = 2  # cls가 보통 group=2
    hide_collision_applied = False
    with KeyPoller() as kp:
        while True:
            ch = kp.poll()
            if ch in ("1", "2", "3", "4"):
                mode = int(ch)
                last_pairs = set()  # 모드 바뀌면 스팸/누락 방지용 초기화
                print(f"\n[MODE] {mode} -> {_mode_str(mode)}\n")

            env.step(
                motion_root_pos[frame_idx],
                motion_root_rot[frame_idx],
                motion_dof_pos[frame_idx],
                rate_limit=True,
            )
            if not hide_collision_applied:
                viewer = _get_mj_viewer(env)
                if viewer is not None and hasattr(viewer, "opt"):
                    # geom group은 0~5, 1이면 보임 / 0이면 숨김
                    viewer.opt.geomgroup[collision_group] = 0
                    hide_collision_applied = True
            # mode=1이면 아예 contact 읽고/출력 안함
            if mode == 1:
                infos = []
            else:
                infos = _contacts_info(model, data, only_penetration=(not args.print_contact))
                infos = _filter_infos_by_mode(model, infos, mode)

            cur_pairs = set(tuple(sorted((b1, b2))) for _, _, _, b1, b2 in infos)

            # 스팸 방지: "새로운 접촉 쌍"이 생겼을 때만 출력
            new_pairs = cur_pairs - last_pairs
            if new_pairs:
                lines = []
                for a, b, dist, b1, b2 in infos:
                    key = tuple(sorted((b1, b2)))
                    if key in new_pairs:
                        lines.append(f"{a} <-> {b} (dist={dist:.6f})")


                if lines:
                    tag = "CONTACT" if args.print_contact else "COLLISION(dist<0)"
                    print(f"[{tag}] frame={frame_idx} n={len(lines)}")
                    for s in lines:
                        print("  -", s)

            last_pairs = cur_pairs

            frame_idx += 1
            if frame_idx >= len(motion_root_pos):
                frame_idx = 0

    env.close()
