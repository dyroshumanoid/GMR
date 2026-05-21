import mink
import mujoco as mj
import numpy as np
import json
import yaml
from scipy.spatial.transform import Rotation as R
from .params import ROBOT_XML_DICT, IK_CONFIG_DICT
from rich import print
from mink.exceptions import TargetNotSet

class CollisionBarrierTask(mink.Task):
    """
    Soft collision avoidance via a logarithmic barrier.

    For each active geom pair (i, j):

        d(q)   : signed distance between geom i and j
        d_min  : minimum safety margin
        h(q)   = d(q) - d_min

    When d(q) < detect_dist, we introduce a barrier term:

        φ(q) = -log(h(q))

    Linearized at the current configuration:

        ∂φ/∂q = -(1 / h(q)) * ∂h/∂q

    which yields an additional task row:

        J_bar = -(1 / h) * J_h
        e_bar = 0

    where:
        J_h = ∂h/∂q  (contact normal Jacobian)

    NOTE:
        The barrier is valid only for h(q) > 0 (no penetration).
        This term contributes as a soft cost in the IK QP.
    """
    def __init__(
        self,
        model: mj.MjModel,
        collision_limits,
        w_bar: float = 0.0,
        name: str = "collision_barrier",
    ):
        super().__init__(cost=np.zeros(0))
        self.model = model
        self.collision_limits = collision_limits
        self.w_bar = float(w_bar)
        self.name = name
        self.h_eps = 1.5e-3  # small epsilon to prevent singularity when h is close to zero
        self._J = np.zeros((0, model.nv))
        self._e = np.zeros((0,))
        self._w = np.zeros((0,))
        self._fromto = np.zeros(6, dtype=np.float64)
        self._normal = np.zeros(3, dtype=np.float64)
        self._jac1 = np.zeros((3, model.nv), dtype=np.float64)
        self._jac2 = np.zeros((3, model.nv), dtype=np.float64)


    def _relaxed_barrier_gradient(self, h: float) -> float:
        delta = self.h_eps
        if h >= delta:
            # Log region: dB/dh = -1/h
            return -1.0 / h
        else:
            # Quadratic region: dB/dh = (h - 2*delta) / delta^2
            return (h - 2 * delta) / (delta ** 2)

    def update(self, configuration: mink.Configuration):
        model = self.model
        data = configuration.data

        J_rows = []
        e_rows = []
        w_rows = []

        mj.mj_fwdPosition(model, data)

        for limit in self.collision_limits:
            d_min = float(limit.minimum_distance_from_collisions)
            detect_dist = float(limit.collision_detection_distance)

            for geom_a, geom_b in limit.geom_id_pairs:
                if geom_a < 0 or geom_b < 0:
                    continue

                dist = mj.mj_geomDistance(model, data, geom_a, geom_b, detect_dist, self._fromto)

                # Activate barrier only when inside the detection band (d < detect_dist)
                if dist >= detect_dist:
                    continue
                
                # Safety function for barrier: h(q) = d(q) - d_min
                h = float(dist - d_min)

                # Contact normal Jacobian: J_h = ∂h/∂q
                J_h = mink.limits.collision_avoidance_limit.compute_contact_normal_jacobian(
                    model, data, geom_a, geom_b,
                    self._fromto, self._normal, self._jac1, self._jac2,
                ).reshape(1, -1)

                dBdh = self._relaxed_barrier_gradient(h)
                J_bar = dBdh * J_h
                e_bar = 0.0
                J_rows.append(J_bar)
                e_rows.append(np.array([e_bar]))
                w_rows.append(np.array([self.w_bar]))

        if len(J_rows) == 0:
            self._J = np.zeros((0, model.nv))
            self._e = np.zeros((0,))
            self._w = np.zeros((0,))
            self.cost = self._w
            return

        self._J = np.vstack(J_rows)
        self._e = np.concatenate(e_rows)
        self._w = np.concatenate(w_rows)

        self.cost = self._w.copy()

    def compute_jacobian(self, configuration):
        return self._J

    def compute_error(self, configuration):
        return self._e

class FootContactLimit(mink.Limit):
    """
    Hard foot contact zero-velocity constraint, implemented as QP inequalities.

    Contact detection is based on the human motion position difference per frame:

        delta = ||human_pos_t - human_pos_{t-1}||

    If  delta  <=  threshold * dt  (i.e. human foot speed <= threshold [m/s]):
        Enforce  |J_c(q)[XY] @ v| <= velocity_bound   (hard QP inequality)

    Two-sided bound  |J_c @ v| <= velocity_bound  in G @ v <= h form.
    """
    def __init__(
        self,
        model: mj.MjModel,
        contact_points: list,       # list of (body_id, local_pos_np)
        human_body_names: list,     # list of human body names, one per contact point
        threshold: float = 0.01,    # m/s — contact activates when human foot speed <= threshold
        velocity_bound: float = 0.0,
        name: str = "foot_contact",
    ):
        self.model = model
        self.contact_points = contact_points
        self.human_body_names = human_body_names
        self.threshold = threshold
        self.velocity_bound = velocity_bound  # max contact-point XY velocity [m/s]
        self.name = name
        self._prev_human_pos = {}              # human_body_name -> np.ndarray(3,)
        self._active_mask = [False] * len(contact_points)  # per-point contact state

    def update_from_human_motion(self, human_data: dict):
        """Update contact active mask from human motion position differences.

        human_data: dict of {human_body_name: (pos, rot)} for the current frame
        """
        threshold_pos = self.threshold / 30  # convert m/s threshold to meters per frame (fixed 30 fps)
        for i, human_name in enumerate(self.human_body_names):
            if human_name not in human_data:
                self._active_mask[i] = False
                continue
            curr_pos = human_data[human_name][0]  # (3,) position
            if human_name in self._prev_human_pos:
                delta = np.linalg.norm(curr_pos - self._prev_human_pos[human_name])
                self._active_mask[i] = delta <= threshold_pos
            else:
                self._active_mask[i] = False  # no previous frame — cannot determine contact
            self._prev_human_pos[human_name] = curr_pos.copy()

    def compute_qp_inequalities(
        self,
        configuration: mink.Configuration,
        dt: float,
    ):
        model = self.model
        data = configuration.data

        mj.mj_fwdPosition(model, data)

        G_rows, h_rows = [], []
        J_pos = np.zeros((3, model.nv))
        J_rot = np.zeros((3, model.nv))

        for i, (body_id, local_pos) in enumerate(self.contact_points):
            if not self._active_mask[i]:
                continue

            body_xpos = data.xpos[body_id]
            body_xmat = data.xmat[body_id].reshape(3, 3)
            world_point = body_xpos + body_xmat @ local_pos

            J_pos[:] = 0.0
            J_rot[:] = 0.0
            mj.mj_jac(model, data, J_pos, J_rot, world_point, body_id)

            J_xy = J_pos[:2]               # XY rows only, (2, nv)
            G_rows.append(J_xy)
            G_rows.append(-J_xy)
            h_rows.append(np.full(2, self.velocity_bound))
            h_rows.append(np.full(2, self.velocity_bound))

        if not G_rows:
            return mink.limits.Constraint()   # inactive (G=None, h=None)

        return mink.limits.Constraint(
            G=np.vstack(G_rows),
            h=np.concatenate(h_rows),
        )


def find_geoms(model, names):
    gids = []
    for n in names:
        gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, n)
        if gid != -1:
            gids.append(gid)
    return gids

class GeneralMotionRetargeting:
    """General Motion Retargeting (GMR).
    """
    def __init__(
        self,
        src_human: str,
        tgt_robot: str,
        actual_human_height: float = None,
        solver: str="daqp", # change from "quadprog" to "daqp".
        damping: float=0.05,
        verbose: bool=True,
        use_velocity_limit: bool=True,
    ) -> None:
        self._warmup_done = False
        self._warmup_iters = 100

        # load the robot model
        self.xml_file = str(ROBOT_XML_DICT[tgt_robot])
        if verbose:
            print("Use robot model: ", self.xml_file)
        self.model = mj.MjModel.from_xml_path(self.xml_file)

        # Print DoF names in order
        print("[GMR] Robot Degrees of Freedom (DoF) names and their order:")
        self.robot_dof_names = {}
        for i in range(self.model.nv):  # 'nv' is the number of DoFs
            dof_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, self.model.dof_jntid[i])
            self.robot_dof_names[dof_name] = i
            if verbose:
                print(f"DoF {i}: {dof_name}")
            
            
        print("[GMR] Robot Body names and their IDs:")
        self.robot_body_names = {}
        for i in range(self.model.nbody):  # 'nbody' is the number of bodies
            body_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, i)
            self.robot_body_names[body_name] = i
            if verbose:
                print(f"Body ID {i}: {body_name}")

        
        print("[GMR] Robot Motor (Actuator) names and their IDs:")
        self.robot_motor_names = {}
        for i in range(self.model.nu):  # 'nu' is the number of actuators (motors)
            motor_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_ACTUATOR, i)
            self.robot_motor_names[motor_name] = i
            if verbose:
                print(f"Motor ID {i}: {motor_name}")

        # Load the IK config
        with open(IK_CONFIG_DICT[src_human][tgt_robot]) as f:
            ik_config = json.load(f)
        if verbose:
            print("Use IK config: ", IK_CONFIG_DICT[src_human][tgt_robot])
        
        # compute the scale ratio based on given human height and the assumption in the IK config
        if actual_human_height is not None:
            ratio = actual_human_height / ik_config["human_height_assumption"]
        else:
            ratio = 1.0
            
        # adjust the human scale table
        for key in ik_config["human_scale_table"].keys():
            ik_config["human_scale_table"][key] = ik_config["human_scale_table"][key] * ratio
    

        # used for retargeting
        self.ik_match_table1 = ik_config["ik_match_table1"]
        self.ik_match_table2 = ik_config["ik_match_table2"]
        self.human_root_name = ik_config["human_root_name"]
        self.use_ik_match_table1 = ik_config["use_ik_match_table1"]
        self.use_ik_match_table2 = ik_config["use_ik_match_table2"]
        self.human_scale_table = ik_config["human_scale_table"]
        self.ground = ik_config["ground_height"] * np.array([0, 0, 1])

        self.max_iter = 10
        self.solver = solver
        self.damping = damping
        self.vel_limit=3.0
        self.collision_weight = 1.0

        self.human_body_to_task1 = {}
        self.human_body_to_task2 = {}
        self.pos_offsets1 = {}
        self.rot_offsets1 = {}
        self.pos_offsets2 = {}
        self.rot_offsets2 = {}

        
        # task -> human body mapping for handling duplicate body_name assignments
        self.task_to_human_body1 = {}
        self.task_to_human_body2 = {}

        # Initialize IK constraints starting with joint configuration limits
        self.ik_limits = [mink.ConfigurationLimit(self.model)]

        # Load robot-specific collision parameters from an external YAML file
        collision_cfg_path = f"assets/{tgt_robot}/collision_cfg.yaml"
        with open(collision_cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)

        # Override IK solver parameters with values from the configuration
        params = cfg.get('parameters', {})
        self.damping = params.get('damping', damping)
        self.max_iter = params.get('max_iter', 10)
        self.vel_limit = params.get('velocity_limit', 10)
        
        # Global weight for the SOFT collision barrier task
        self.collision_weight = params.get('collision_avoidance_weight', 1.0)



        if use_velocity_limit:
            VELOCITY_LIMITS = {}
            for a_id in range(self.model.nu):
                j_id = int(self.model.actuator_trnid[a_id, 0])
                if j_id < 0:
                    continue
                j_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, j_id)
                if j_name is None:
                    continue
                VELOCITY_LIMITS[j_name] = self.vel_limit

            # Hard velocity bound
            self.ik_limits.append(mink.VelocityLimit(self.model, VELOCITY_LIMITS))



        
        if verbose:
            print(f"[GMR] Final Parameters ->  Damping: {self.damping}, Max Iterations: {self.max_iter}")
            print(f"Velocity Limit: {self.vel_limit}, Collision Avoidance weight: {self.collision_weight}"
)
        
        # Resolve collision groups and individual geometries
        self.groups = cfg['groups']
        self.all_collision_limits = []


        # Build collision pair metadata for the soft barrier task
        for limit_cfg in cfg['collision_limits']:
            geom_pairs = []
            
            for p_a, p_b in limit_cfg['pairs']:
                list_a = self.groups.get(p_a, [p_a] if isinstance(p_a, str) else p_a)
                list_b = self.groups.get(p_b, [p_b] if isinstance(p_b, str) else p_b)
                geom_pairs.append((list_a, list_b))

            # Provides geom pairs, d_min and detect_dist used in h(q) = d - d_min
            limit_obj = mink.CollisionAvoidanceLimit(
                model=self.model,
                geom_pairs=geom_pairs,
                minimum_distance_from_collisions=limit_cfg['margin'],
                collision_detection_distance=limit_cfg['detect_dist'],
            )
            self.all_collision_limits.append(limit_obj)            

        # Store foot contact config for setup after ik_match_tables are loaded
        self._fc_cfg = cfg.get('foot_contact', {})

        self.setup_retarget_configuration()

        # Set up foot contact limit after ik_match_tables are available (human body names derived from them)
        self.foot_contact_limit = None
        fc_cfg = self._fc_cfg
        if fc_cfg.get('enabled', False):
            fc_threshold      = fc_cfg.get('threshold', 0.01)
            fc_velocity_bound = fc_cfg.get('velocity_bound', 0.0)
            fc_points_cfg     = fc_cfg.get('contact_points', [])

            # Build reverse mapping: robot_body_name -> human_body_name from IK tables
            robot_to_human = {}
            for robot_body, entry in self.ik_match_table1.items():
                robot_to_human[robot_body] = entry[0]
            for robot_body, entry in self.ik_match_table2.items():
                if robot_body not in robot_to_human:
                    robot_to_human[robot_body] = entry[0]

            contact_points = []
            human_body_names = []
            for cp in fc_points_cfg:
                body_name = cp['body_name']
                human_body_name = robot_to_human.get(body_name, '')
                if not human_body_name:
                    print(f"[GMR][FootContact] WARNING: '{body_name}' not in IK match tables, skipping.")
                    continue
                local_pos = np.array(cp.get('local_pos', [0.0, 0.0, 0.0]), dtype=float)
                body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, body_name)
                if body_id == -1:
                    print(f"[GMR][FootContact] WARNING: robot body '{body_name}' not found in model, skipping.")
                    continue
                contact_points.append((body_id, local_pos))
                human_body_names.append(human_body_name)

            if contact_points:
                self.foot_contact_limit = FootContactLimit(
                    model=self.model,
                    contact_points=contact_points,
                    human_body_names=human_body_names,
                    threshold=fc_threshold,
                    velocity_bound=fc_velocity_bound,
                )
                self.ik_limits.append(self.foot_contact_limit)
                if verbose:
                    # contact_points contains resolved (body_id, local_pos); use human_body_names for display
                    robot_names = [mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, bid) for bid, _ in contact_points]
                    pairs = list(zip(robot_names, human_body_names))
                    print(f"[GMR] FootContactLimit enabled | {pairs} | "
                          f"threshold: {fc_threshold} m/s | human_fps: 30 Hz | "
                          f"threshold_pos: {fc_threshold / 30 * 1000:.3f} mm/frame | "
                          f"velocity_bound: {fc_velocity_bound} m/s")
            else:
                print("[GMR][FootContact] WARNING: no valid contact points found, limit disabled.")

        self.ground_offset = -0.01
        self.floor_gid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "floor")
        left_candidates  = ["Left_Foot", "Left_Inner_Foot", "Left_Outer_Foot", "left_ankle_roll_link",
                             "left_foot1_collision", "left_foot2_collision", "left_foot3_collision",
                             "left_foot4_collision", "left_foot5_collision", "left_foot6_collision",
                             "left_foot7_collision"]
        right_candidates = ["Right_Foot", "Right_Inner_Foot", "Right_Outer_Foot", "right_ankle_roll_link",
                             "right_foot1_collision", "right_foot2_collision", "right_foot3_collision",
                             "right_foot4_collision", "right_foot5_collision", "right_foot6_collision",
                             "right_foot7_collision"]

        self.left_foot_gids  = find_geoms(self.model, left_candidates)
        self.right_foot_gids = find_geoms(self.model, right_candidates)

        if not self.left_foot_gids:
            raise ValueError(f"No left foot geom found. Tried {left_candidates}")
        if not self.right_foot_gids:
            raise ValueError(f"No right foot geom found. Tried {right_candidates}")

        

        assert self.floor_gid != -1, "geom 'floor' not found"





    def setup_retarget_configuration(self):
        self.configuration = mink.Configuration(self.model)
        
        # targets: tasks that require set_target() every frame (from human_data)
        # solver : tasks actually passed to solve_ik() (targets + regularizers/barriers)
        self.tasks1_targets = []
        self.tasks1_solver = []

        self.tasks2_targets = []
        self.tasks2_solver = [] 
        
        # reset mappings
        self.human_body_to_task1 = {}
        self.human_body_to_task2 = {}
        self.task_to_human_body1 = {}
        self.task_to_human_body2 = {}

        self.pos_offsets1 = {}
        self.rot_offsets1 = {}
        self.pos_offsets2 = {}
        self.rot_offsets2 = {}


        # ----------------------------
        # Table 1: build FrameTasks
        # ----------------------------
        for frame_name, entry in self.ik_match_table1.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight == 0 and rot_weight == 0:
                continue

            task = mink.FrameTask(
                frame_name=frame_name,
                frame_type="body",
                position_cost=pos_weight,
                orientation_cost=rot_weight,
                lm_damping=1,
            )
            
            # NOTE: Multiple robot tasks may track the same human body part. 
            # Using the Task object as a key prevents data loss from duplicate body names.
            self.human_body_to_task1[body_name] = task
            self.task_to_human_body1[task] = body_name

            self.pos_offsets1[body_name] = np.array(pos_offset) - self.ground
            self.rot_offsets1[body_name] = R.from_quat(rot_offset, scalar_first=True)

            self.tasks1_targets.append(task)
            self.tasks1_solver.append(task)

        # ----------------------------
        # Table 2: build FrameTasks
        # ----------------------------
        for frame_name, entry in self.ik_match_table2.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight == 0 and rot_weight == 0:
                continue

            task = mink.FrameTask(
                frame_name=frame_name,
                frame_type="body",
                position_cost=pos_weight,
                orientation_cost=rot_weight,
                lm_damping=1,
            )

            self.human_body_to_task2[body_name] = task
            self.task_to_human_body2[task] = body_name

            self.pos_offsets2[body_name] = np.array(pos_offset) - self.ground
            self.rot_offsets2[body_name] = R.from_quat(rot_offset, scalar_first=True)

            self.tasks2_targets.append(task)
            self.tasks2_solver.append(task)

        # Soft collision cost added to both stages
        self.collision_barrier_task = CollisionBarrierTask(
            model=self.model,
            collision_limits=self.all_collision_limits,
            w_bar = self.collision_weight,
        )
        self.tasks1_solver.append(self.collision_barrier_task)
        self.tasks2_solver.append(self.collision_barrier_task)


    def calibrate_offsets_from_initial_pose(self, human_data_init, verbose=True):
        """Replace JSON-loaded rotation offsets with values derived from the
        initial robot pose (qpos0) and the initial human motion frame.

        For each (robot_body, human_body) pair active in the IK match tables:
            R_off = R_human_init^{-1} @ R_robot_init
        which guarantees R_target = R_human * R_off equals R_robot at frame 0.

        Must be called before the first retarget(). Robot configuration is
        not modified.
        """
        tmp_data = mj.MjData(self.model)
        mj.mj_forward(self.model, tmp_data)

        def _calibrate(match_table, rot_offsets_store, tag):
            for robot_body, entry in match_table.items():
                human_body, pos_w, rot_w = entry[0], entry[1], entry[2]
                if pos_w == 0 and rot_w == 0:
                    continue
                bid = self.robot_body_names.get(robot_body)
                if bid is None:
                    print(f"[CAL][{tag}] robot body '{robot_body}' not in model, keep JSON value.")
                    continue
                if human_body not in human_data_init:
                    print(f"[CAL][{tag}] human body '{human_body}' missing in init frame, keep JSON value.")
                    continue
                R_robot = R.from_matrix(tmp_data.xmat[bid].reshape(3, 3).copy())
                quat = np.asarray(human_data_init[human_body][1])
                R_human = R.from_quat(quat, scalar_first=True)
                R_off = R_human.inv() * R_robot
                rot_offsets_store[human_body] = R_off
                if verbose:
                    eh = R_human.as_euler('xyz', degrees=True)
                    er = R_robot.as_euler('xyz', degrees=True)
                    eo = R_off.as_euler('xyz', degrees=True)
                    ej = R.from_quat(np.asarray(entry[4]), scalar_first=True).as_euler('xyz', degrees=True)
                    def _fmt(e):
                        return f"[{e[0]:+7.2f}, {e[1]:+7.2f}, {e[2]:+7.2f}]"
                    print(f"[CAL][{tag}] {robot_body:26s} <- {human_body:14s} | "
                          f"R_h={_fmt(eh)} R_r={_fmt(er)} off_auto={_fmt(eo)} off_json={_fmt(ej)} (deg, xyz)")

        _calibrate(self.ik_match_table1, self.rot_offsets1, "Table1")
        _calibrate(self.ik_match_table2, self.rot_offsets2, "Table2")




    def update_targets(self, human_data, offset_to_ground=False):
        # scale/offset human data
        human_data = self.to_numpy(human_data)
        human_data = self.scale_human_data(human_data, self.human_root_name, self.human_scale_table)

        # Apply Table 1 spatial offsets (only for bodies with defined offset entries)
        human_data = self.offset_human_data(human_data, self.pos_offsets1, self.rot_offsets1)
        human_data = self.apply_ground_offset(human_data)

        if offset_to_ground:
            human_data = self.offset_human_data_to_ground(human_data)

        self.scaled_human_data = human_data

        # CORE: Iterate through all tasks in Table 1 and set IK targets using the mapping
        if self.use_ik_match_table1:
            for task in self.tasks1_targets:
                body_name = self.task_to_human_body1[task]
                if body_name not in human_data:
                    # Skip target update and log a warning if the required body data is missing.
                    print(f"[WARN] human_data missing key for tasks1: {body_name}")
                    continue
                pos, rot = human_data[body_name]
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))

        if self.use_ik_match_table2:
            for task in self.tasks2_targets:
                body_name = self.task_to_human_body2[task]
                if body_name not in human_data:
                    print(f"[WARN] human_data missing key for tasks2: {body_name}")
                    continue
                pos, rot = human_data[body_name]
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))

    # Utility function to monitor and log active collisions during the IK process
    def log_collision_warning(self, stage_name, num_iter):
        mj.mj_fwdPosition(self.model, self.configuration.data)
        
        fromto = np.zeros(6, dtype=np.float64)

        for limit_obj in self.all_collision_limits:
            current_limit = limit_obj.minimum_distance_from_collisions
            
            for id_a, id_b in limit_obj.geom_id_pairs:
                if id_a == -1 or id_b == -1:
                    continue

                # Compute the shortest distance between two geometries
                dist = mj.mj_geomDistance(self.model, self.configuration.data, id_a, id_b, 8.0, fromto)

                # Log a detailed warning if a penetration (collision) is detected
                if dist <= -0.02:
                    name_a = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, id_a)
                    name_b = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, id_b)
                    
                    p1 = self.configuration.data.geom_xpos[id_a]
                    p2 = self.configuration.data.geom_xpos[id_b]
                    c_dist = np.linalg.norm(p1 - p2)
                    
                    msg = (f"[bold red][COLLISION][/bold red] {stage_name} | Iter:{num_iter} | "
                           f"{name_a} <-> {name_b} | "
                           f"Dist:{dist:.4f} (Limit:{current_limit:.3f}) | CenterDist:{c_dist:.4f}")
                    print(msg)

    def retarget(self, human_data, offset_to_ground=False):
        self.update_targets(human_data, offset_to_ground)
        dt = self.configuration.model.opt.timestep

        if not self._warmup_done:
            for _ in range(self._warmup_iters):
                self.collision_barrier_task.update(self.configuration)
                vel = mink.solve_ik(
                    self.configuration,
                    self.tasks1_solver,
                    dt,
                    self.solver,
                    damping=self.damping,
                    limits=None,
                )
                self.configuration.integrate_inplace(vel, dt)
            self._warmup_done = True

        # Update foot contact mask once per frame from human motion position differences
        if self.foot_contact_limit is not None:
            self.foot_contact_limit.update_from_human_motion(self.scaled_human_data)

        if self.use_ik_match_table1 and len(self.tasks1_solver) > 0:
            num_iter = 0
            curr_error = self.error1()

            while num_iter < self.max_iter:
                # Recompute barrier rows at the current q before each IK QP solve
                self.collision_barrier_task.update(self.configuration)
                vel1 = mink.solve_ik(
                    self.configuration,
                    self.tasks1_solver,
                    dt,
                    self.solver,
                    damping=self.damping,
                    limits=self.ik_limits,
                )
                self.configuration.integrate_inplace(vel1, dt)
                next_error = self.error1()

                if abs(curr_error - next_error) < 1e-8:
                    break
                curr_error = next_error
                num_iter += 1

            self.log_collision_warning("Table1", num_iter)

        if self.use_ik_match_table2 and len(self.tasks2_solver) > 0:
            num_iter = 0
            curr_error = self.error2()

            while num_iter < self.max_iter:
                self.collision_barrier_task.update(self.configuration)
                vel2 = mink.solve_ik(
                    self.configuration,
                    self.tasks2_solver,
                    dt,
                    self.solver,
                    damping=self.damping,
                    limits=self.ik_limits,
                )
                self.configuration.integrate_inplace(vel2, dt)
                next_error = self.error2()
                if abs(curr_error - next_error) < 1e-8:
                    break
                curr_error = next_error
                num_iter += 1

            self.log_collision_warning("Table2", num_iter)
            perf = {"stage": "Table2", "iter": num_iter}

        mj.mj_fwdPosition(self.model, self.configuration.data)
        mj.mj_forward(self.model, self.configuration.data)

        return self.configuration.data.qpos.copy()

    def get_contact_point_positions(self) -> list:
        """Return [(world_pos, is_active), ...] for each foot contact point.
        is_active=True  → constraint was enforced in the last IK solve (foot in contact).
        is_active=False → foot moving freely.
        Returns empty list if foot contact is disabled.
        """
        if self.foot_contact_limit is None:
            return []
        data = self.configuration.data
        result = []
        for i, (body_id, local_pos) in enumerate(self.foot_contact_limit.contact_points):
            body_xpos = data.xpos[body_id]
            body_xmat = data.xmat[body_id].reshape(3, 3)
            pos = body_xpos + body_xmat @ local_pos
            result.append((pos, self.foot_contact_limit._active_mask[i]))
        return result

    def error1(self):
        errs = []
        unset = []
        for task in self.tasks1_targets:
            try:
                errs.append(task.compute_error(self.configuration))
            except TargetNotSet:
                name = getattr(task, "frame_name", None) or getattr(task, "name", None) or task.__class__.__name__
                unset.append(str(name))

        if unset:
            print(f"[WARN] FrameTask target not set (tasks1, skipped): {unset[:10]}{' ...' if len(unset) > 10 else ''}")

        if len(errs) == 0:
            return 0.0
        return np.linalg.norm(np.concatenate(errs))


    def error2(self):
        errs = []
        unset = []
        for task in self.tasks2_targets:
            try:
                errs.append(task.compute_error(self.configuration))
            except TargetNotSet:
                name = getattr(task, "frame_name", None) or getattr(task, "name", None) or task.__class__.__name__
                unset.append(str(name))

        if unset:
            print(f"[WARN] FrameTask target not set (tasks2, skipped): {unset[:10]}{' ...' if len(unset) > 10 else ''}")

        if len(errs) == 0:
            return 0.0
        return np.linalg.norm(np.concatenate(errs))


    def to_numpy(self, human_data):
        for body_name in human_data.keys():
            human_data[body_name] = [np.asarray(human_data[body_name][0]), np.asarray(human_data[body_name][1])]
        return human_data


    def scale_human_data(self, human_data, human_root_name, human_scale_table):
        
        human_data_local = {}
        root_pos, root_quat = human_data[human_root_name]
        
        # scale root
        scaled_root_pos = human_scale_table[human_root_name] * root_pos
        
        # scale other body parts in local frame
        for body_name in human_data.keys():
            if body_name not in human_scale_table:
                continue
            if body_name == human_root_name:
                continue
            else:
                # transform to local frame (only position)
                human_data_local[body_name] = (human_data[body_name][0] - root_pos) * human_scale_table[body_name]
            
        # transform the human data back to the global frame
        human_data_global = {human_root_name: (scaled_root_pos, root_quat)}
        for body_name in human_data_local.keys():
            human_data_global[body_name] = (human_data_local[body_name] + scaled_root_pos, human_data[body_name][1])

        return human_data_global
    
    def offset_human_data(self, human_data, pos_offsets, rot_offsets):
        """the pos offsets are applied in the local frame"""
        offset_human_data = {}
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            offset_human_data[body_name] = [pos, quat]
            # apply rotation offset first
            updated_quat = (R.from_quat(quat, scalar_first=True) * rot_offsets[body_name]).as_quat(scalar_first=True)
            offset_human_data[body_name][1] = updated_quat
            
            local_offset = pos_offsets[body_name]
            # compute the global position offset using the updated rotation
            global_pos_offset = R.from_quat(updated_quat, scalar_first=True).apply(local_offset)
            
            offset_human_data[body_name][0] = pos + global_pos_offset
           
        return offset_human_data
            
    def offset_human_data_to_ground(self, human_data):
        """find the lowest point of the human data and offset the human data to the ground"""
        offset_human_data = {}
        ground_offset = 0.1
        lowest_pos = np.inf

        for body_name in human_data.keys():
            # only consider the foot/Foot
            if "Foot" not in body_name and "foot" not in body_name:
                continue
            pos, quat = human_data[body_name]
            if pos[2] < lowest_pos:
                lowest_pos = pos[2]
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            offset_human_data[body_name] = [pos, quat]
            offset_human_data[body_name][0] = pos - np.array([0, 0, lowest_pos]) + np.array([0, 0, ground_offset])
        return offset_human_data

    def set_ground_offset(self, ground_offset):
        self.ground_offset = ground_offset

    def apply_ground_offset(self, human_data):
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            human_data[body_name][0] = pos - np.array([0, 0, self.ground_offset])
        return human_data
