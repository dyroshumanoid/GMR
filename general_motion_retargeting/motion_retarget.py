import mink
import mujoco as mj
import numpy as np
import json
import yaml
from scipy.spatial.transform import Rotation as R
from .params import ROBOT_XML_DICT, IK_CONFIG_DICT
from rich import print
from mink.exceptions import TargetNotSet
from mink.tasks.damping_task import DampingTask
<<<<<<< Updated upstream
def cubic_hermite(time, time_0, time_f, x_0, x_f, x_dot_0=0.0, x_dot_f=0.0):

    if time < time_0:
        return float(x_0)
    if time > time_f:
        return float(x_f)

    elapsed = time - time_0
    T = time_f - time_0
    T2 = T * T
    T3 = T2 * T
    dx = x_f - x_0

    return float(
        x_0
        + x_dot_0 * elapsed
        + (3 * dx / T2 - 2 * x_dot_0 / T - x_dot_f / T) * elapsed * elapsed
        + (-2 * dx / T3 + (x_dot_0 + x_dot_f) / T2) * elapsed * elapsed * elapsed
    )

def damping_from_distance(dist, eps, detect_dist, d_min, d_max, slope_eps=0.0, slope_detect=0.0):

    if detect_dist <= eps:
        return float(d_max if dist <= eps else d_min)

    return cubic_hermite(
        time=dist,
        time_0=eps,
        time_f=detect_dist,
        x_0=d_max,
        x_f=d_min,
        x_dot_0=slope_eps,
        x_dot_f=slope_detect,
    )


class AdaptiveJointDamping:
    def __init__(
        self,
        model,
        all_collision_limits,
        d_min,
        d_max,
        *,
        log_every=200,
        log_topk=12,
        verbose_pairs=False,
        concise_log=True,
        round_digits=3,
        name_by_limit=None,
    ):
        self.model = model
        self.limits = all_collision_limits
        self.d_min = float(d_min)
        self.d_max = float(d_max)

        self.lambda_vec = np.full(model.nv, self.d_min, dtype=float)
        self.task = DampingTask(model, cost=self.lambda_vec.copy())
        self._fromto = np.zeros(6, dtype=np.float64)

        self._step = 0
        self.log_every = int(log_every)
        self.log_topk = int(log_topk)
        self.verbose_pairs = bool(verbose_pairs)
        self.name_by_limit = name_by_limit or {}
        
        self._prev_lam = self.lambda_vec.copy()
        self.concise_log = bool(concise_log)
        self.round_digits = int(round_digits)

        self._left_arm_joint_prefixes = (
            "left_shoulder_", "left_elbow_", "left_wrist_"
        )

    def _dof_to_joint_name(self, dof_i: int) -> str:
        j_id = int(self.model.dof_jntid[dof_i])
        name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, j_id)
        return name if name is not None else f"joint#{j_id}"
    
    def _joint_and_dof_label(self, dof_i: int) -> str:
        j_id = int(self.model.dof_jntid[dof_i])
        j_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, j_id)
        if j_name is None:
            j_name = f"joint#{j_id}"
        return j_name


    def update(self, data):
        self._step += 1
        mj.mj_fwdPosition(self.model, data)

        lam = np.full(self.model.nv, self.d_min, dtype=float)


        left_arm_cause_by_dof = {}

        for li, limit_obj in enumerate(self.limits):
            detect_dist = float(getattr(limit_obj, "collision_detection_distance"))
            eps = float(getattr(limit_obj, "minimum_distance_from_collisions"))

            for id_a, id_b in limit_obj.geom_id_pairs:
                if id_a == -1 or id_b == -1:
                    continue

                dist = mj.mj_geomDistance(self.model, data, id_a, id_b, 8.0, self._fromto)
                if dist >= detect_dist:
                    continue

                d_val = damping_from_distance(
                    dist=dist, eps=eps, detect_dist=detect_dist,
                    d_min=self.d_min, d_max=self.d_max
                )

                dof_subset = self._get_affected_dofs(id_a, id_b)
                


                lam[dof_subset] = np.maximum(lam[dof_subset], d_val)

                name_a = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, id_a) or f"geom#{id_a}"
                name_b = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, id_b) or f"geom#{id_b}"
                # if "hand" in name_a.lower() or "hand" in name_b.lower():
                #     print(f"\n[DEBUG_FOOT] Triggered by: {name_a} <-> {name_b}")
                #     print(f"[DEBUG_FOOT] Affected Joint Names (Chain to Root):")
                #     # DoF 인덱스를 Joint 이름으로 변환해서 출력
                #     joint_names = sorted(list(set([self._dof_to_joint_name(d) for d in dof_subset])))
                #     for j_name in joint_names:
                #         print(f"  - {j_name}")
                #     print("-" * 30)

                for dof_i in dof_subset:
                    if not self._is_left_arm_dof(dof_i):
                        continue
                    prev = left_arm_cause_by_dof.get(dof_i)
                    if (prev is None) or (dist < prev[0]):
                        left_arm_cause_by_dof[dof_i] = (float(dist), name_a, name_b, float(d_val))

        # finalize
        self.lambda_vec[:] = lam
        self.task.cost = self.lambda_vec.copy()

        if self.log_every > 0 and (self._step % self.log_every == 0):
            left_arm_dofs = [i for i in range(self.model.nv) if self._is_left_arm_dof(i)]

            inc_rows = []
            for i in left_arm_dofs:
                old = float(self._prev_lam[i])
                new = float(lam[i])
                if new > old + 1e-12:
                    inc_rows.append((new - old, i, old, new))

            if len(inc_rows) == 0:
                print(f"[ADAPT_DAMP][L-ARM] step={self._step} no change")
            else:
                inc_rows.sort(reverse=True, key=lambda x: x[0])
                top = inc_rows[: self.log_topk]

                print(f"[ADAPT_DAMP][L-ARM] step={self._step} changed={len(inc_rows)}/{len(left_arm_dofs)}")
                for _, dof_i, old, new in top:
                    jname = self._joint_and_dof_label(dof_i)  # dof->joint name
                    cause = left_arm_cause_by_dof.get(dof_i)

                    if cause is None:
                        print(f"  dof[{dof_i:02d}] {jname}: {old:.3f} -> {new:.3f}")
                    else:
                        dist, ga, gb, dv = cause
                        print(
                            f"  dof[{dof_i:02d}] {jname}: {old:.3f} -> {new:.3f} | "
                            f"cause={ga} <-> {gb} (dist={dist:.4f}, d_val={dv:.3f})"
                        )

        self._prev_lam[:] = lam

    def _is_left_arm_dof(self, dof_i: int) -> bool:
        j_id = int(self.model.dof_jntid[dof_i])
        j_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, j_id) or ""
        return any(j_name.startswith(p) for p in self._left_arm_joint_prefixes)

    def _get_affected_dofs(self, id_a, id_b):
        body_a = self.model.geom_bodyid[id_a]
        body_b = self.model.geom_bodyid[id_b]

        dofs = set()
        for b_id in [body_a, body_b]:
            curr = b_id
            while curr > 1:
                jnt_adr = self.model.body_jntadr[curr]
                for k in range(self.model.body_jntnum[curr]):
                    j_id = jnt_adr + k
                    d_adr = self.model.jnt_dofadr[j_id]
                    j_type = self.model.jnt_type[j_id]
                    d_num = {mj.mjtJoint.mjJNT_FREE: 6, mj.mjtJoint.mjJNT_BALL: 3}.get(j_type, 1)
                    dofs.update(range(d_adr, d_adr + d_num))
                curr = self.model.body_parentid[curr]
        return list(dofs)

=======
from mink.limits.limit import Limit, Constraint

class CollisionBarrierTask(mink.Task):
    def __init__(
        self,
        model: mj.MjModel,
        collision_limits,
        w_bar: float = 0.0,
        h_eps: float = 1e-4,
        invh_max: float = 1e3,
        name: str = "collision_barrier",
    ):
        super().__init__(cost=np.zeros(0))
        self.model = model
        self.collision_limits = collision_limits

        self.w_bar = float(w_bar)
        self.h_eps = float(h_eps)
        self.invh_max = float(invh_max)
        self.name = name

        self._J = np.zeros((0, model.nv))
        self._e = np.zeros((0,))
        self._w = np.zeros((0,))
        self._fromto = np.zeros(6, dtype=np.float64)

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

                if dist >= detect_dist:
                    continue

                h = float(dist - d_min)


                J_h = mink.limits.collision_avoidance_limit.compute_contact_normal_jacobian(
                    model, data,
                    mink.limits.collision_avoidance_limit.Contact(
                        dist=dist,
                        fromto=self._fromto.copy(),
                        geom1=geom_a,
                        geom2=geom_b,
                        distmax=detect_dist,
                    )
                ).reshape(1, -1)

                h_eff = max(h, self.h_eps)
                invh = 1.0 / h_eff
                invh = min(invh, self.invh_max)

                J_bar = -invh * J_h
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

def find_geoms(model, names):
    gids = []
    for n in names:
        gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, n)
        if gid != -1:
            gids.append(gid)
    return gids
def find_free_base_body_id(model: mj.MjModel):
    for j_id in range(model.njnt):
        if model.jnt_type[j_id] == mj.mjtJoint.mjJNT_FREE:
            return int(model.jnt_bodyid[j_id])
    return -1
>>>>>>> Stashed changes

def compute_cam_yaw_jacobian(model, data, root_body_id=1):
    nv = model.nv
    J_yaw = np.zeros(nv)

    qvel_orig = data.qvel.copy()
    qacc_orig = data.qacc.copy()

    for i in range(nv):
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        data.qvel[i] = 1.0

        mj.mj_fwdVelocity(model, data)
        mj.mj_subtreeVel(model, data)

        J_yaw[i] = data.subtree_angmom[root_body_id][2]

    data.qvel[:] = qvel_orig
    data.qacc[:] = qacc_orig
    mj.mj_forward(model, data)

    return J_yaw.reshape(1, -1)

class CentroidalYawTask(mink.Task):
    def __init__(self, model, weight=1.0, root_body_id=1):
        super().__init__(cost=np.array([float(weight)])) 
        self.model = model
        self.root_body_id = root_body_id
        self._dbg_count = 0

    
    def compute_jacobian(self, configuration):
        J = compute_cam_yaw_jacobian(self.model, configuration.data, self.root_body_id)
        self._dbg_count += 1
        if self._dbg_count % 100 == 0:
            print(f"[CAM] cost={self.cost}, ||J||={np.linalg.norm(J):.4f}")
        return J
    def compute_error(self, configuration):
        return np.zeros(1)

class GeneralMotionRetargeting:
    """General Motion Retargeting (GMR).
    """
    def __init__(
        self,
        src_human: str,
        tgt_robot: str,
        actual_human_height: float = None,
        solver: str="daqp", # change from "quadprog" to "daqp".
        damping: float=0.05, # Default value; will be overwritten by collision_cfg.yaml if provided.
        verbose: bool=True,
        use_velocity_limit: bool=True,
        cam_weight: float = 1000.0
    ) -> None:

        # load the robot model
        self.xml_file = str(ROBOT_XML_DICT[tgt_robot])
        if verbose:
            print("Use robot model: ", self.xml_file)
        self.model = mj.MjModel.from_xml_path(self.xml_file)

        self.total_steps = 0
        self.performance_logs = []
        self._dbg_count = 0


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
        self.robot_root_name = ik_config["robot_root_name"]
        self.use_ik_match_table1 = ik_config["use_ik_match_table1"]
        self.use_ik_match_table2 = ik_config["use_ik_match_table2"]
        self.human_scale_table = ik_config["human_scale_table"]
        self.ground = ik_config["ground_height"] * np.array([0, 0, 1])

        self.max_iter = 10 # Default value; will be overwritten by collision_cfg.yaml if provided.

        self.solver = solver
        self.damping = damping
        self.vel_limit=3.0
        self.cam_cost = 100
        self.collision_weight = 1.0

        self.human_body_to_task1 = {}
        self.human_body_to_task2 = {}
        self.pos_offsets1 = {}
        self.rot_offsets1 = {}
        self.pos_offsets2 = {}
        self.rot_offsets2 = {}

        self.task_errors1 = {}
        self.task_errors2 = {}
        
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
<<<<<<< Updated upstream
        self.damping_max = params.get('damping_max', self.damping * 50.0)
=======
        self.vel_limit = params.get('velocity_limit', 10)
        self.cam_cost = params.get('cam_weight', 100)
        self.collision_weight = params.get('collision_avoidance_weight', 1.0)
        # self.damping_max = params.get('damping_max', self.damping * 50.0)
        self.cam_weight = self.cam_cost



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

            self.ik_limits.append(mink.VelocityLimit(self.model, VELOCITY_LIMITS))



>>>>>>> Stashed changes
        
        if verbose:
            print(f"[GMR] Final Parameters ->  Damping: {self.damping}, Max Iterations: {self.max_iter}")
            print(f"Velocity Limit: {self.vel_limit}, Cam weight: {self.cam_cost}, Collision Avoidance weight: {self.collision_weight}"
)
        
        # Resolve collision groups and individual geometries
        self.groups = cfg['groups']
        self.all_collision_limits = []

        self.limit_name_by_id = {}


        # Iterate through defined collision pairs and register them as IK limits
        for limit_cfg in cfg['collision_limits']:
            geom_pairs = []
            
            for p_a, p_b in limit_cfg['pairs']:
                # Support both pre-defined groups and individual geometry names
                list_a = self.groups.get(p_a, [p_a] if isinstance(p_a, str) else p_a)
                list_b = self.groups.get(p_b, [p_b] if isinstance(p_b, str) else p_b)
                geom_pairs.append((list_a, list_b))

            # Define the collision avoidance limit with safety margins and gains
            limit_obj = mink.CollisionAvoidanceLimit(
                model=self.model,
                geom_pairs=geom_pairs,
                minimum_distance_from_collisions=limit_cfg['margin'],
                collision_detection_distance=limit_cfg['detect_dist'],
                gain=limit_cfg.get('gain', 500.0),
            )

            # self.ik_limits.append(limit_obj)
            
            self.all_collision_limits.append(limit_obj)
            self.limit_name_by_id[id(limit_obj)] = limit_cfg.get("name", "unnamed")
            

        self.setup_retarget_configuration()
        self.ground_offset = 0.0
        self.floor_gid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "floor")
        left_candidates  = ["Left_Foot", "Left_Inner_Foot", "Left_Outer_Foot"]
        right_candidates = ["Right_Foot", "Right_Inner_Foot", "Right_Outer_Foot"]

        self.left_foot_gids  = find_geoms(self.model, left_candidates)
        self.right_foot_gids = find_geoms(self.model, right_candidates)

        if not self.left_foot_gids:
            raise ValueError(f"No left foot geom found. Tried {left_candidates}")
        if not self.right_foot_gids:
            raise ValueError(f"No right foot geom found. Tried {right_candidates}")

        

        assert self.floor_gid != -1, "geom 'floor' not found"



        self.adaptive_damping = AdaptiveJointDamping(
            self.model,
            self.all_collision_limits,
            d_min=self.damping,
            d_max=self.damping_max,
            log_every=20,
            log_topk=10,
            verbose_pairs=True,
            name_by_limit=self.limit_name_by_id
        )



    def setup_retarget_configuration(self):
        self.configuration = mink.Configuration(self.model)
        self.tasks1_targets = []
        self.tasks1_solver = []

        # self.tasks2 = []
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

        self.task_errors1 = {}
        self.task_errors2 = {}

        # table1
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
            self.task_errors1[task] = []

        # table2
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
            self.task_errors2[task] = []

        root_body_id = find_free_base_body_id(self.model)
        assert root_body_id != -1, "No FREE joint found (floating base)."
        self.cam_yaw_task = CentroidalYawTask(
            self.model,
            weight=self.cam_weight,
            root_body_id=root_body_id,
        )
        self.tasks2_solver.append(self.cam_yaw_task)
        self.collision_barrier_task = CollisionBarrierTask(
            model=self.model,
            collision_limits=self.all_collision_limits,
            w_bar = self.collision_weight,
            h_eps = 1e-4,
            invh_max = 1e3,
        )
        self.tasks2_solver.append(self.collision_barrier_task)
        self.tasks1_solver.append(self.collision_barrier_task)





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
                if dist <= 0:
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


        if self.use_ik_match_table1 and len(self.tasks1_solver) > 0:
            num_iter = 0
            curr_error = self.error1()

            while num_iter < self.max_iter:
<<<<<<< Updated upstream
                self.adaptive_damping.update(self.configuration.data)

                vel1 = mink.solve_ik(
                    self.configuration,
                    self.tasks1 + [self.adaptive_damping.task], 
=======
                self.collision_barrier_task.update(self.configuration)
                vel1 = mink.solve_ik(
                    self.configuration,
                    self.tasks1_solver, 
>>>>>>> Stashed changes
                    dt,
                    self.solver,
                    damping=0.0,
                    limits=self.ik_limits,
                )
                self.configuration.integrate_inplace(vel1, dt)
                self._vprev_t1 = vel1.copy()
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
<<<<<<< Updated upstream
                self.adaptive_damping.update(self.configuration.data)

=======
                self.collision_barrier_task.update(self.configuration)
>>>>>>> Stashed changes
                vel2 = mink.solve_ik(
                    self.configuration,
                    self.tasks2_solver + [self.adaptive_damping.task], 
                    dt,
                    self.solver,
                    damping=0.0,
                    limits=self.ik_limits,
                )

                if num_iter == 0:
                    J_yaw = self.cam_yaw_task.compute_jacobian(self.configuration)
                    predicted_cam = (J_yaw @ vel2)[0]
                    self._dbg_count += 1
                    if self._dbg_count % 10 == 0:
                        print(f"[RESULT] Yaw CAM: {predicted_cam:.8f}")

                self.configuration.integrate_inplace(vel2, dt)
                next_error = self.error2()
                if abs(curr_error - next_error) < 1e-8:
                    break
                curr_error = next_error
                num_iter += 1

            self.log_collision_warning("Table2", num_iter)
            perf = {"stage": "Table2", "iter": num_iter}
            self.performance_logs.append(perf)

        mj.mj_fwdPosition(self.model, self.configuration.data)
        mj.mj_forward(self.model, self.configuration.data)
        
        return self.configuration.data.qpos.copy()


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
                lowest_body_name = body_name
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
