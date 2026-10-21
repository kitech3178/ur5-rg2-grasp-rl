"""MuJoCo backend (rigid-body dynamics + soft contacts + OpenGL depth camera).

Runs the *same* MDP as the simple backend:
  * the robot is the URDF generated from the config (grasp_rl/urdf.py), loaded
    with MuJoCo's URDF importer and extended with actuators / camera via MjSpec,
  * RL actions -> TCP target -> the shared UR5 IK -> joint position servos
    (interpolated in joint space at `mujoco.move_speed`, gravity compensated),
  * the RG2 fingers are force-limited position servos (|force| <= grip force)
    that close at the RG2 speed for the configured force,
  * the wrist camera is a MuJoCo camera on the TCP link with the same
    intrinsics / mounting as `physics/depth_camera.py`, rendered off-screen,
  * contacts, friction (incl. torsional), slip and dropping come from MuJoCo
    instead of the analytic model.

Only pip packages are needed (`pip install mujoco`), CPU only. Off-screen
rendering uses `MUJOCO_GL` (glfw when a display exists, else egl / osmesa).
"""

from __future__ import annotations

import copy
import os
import time

import numpy as np

from grasp_rl.backends.base import Backend, MoveResult
from grasp_rl.physics.objects import sample_object
from grasp_rl.physics.rg2_gripper import GraspResult
from grasp_rl.physics.ur5_kinematics import rot_x
from grasp_rl.urdf import ARM_JOINTS, FINGER_JOINTS, generate_urdf

# geom groups: robot visual meshes come in as group 1 from the URDF importer
GROUP_ROBOT_VISUAL, GROUP_SCENE, GROUP_ROBOT_COLLISION = 1, 2, 3
# collision bits: robot <-> scene and scene <-> scene collide, robot <-> robot does not
ROBOT_BITS, SCENE_BITS = (2, 1), (1, 3)


def _import_mujoco(gl):
    if gl and "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = str(gl)
    elif "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "glfw" if os.environ.get("DISPLAY") else "egl"
    import mujoco
    return mujoco


def _mat2quat(mujoco, R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=float).reshape(-1))
    return q


def _yaw_from_quat(q):
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _box_inertia(m, s):
    sx, sy, sz = s
    return m / 12.0 * np.array([sy * sy + sz * sz, sx * sx + sz * sz, sx * sx + sy * sy])


def _cyl_inertia(m, r, h):
    ixx = m * (3 * r * r + h * h) / 12.0
    return np.array([ixx, ixx, 0.5 * m * r * r])


class MujocoBackend(Backend):
    def __init__(self, cfg):
        super().__init__(cfg)
        mc = self.mc = cfg.mujoco
        mujoco = self.mj = _import_mujoco(mc.gl)
        self.dt = float(mc.timestep)

        spec = self._build_spec(mujoco)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        m = self.model

        self.arm_qadr = np.array([m.joint(n).qposadr[0] for n in ARM_JOINTS])
        self.arm_dadr = np.array([m.joint(n).dofadr[0] for n in ARM_JOINTS])
        self.fin_qadr = np.array([m.joint(n).qposadr[0] for n in FINGER_JOINTS])
        self.arm_act = np.array([m.actuator(f"act_{n}").id for n in ARM_JOINTS])
        self.fin_act = np.array([m.actuator(f"act_{n}").id for n in FINGER_JOINTS])
        self.obj_body = m.body("target").id
        self.obj_geom = m.geom("target_geom").id
        self.obj_qadr = m.joint("target_free").qposadr[0]
        self.obj_dadr = m.joint("target_free").dofadr[0]
        self.cam_id = m.camera("wrist_depth_camera").id

        c = cfg.camera
        self.renderer = mujoco.Renderer(m, int(c.height), int(c.width))
        self.renderer.enable_depth_rendering()
        self.scene_opt = mujoco.MjvOption()
        self.scene_opt.geomgroup[:] = 0
        self.scene_opt.geomgroup[GROUP_SCENE] = 1
        if mc.render_robot_in_depth:
            self.scene_opt.geomgroup[GROUP_ROBOT_VISUAL] = 1

        self.viewer = None
        if mc.viewer:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(m, self.data)
        self.obj = None
        self._fin_cmd = self.gripper.open_width / 2

    # ------------------------------------------------------------- model
    def _build_spec(self, mujoco):
        cfg, mc, g, o = self.cfg, self.mc, self.cfg.gripper, self.cfg.object
        urdf = generate_urdf(cfg).replace(
            '<robot name="ur5_rg2">',
            '<robot name="ur5_rg2"><mujoco><compiler fusestatic="false" discardvisual="false" '
            'balanceinertia="true"/></mujoco>', 1)
        spec = mujoco.MjSpec.from_string(urdf)
        spec.option.timestep = self.dt
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        spec.option.impratio = float(mc.impratio)
        spec.stat.extent = 1.0
        spec.visual.map.znear = 0.01                 # x extent -> 1 cm near clip
        spec.visual.map.zfar = 5.0

        # robot: no self collision, gravity compensated, armature for stiff servos
        for body in spec.bodies:
            if body.name == "world":
                continue
            body.gravcomp = 1.0
            for geom in body.geoms:
                if geom.contype == 0 and geom.conaffinity == 0:
                    continue                          # visual geom
                geom.group = GROUP_ROBOT_COLLISION
                geom.contype, geom.conaffinity = ROBOT_BITS
                geom.friction = [float(g.friction), 0.005, 0.0001]
        for name in ARM_JOINTS:
            j = spec.joint(name)
            j.armature = float(mc.arm_armature)
            j.damping = [float(mc.arm_joint_damping), 0.0, 0.0]
        for name in FINGER_JOINTS:
            spec.joint(name).armature = 0.01

        for name in ARM_JOINTS:
            a = spec.add_actuator(name=f"act_{name}", target=name,
                                  trntype=mujoco.mjtTrn.mjTRN_JOINT)
            a.set_to_position(kp=float(mc.arm_kp), kv=float(mc.arm_kv))
        for name in FINGER_JOINTS:
            a = spec.add_actuator(name=f"act_{name}", target=name,
                                  trntype=mujoco.mjtTrn.mjTRN_JOINT)
            a.set_to_position(kp=float(mc.finger_kp), kv=float(mc.finger_kv))
            a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
            a.forcerange = [-self.gripper.force, self.gripper.force]

        # wrist camera: ROS optical frame (z forward, y down) -> MuJoCo camera (-z forward, y up)
        c = cfg.camera
        R = rot_x(np.deg2rad(float(c.tilt_deg))) @ np.diag([1.0, -1.0, -1.0])
        fx = int(c.width) / 2.0 / np.tan(np.deg2rad(float(c.hfov_deg)) / 2.0)
        fovy = np.rad2deg(2.0 * np.arctan(int(c.height) / 2.0 / fx))
        spec.body("tcp").add_camera(name="wrist_depth_camera", pos=list(c.mount_xyz),
                                    quat=_mat2quat(mujoco, R), fovy=float(fovy))

        # table (top at z = 0) and floor; highest priority -> object/table friction = table_friction
        wb = spec.worldbody
        cx, cy = o.spawn_center
        wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.7, 0.6, 0.025],
                    pos=[cx * 0.6, cy, -0.025], rgba=[0.55, 0.45, 0.35, 1],
                    friction=[float(o.table_friction), 0.005, 0.0001], priority=2,
                    group=GROUP_SCENE, contype=SCENE_BITS[0], conaffinity=SCENE_BITS[1])
        wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1],
                    pos=[0, 0, -0.75], rgba=[0.3, 0.3, 0.32, 1], group=GROUP_SCENE,
                    contype=SCENE_BITS[0], conaffinity=SCENE_BITS[1])
        wb.add_light(pos=[0, 0, 2.5], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])

        # target object: priority 1 -> finger/object contacts use the object's friction
        size = np.array(o.size, dtype=float)
        body = wb.add_body(name="target", pos=[cx, cy, size[2] / 2 + 0.001])
        body.add_freejoint(name="target_free")
        kw = dict(name="target_geom", rgba=[0.9, 0.3, 0.2, 1], mass=float(o.mass), priority=1,
                  condim=4, group=GROUP_SCENE, contype=SCENE_BITS[0], conaffinity=SCENE_BITS[1],
                  friction=[float(o.friction), float(o.friction) * float(g.contact_patch_radius),
                            0.0001])
        if o.shape == "cylinder":
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[size[0] / 2, size[2] / 2, 0], **kw)
        else:
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=list(size / 2), **kw)
        return spec

    def _set_object_params(self, o):
        """Apply this episode's sampled size / mass / friction to the compiled model."""
        m, gid, bid = self.model, self.obj_geom, self.obj_body
        s = np.asarray(o.size, dtype=float)
        if o.shape == "cylinder":
            r, h = s[0] / 2, s[2]
            m.geom_size[gid] = [r, h / 2, 0]
            m.geom_aabb[gid] = [0, 0, 0, r, r, h / 2]
            m.geom_rbound[gid] = np.hypot(r, h / 2)
            m.body_inertia[bid] = _cyl_inertia(o.mass, r, h)
        else:
            m.geom_size[gid] = s / 2
            m.geom_aabb[gid] = np.r_[0, 0, 0, s / 2]
            m.geom_rbound[gid] = np.linalg.norm(s / 2)
            m.body_inertia[bid] = _box_inertia(o.mass, s)
        m.body_mass[bid] = o.mass
        m.geom_friction[gid] = [o.mu_s, o.mu_s * float(self.cfg.gripper.contact_patch_radius), 0.0001]

    # ------------------------------------------------------------ sim utils
    def _step(self, n=1):
        n = int(n)
        if self.viewer is None:
            self.mj.mj_step(self.model, self.data, nstep=max(n, 1))
            return
        every = max(1, int(self.mc.viewer_sync_every))
        for k in range(n):
            t0 = time.time()
            self.mj.mj_step(self.model, self.data)
            if k % every == 0:
                self.viewer.sync()
            if self.mc.realtime:
                time.sleep(max(0.0, self.dt - (time.time() - t0)))

    def _set_arm_target(self, q):
        self.data.ctrl[self.arm_act] = q

    def _set_finger_target(self, half_width):
        self._fin_cmd = float(half_width)
        self.data.ctrl[self.fin_act] = self._fin_cmd

    def _drive_arm(self, q_goal, duration):
        """Joint-space linear interpolation to q_goal, then settle."""
        q0 = self.data.ctrl[self.arm_act].copy()
        n = max(1, int(np.ceil(duration / self.dt)))
        for k in range(1, n + 1):
            self._set_arm_target(q0 + (q_goal - q0) * k / n)
            self._step()
        for _ in range(int(self.mc.settle_steps)):
            if np.abs(self.joint_positions() - q_goal).max() < float(self.mc.joint_tol):
                break
            self._step()

    def _read_object(self):
        d = self.data
        pos = d.qpos[self.obj_qadr:self.obj_qadr + 3].copy()
        quat = d.qpos[self.obj_qadr + 3:self.obj_qadr + 7]
        o = copy.copy(self.obj)
        o.x, o.y = float(pos[0]), float(pos[1])
        o.yaw = _yaw_from_quat(quat)
        return o, pos

    def finger_width(self):
        return float(np.sum(self.data.qpos[self.fin_qadr]))

    # ------------------------------------------------------------ interface
    def reset(self, rng):
        self.rng = rng
        self.obj = o = sample_object(self.cfg.object, rng)
        mj, m, d = self.mj, self.model, self.data
        mj.mj_resetData(m, d)
        self._set_object_params(o)
        a = self.obj_qadr
        d.qpos[a:a + 3] = [o.x, o.y, o.height / 2 + 0.001]
        d.qpos[a + 3:a + 7] = [np.cos(o.yaw / 2), 0, 0, np.sin(o.yaw / 2)]

        pos, yaw = self.home_tcp(rng)
        q_arm, ok = self.solve_ik(pos, yaw, self.cfg.robot.ik_seed)
        if not ok:
            raise RuntimeError("home pose is not reachable; check task.start_height / spawn_center")
        d.qpos[self.arm_qadr] = q_arm
        d.qpos[self.fin_qadr] = self.gripper.open_width / 2
        self._set_arm_target(q_arm)
        self._set_finger_target(self.gripper.open_width / 2)
        mj.mj_forward(m, d)
        self._step(int(self.mc.reset_settle_steps))

    def joint_positions(self):
        return self.data.qpos[self.arm_qadr].copy()

    def move_tcp(self, pos, yaw):
        pos = np.array(pos, dtype=float)
        std = float(self.cfg.robot.execution_noise_std)
        if std > 0:
            pos = pos + self.rng.normal(0.0, std, 3) * np.array([1, 1, 0.5])
        obj, before = self._read_object()
        hit = self.gripper.check_collision(obj, pos, yaw)
        if hit:                                   # same rule as the other backends
            return MoveResult(f"collision_{hit}")
        q, ok = self.solve_ik(pos, yaw, self.joint_positions())
        if not ok:
            return MoveResult("ik_fail")
        cur_pos, cur_yaw = self.tcp_pose()
        duration = max(np.linalg.norm(pos - cur_pos) / float(self.mc.move_speed),
                       abs(yaw - cur_yaw) / float(self.mc.yaw_speed), float(self.mc.min_move_time))
        self._drive_arm(q, duration)
        _, after = self._read_object()
        if np.linalg.norm(after[:2] - before[:2]) > float(self.mc.knock_threshold):
            return MoveResult("collision_object")
        return MoveResult("ok")

    def grasp(self):
        g, t = self.gripper, self.cfg.task
        obj0, p0 = self._read_object()
        # close at the RG2 speed for this force; the actuator force limit is the grip force
        half = self._fin_cmd
        while half > 0:
            half = max(0.0, half - g.speed / 2 * self.dt)
            self._set_finger_target(half)
            self._step()
        self._step(float(self.mc.grip_settle_time) / self.dt)
        width = self.finger_width()
        info = {"width": width, "force": g.force, "mass": obj0.mass, "mu_s": obj0.mu_s}
        if width < 0.003:
            self._release()
            return GraspResult(False, "miss", info)

        # lift: trapezoidal velocity profile (lift_max_speed, lift_accel), IK waypoint per step
        pos, yaw = self.tcp_pose()
        h, vmax, acc = float(t.lift_height), float(t.lift_max_speed), float(t.lift_accel)
        q = self.joint_positions()
        z, v = 0.0, 0.0
        while z < h:
            brake = v * v / (2 * acc) >= h - z
            v = max(0.2 * vmax, v - acc * self.dt) if brake else min(vmax, v + acc * self.dt)
            z = min(h, z + v * self.dt)
            q, _ = self.solve_ik(pos + np.array([0, 0, z]), yaw, q)
            self._set_arm_target(q)
            self._step()
        self._step(float(self.mc.hold_time) / self.dt)
        _, p1 = self._read_object()
        lift = float(p1[2] - p0[2])
        info["object_lift"] = lift
        if lift >= 0.8 * h:
            return GraspResult(True, "success", info)
        self._release()
        return GraspResult(False, "dropped" if lift > 0.01 else "no_lift", info)

    def _release(self):
        self._set_finger_target(self.gripper.open_width / 2)
        self._step(float(self.mc.grip_settle_time) / self.dt)

    def depth_image(self):
        self.renderer.update_scene(self.data, camera=self.cam_id, scene_option=self.scene_opt)
        d = np.asarray(self.renderer.render(), dtype=np.float64)
        return self.camera.add_noise(d, self.rng)

    def object_state(self):
        return self._read_object()[0]

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
        self.renderer.close()
