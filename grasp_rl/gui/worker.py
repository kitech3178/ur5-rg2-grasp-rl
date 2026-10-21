"""Background jobs for the GUI: training and step-by-step policy preview.

Each job runs in its own process (multiprocessing 'spawn') so the simulator's
OpenGL context and the CPU-heavy learning loop never block the Qt window.
Messages go back through a queue as (kind, payload) tuples:

    frame    camera images + features + reward of the latest step
    episode  one row of episodes.csv          eval  one row of eval.csv
    step     one preview transition           preview_episode  preview result
    done     final summary                    error  traceback text
"""

from __future__ import annotations

import time
import traceback


def _feat(f):
    if f is None or not f.visible:
        return None
    return {"dx": f.dx, "dy": f.dy, "dz": f.dz, "yaw": f.yaw}


def _make_display(cfg):
    """A MuJoCo model used only to draw backends without a renderer of their own (simple)."""
    if cfg.backend == "mujoco":
        return None
    try:
        from grasp_rl.backends.mujoco_backend import MujocoBackend
        return MujocoBackend(cfg)
    except Exception:  # noqa: BLE001 - mujoco not installed / no OpenGL: depth view only
        return None


def _frame(env, info, display=None, **extra):
    b, last = env.backend, env.last
    cam, scene = b.camera_rgb(), b.scene_rgb()
    if scene is None and display is not None:
        display.show_state(b.joint_positions(), b.object_state())
        cam, scene = display.camera_rgb(), display.scene_rgb()
    return {"depth": last["depth"], "cam_rgb": cam, "scene_rgb": scene,
            "perceived": _feat(last["perceived"]), "gt": _feat(last["gt"]),
            "step": int(info.get("step", 0)), "action": info.get("action_name", "reset"),
            "terms": info.get("reward_terms", {}), "events": info.get("events", []), **extra}


def train_worker(job, q, stop):
    try:
        from grasp_rl.config import load_config
        from grasp_rl.train_loop import train

        cfg = load_config(None, job["overrides"])
        hz, delay = float(job["frame_hz"]), float(job["step_delay"])
        display = _make_display(cfg) if hz > 0 else None
        st = {"t": 0.0, "ep": 0, "ret": 0.0}

        def recorder(env, a, r, info):
            st["ret"] = r if info["step"] == 1 else st["ret"] + r
            now = time.time()
            if hz > 0 and (delay > 0 or now - st["t"] >= 1.0 / hz):
                st["t"] = now
                q.put(("frame", _frame(env, info, display, episode=st["ep"] + 1, reward=r,
                                       ret=st["ret"])))
            if delay > 0:
                time.sleep(delay)

        def callback(kind, row):
            if kind == "episode":
                st["ep"] = row["episode"]
            q.put((kind, row))

        _, summary, out = train(cfg, job["algo"], seed=job["seed"], out_dir=job["out_dir"],
                                episodes=job["episodes"], verbose=False, label=job["label"],
                                callback=callback, recorder=recorder, should_stop=stop.is_set)
        q.put(("done", {"summary": summary, "out_dir": str(out)}))
    except Exception:  # noqa: BLE001 - everything is reported to the GUI
        q.put(("error", traceback.format_exc()))


def preview_worker(job, q, stop):
    """Run a few greedy episodes and stream every step (for checking rewards)."""
    try:
        from grasp_rl.agents import make_agent, observation_mode
        from grasp_rl.config import load_config
        from grasp_rl.envs import GraspEnv

        cfg = load_config(None, job["overrides"])
        algo = job["policy"]
        env = GraspEnv(cfg, observation_mode(algo, cfg))
        display = _make_display(cfg)
        agent = make_agent(algo, cfg, env)
        if job.get("model"):
            agent.load(job["model"])
        delay = float(job["step_delay"])
        for i in range(int(job["episodes"])):
            if stop.is_set():
                break
            obs, info = env.reset(seed=int(job["seed"]) + i)
            ret = 0.0
            q.put(("frame", _frame(env, info, display, episode=i + 1, reward=0.0, ret=0.0)))
            while True:
                obs, r, te, tr, info = env.step(agent.act(obs, explore=False))
                ret += r
                q.put(("frame", _frame(env, info, display, episode=i + 1, reward=r, ret=ret)))
                q.put(("step", {"episode": i + 1, "step": info["step"], "action": info["action_name"],
                                "reward": r, "terms": info["reward_terms"], "events": info["events"]}))
                time.sleep(delay)
                if te or tr or stop.is_set():
                    break
            outcome = "success" if info["success"] else info.get("grasp_reason", "timeout")
            q.put(("preview_episode", {"episode": i + 1, "outcome": outcome, "return": ret,
                                       "length": info["step"]}))
        env.close()
        q.put(("done", None))
    except Exception:  # noqa: BLE001
        q.put(("error", traceback.format_exc()))
