#!/usr/bin/env python3
"""Offscreen, GPU render of the simulator scene (air_bearing_table.xml): a
still, or a frame sequence, at any resolution, with no window and no X server.

    MUJOCO_GL=egl python3 tools/render/render_scene.py --out scene.png
    MUJOCO_GL=egl python3 tools/render/render_scene.py --preset top --bg white --out top.png
    MUJOCO_GL=egl python3 tools/render/render_scene.py \\
        --series data/thesis_eval/tuned/balanced/series/circle_p-hold_sqp.csv --time 6.5 --out mid.png
    MUJOCO_GL=egl python3 tools/render/render_scene.py \\
        --series <same csv> --frames 0:20:0.1 --out frames/f.png   # f_0000.png, f_0001.png, ...

It loads the same generated MJCF the simulation node loads, so it looks the
same as the viewer: the collision-debug primitives (geom group 3) are hidden,
and the arms start at simulation.yaml's initial_arm_qpos.

The pose:
  default        simulation.yaml's initial_arm_qpos, chaser and target at rest
  --series CSV   a contact_bench per-tick series (data/thesis_eval/*/series/):
                 its arm (--arm, the bench's default left) takes q0..q2 at
                 --time, the chaser base_x / base_y / base_yaw. The other arm
                 keeps the initial pose (correct for a 'hold' partner).
  --qpos J=V     any joint, after everything else (repeatable): joint2_R=0.8

Needs only the `mujoco` and `numpy` Python packages (pip install --user mujoco);
the PNG is written without any imaging library. Rendering backend: MUJOCO_GL=egl
for the NVIDIA GPU headless, osmesa for CPU-only, unset for a desktop (GLFW).
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import struct
import sys
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
SCENE_REL = os.path.join("share", "pat_simulation", "models", "environment",
                         "air_bearing_table.xml")
SIM_YAML = os.path.join(REPO, "src", "pat_simulation", "config", "simulation.yaml")

#: Camera presets: lookat [m], distance [m], azimuth [deg], elevation [deg].
PRESETS = {
    "overview": dict(lookat=(-0.10, 0.0, 0.10), distance=2.3, azimuth=135, elevation=-30),
    "chaser":   dict(lookat=(-0.45, 0.0, 0.20), distance=1.7, azimuth=150, elevation=-25),
    "arms":     dict(lookat=(-0.15, 0.0, 0.10), distance=1.3, azimuth=180, elevation=-50),
    "top":      dict(lookat=(0.0, 0.0, 0.00), distance=3.4, azimuth=90, elevation=-89.9),
    "side":     dict(lookat=(-0.05, 0.0, 0.20), distance=2.2, azimuth=90, elevation=-8),
}


def find_scene() -> str | None:
    """The installed scene: this checkout's install/ first, then any overlay on
    AMENT_PREFIX_PATH (a sourced workspace)."""
    prefixes = [os.path.join(REPO, "install", "pat_simulation"), os.path.join(REPO, "install")]
    prefixes += [p for p in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep) if p]
    for p in prefixes:
        f = os.path.join(p, SCENE_REL)
        if os.path.isfile(f):
            return f
    return None


def yaml_list(key: str) -> list[str] | None:
    """A one-line `key: [a, b, ...]` from simulation.yaml, without PyYAML."""
    try:
        text = open(SIM_YAML, encoding="utf-8").read()
    except OSError:
        return None
    m = re.search(rf"^\s*{key}:\s*\[([^\]]*)\]", text, re.M)
    return [v.strip().strip("\"'") for v in m.group(1).split(",")] if m else None


def initial_pose() -> dict[str, float]:
    names = yaml_list("arm_joint_names") or \
        ["joint2_L", "joint3_L", "joint5_L", "joint2_R", "joint3_R", "joint5_R"]
    vals = yaml_list("initial_arm_qpos") or \
        ["0.6108652382", "-0.2617993878", "0.0", "0.6108652382", "-0.2617993878", "0.0"]
    return {n: float(v) for n, v in zip(names, vals)}


def read_series(path: str) -> dict[str, list[float]]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"{path}: empty series")
    need = {"t_s", "q0", "q1", "q2", "base_x", "base_y", "base_yaw"}
    if not need <= rows[0].keys():
        sys.exit(f"{path}: not a contact_bench series (needs {', '.join(sorted(need))})")
    return {k: [float(r[k]) for r in rows] for k in need}


def series_pose(s: dict[str, list[float]], t: float, arm: str) -> dict[str, float]:
    """The joint values at the tick nearest t."""
    i = min(range(len(s["t_s"])), key=lambda k: abs(s["t_s"][k] - t))
    side = "L" if arm == "left" else "R"
    return {f"joint2_{side}": s["q0"][i], f"joint3_{side}": s["q1"][i],
            f"joint5_{side}": s["q2"][i], "chaser_x": s["base_x"][i],
            "chaser_y": s["base_y"][i], "chaser_yaw": s["base_yaw"][i]}


def write_png(path: str, img) -> None:
    """RGB or RGBA uint8 (H, W, C) to PNG, with zlib alone."""
    h, w, c = img.shape
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + \
            struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6 if c == 4 else 2, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(raw, 6)))
        f.write(chunk(b"IEND", b""))


def frame_times(spec: str) -> list[float]:
    a, b, step = (float(v) for v in spec.split(":"))
    n = int(round((b - a) / step)) + 1
    return [a + k * step for k in range(n)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("--model", help="MJCF to render (default: the installed "
                    "air_bearing_table.xml, found under install/ or AMENT_PREFIX_PATH)")
    ap.add_argument("--out", default="scene.png", help="PNG path (a sequence appends _0000, ...)")
    ap.add_argument("--width", type=int, default=2400)
    ap.add_argument("--height", type=int, default=1600)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="overview")
    ap.add_argument("--lookat", help="x,y,z [m], overrides the preset")
    ap.add_argument("--distance", type=float)
    ap.add_argument("--azimuth", type=float, help="[deg]")
    ap.add_argument("--elevation", type=float, help="[deg], negative looks down")
    ap.add_argument("--track", action="store_true",
                    help="centre the camera on the chaser wherever it has drifted")
    ap.add_argument("--bg", choices=("black", "white", "transparent"), default="black",
                    help="what shows where nothing is drawn (beyond the table)")
    ap.add_argument("--samples", type=int, default=8, help="multisample anti-aliasing")
    ap.add_argument("--no-shadows", action="store_true")
    ap.add_argument("--show-collision", action="store_true",
                    help="also draw the collision primitives (geom group 3)")
    ap.add_argument("--show-sites", action="store_true",
                    help="also draw the sites (IMU, thrusters, EE), hidden by default")
    ap.add_argument("--series", help="contact_bench series CSV to take the pose from")
    ap.add_argument("--arm", choices=("left", "right"), default="left",
                    help="the arm the series' q0..q2 belong to")
    ap.add_argument("--time", type=float, default=0.0, help="[s] into the series")
    ap.add_argument("--frames", help="start:stop:step [s] into the series: a PNG sequence")
    ap.add_argument("--qpos", action="append", default=[], metavar="JOINT=VALUE",
                    help="set one joint [m or rad], applied last; repeatable")
    a = ap.parse_args()

    try:
        import mujoco
        import numpy as np
    except ImportError as e:
        sys.exit(f"{e}: pip install --user mujoco numpy")

    path = a.model or find_scene()
    if not path or not os.path.isfile(path):
        sys.exit("no scene: build pat_simulation (colcon build) or pass --model "
                 "<install>/share/pat_simulation/models/environment/air_bearing_table.xml")
    if a.frames and not a.series:
        sys.exit("--frames needs --series")
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)

    # Offscreen buffer and quality: the defaults are 640x480, 4 samples.
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, a.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, a.height)
    model.vis.quality.offsamples = a.samples
    model.vis.quality.shadowsize = 8192

    fixed = {}
    for kv in a.qpos:
        name, _, val = kv.partition("=")
        if not val:
            sys.exit(f"--qpos {kv}: expected JOINT=VALUE")
        fixed[name] = float(val)
    series = read_series(a.series) if a.series else None

    def pose_at(t: float) -> None:
        mujoco.mj_resetData(model, data)
        values = initial_pose()
        if series is not None:
            values.update(series_pose(series, t, a.arm))
        values.update(fixed)
        for name, v in values.items():
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if j < 0:
                sys.exit(f"no joint '{name}' in {os.path.basename(path)}")
            data.qpos[model.jnt_qposadr[j]] = v
        mujoco.mj_forward(model, data)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    p = dict(PRESETS[a.preset])
    if a.lookat:
        p["lookat"] = tuple(float(v) for v in a.lookat.split(","))
    for k in ("distance", "azimuth", "elevation"):
        if getattr(a, k) is not None:
            p[k] = getattr(a, k)
    cam.distance, cam.azimuth, cam.elevation = p["distance"], p["azimuth"], p["elevation"]

    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 1 if a.show_collision else 0
    if not a.show_sites:          # the IMU / thruster / EE marker dots
        for g in range(len(opt.sitegroup)):
            opt.sitegroup[g] = 0

    renderer = mujoco.Renderer(model, height=a.height, width=a.width)
    if a.no_shadows:
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    far = model.stat.extent * model.vis.map.zfar

    def render() -> "np.ndarray":
        chaser = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chaser")
        lookat = np.array(p["lookat"], dtype=float)
        if a.track and chaser >= 0:
            lookat[:2] += data.xpos[chaser][:2] - model.body_pos[chaser][:2]
        cam.lookat[:] = lookat
        renderer.update_scene(data, camera=cam, scene_option=opt)
        rgb = renderer.render().copy()
        if a.bg == "black":
            return rgb
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=cam, scene_option=opt)
        empty = renderer.render() >= 0.999 * far
        renderer.disable_depth_rendering()
        if a.bg == "white":
            rgb[empty] = 255
            return rgb
        alpha = np.where(empty, 0, 255).astype(np.uint8)[..., None]
        return np.concatenate([rgb, alpha], axis=2)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if a.frames:
        stem, ext = os.path.splitext(a.out)
        times = frame_times(a.frames)
        for k, t in enumerate(times):
            pose_at(t)
            write_png(f"{stem}_{k:04d}{ext or '.png'}", render())
        print(f"wrote {len(times)} frames {stem}_0000{ext or '.png'} ... "
              f"({a.width}x{a.height}); a video: ffmpeg -framerate "
              f"{round(1 / float(a.frames.split(':')[2]))} -i {stem}_%04d.png "
              f"-pix_fmt yuv420p {stem}.mp4")
    else:
        pose_at(a.time)
        write_png(a.out, render())
        print(f"wrote {a.out} ({a.width}x{a.height}, {a.preset}, "
              f"backend {os.environ.get('MUJOCO_GL', 'default')})")
    renderer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
