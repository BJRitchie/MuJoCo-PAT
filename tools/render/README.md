# render: offscreen images of the simulator scene

`render_scene.py` renders the scene the simulation node loads
(`air_bearing_table.xml`) straight to PNG, with no window and no X server. On
a headless machine with an NVIDIA GPU it renders on the GPU through EGL, so it
is the way to get print-quality pictures off the DGX. The viewer under Xvfb
works too, but it renders in software on the CPU and is slow.

## Setup (no sudo)

```bash
docker compose exec dev bash
export HOME=/tmp/$(id -u) && mkdir -p $HOME     # the container user has no home
pip install --user mujoco                        # numpy comes with it
colcon build --packages-select pat_simulation    # if install/ has no scene yet
```

## Use

```bash
export MUJOCO_GL=egl                             # GPU, headless; osmesa = CPU fallback
python3 tools/render/render_scene.py --out docs/assets/thesis/scene.png
python3 tools/render/render_scene.py --preset top --bg white --out top.png
python3 tools/render/render_scene.py --preset chaser --width 3600 --height 2400 --out big.png

# pose from a contact_bench run: the scored arm and the drifted, yawed base
python3 tools/render/render_scene.py \
    --series data/thesis_eval/tuned/balanced/series/circle_p-hold_sqp.csv --time 6.5 --out mid.png

# a frame sequence, then a video
python3 tools/render/render_scene.py --series <same csv> --frames 0:20:0.04 \
    --preset chaser --track --out frames/f.png
ffmpeg -framerate 25 -i frames/f_%04d.png -pix_fmt yuv420p circle.mp4
```

| Option | What it does |
|---|---|
| `--preset overview\|chaser\|arms\|top\|side` | a starting camera; `--lookat x,y,z`, `--distance`, `--azimuth`, `--elevation` override any part of it |
| `--width`, `--height` | any size (default 2400×1600); `--samples` sets the anti-aliasing |
| `--bg black\|white\|transparent` | what shows beyond the table; `transparent` writes an RGBA PNG |
| `--series CSV --time T` | arm (`--arm left\|right`, the bench's default left) and base pose from a per-tick series; the other arm keeps the start pose |
| `--qpos joint3_R=-0.5` | set any joint, last; repeatable |
| `--track` | follow the chaser as it drifts |
| `--show-collision`, `--show-sites` | draw the collision primitives / site markers, hidden by default as in the viewer |

The arms start at `simulation.yaml`'s `initial_arm_qpos`, the pose every launch
starts in. The script needs only `mujoco` and `numpy`.
