# Visualization and RGB-D tools

Run these from the repository root through `scripts/rapid-python`.

## Preview a packaged scene

```bash
# By name, looked up in the brace package's assets/scenes; or by directory.
./scripts/rapid-real2sim view push_scene
./scripts/rapid-real2sim view path/to/scenes/push_scene --viewer
```

Without `--viewer` this writes a PNG to `results/real2sim/previews/<scene>.png`
(`-o` overrides it), using the default camera stored in the scene's
`scene.json`. `--viewer` opens MuJoCo's interactive viewer and needs a display.
The implementation is `vistools/visualize_scene.py`.

## Inspect any MJCF file

```bash
./scripts/rapid-python -m vistools.visualize_mujoco scene.xml
./scripts/rapid-python -m vistools.visualize_mujoco scene.xml --orbit-video orbit.mp4
```

## RGB-D helpers

`rgbd_format.py` and `rgbd_input.py` define and load the calibrated RGB-D
capture layout that `rapid-real2sim` accepts. `clip_rgbd_demo.py` (the
`rapid-rgbd-demo-clip` script) trims a synchronized RGB-D demo.
