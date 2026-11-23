# `scripts/` - Entry Points

Thin shell wrappers. Each one calls `rapid-python`, which puts this repository
on `PYTHONPATH` and runs in the `brace` conda environment (override with
`RAPID_CONDA_ENV`). They work whether or not that environment is activated.

| Script | Use for |
| --- | --- |
| `rapid-python` | Run any module or script from this repository. `--rapid-env` prints the resolved environment. |
| `rapid-real2sim` | Reconstruct and package a scene from RGB or calibrated RGB-D input; `view` previews a packaged scene. Requires the model services in [SETUP.md](../SETUP.md). |
| `rapid-real2sim-capture` | Capture a RealSense scene or synchronized demonstration. |
| `rapid-rgbd-demo-clip` | Trim a synchronized RGB-D demo. |
