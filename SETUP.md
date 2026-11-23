# Real2Sim Setup

This branch keeps only RAPID's Real2Sim pipeline and serves the
[BRACE](../../README.md) project. The Python environment is the `brace` conda
environment described in BRACE's README; there is nothing else to install on
the client side. This guide covers what a real reconstruction needs beyond
that: a VLM endpoint and the model services.

## Reconstruct new scenes

The `real2sim/` pipeline accepts an RGB image, the first frame of a video, or a
calibrated RGB-D directory. It identifies objects with a VLM, segments them with
SAM3, reconstructs their meshes with SAM 3D Objects, estimates their poses with
FoundationPose, and packages a MuJoCo task scene. RGB inputs additionally use
Depth Anything 3 for depth and camera estimation. A calibrated RGB-D input uses
its measured depth and calibration directly.

### Client dependencies and VLM configuration

Run reconstruction in the `brace` environment, which already includes Open3D
for mesh simplification and ICP. Install the camera package only if capturing
new data:

```bash
# Only for a connected RealSense camera:
conda run -n brace python -m pip install pyrealsense2
./scripts/rapid-real2sim --help
```

Configure a vision-capable OpenAI-compatible model endpoint. These credentials
are separate from the Codex login used by the scene-repair agent:

```bash
export REAL2SIM_VLM_BACKEND=CUSTOM
export CUSTOM_OPENAI_BASE_URL="https://your-provider.example/v1"
export CUSTOM_OPENAI_API_KEY="..."
export RAPID_VLM_MODEL="your-vision-model"
```

For the OpenAI endpoint, set `REAL2SIM_VLM_BACKEND=OPENAI` and `OPENAI_API_KEY`
instead. Pass the model name with `--vlm-model`; if the provider does not support
reasoning effort, also pass `--vlm-reasoning-effort none`.

### Install the model services

The adapters in `server/` use upstream model installations and checkpoints.
Install each model in the environment or container recommended by its project;
these GPU dependencies are independent of the `brace` environment.

| Service | Upstream installation and weights | Adapter | Default port |
| --- | --- | --- | --- |
| SAM3 | [SAM3](https://github.com/facebookresearch/sam3#installation) | `server.sam3_server` | 4002 |
| SAM 3D Objects | [SAM 3D setup](https://github.com/facebookresearch/sam-3d-objects/blob/main/doc/setup.md) | `server.sam3d_server` | 4001 |
| FoundationPose | [FoundationPose](https://github.com/NVlabs/FoundationPose) | `server.foundation_pose_server` | 4004 |
| Depth Anything 3 | [DA3](https://github.com/ByteDance-Seed/Depth-Anything-3#installation) | `server.depth_anything_server` | 4003 |

Follow the upstream checkpoint access/download instructions, including any
Hugging Face access requirements. Model code and weights remain subject to their
upstream licenses; this repository does not redistribute them. The SAM3, SAM 3D
and FoundationPose HTTP interfaces follow the service pattern in
[GROW²](https://github.com/dengyh16code/improvised_tooluse/tree/main/server).
FoundationPose works with its official single-pose return API; no estimator
source patch is required.

In each model's activated environment, install the lightweight adapter dependencies:

```bash
python -m pip install -r /absolute/path/to/RAPID/server/requirements.txt
```

Run each service in a separate terminal, from the RAPID repository root, using
that model's environment. Select the GPU with `CUDA_VISIBLE_DEVICES` if needed.
The examples below use localhost and match the reconstruction client's defaults:

```bash
# SAM3 environment; --checkpoint /path/to/sam3.pt can select a local checkpoint.
python -m server.sam3_server --port 4002

# SAM 3D Objects environment, with its checkpoint configuration downloaded.
python -m server.sam3d_server \
  --repo /absolute/path/to/sam-3d-objects \
  --config checkpoints/hf/pipeline.yaml --port 4001

# FoundationPose environment, with its weights/ populated as documented upstream.
python -m server.foundation_pose_server \
  --repo /absolute/path/to/FoundationPose --port 4004

# DA3 environment; omit this service when supplying calibrated RGB-D inputs.
python -m server.depth_anything_server \
  --model depth-anything/DA3NESTED-GIANT-LARGE-1.1 --port 4003
```

Each process loads its model before serving requests and processes one inference
at a time. Check readiness with, for example:

```bash
curl --fail http://127.0.0.1:4002/health
```

For services on another machine, bind with `--host 0.0.0.0` on a private network
and point the client at that machine's reachable address. Full endpoint options
are `--sam3-url`, `--sam3d-url`, `--foundation-pose-url`, and `--depth-url`.
For example:

```bash
./scripts/rapid-real2sim capture/ --name captured_scene \
  --vlm-model "$RAPID_VLM_MODEL" \
  --sam3-url http://gpu.example.org:4002/sam3_query \
  --sam3d-url http://gpu.example.org:4001/sam3d_query \
  --foundation-pose-url http://gpu.example.org:4004/fp_registration_query
```

Alternatively set `RAPID_SAM3_HOST` / `RAPID_SAM3_PORT`,
`RAPID_SAM3D_HOST` / `RAPID_SAM3D_PORT`,
`RAPID_FOUNDATION_POSE_HOST` / `RAPID_FOUNDATION_POSE_PORT`, or
`RAPID_DEPTH_ANYTHING_HOST` / `RAPID_DEPTH_ANYTHING_PORT` before starting the
client. Explicit URL flags take precedence. Services can also be forwarded to
the default local ports with SSH tunnels.

### Reconstruct and inspect a scene

A calibrated RGB-D input directory contains `color.png`, metric `depth.npy`,
`intrinsics.json`, and an Easy Handeye `*_eob_calib.yaml`:

```bash
./scripts/rapid-real2sim demos/primitive/push \
  --name reconstructed_push --vlm-model "$RAPID_VLM_MODEL" --preview
```

For an RGB image or a video, start DA3 as well:

```bash
./scripts/rapid-real2sim /path/to/frame.png \
  --name reconstructed_frame --vlm-model "$RAPID_VLM_MODEL" --preview
```

Reconstruction artifacts are written to `results/real2sim/<name>/`, and packaged
scenes to `assets/scenes/<name>/` inside the `brace` package (override with
`--tasks-root`). Packaging checks visual/collision alignment, compiles the
MuJoCo model, and settles it before saving the resting poses. See
[README.md](README.md#packaged-scene-layout) for the layout.
The optional scene-repair agent uses the included
[`repair-real2sim-scene` skill](real2sim/.agents/skills/repair-real2sim-scene/SKILL.md).
Use `--no-agent-fallback` to disable agent repair; `--no-package` keeps only the
reconstruction bundle. To omit FoundationPose, use `--no-foundation-pose`; this
uses SAM 3D placement, with local ICP when the support correction triggers it,
and can change reconstruction quality. Use `--no-fp-fallback` to disable SAM 3D
placement as a fallback when FoundationPose fails. By default, failed objects
are recorded in `skipped_objects` and reconstruction continues with the remaining
objects. When using the Python API, set
`PipelineConfig(continue_on_object_error=False)` to stop on an object failure.

```bash
./scripts/rapid-real2sim view reconstructed_push --viewer
```

### Test the interfaces without model weights

The following tests start local HTTP servers with deterministic model substitutes,
call them through the production clients, and exercise both RGB and RGB-D inputs
through scene export, packaging, and MuJoCo simulation. They also check malformed
requests, depth/camera scaling, and FoundationPose tracking-frame conventions.
They do not establish reconstruction accuracy or validate GPU model installations.

```bash
./scripts/rapid-python -m pytest -q \
  tests/test_reconstruction_servers.py tests/test_vfm_client_config.py
./scripts/rapid-python -m pytest -q
```
