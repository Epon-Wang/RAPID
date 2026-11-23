# RAPID Real2Sim for BRACE

This branch (`brace`) is a trimmed fork of
[RAPID](https://github.com/nature21/RAPID) that keeps only its **Real2Sim**
pipeline and writes its results straight into the
[BRACE](../../README.md) project. Everything else RAPID does (primitive
construction, strategy composition, the benchmark, the Codex invention agents,
Judo) has been removed here; it is all still on the `main` branch.

Real2Sim takes an RGB image, the first frame of a video, or a calibrated RGB-D
capture, and produces a MuJoCo scene: a VLM lists the objects, SAM3 segments
them, SAM 3D Objects reconstructs their meshes, FoundationPose refines their
poses, and the result is validated, given box collisions, and settled.

## Layout

| Path | Purpose |
| --- | --- |
| `real2sim/` | The pipeline, packaging, and the scene-repair agent skill. |
| `server/` | HTTP adapters for the model services (SAM3, SAM 3D Objects, FoundationPose, Depth Anything 3). |
| `vistools/` | RGB-D input helpers and scene preview. |
| `scripts/` | Entry points; see [scripts/README.md](scripts/README.md). |
| `demos/primitive/push/` | One calibrated RGB-D capture, usable as a reconstruction input. |
| `examples/push_scene/` | Its finished reconstruction, for trying the packaging step without any model service. |
| `calibration/` | Default hand-eye calibration used when capturing. |
| `tests/` | Tests; they run against local stand-ins for the model services. |

## Usage

The environment is the `brace` conda environment from BRACE's README.

To try packaging without any model service, use the bundled reconstruction:

```bash
./scripts/rapid-real2sim examples/push_scene --preview
```

A real reconstruction also needs a VLM endpoint and the model services; see
[SETUP.md](SETUP.md).

```bash
# Reconstruct from a calibrated RGB-D capture and package the scene.
./scripts/rapid-real2sim demos/primitive/push --name my_scene \
  --vlm-model "$RAPID_VLM_MODEL" --preview

# Re-package an existing reconstruction output.
./scripts/rapid-real2sim results/real2sim/my_scene --force

# Look at a packaged scene.
./scripts/rapid-real2sim view my_scene --viewer
```

Intermediate reconstruction artifacts go to `results/real2sim/<name>/`. The
packaged scene goes to `assets/scenes/<name>/` inside the installed `brace`
package; pass `--tasks-root` to write somewhere else.

## Packaged scene layout

```
<name>/
├── scene.json        # index: floor, default camera, and every object's resting pose
├── objects/<body>.xml  # one standalone MJCF per object, body at the origin
├── meshes/           # visual meshes shared by scene.xml and objects/
├── scene.xml         # the whole scene in one MJCF: floor, lights, all objects
├── metadata.json     # object labels
└── settling.json     # report of the settling simulation
```

`scene.json` lists, for each object: `name` (label), `body` (MuJoCo body name),
`file`, `fixed`, `pos`, `quat` (w, x, y, z), `mass`, and `friction`. Poses are in
the scene frame, which for a calibrated RGB-D capture is the robot base frame.
`floor.height` is the height of the support plane in that frame.

BRACE loads these directories with `brace.utils.scenes.load_scene`.

## Tests

```bash
./scripts/rapid-python -m pytest -q
```

## License and citation

RAPID is released under the [MIT license](LICENSE); cite it with
[CITATION.cff](CITATION.cff). The original README, the full setup guide, and
the rest of the code are on the `main` branch and upstream.
