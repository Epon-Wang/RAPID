"""Render or interactively inspect a packaged Real2Sim scene directory.

Examples:
    ./scripts/rapid-python -m vistools.visualize_scene demo_5_scene
    ./scripts/rapid-python -m vistools.visualize_scene demo_5_scene --viewer
    ./scripts/rapid-python -m vistools.visualize_scene path/to/scenes/demo_5_scene
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import mujoco

from vistools.visualize_mujoco import (
    launch_model_viewer,
    render_model,
    _parse_vec3,
)


def resolve_scene_dir(scene_ref: str | Path) -> Path:
    """Return the packaged scene directory named by a path or a bare scene name.

    A bare name is looked up under the default scene root of the brace package.
    """

    path = Path(str(scene_ref)).expanduser()
    if path.is_file():
        path = path.parent
    if not path.is_dir() and path.name == str(scene_ref):
        from real2sim.package_scene import default_scenes_root

        path = default_scenes_root() / path.name
    if not (path / "scene.xml").is_file():
        raise FileNotFoundError(f"packaged scene not found: {scene_ref}")
    return path.resolve()


def render_scene(
    scene_ref: str | Path,
    output_path: str | Path | None = None,
    *,
    width: int = 1280,
    height: int = 720,
    azimuth: float | None = None,
    elevation: float | None = None,
    distance: float | None = None,
    lookat: tuple[float, float, float] | None = None,
    step_count: int = 0,
) -> Path:
    """Render a PNG preview of a packaged scene."""

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    scene_dir = resolve_scene_dir(scene_ref)
    camera = _camera_options(
        scene_dir,
        azimuth=azimuth,
        elevation=elevation,
        distance=distance,
        lookat=lookat,
    )
    output = (
        Path(output_path)
        if output_path
        else Path("results") / "real2sim" / "previews" / f"{scene_dir.name}.png"
    )
    return render_model(
        _load_model(scene_dir),
        output,
        width=width,
        height=height,
        step_count=step_count,
        **camera,
    )


def launch_scene_viewer(
    scene_ref: str | Path,
    *,
    azimuth: float | None = None,
    elevation: float | None = None,
    distance: float | None = None,
    lookat: tuple[float, float, float] | None = None,
) -> None:
    """Open MuJoCo's interactive viewer for a packaged scene."""

    scene_dir = resolve_scene_dir(scene_ref)
    camera = _camera_options(
        scene_dir,
        azimuth=azimuth,
        elevation=elevation,
        distance=distance,
        lookat=lookat,
    )
    launch_model_viewer(_load_model(scene_dir), **camera)


def _load_model(scene_dir: Path) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(scene_dir / "scene.xml"))


def _default_camera(scene_dir: Path) -> dict[str, Any]:
    scene_json = scene_dir / "scene.json"
    if not scene_json.is_file():
        return {}
    return json.loads(scene_json.read_text(encoding="utf-8")).get("camera") or {}


def _camera_options(
    scene_dir: Path,
    *,
    azimuth: float | None,
    elevation: float | None,
    distance: float | None,
    lookat: tuple[float, float, float] | None,
) -> dict[str, Any]:
    default_camera = _default_camera(scene_dir)
    return {
        "azimuth": _coalesce_float(azimuth, default_camera.get("camera_azimuth"), 135.0),
        "elevation": _coalesce_float(
            elevation,
            default_camera.get("camera_elevation"),
            -25.0,
        ),
        "distance": _coalesce_float_or_none(
            distance,
            default_camera.get("camera_distance"),
        ),
        "lookat": lookat
        if lookat is not None
        else _coerce_vec3_or_none(default_camera.get("camera_lookat")),
    }


def _coalesce_float(*values: Any) -> float:
    for value in values:
        if value is not None:
            return float(value)
    raise ValueError("expected at least one non-None value")


def _coalesce_float_or_none(*values: Any) -> float | None:
    for value in values:
        if value is not None:
            return float(value)
    return None


def _coerce_vec3_or_none(value: Any) -> tuple[float, float, float] | None:
    if value is None:
        return None
    parts = [float(part) for part in value]
    if len(parts) != 3:
        return None
    return parts[0], parts[1], parts[2]


def build_arg_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "scene",
        help=(
            "Packaged scene directory, or the name of a scene under the brace "
            "package's assets/scenes."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="PNG output path. Defaults to results/real2sim/previews/<scene>.png.",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--azimuth", type=float, default=None)
    parser.add_argument("--elevation", type=float, default=None)
    parser.add_argument("--distance", type=float, default=None)
    parser.add_argument("--lookat", type=_parse_vec3, default=None)
    parser.add_argument(
        "--step-count",
        type=int,
        default=0,
        help="Simulation steps to advance before rendering a PNG.",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Launch the interactive MuJoCo viewer instead of writing a PNG.",
    )
    return parser


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    args = build_arg_parser(prog).parse_args(argv)
    if args.viewer:
        launch_scene_viewer(
            args.scene,
            azimuth=args.azimuth,
            elevation=args.elevation,
            distance=args.distance,
            lookat=args.lookat,
        )
    else:
        output_path = render_scene(
            args.scene,
            args.output,
            width=args.width,
            height=args.height,
            azimuth=args.azimuth,
            elevation=args.elevation,
            distance=args.distance,
            lookat=args.lookat,
            step_count=args.step_count,
        )
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
