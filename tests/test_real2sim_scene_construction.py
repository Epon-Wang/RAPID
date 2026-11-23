"""Regression tests for tool-free Real2Sim scene construction."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import patch

import mujoco
import numpy as np
import trimesh

from real2sim.mujoco_export import (
    InitialObjectPenetrationError,
    _obb_3d_from_vertices,
    _quat_to_rotation_matrix,
    compile_mujoco_scene,
    write_mujoco_scene,
)
from real2sim.schema import PipelineConfig
from real2sim.workflow import package_reconstruction


def _arbitrary_rotation() -> np.ndarray:
    roll, pitch, yaw = 0.31, -0.42, 0.47
    rotation_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(roll), -np.sin(roll)],
            [0.0, np.sin(roll), np.cos(roll)],
        ],
        dtype=float,
    )
    rotation_y = np.array(
        [
            [np.cos(pitch), 0.0, np.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-np.sin(pitch), 0.0, np.cos(pitch)],
        ],
        dtype=float,
    )
    rotation_z = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return rotation_z @ rotation_y @ rotation_x


def _rotated_box_mesh() -> tuple[trimesh.Trimesh, np.ndarray]:
    mesh = trimesh.creation.box(extents=[0.4, 0.2, 0.1])
    rotation = _arbitrary_rotation()
    mesh.vertices = np.asarray(mesh.vertices) @ rotation.T
    mesh.apply_translation([0.03, -0.04, 0.052])
    return mesh, np.asarray(mesh.vertices, dtype=float).copy()


def _object(mesh_file: str, vertices: np.ndarray, *, is_static: bool = False):
    bounds = np.stack([vertices.min(axis=0), vertices.max(axis=0)])
    return SimpleNamespace(
        name="test object",
        safe_name="test_object",
        description="",
        mesh_file=mesh_file,
        obb_source_mesh_file=mesh_file,
        visual_mesh_files=[],
        body_pos=[0.0, 0.0, 0.0],
        local_bounds=bounds.tolist(),
        world_bounds=bounds.tolist(),
        is_static=is_static,
    )


class OrientedCollisionBoxTest(unittest.TestCase):
    def test_collision_box_is_tight_3d_obb(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mesh, vertices = _rotated_box_mesh()
            mesh_path = root / "object.obj"
            mesh.export(mesh_path)
            scene_xml = root / "scene.xml"

            write_mujoco_scene(
                scene_xml,
                [_object(mesh_path.name, vertices)],
                PipelineConfig(verbose=False),
            )

            xml_root = ET.parse(scene_xml).getroot()
            collision = xml_root.find(
                "./worldbody/body[@name='test_object']/geom[@name='test_object_collision']"
            )
            self.assertIsNotNone(collision)
            assert collision is not None
            self.assertEqual(collision.get("type"), "box")
            self.assertIsNone(collision.get("mesh"))
            self.assertFalse(
                any(
                    "collision_mesh" in (asset.get("name") or "")
                    for asset in xml_root.findall("./asset/mesh")
                )
            )

            model = mujoco.MjModel.from_xml_path(str(scene_xml))
            geom_id = model.geom("test_object_collision").id
            rotation = np.empty(9, dtype=float)
            mujoco.mju_quat2Mat(rotation, model.geom_quat[geom_id])
            rotation = rotation.reshape(3, 3)
            center = np.asarray(model.geom_pos[geom_id], dtype=float)
            size = np.asarray(model.geom_size[geom_id], dtype=float)
            projected = (vertices - center) @ rotation

            np.testing.assert_allclose(projected.min(axis=0), -size, atol=1e-8)
            np.testing.assert_allclose(projected.max(axis=0), size, atol=1e-8)
            self.assertLess(float(np.max(np.abs(rotation[2, :]))), 0.99)
            np.testing.assert_allclose(
                np.sort(2.0 * size), np.sort([0.4, 0.2, 0.1]), atol=1e-8
            )

            body_id = model.body("test_object").id
            np.testing.assert_allclose(model.body_ipos[body_id], center, atol=1e-10)
            self.assertEqual(model.body_jntnum[body_id], 1)

    def test_board_obb_includes_nail_protrusions(self) -> None:
        board = trimesh.creation.box(extents=[1.2, 0.18, 0.02])
        vertices = np.asarray(board.vertices)
        faces = np.asarray(board.faces)
        for _ in range(3):
            vertices, faces = trimesh.remesh.subdivide(vertices, faces)

        point_sets = [vertices]
        nail_positions = (
            (-0.42, -0.04),
            (-0.12, 0.05),
            (0.21, -0.03),
            (0.45, 0.04),
        )
        for x, y in nail_positions:
            nail = trimesh.creation.cylinder(radius=0.012, height=0.05, sections=16)
            nail.apply_translation([x, y, 0.035])
            point_sets.append(np.asarray(nail.vertices))

        rotation = _arbitrary_rotation()
        center = np.array([0.08, -0.03, 0.12])
        vertices_with_nails = np.vstack(point_sets) @ rotation.T + center
        collision_box = _obb_3d_from_vertices(vertices_with_nails)

        np.testing.assert_allclose(
            np.sort(2.0 * collision_box.half_extents),
            np.sort([1.2, 0.18, 0.07]),
            atol=2e-3,
        )
        np.testing.assert_allclose(
            collision_box.center,
            center + rotation @ np.array([0.0, 0.0, 0.025]),
            atol=2e-3,
        )

    def test_box_obb_uses_both_broad_faces(self) -> None:
        # Model two dense, slightly uneven broad faces plus the four side faces.
        # Collision fitting must always retain the complete vertex set.
        y = np.linspace(-0.06, 0.06, 10)
        z = np.linspace(-0.086, 0.086, 16)
        yy, zz = np.meshgrid(y, z, indexing="ij")
        ripple = 0.002 * np.sin(yy * 40.0) * np.cos(zz * 30.0)
        broad_faces = [
            np.column_stack(
                [
                    np.full(yy.size, side * 0.0205) + ripple.ravel(),
                    yy.ravel(),
                    zz.ravel(),
                ]
            )
            for side in (-1.0, 1.0)
        ]
        side_points: list[np.ndarray] = []
        x = np.linspace(-0.0225, 0.0225, 5)
        for coordinate, extent, axis in (
            (-0.06, 0.086, 1),
            (0.06, 0.086, 1),
            (-0.086, 0.06, 2),
            (0.086, 0.06, 2),
        ):
            other = np.linspace(-extent, extent, 10)
            points = np.zeros((len(x) * len(other), 3), dtype=float)
            points[:, 0] = np.repeat(x, len(other))
            points[:, axis] = coordinate
            points[:, 3 - axis] = np.tile(other, len(x))
            side_points.append(points)

        vertices = np.vstack([*broad_faces, *side_points])
        rotation = _arbitrary_rotation()
        vertices = vertices @ rotation.T
        collision_box = _obb_3d_from_vertices(vertices)

        np.testing.assert_allclose(
            np.sort(2.0 * collision_box.half_extents),
            np.sort([0.045, 0.12, 0.172]),
            atol=5e-3,
        )

    def test_fixed_objects_still_receive_box_collisions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mesh, vertices = _rotated_box_mesh()
            mesh_path = root / "object.obj"
            mesh.export(mesh_path)
            scene_xml = root / "scene.xml"
            original_box = _obb_3d_from_vertices(vertices)
            original_rotation = _quat_to_rotation_matrix(original_box.quat)
            expected_angle = np.arccos(
                np.max(np.abs(original_rotation[2, :]))
            )
            original_bottom = float(
                original_box.center[2]
                - np.sum(
                    np.abs(original_rotation[2, :])
                    * original_box.half_extents
                )
            )

            write_mujoco_scene(
                scene_xml,
                [_object(mesh_path.name, vertices, is_static=True)],
                PipelineConfig(dynamic_objects=False, verbose=False),
            )

            model = mujoco.MjModel.from_xml_path(str(scene_xml))
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            body_id = model.body("test_object").id
            geom_id = model.geom("test_object_collision").id
            self.assertEqual(model.body_jntnum[body_id], 0)
            self.assertEqual(
                model.geom_type[geom_id],
                mujoco.mjtGeom.mjGEOM_BOX,
            )
            world_rotation = data.geom_xmat[geom_id].reshape(3, 3)
            self.assertAlmostEqual(
                float(np.max(np.abs(world_rotation[2, :]))),
                1.0,
                places=10,
            )
            corrected_bottom = float(
                data.geom_xpos[geom_id, 2]
                - np.sum(
                    np.abs(world_rotation[2, :]) * model.geom_size[geom_id]
                )
            )
            self.assertAlmostEqual(corrected_bottom, original_bottom, places=7)
            actual_angle = 2.0 * np.arccos(abs(float(model.body_quat[body_id, 0])))
            self.assertAlmostEqual(actual_angle, expected_angle, places=7)


class InitialObjectPenetrationTest(unittest.TestCase):
    def test_rejects_deep_object_overlap_but_ignores_floor_contact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mesh = trimesh.creation.box(extents=[0.2, 0.2, 0.2])
            first_path = root / "first.obj"
            second_path = root / "second.obj"
            mesh.export(first_path)
            mesh.export(second_path)
            vertices = np.asarray(mesh.vertices)
            first = _object(first_path.name, vertices)
            first.name = "first"
            first.safe_name = "first"
            second = _object(second_path.name, vertices)
            second.name = "second"
            second.safe_name = "second"
            scene_xml = root / "scene.xml"
            write_mujoco_scene(
                scene_xml,
                [first, second],
                PipelineConfig(verbose=False),
            )

            with self.assertRaises(InitialObjectPenetrationError) as raised:
                compile_mujoco_scene(
                    scene_xml,
                    object_safe_names=["first", "second"],
                    max_initial_object_penetration_meters=0.03,
                )

            self.assertGreater(
                raised.exception.report["maximum_penetration_meters"],
                0.03,
            )
            self.assertNotIn(
                "floor",
                raised.exception.report["maximum_penetration_pair"],
            )


_SOURCE_SCENE_XML = """<mujoco model="source">
  <compiler meshdir="."/>
  <asset>
    <mesh name="object_visual_mesh" file="object.obj"/>
    <mesh name="object_collision_mesh" file="object.obj"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.01" friction="0.44 0.01 0.001"/>
    <body name="object">
      <freejoint/>
      <inertial mass="0.2" pos="0 0 0" diaginertia="0.001 0.001 0.001"/>
      <geom name="object_visual" type="mesh" mesh="object_visual_mesh"
            contype="0" conaffinity="0"/>
      <geom name="object_collision" type="box" size="0.2 0.1 0.05" friction="0.38 0.007 0.001"/>
    </body>
    <body name="fixture" pos="0.7 0.3 0.05">
      <geom name="fixture_collision" type="box" size="0.05 0.05 0.05"/>
    </body>
  </worldbody>
</mujoco>
"""


def _write_reconstruction(source: Path) -> None:
    """Write a minimal Real2Sim output: one free object, one fixed object."""

    source.mkdir()
    mesh, _ = _rotated_box_mesh()
    mesh.export(source / "object.obj")
    (source / "scene.xml").write_text(_SOURCE_SCENE_XML, encoding="utf-8")
    (source / "metadata.json").write_text(
        json.dumps(
            {
                "objects": [
                    {
                        "name": "test box",
                        "safe_name": "object",
                        "obb_source_mesh_file": "object.obj",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def _scene_entry(scene_dir: Path, body: str) -> dict:
    scene = json.loads((scene_dir / "scene.json").read_text(encoding="utf-8"))
    return next(item for item in scene["objects"] if item["body"] == body)


class BraceScenePackagingTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        _write_reconstruction(self.root / "source")
        self.result = package_reconstruction(
            self.root / "source",
            name="tool_free",
            tasks_root=self.root / "scenes",
        )
        self.scene_dir = self.result.scene_dir

    def test_writes_scene_directory_without_python_wrappers(self) -> None:
        self.assertEqual(self.scene_dir, (self.root / "scenes" / "tool_free").resolve())
        for name in ("scene.json", "scene.xml", "objects/object.xml", "objects/fixture.xml"):
            self.assertTrue((self.scene_dir / name).is_file(), name)
        for name in ("scene.py", "replay.py", "__init__.py", "bundle"):
            self.assertFalse((self.scene_dir / name).exists(), name)
        self.assertFalse((self.root / "scenes" / "__init__.py").exists())

    def test_object_file_compiles_alone_as_one_free_body_at_the_origin(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(self.scene_dir / "objects" / "object.xml"))

        self.assertEqual(model.nbody, 2)
        self.assertEqual(model.body(1).name, "object")
        self.assertEqual(model.njnt, 1)
        self.assertEqual(model.jnt_type[0], mujoco.mjtJoint.mjJNT_FREE)
        np.testing.assert_allclose(model.qpos0, [0, 0, 0, 1, 0, 0, 0], atol=1e-12)
        self.assertAlmostEqual(float(model.body_mass[1]), 0.2)
        geom_types = [int(kind) for kind in model.geom_type]
        self.assertIn(int(mujoco.mjtGeom.mjGEOM_MESH), geom_types)
        self.assertIn(int(mujoco.mjtGeom.mjGEOM_BOX), geom_types)

    def test_scene_json_records_the_resting_pose_of_each_object(self) -> None:
        entry = _scene_entry(self.scene_dir, "object")
        scene_model = mujoco.MjModel.from_xml_path(str(self.scene_dir / "scene.xml"))
        address = scene_model.jnt_qposadr[scene_model.body("object").jntadr[0]]

        self.assertEqual(entry["file"], "objects/object.xml")
        self.assertEqual(entry["name"], "test box")
        self.assertFalse(entry["fixed"])
        np.testing.assert_allclose(entry["pos"], scene_model.qpos0[address : address + 3], atol=1e-6)
        np.testing.assert_allclose(entry["quat"], scene_model.qpos0[address + 3 : address + 7], atol=1e-6)
        self.assertAlmostEqual(entry["mass"], 0.2)
        np.testing.assert_allclose(entry["friction"], [0.38, 0.007, 0.001])

    def test_fixed_object_keeps_its_pose_in_scene_json_and_has_no_joint(self) -> None:
        entry = _scene_entry(self.scene_dir, "fixture")
        model = mujoco.MjModel.from_xml_path(str(self.scene_dir / "objects" / "fixture.xml"))

        self.assertTrue(entry["fixed"])
        np.testing.assert_allclose(entry["pos"], [0.7, 0.3, 0.05])
        np.testing.assert_allclose(entry["quat"], [1, 0, 0, 0])
        self.assertEqual(model.njnt, 0)
        np.testing.assert_allclose(model.body_pos[1], [0, 0, 0])

    def test_scene_json_records_floor_height_and_friction(self) -> None:
        source = self.root / "raised"
        source.mkdir()
        (source / "scene.xml").write_text(
            '<mujoco><worldbody><geom name="floor" type="plane" pos="0 0 0.05" '
            'size="1 1 0.01" friction="0.44 0.01 0.001"/></worldbody></mujoco>',
            encoding="utf-8",
        )

        result = package_reconstruction(source, name="raised", tasks_root=self.root / "scenes")
        scene = json.loads((result.scene_dir / "scene.json").read_text(encoding="utf-8"))

        self.assertEqual(scene["name"], "raised")
        self.assertEqual(scene["objects"], [])
        self.assertAlmostEqual(scene["floor"]["height"], 0.05)
        np.testing.assert_allclose(scene["floor"]["friction"], [0.44, 0.01, 0.001])


class DefaultSceneRootTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        _write_reconstruction(self.root / "source")

    def test_defaults_to_assets_scenes_inside_the_brace_package(self) -> None:
        package_dir = self.root / "site" / "brace"
        package_dir.mkdir(parents=True)
        fake_brace = types.ModuleType("brace")
        fake_brace.__file__ = str(package_dir / "__init__.py")

        with patch.dict(sys.modules, {"brace": fake_brace}):
            result = package_reconstruction(self.root / "source", name="tool_free")

        self.assertEqual(
            result.scene_dir,
            (package_dir / "assets" / "scenes" / "tool_free").resolve(),
        )
        self.assertTrue((result.scene_dir / "scene.json").is_file())

    def test_explains_how_to_proceed_when_brace_is_not_installed(self) -> None:
        with patch.dict(sys.modules, {"brace": None}):
            with self.assertRaisesRegex(RuntimeError, "--tasks-root"):
                package_reconstruction(self.root / "source", name="tool_free")


if __name__ == "__main__":
    unittest.main()
