"""Previewing packaged Real2Sim scene directories."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from real2sim.workflow import package_reconstruction
from tests.test_real2sim_scene_construction import _write_reconstruction
from vistools.visualize_scene import render_scene, resolve_scene_dir


class ScenePreviewTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        _write_reconstruction(self.root / "source")
        self.scene_dir = package_reconstruction(
            self.root / "source",
            name="tool_free",
            tasks_root=self.root / "scenes",
        ).scene_dir

    def test_renders_a_png_from_a_packaged_scene_directory(self) -> None:
        output = render_scene(self.scene_dir, self.root / "preview.png", width=160, height=120)

        self.assertEqual(output, self.root / "preview.png")
        self.assertEqual(output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_bare_name_resolves_under_the_brace_scene_root(self) -> None:
        package_dir = self.root / "scenes_site" / "brace"
        scenes = package_dir / "assets" / "scenes"
        scenes.mkdir(parents=True)
        self.scene_dir.rename(scenes / "tool_free")
        fake_brace = types.ModuleType("brace")
        fake_brace.__file__ = str(package_dir / "__init__.py")

        with patch.dict(sys.modules, {"brace": fake_brace}):
            resolved = resolve_scene_dir("tool_free")

        self.assertEqual(resolved, (scenes / "tool_free").resolve())

    def test_unknown_scene_is_reported_as_missing(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "no_such_scene"):
            resolve_scene_dir(self.root / "no_such_scene")


if __name__ == "__main__":
    unittest.main()
