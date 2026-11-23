"""The bundled example reconstruction stays packageable without model services."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from real2sim.workflow import package_reconstruction

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "push_scene"


class BundledExampleTest(unittest.TestCase):
    def test_push_scene_example_packages_into_three_objects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            packaged = package_reconstruction(EXAMPLE, tasks_root=Path(tmp) / "scenes")
            scene = json.loads((packaged.scene_dir / "scene.json").read_text(encoding="utf-8"))

            self.assertEqual(packaged.scene_dir.name, "push_scene")
            self.assertEqual(
                [item["body"] for item in scene["objects"]],
                ["brown_cardboard_box", "cheez_it_box", "white_divider"],
            )
            for item in scene["objects"]:
                self.assertTrue((packaged.scene_dir / item["file"]).is_file(), item["file"])


if __name__ == "__main__":
    unittest.main()
