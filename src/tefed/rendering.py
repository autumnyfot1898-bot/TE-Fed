"""Render user-supplied Blender scenes; NIR uses luminance and noise."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any


class BlenderRenderer:
    def __init__(
        self,
        blender_executable: str | Path,
        scene_path: str | Path | None = None,
        *,
        timeout: float = 600,
        samples: int = 8,
        script_path: str | Path | None = None,
        generic_script_path: str | Path | None = None,
    ) -> None:
        resolved = shutil.which(str(blender_executable))
        self.executable = Path(resolved or blender_executable).expanduser().resolve()
        if not self.executable.is_file():
            raise FileNotFoundError(f"Blender executable not found: {self.executable}")
        self.scene_path = Path(scene_path).expanduser().resolve() if scene_path else None
        self.timeout = float(timeout)
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        if not isinstance(samples, int) or isinstance(samples, bool) or not 1 <= samples <= 128:
            raise ValueError("samples must be an integer in [1, 128]")
        self.samples = samples
        project_root = Path(__file__).resolve().parents[2]
        self.script_path = Path(script_path or project_root / "scripts" / "blender_render.py").resolve()
        if not self.script_path.is_file():
            raise FileNotFoundError(f"Blender rendering script not found: {self.script_path}")
        self.generic_script_path = Path(generic_script_path or project_root / "scripts" / "blender_generic_render.py").resolve()

    def render(self, request: dict[str, Any], output_dir: Path) -> dict[str, Any]:
        """Render aligned images, targets, and measurements into a fresh directory.

        Require descriptor (eight attrs/attributes and source_class), seed,
        and case_id. scene_path may override the configured scene."""
        if not isinstance(request, dict):
            raise TypeError("request must be a dictionary")
        descriptor = request.get("descriptor")
        if not isinstance(descriptor, dict):
            raise ValueError("request.descriptor must be a dictionary")
        attrs = descriptor.get("attrs", descriptor.get("attributes"))
        if not isinstance(attrs, (list, tuple)) or len(attrs) != 8:
            raise ValueError("descriptor.attrs must contain eight values")
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x <= 1 for x in attrs):
            raise ValueError("descriptor attributes must be finite numbers in [0, 1]")
        source_class = descriptor.get("source_class")
        class_map = request.get("scene_class_map")
        if class_map is not None:
            if not isinstance(class_map, dict) or not class_map or any(not isinstance(k, str) or not k.strip() or not isinstance(v, int) or isinstance(v, bool) or v < 0 for k, v in class_map.items()):
                raise ValueError("scene_class_map must map nonempty object/collection root names to nonnegative integer classes")
            if not isinstance(source_class, int) or isinstance(source_class, bool) or source_class not in class_map.values():
                raise ValueError("source_class must appear in scene_class_map")
            if not self.generic_script_path.is_file():
                raise FileNotFoundError(f"Generic rendering script not found: {self.generic_script_path}")
        elif not isinstance(source_class, int) or isinstance(source_class, bool) or source_class not in (0, 1):
            raise ValueError("tomato renderer supports source_class 0/1; other classes require scene_class_map")
        seed = request.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
            raise ValueError("seed must be a uint32 integer")
        case_id = request.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("case_id must be a nonempty string")
        image_size = request.get("image_size", [640, 640])
        if isinstance(image_size, int) and not isinstance(image_size, bool):
            image_size = [image_size, image_size]
        if not isinstance(image_size, (list, tuple)) or len(image_size) != 2 or any(not isinstance(v, int) or isinstance(v, bool) or not 32 <= v <= 4096 for v in image_size):
            raise ValueError("image_size must be [width, height], both integers in [32, 4096]")
        scene_value = request.get("scene_path") or self.scene_path
        if scene_value is None:
            raise ValueError("scene_path is required; local assets are not bundled")
        scene = Path(scene_value).expanduser().resolve(strict=True)
        if not scene.is_file() or scene.suffix.lower() != ".blend":
            raise ValueError("scene_path must refer to an existing .blend file")
        out = Path(output_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        if any(out.iterdir()):
            raise FileExistsError(f"Render output directory must be empty: {out}")
        payload = dict(request)
        payload.update(scene_path=str(scene), image_size=list(image_size), samples=self.samples)
        payload["descriptor"] = dict(descriptor, attrs=list(attrs))
        request_path, response_path = out / "request.json", out / "response.json"
        request_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        command = [
            str(self.executable), "--background", "--factory-startup",
            "--python", str(self.generic_script_path if class_map is not None else self.script_path), "--",
            "--request", str(request_path), "--response", str(response_path),
        ]
        try:
            result = subprocess.run(command, shell=False, capture_output=True, text=True,
                                    encoding="utf-8", errors="replace", timeout=self.timeout)
        except subprocess.TimeoutExpired as exc:
            (out / "blender.log").write_text(f"Rendering timed out after {self.timeout} seconds.\n{exc}", encoding="utf-8")
            raise RuntimeError(f"Blender rendering timed out; see {out / 'blender.log'}") from exc
        (out / "blender.log").write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
        if result.returncode != 0 or not response_path.is_file():
            raise RuntimeError(f"Blender failed (exit {result.returncode}); see {out / 'blender.log'}")
        response = json.loads(response_path.read_text(encoding="utf-8"))
        if response.get("status") != "ok":
            raise RuntimeError(f"Blender rejected case: {response.get('error', 'unknown error')}")
        for key in ("rgb", "depth", "nir", "instance_mask"):
            path = Path(response[key]).resolve()
            if not path.is_relative_to(out) or not path.is_file():
                raise RuntimeError(f"Invalid or missing renderer output {key}: {path}")
        if not response.get("targets") or response.get("focal_target_index") is None:
            raise RuntimeError("Renderer produced no visible source-class focal target")
        measured = response.get("attrs_measured")
        if not isinstance(measured, list) or len(measured) != 8:
            raise RuntimeError("Renderer measurements must contain eight entries")
        return response


Renderer = BlenderRenderer
