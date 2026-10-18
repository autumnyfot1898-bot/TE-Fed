"""Inspect a Blender asset with Blender Python.

blender --background --factory-startup --python scripts/inspect_blend.py -- \
    --scene /path/to/asset.blend --output /path/to/inventory.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    path = Path(args.scene).expanduser().resolve(strict=True)
    bpy.ops.wm.open_mainfile(filepath=str(path), load_ui=False)
    scene = bpy.context.scene
    result = {
        "scene_path": str(path),
        "blender_version": bpy.app.version_string,
        "camera": scene.camera.name if scene.camera else None,
        "render_engine": scene.render.engine,
        "resolution": [scene.render.resolution_x, scene.render.resolution_y],
        "object_count": len(scene.objects),
        "mesh_count": sum(obj.type == "MESH" for obj in scene.objects),
        "objects": [
            {
                "name": obj.name,
                "type": obj.type,
                "hidden": bool(obj.hide_render),
                "location": list(obj.location),
                "dimensions": list(obj.dimensions),
                "materials": [s.material.name for s in obj.material_slots if s.material],
            }
            for obj in scene.objects
            if obj.type in {"MESH", "CAMERA", "LIGHT"}
        ],
        "external_images": [
            {"name": im.name, "filepath": im.filepath, "packed": bool(im.packed_file)}
            for im in bpy.data.images
            if im.source == "FILE"
        ],
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"TEFED_ASSET_INVENTORY={output}", flush=True)


if __name__ == "__main__":
    main()
