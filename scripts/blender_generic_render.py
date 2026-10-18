"""Render class-mapped Blender scenes; each root defines one object instance.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blender_render as common


def mapped_instances(scene, request):
    mapping = request.get("scene_class_map")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("A nonempty scene_class_map is required")
    groups, owner = {}, {}
    scene_objects = set(scene.objects)
    for root_name, class_id in sorted(mapping.items()):
        if not isinstance(root_name, str) or not root_name.strip() or not isinstance(class_id, int) or isinstance(class_id, bool) or class_id < 0:
            raise ValueError("scene_class_map maps root names to nonnegative integer classes")
        kind, name = root_name.split(":", 1) if root_name.startswith(("object:", "collection:")) else (None, root_name)
        obj = bpy.data.objects.get(name) if kind != "collection" else None
        collection = bpy.data.collections.get(name) if kind != "object" else None
        if obj is not None and collection is not None:
            raise ValueError(f"Ambiguous map root {name!r}; prefix it with object: or collection:")
        if obj is not None:
            candidates = [obj, *obj.children_recursive]
        elif collection is not None:
            if collection.hide_render:
                raise ValueError(f"Mapped collection is hidden from rendering: {name}")
            candidates = list(collection.all_objects)
        else:
            raise ValueError(f"Mapped object/collection does not exist: {root_name}")
        if any(item.instance_type == "COLLECTION" for item in candidates):
            raise ValueError(f"Mapped root {root_name} contains a collection instancer; make instances real and map each instance root")
        meshes = [item for item in candidates if item in scene_objects and item.type == "MESH" and not item.hide_render]
        if not meshes:
            raise ValueError(f"Mapped root {root_name} has no renderable scene meshes")
        for item in meshes:
            if item.name in owner:
                raise ValueError(f"Mesh {item.name} belongs to overlapping instance roots {owner[item.name]!r} and {root_name!r}")
            owner[item.name] = root_name
        groups[root_name] = {"id": root_name, "class_id": class_id, "all": meshes, "body": meshes}
    if request["descriptor"]["source_class"] not in {group["class_id"] for group in groups.values()}:
        raise ValueError("No mapped instance matches the requested source class")
    return groups


def perturb_generic(scene, camera, groups, focus, attrs, rng):
    return common.perturb(scene, camera, groups, focus, attrs, rng, tomato_color_control=False)


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    args = parser.parse_args(argv)
    response_path = Path(args.response).resolve()
    try:
        common.main(Path(args.request).resolve(), response_path,
                    group_adapter=mapped_instances, perturb_adapter=perturb_generic,
                    adapter_name="explicit_scene_class_map")
    except Exception as exc:
        response_path.write_text(json.dumps({"status": "error", "error": str(exc), "traceback": traceback.format_exc()}, indent=2), encoding="utf-8")
        traceback.print_exc()
        raise
