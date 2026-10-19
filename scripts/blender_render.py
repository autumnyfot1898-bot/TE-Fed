"""Render one asset-backed case with Blender Python."""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys
import traceback
import zlib
from pathlib import Path

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector


ATTRIBUTE_NAMES = ["size", "cls", "occ", "illum", "view", "geo", "ctx", "sens"]
FRUIT_PATTERN = re.compile(r"^(project_asset_tomato_(red|green|yellow)_\d+_\d+_\d+)_")


def write_png(path: Path, values: np.ndarray) -> None:
    """Write 8-bit PNG without Pillow."""
    data = np.asarray(np.clip(values, 0, 255), dtype=np.uint8)
    if data.ndim == 2:
        color_type = 0
    elif data.ndim == 3 and data.shape[2] == 3:
        color_type = 2
    else:
        raise ValueError("PNG expects an HxW grayscale or HxWx3 RGB array")
    height, width = data.shape[:2]
    def chunk(kind: bytes, content: bytes) -> bytes:
        return struct.pack(">I", len(content)) + kind + content + struct.pack(">I", zlib.crc32(kind + content) & 0xFFFFFFFF)
    rows = b"".join(b"\x00" + row.tobytes() for row in data)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(rows, 6)) + chunk(b"IEND", b""))


def read_image(path: Path, *, raw: bool = True) -> np.ndarray:
    image = bpy.data.images.load(str(path), check_existing=False)
    if raw:
        image.colorspace_settings.name = "Non-Color"
    width, height = image.size
    pixels = np.empty(width * height * image.channels, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    result = pixels.reshape(height, width, image.channels)[::-1].copy()
    bpy.data.images.remove(image)
    return result


def body_objects(scene):
    groups = {}
    for obj in scene.objects:
        match = FRUIT_PATTERN.match(obj.name)
        if obj.type != "MESH" or obj.hide_render or match is None:
            continue
        key, color = match.groups()
        group = groups.setdefault(key, {"id": key, "color": color, "class_id": 0 if color == "red" else 1, "all": [], "body": []})
        group["all"].append(obj)
        materials = " ".join(s.material.name.lower() for s in obj.material_slots if s.material)
        if "candy" in materials or "color" in materials or "yellow_tomato" in materials:
            group["body"].append(obj)
    groups = {key: value for key, value in groups.items() if value["body"]}
    if not groups:
        raise ValueError("No recognized fruit bodies. Expected project_asset_tomato_{red,green,yellow}_<plant>_<branch>_<fruit>_* with body material containing candy/color/yellow_tomato")
    return groups


def corners(objects):
    return [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]


def center(objects):
    points = corners(objects)
    low = Vector(tuple(min(p[i] for p in points) for i in range(3)))
    high = Vector(tuple(max(p[i] for p in points) for i in range(3)))
    return (low + high) * 0.5


def transform_group(objects, transform):
    """Transform components in parent order to avoid double transforms."""
    matrices = {obj: transform @ obj.matrix_world.copy() for obj in objects}
    def depth(obj):
        count, parent = 0, obj.parent
        while parent is not None:
            count += 1
            parent = parent.parent
        return count
    for obj in sorted(objects, key=depth):
        obj.matrix_world = matrices[obj]
    bpy.context.view_layer.update()


def projected_box(scene, camera, objects):
    coordinates = [world_to_camera_view(scene, camera, point) for point in corners(objects)]
    if not coordinates or any(co.z <= camera.data.clip_start for co in coordinates):
        return None
    return [min(co.x for co in coordinates), 1 - max(co.y for co in coordinates), max(co.x for co in coordinates), 1 - min(co.y for co in coordinates)]


def configure_passes(scene, out: Path, prefix: str):
    layer = scene.view_layers[0]
    layer.use_pass_z = True
    layer.use_pass_object_index = True
    if hasattr(scene, "compositing_node_group"):
        tree = bpy.data.node_groups.new("TEFedCompositor", "CompositorNodeTree")
        scene.compositing_node_group = tree
        tree.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
        composite = tree.nodes.new("NodeGroupOutput")
    else:
        scene.use_nodes = True
        tree = scene.node_tree
        tree.nodes.clear()
        composite = tree.nodes.new("CompositorNodeComposite")
    render = tree.nodes.new("CompositorNodeRLayers")
    tree.links.new(render.outputs["Image"], composite.inputs["Image"])
    for socket_names, name in [(["Depth"], "depth"), (["Object Index", "IndexOB"], "instances")]:
        node = tree.nodes.new("CompositorNodeOutputFile")
        if hasattr(node.format, "media_type"):
            node.format.media_type = "IMAGE"
        node.format.file_format = "OPEN_EXR"
        node.format.color_depth = "32"
        node.format.color_mode = "RGB"
        node.format.exr_codec = "ZIP"
        if hasattr(node, "file_output_items"):
            node.directory = str(out)
            node.file_name = f"{prefix}_{name}"
            node.file_output_items.new("FLOAT", name)
            node.file_output_items[0].save_as_render = False
            socket = node.inputs[name]
        else:
            node.base_path = str(out)
            node.file_slots[0].path = f"{prefix}_{name}"
            node.file_slots[0].save_as_render = False
            socket = node.inputs[0]
        source = next((render.outputs.get(s) for s in socket_names if render.outputs.get(s) is not None), None)
        if source is None:
            raise RuntimeError(f"Blender does not provide {socket_names} pass")
        tree.links.new(source, socket)
    return tree


def pass_file(out: Path, prefix: str, name: str) -> Path:
    candidates = sorted(out.glob(f"{prefix}_{name}*.exr"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one {prefix} {name} EXR, found {candidates}")
    return candidates[0]


def sensor_cues(rgb8, depth, nir):
    """Measure crop-level sensor degradation."""
    if rgb8.size == 0:
        return None, {}
    gray255 = 0.299 * rgb8[:, :, 0] + 0.587 * rgb8[:, :, 1] + 0.114 * rgb8[:, :, 2]
    if min(gray255.shape) < 16:
        q_blur = 1.0
        variance = None
    else:
        # Match the four-neighbor Laplacian in cues.py.
        p = np.pad(gray255, 1, mode="reflect")
        laplacian = p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4 * p[1:-1, 1:-1]
        variance = float(laplacian.var())
        q_blur = float(np.clip(1 - variance / 120, 0, 1))
    clipping_fraction = float(np.any((rgb8 == 0) | (rgb8 == 255), axis=2).mean())
    q_clip = float(np.clip(clipping_fraction / 0.20, 0, 1))
    q_rgb = 0.7 * q_blur + 0.3 * q_clip
    valid_depth_fraction = float((np.isfinite(depth) & (depth > 0)).mean())
    padded = np.pad(nir, 1, mode="reflect" if min(nir.shape) > 1 else "edge")
    local_median = np.median(np.lib.stride_tricks.sliding_window_view(padded, (3, 3)), axis=(-2, -1))
    residual = nir - local_median
    sigma = float(1.4826 * np.median(np.abs(residual - np.median(residual))))
    q_nir = float(np.clip(sigma / 0.08, 0, 1))
    score = float(max(1 - valid_depth_fraction, q_nir, q_rgb))
    return score, {"laplacian_variance": variance, "rgb_clipping_fraction": clipping_fraction,
                   "q_rgb": q_rgb, "valid_depth_fraction": valid_depth_fraction,
                   "nir_residual_robust_sigma": sigma, "q_nir_luminance_residual": q_nir}


def viewpoint_cue(depth_crop, box, source_class, width, height):
    """Measure depth-plane and aspect residuals, with sparse-depth fallback."""
    x1, y1, x2, y2 = box
    aspect = ((x2 - x1) * width) / max((y2 - y1) * height, 1e-6)
    class_aspect_median = 1.03 if source_class == 0 else 0.97
    r_aspect = float(np.clip(abs(math.log((aspect + 1e-6) / (class_aspect_median + 1e-6))) / math.log(1.7), 0, 1))
    valid = np.isfinite(depth_crop) & (depth_crop > 0)
    used_fallback = valid.sum() < 64 or valid.mean() < 0.35
    r_perspective = 0.5
    if not used_fallback:
        lo, hi = np.percentile(depth_crop[valid], [10, 90])
        keep = valid & (depth_crop >= lo) & (depth_crop <= hi)
        yy, xx = np.indices(depth_crop.shape)
        xx = 2 * (xx + round(x1 * width) + 0.5) / width - 1
        yy = 2 * (yy + round(y1 * height) + 0.5) / height - 1
        design = np.column_stack((xx[keep], yy[keep], np.ones(int(keep.sum()))))
        if len(design) >= 3:
            coefficients, _, rank, _ = np.linalg.lstsq(design, depth_crop[keep], rcond=None)
            if rank == 3 and np.isfinite(coefficients).all():
                a, b, c = coefficients
                r_perspective = float(np.clip(math.sqrt(a * a + b * b) / (0.35 * abs(c) + 1e-6), 0, 1))
            else:
                used_fallback = True
    return float(np.clip(0.6 * r_perspective + 0.4 * r_aspect, 0, 1)), {"r_perspective": r_perspective, "r_aspect": r_aspect, "perspective_fallback": bool(used_fallback)}


def perturb(scene, camera, groups, focus, attrs, rng, *, tomato_color_control=True):
    """Apply seeded scene perturbations."""
    focus_center = center(focus["body"])
    # Orbit the camera around the focal object.
    angle = float(rng.uniform(-1, 1)) * math.radians(12) * attrs[4]
    relative = camera.location - focus_center
    camera.location = focus_center + Matrix.Rotation(angle, 3, "Z") @ relative
    direction = focus_center - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    bpy.context.view_layer.update()
    box = projected_box(scene, camera, focus["body"])
    if box is None:
        raise ValueError("Focal fruit is behind the camera")
    current_linear = math.sqrt(max((box[2] - box[0]) * (box[3] - box[1]), 1e-10))
    desired_linear = float(np.clip(0.35 * (1 - attrs[0]), 0.04, 0.28))
    scale = float(np.clip(desired_linear / current_linear, 0.55, 2.0))
    transform = Matrix.Translation(focus_center) @ Matrix.Diagonal((scale, scale, scale, 1)) @ Matrix.Translation(-focus_center)
    if tomato_color_control:
        for obj in focus["all"]:
            obj.matrix_world = transform @ obj.matrix_world
    else:
        transform_group(focus["all"], transform)
    # Shift along the viewing ray.
    shift = direction.normalized() * float(rng.uniform(-0.04, 0.04) * attrs[5])
    if tomato_color_control:
        for obj in focus["all"]:
            obj.matrix_world = Matrix.Translation(shift) @ obj.matrix_world
    else:
        transform_group(focus["all"], Matrix.Translation(shift))
    ambiguity = attrs[1]
    body_color = None
    if tomato_color_control:
        body_color = (0.72, 0.04 + 0.20 * ambiguity, 0.025, 1) if focus["class_id"] == 0 else (0.12 + 0.28 * ambiguity, 0.52 - 0.12 * ambiguity, 0.035, 1)
        material = bpy.data.materials.new("TEFedFocalBodyClassPreservingColor")
        material.use_nodes = True
        shader = material.node_tree.nodes.get("Principled BSDF")
        shader.inputs["Base Color"].default_value = body_color
        shader.inputs["Roughness"].default_value = 0.42
        for obj in focus["body"]:
            obj.data = obj.data.copy()
            obj.data.materials.clear()
            obj.data.materials.append(material)
            for poly in obj.data.polygons:
                poly.material_index = 0
    else:
        # Preserve mapped class identity and linked textures.
        for obj in focus["body"]:
            obj.data = obj.data.copy()
            for slot in obj.material_slots:
                if slot.material is None:
                    continue
                material = slot.material.copy()
                slot.material = material
                if not material.use_nodes:
                    material.diffuse_color = tuple(float(np.clip(v * rng.uniform(1 - .15 * ambiguity, 1 + .15 * ambiguity), 0, 1)) for v in material.diffuse_color[:3]) + (material.diffuse_color[3],)
                    continue
                for shader in material.node_tree.nodes:
                    if shader.type == "BSDF_PRINCIPLED" and not shader.inputs["Base Color"].is_linked:
                        previous = shader.inputs["Base Color"].default_value
                        shader.inputs["Base Color"].default_value = tuple(float(np.clip(v * rng.uniform(1 - .15 * ambiguity, 1 + .15 * ambiguity), 0, 1)) for v in previous[:3]) + (previous[3],)
    illumination_scale = 2.5 ** (-attrs[3])
    for obj in scene.objects:
        if obj.type == "LIGHT":
            obj.data.energy *= illumination_scale
    if scene.world:
        scene.world.color = tuple(float(v) * (0.85 + 0.15 * (1 - attrs[6])) for v in scene.world.color)
    bpy.context.view_layer.update()
    focus_center = center(focus["body"])
    points = corners(focus["body"])
    diameter = max((point - focus_center).length for point in points) * 2
    if attrs[2] > 0.05:
        # Exclude the leaf-like occluder from target annotations.
        towards_camera = (camera.location - focus_center).normalized()
        right = camera.matrix_world.to_quaternion() @ Vector((1, 0, 0))
        location = focus_center + towards_camera * diameter * 0.70 + right * diameter * (0.45 - 0.30 * attrs[2])
        bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, location=location)
        occluder = bpy.context.object
        occluder.name = "TEFedNonTargetLeafOccluder"
        occluder.scale = (diameter * (0.08 + 0.25 * attrs[2]), diameter * 0.07, diameter * 0.7)
        leaf = bpy.data.materials.new("TEFedOccluderGreen")
        leaf.diffuse_color = (0.03, 0.22, 0.025, 1)
        occluder.data.materials.append(leaf)
    return {
        "focal_scale_multiplier": scale,
        "view_yaw_radians": angle,
        "illumination_energy_multiplier": illumination_scale,
        "focal_color_rgba": body_color,
        "geometry_translation_scene_units": list(shift),
        "control_values_are_measurements": False,
    }


def main(request_path: Path, response_path: Path, *, group_adapter=None, perturb_adapter=None, adapter_name="tomato_instance_names"):
    request = json.loads(request_path.read_text(encoding="utf-8"))
    out = response_path.parent
    descriptor = request["descriptor"]
    attrs = np.asarray(descriptor.get("attrs", descriptor.get("attributes")), dtype=np.float64)
    if attrs.shape != (8,) or not np.isfinite(attrs).all() or ((attrs < 0) | (attrs > 1)).any():
        raise ValueError("Invalid descriptor attributes")
    availability = np.asarray(descriptor.get("availability", [1] * 8))
    if availability.shape != (8,) or not np.isin(availability, [0, 1]).all():
        raise ValueError("descriptor.availability must contain eight binary flags")
    control_attrs = np.where(availability.astype(bool), attrs, 0)
    rng = np.random.default_rng(int(request["seed"]))
    bpy.ops.wm.open_mainfile(filepath=str(Path(request["scene_path"]).resolve(strict=True)), load_ui=False)
    scene = bpy.context.scene
    camera = scene.camera
    if camera is None:
        raise ValueError("Scene has no active camera")
    width, height = request.get("image_size", [640, 640])
    scene.render.resolution_x, scene.render.resolution_y = int(width), int(height)
    scene.render.resolution_percentage = 100
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = int(request.get("samples", 8))
    scene.cycles.use_denoising = False
    scene.cycles.seed = int(request["seed"]) % 2147483647
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.frame_set(1)
    groups = body_objects(scene) if group_adapter is None else group_adapter(scene, request)
    candidates = []
    for key, group in sorted(groups.items()):
        if group["class_id"] != descriptor["source_class"]:
            continue
        box = projected_box(scene, camera, group["body"])
        if box and box[2] > 0 and box[0] < 1 and box[3] > 0 and box[1] < 1:
            area = max(min(box[2], 1) - max(box[0], 0), 0) * max(min(box[3], 1) - max(box[1], 0), 0)
            if area > 0.00015:
                candidates.append((area, key))
    if not candidates:
        raise ValueError("No source-class fruit projects into the camera image")
    # Prefer large visible objects for stable measurements.
    candidates.sort(reverse=True)
    focus_id = candidates[int(rng.integers(min(5, len(candidates))))][1]
    focus = groups[focus_id]
    controls = (perturb_adapter or perturb)(scene, camera, groups, focus, control_attrs, rng)
    bpy.context.view_layer.update()
    group_ids = {}
    for obj in scene.objects:
        obj.pass_index = 0
    for pass_id, (key, group) in enumerate(sorted(groups.items()), 1):
        group_ids[pass_id] = group
        group["pass_index"] = pass_id
        for obj in group["body"]:
            obj.pass_index = pass_id
    configure_passes(scene, out, "main")
    raw_rgb = out / "rgb_raw.png"
    scene.render.filepath = str(raw_rgb)
    bpy.ops.render.render(write_still=True)
    rgb = read_image(raw_rgb)[:, :, :3]
    depth = read_image(pass_file(out, "main", "depth"))[:, :, 0]
    indices = np.rint(read_image(pass_file(out, "main", "instances"))[:, :, 0]).astype(np.int32)
    main_focus_mask = indices == focus["pass_index"]
    if main_focus_mask.sum() < 4:
        raise ValueError("Focal source-class fruit has fewer than four visible pixels")
    # Render isolated visibility for the occlusion denominator.
    keep = set(focus["body"])
    for obj in scene.objects:
        if obj.type in {"MESH", "CURVE", "SURFACE", "META", "FONT", "VOLUME"}:
            obj.hide_render = obj not in keep
    scene.cycles.samples = 1
    configure_passes(scene, out, "isolated")
    scene.render.filepath = str(out / "isolated_focal.png")
    bpy.ops.render.render(write_still=True)
    isolated_indices = np.rint(read_image(pass_file(out, "isolated", "instances"))[:, :, 0]).astype(np.int32)
    isolated_mask = isolated_indices == focus["pass_index"]
    if isolated_mask.sum() == 0:
        raise RuntimeError("Unable to measure isolated focal silhouette")
    targets = []
    focal_index = None
    for pass_id, group in sorted(group_ids.items()):
        ys, xs = np.where(indices == pass_id)
        if len(xs) < 4:
            continue
        box = [float(xs.min() / width), float(ys.min() / height), float((xs.max() + 1) / width), float((ys.max() + 1) / height)]
        target = {"box": box, "class_id": group["class_id"], "object_id": group["id"], "visible_pixels": int(len(xs)), "annotation_type": "visible_instance_mask_bbox"}
        if group["id"] == focus_id:
            focal_index = len(targets)
        targets.append(target)
    if focal_index is None:
        raise RuntimeError("Focal annotation missing")
    target_box = targets[focal_index]["box"]
    # Apply sensor noise before measuring degradation.
    valid_depth = np.isfinite(depth) & (depth > 0) & (depth < 1e10)
    depth = np.where(valid_depth, depth, 0).astype(np.float32)
    dropout = rng.random(depth.shape) < 0.45 * control_attrs[7]
    depth[dropout] = 0
    rgb = np.clip(rgb + rng.normal(0, 0.025 * control_attrs[7], rgb.shape), 0, 1)
    rgb8 = np.rint(rgb * 255).astype(np.uint8)
    # NIR from registered luminance and noise.
    gray = (0.299 * rgb8[:, :, 0] + 0.587 * rgb8[:, :, 1] + 0.114 * rgb8[:, :, 2]) / 255
    nir = np.clip(gray + rng.normal(0, 0.08 * control_attrs[7], gray.shape), 0, 1).astype(np.float32)
    write_png(out / "rgb.png", rgb8)
    write_png(out / "nir.png", np.rint(nir * 255))
    np.save(out / "depth.npy", depth, allow_pickle=False)
    np.save(out / "nir.npy", nir, allow_pickle=False)
    np.save(out / "instances.npy", indices, allow_pickle=False)
    np.save(out / "focal_visible_mask.npy", main_focus_mask, allow_pickle=False)
    np.save(out / "focal_isolated_mask.npy", isolated_mask, allow_pickle=False)
    x1, y1, x2, y2 = target_box
    xa, ya, xb, yb = int(x1 * width), int(y1 * height), int(round(x2 * width)), int(round(y2 * height))
    crop = gray[ya:yb, xa:xb]
    size = float(np.clip(1 - math.sqrt((x2 - x1) * (y2 - y1)) / 0.35, 0, 1))
    visibility = float(np.clip(main_focus_mask.sum() / isolated_mask.sum(), 0, 1))
    occlusion = 1 - visibility
    mu_cal = descriptor.get("calibration", {}).get("luminance_median")
    luminance = float(crop.mean()) if crop.size else None
    illumination = None
    if mu_cal is not None and math.isfinite(float(mu_cal)) and float(mu_cal) > 0 and luminance is not None:
        illumination = 1.0 if luminance <= 0 else float(np.clip(abs(math.log(luminance / float(mu_cal))) / math.log(2.5), 0, 1))
    sensor, sensor_measurements = sensor_cues(rgb8[ya:yb, xa:xb], depth[ya:yb, xa:xb], nir[ya:yb, xa:xb])
    if adapter_name == "tomato_instance_names":
        viewpoint, viewpoint_measurements = viewpoint_cue(depth[ya:yb, xa:xb], target_box, focus["class_id"], width, height)
    else:
        viewpoint, viewpoint_measurements = None, {"reason": "missing_generic_class_aspect_prior"}
    measured = [size, None, occlusion, illumination, viewpoint, None, None, sensor]
    measurement_types = ["SI_inverse_visible_bbox_scale", "missing_frozen_expert_class_probabilities", "one_minus_rendered_visible_to_isolated_silhouette_ratio", "SI_crop_luminance" if illumination is not None else "missing_validation_luminance_median", "SI_plane_and_class_aspect_residual" if viewpoint is not None else "missing_generic_class_aspect_prior", "missing_validation_depth_and_temporal_reference", "missing_frozen_detector_proposal_count", "SI_arrays_with_luminance_noise_NIR"]
    labels = "\n".join(f"{t['class_id']} {(t['box'][0] + t['box'][2]) / 2:.8f} {(t['box'][1] + t['box'][3]) / 2:.8f} {t['box'][2] - t['box'][0]:.8f} {t['box'][3] - t['box'][1]:.8f}" for t in targets) + "\n"
    (out / "labels.txt").write_text(labels, encoding="utf-8")
    result = {
        "status": "ok", "id": request["case_id"], "seed": request["seed"],
        "specialist_id": request.get("specialist_id"), "scene_path": request["scene_path"],
        "rgb": str(out / "rgb.png"), "depth": str(out / "depth.npy"), "nir": str(out / "nir.npy"),
        "instance_mask": str(out / "instances.npy"), "labels": str(out / "labels.txt"),
        "targets": targets, "focal_target_index": focal_index, "target_box": target_box,
        "attrs_measured": measured, "measured_attributes": measured,
        "geometry": None,
        "measurements": {"attribute_names": ATTRIBUTE_NAMES, "types": measurement_types, "focal_visible_pixels": int(main_focus_mask.sum()), "focal_isolated_pixels": int(isolated_mask.sum()), "focal_visibility": visibility, "crop_luminance": luminance, "valid_depth_fraction": float((depth[ya:yb, xa:xb] > 0).mean()), "image_size": [width, height], "sensor": sensor_measurements, "viewpoint": viewpoint_measurements},
        "metadata": {"implementation": "tefed", "blender_version": bpy.app.version_string, "renderer": "Cycles CPU", "rgb_depth_nir_registration": "same_camera_same_frame", "depth_units": "asset_scene_units_not_calibrated_metres", "nir_model": "rendered_rgb_luminance_plus_seeded_noise_no_spectral_materials", "occlusion_measurement": "one_minus_visible_to_isolated_silhouette_ratio", "annotation_type": "visible_instance_mask_bbox", "minimum_annotation_pixels": 4, "controls": controls, "unknown_attributes_are_missing": True},
    }
    result["metadata"].update(renderer_adapter=adapter_name, scene_data_origin=str(scene.get("tefed_data_origin", "asset")))
    if request.get("scene_class_map") is not None:
        result["metadata"].update(scene_class_map=request["scene_class_map"],
            class_identity="user_provided_asset_map_not_tomato_color_rule",
            nir_and_thermal_ir_model="rendered_luminance_plus_seeded_noise_no_spectral_materials")
    (out / "targets.json").write_text(json.dumps(targets, indent=2), encoding="utf-8")
    response_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(f"TEFED_RENDER_RESPONSE={response_path}", flush=True)


if __name__ == "__main__":
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    options = parser.parse_args(arguments)
    response = Path(options.response).resolve()
    try:
        main(Path(options.request).resolve(), response)
    except Exception as exc:
        response.write_text(json.dumps({"status": "error", "error": str(exc), "traceback": traceback.format_exc()}, indent=2), encoding="utf-8")
        traceback.print_exc()
        raise
