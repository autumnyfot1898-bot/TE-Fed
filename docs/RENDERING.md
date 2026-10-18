# Asset-backed rendering

TE-Fed renders correction cases from a local `.blend` scene without saving
changes to the asset. Configure the scene path locally; scene files are supplied
separately.

## Supported scene convention

The tested local scene is `tomato_greenhouse_project_assets_style.blend`. Fruit
instances have names `project_asset_tomato_{red,green,yellow}_<plant>_<branch>_<fruit>_*`.
Body materials contain `candy`, `color`, or `yellow_tomato`; calyx/stem meshes are
excluded from fruit bounding boxes. Red is class 0 (ripe), green/yellow class 1
(nonripe). Other scene conventions require an explicit annotation adapter.

The complete frame receives labels for every fruit with at least four visible
body pixels. Boxes enclose the **visible instance mask**, not the occluded full
3-D silhouette. A second, isolated focal-object mask measures visibility; this
auxiliary image is not an additional training case.

## API

```python
from pathlib import Path
from tefed.rendering import BlenderRenderer

renderer = BlenderRenderer("/path/to/blender", "/path/to/scene.blend", samples=8)
case = renderer.render({
    "descriptor": {
        "attrs": [0.6, 0.3, 0.4, 0.5, 0.2, 0.2, 0.3, 0.1],
        "source_class": 0,
        "calibration": {"luminance_median": 0.45},
    },
    "seed": 123,
    "case_id": "r00-d001-s1-v0",
    "specialist_id": 1,
    "image_size": [640, 640],
}, Path("outputs/new-empty-case-directory"))
```

The subprocess uses an argument list, `shell=False`, an execution timeout, and an
empty output directory. Failed rendering, missing focal targets, invalid inputs,
and absent output files raise errors. No keep-all fallback exists.

## Eight controls versus actual measurements

Attribute order is `size, cls, occ, illum, view, geo, ctx, sens`. Descriptor values
drive bounded transforms, class-preserving body color interpolation, foreground
occlusion, illumination, camera yaw, geometric displacement, background tint, and
array-level sensor corruption. Actual rendered measurements are stored separately.
Control values are **never copied into `attrs_measured`**.

The sensor-quality measurement uses the same adapter as `tefed.cues`: a four-neighbor 3×3 cross Laplacian with center coefficient -4,
reflection padding, and BT.601 grayscale. The 3×3 NIR median filter also uses
reflection padding. Verification recomputes all available attributes with the common
cue adapter and preserves the renderer's original measurements for audit.

| Attribute | Renderer measurement | Status |
| --- | --- | --- |
| size | Inverse linear scale of the visible bounding box, using the SI scale 0.35 | Available |
| cls | Requires the frozen expert class distribution | `None`; verification must recompute |
| occ | Visible focal mask pixels divided by isolated focal silhouette pixels | Explicit visibility proxy, not the SI detector-box-overlap cue |
| illum | Crop luminance deviation from a supplied validation median | Available only with valid `calibration.luminance_median` |
| view | Trimmed depth-plane slope and class-aspect residual with the SI fallback | Available, including the recorded fallback flag |
| geo | Requires validation depth calibration and temporal reference | `None` |
| ctx | Requires frozen-detector proposal density and local entropy | `None` |
| sens | Depth dropout, crop Laplacian/clipping and robust NIR residual on stored arrays | Available, with NIR explicitly a proxy |

Availability and fidelity activation masks are separate descriptor fields. An
active fidelity attribute without a valid measurement must make verification
reject the case or explicitly mask it according to the reported protocol. It
must not be filled with the requested value. Round-start inference can supply
missing class, context and geometric observations, with provenance recorded.

## Synchronized modalities and files

- `rgb.png`: 8-bit RGB from the asset scene, followed by seeded noise.
- `depth.npy`: float32 camera Z-pass values in **scene units**, with zero for
  invalid/background/dropout. Scene units have not been calibrated to metres.
- `nir.npy` and `nir_proxy.png`: co-registered RGB-luminance/noise **synthetic
  proxy**, with no wavelength-dependent material reflectance or physical NIR model.
- `instances.npy`: int32 instance IDs; zero denotes unannotated/background surfaces.
- `focal_visible_mask.npy`, `focal_isolated_mask.npy`: measured focal visibility.
- `targets.json`, `labels.txt`: complete-frame instance boxes and YOLO labels.
- `request.json`, `response.json`, `blender.log`: controls, measurements, software,
  source path, seed, result and diagnostics.

The proxy NIR output is suitable for interface/smoke testing and explicit proxy
experiments. It does not establish physical multisensor generalization. Actual
spectral materials and calibrated sensor simulation require a separate adapter.

## Inspect without changing the asset

```text
blender --background --factory-startup --python scripts/inspect_blend.py -- --scene /path/to/scene.blend --output /path/to/inventory.json
```

The tested scene loaded under Blender 5.2.2 LTS with 4,009 objects and 4,003 meshes.
Source scene hashes can be recorded externally before and after rendering. No
local asset is granted a new distribution license by this code package.

## Generic scene class-map adapter

Non-tomato scenes can use `scripts/blender_generic_render.py` through the same
`BlenderRenderer.render` API. Supply `scene_class_map` in the request; it maps
**each object-instance root** to a nonnegative class ID matching the dataset
manifest. No vehicle is silently mapped to a tomato class.

```python
request["scene_class_map"] = {
    "object:instance_000": 0,
    "object:instance_001": 2,
    "collection:instance_002_components": 4,
}
request["descriptor"]["source_class"] = 2
```

An object root includes its descendant meshes. A collection root includes its
member meshes and child collections. Each root must describe **one object
instance**, so a collection containing several distinct vehicles needs separate
per-vehicle roots. Prefixes `object:` and `collection:` disambiguate identical
names; an unprefixed root is accepted when unique. Missing roots, overlapping
mesh ownership, hidden/empty roots, unmapped source classes and collection
instancers are rejected. Materialize collection instances before mapping them.

Instance masks and full-frame visible-object boxes are measured from the actual
mapped meshes. The focal object is selected only from the requested source class.
Class identity comes from the user-supplied asset mapping. Generic appearance
variation changes unlinked material colors by a bounded amount; it preserves
geometry and the mapped label and does not use the tomato red/green rule.
Conditional class ambiguity is still remeasured by the frozen expert bank.

Generic `view` is `None` until a task-specific class-aspect/depth adapter supplies
it. The remaining missing-measurement rules, actual size/visibility measurements,
bounded controls and synchronized output formats are shared with the tomato
renderer. NIR remains a synthetic luminance/noise proxy; **thermal IR is also not
physically modeled**. Genuine multisensor vehicle evaluation requires suitable
user-provided scene assets and calibrated sensor materials.

A five-class primitive-geometry scene was used for a Blender software test.
It rendered a class-2 focal instance and all five mapped classes. The response
records `renderer_adapter="explicit_scene_class_map"` and
`scene["tefed_data_origin"]` as `metadata.scene_data_origin`; the default is `"asset"`.
