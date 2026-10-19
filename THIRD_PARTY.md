# Third-party components

- **Ultralytics 8.4.0**, AGPL-3.0. This project imports its model definitions and native losses; it does not bundle the upstream source or weights. [Pinned model source](https://github.com/ultralytics/ultralytics/tree/v8.4.0/ultralytics), [license](https://github.com/ultralytics/ultralytics/blob/v8.4.0/LICENSE), [licensing guidance](https://www.ultralytics.com/license).
- **PyTorch / torchvision**, BSD-style licenses. [Source](https://github.com/pytorch/pytorch), [vision](https://github.com/pytorch/vision). Install matching builds for the target device.
- **NumPy**, BSD-3-Clause; **Pillow**, HPND; **SciPy**, BSD-3-Clause. These are dependencies, not bundled copies.
- **Blender**, GPL. The renderer invokes a separately installed executable. Blender, local `.blend` scenes and source meshes are not bundled.

Local scene/model/data assets have their own provenance and permissions. The code license does not grant rights to redistribute them.
