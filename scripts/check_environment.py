"""Inspect local packages and optionally exercise device kernels."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys


def read_text(path):
    try:
        return Path(path).read_text(encoding="utf-8").strip().rstrip("\x00")
    except (OSError, UnicodeError):
        return None


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def jetpack_version():
    if platform.system() != "Linux" or not shutil.which("dpkg-query"):
        return None
    try:
        result = subprocess.run(
            ["dpkg-query", "-W", "-f=${Version}", "nvidia-jetpack"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def smoke(torch, torchvision, device):
    import numpy as np

    values = np.arange(4, dtype=np.float32).reshape(2, 2)
    bridge = torch.from_numpy(values).numpy()
    if not np.array_equal(bridge, values):
        raise RuntimeError("NumPy tensor conversion failed")
    model = torch.nn.Conv2d(3, 4, 3, padding=1).to(device)
    inputs = torch.randn(2, 3, 32, 32, device=device)
    loss = model(inputs).square().mean()
    loss.backward()
    if not torch.isfinite(loss).item() or not torch.isfinite(model.weight.grad).all().item():
        raise RuntimeError("Nonfinite convolution loss or gradient")
    boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 9, 9], [20, 20, 30, 30]],
                         dtype=torch.float32, device=device)
    scores = torch.tensor([0.9, 0.8, 0.7], device=device)
    kept = torchvision.ops.nms(boxes, scores, 0.5).cpu().tolist()
    if kept != [0, 2]:
        raise RuntimeError(f"Unexpected torchvision NMS result: {kept}")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {"device": str(device), "numpy_bridge": "passed",
            "convolution_forward_backward": "passed", "torchvision_nms": "passed"}


def check_profile(report, torch, device, profile):
    if profile == "auto":
        return
    if profile == "cpu":
        if device.type != "cpu":
            raise RuntimeError("The CPU profile requires --device cpu")
        return
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"The {profile} profile requires a CUDA device")
    version = tuple(int(part) for part in (torch.version.cuda or "0.0").split(".")[:2])
    if profile == "rtx5080":
        name = torch.cuda.get_device_name(device).lower().replace(" ", "")
        if "rtx5080" not in name or version < (12, 8):
            raise RuntimeError("The RTX 5080 profile requires that GPU and a CUDA 12.8+ torch build")
    else:
        host = report["platform"]
        if host["system"] != "Linux" or host["machine"].lower() not in {"aarch64", "arm64"}:
            raise RuntimeError("The Jetson profile requires Linux aarch64")
        l4t = report["jetson"]["l4t"] or ""
        if not re.search(r"R39\s*\(release\),\s*REVISION:\s*2\.", l4t):
            raise RuntimeError("The Jetson profile requires L4T 39.2.x (JetPack 7.2.x)")
        if version < (13, 2) or torch.cuda.get_device_capability(device) != (8, 7):
            raise RuntimeError("The Jetson profile requires an Orin GPU and a CUDA 13.2+ torch build")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda:0")
    parser.add_argument("--profile", choices=("auto", "cpu", "rtx5080", "jetson"), default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run small local kernel checks")
    parser.add_argument("--output", type=Path, help="Save the JSON report")
    args = parser.parse_args(argv)
    report = {
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "platform": {"system": platform.system(), "release": platform.release(),
                     "machine": platform.machine()},
        "profile": args.profile,
        "jetson": {"model": read_text("/proc/device-tree/model"),
                   "l4t": read_text("/etc/nv_tegra_release"),
                   "jetpack_package": jetpack_version()},
        "packages": {name: package_version(name) for name in
                     ("torch", "torchvision", "numpy", "Pillow", "ultralytics", "scipy")},
        "errors": [],
    }
    if sys.version_info[:2] != (3, 12):
        report["errors"].append("This release requires Python 3.12")
    for name, version in report["packages"].items():
        if version is None:
            report["errors"].append(f"Missing package: {name}")
    if report["packages"]["ultralytics"] not in (None, "8.4.0"):
        report["errors"].append("This adapter requires ultralytics==8.4.0")
    try:
        import torch
        import torchvision

        available = torch.cuda.is_available()
        report["cuda"] = {"available": available, "torch_build": torch.version.cuda,
                          "cudnn": torch.backends.cudnn.version(),
                          "compiled_architectures": torch.cuda.get_arch_list(), "devices": []}
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            report["cuda"]["devices"].append({
                "index": index, "name": props.name,
                "capability": list(torch.cuda.get_device_capability(index)),
                "memory_bytes": props.total_memory,
            })
        device = torch.device("cuda:0" if available else "cpu") if args.device == "auto" else torch.device(args.device)
        report["selected_device"] = str(device)
        if args.require_cuda and (not available or device.type != "cuda"):
            raise RuntimeError("CUDA is required but no CUDA device is selected and available")
        if device.type == "cuda" and not available:
            raise RuntimeError("The requested CUDA device is unavailable")
        check_profile(report, torch, device, args.profile)
        if args.smoke:
            report["smoke"] = smoke(torch, torchvision, device)
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    report["status"] = "failed" if report["errors"] else "passed"
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
