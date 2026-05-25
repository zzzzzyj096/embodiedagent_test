#!/usr/bin/env python3
"""检查 Web + Agent 所需 import；缺包时打印安装命令。"""
from __future__ import annotations

import subprocess
import sys

CHECKS = [
    ("fastapi", "fastapi", "pip install fastapi uvicorn pydantic"),
    ("uvicorn", "uvicorn", "pip install uvicorn"),
    ("cv2", "opencv-python-headless", "pip install opencv-python-headless"),
    ("PIL", "Pillow", "pip install Pillow"),
    ("numpy", "numpy", None),
    ("torch", "torch", None),
    ("habitat_sim", "habitat-sim", "请使用已安装 habitat-sim 的 conda 环境"),
    ("dashscope", "dashscope", "pip install dashscope"),
]

PIP_INSTALL = [
    "fastapi",
    "uvicorn",
    "pydantic",
    "opencv-python-headless",
    "Pillow",
    "dashscope",
]


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--light",
        action="store_true",
        help="仅检查 Web 轻量依赖，不 import torch/habitat_sim（避免 OOM）",
    )
    args = ap.parse_args()
    checks = CHECKS if not args.light else CHECKS[:5]  # fastapi..numpy

    print(f"当前解释器: {sys.executable}\n")
    missing = []
    for mod, pkg, hint in checks:
        try:
            __import__(mod)
            print(f"  OK  {mod}")
        except ImportError:
            print(f"  MISSING  {mod}  →  {hint or pkg}")
            if hint and hint.startswith("pip"):
                missing.append(pkg)

    if missing:
        print()
        if "habitat_sim" in [m.split()[0] for m in missing if "habitat" in m]:
            print("habitat_sim 未找到 → 请先: conda activate habitat")
            print("  然后: bash fix_habitat_env.sh")
            print("  再:   python start_web.py")
            print()
        print("可一次性安装 Web 侧常见缺包（在 habitat 环境里）：")
        print(f"  pip install {' '.join(dict.fromkeys(missing))}")
        print("或：")
        print("  pip install -r requirements-web.txt")
        return 1

    print("\n依赖检查通过，可运行: python start_web.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
