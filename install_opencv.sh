#!/usr/bin/env bash
# 在 habitat 环境中安装 OpenCV（解决 pip 代理/镜像连不上）
set -euo pipefail

echo "=== 当前代理环境变量 ==="
env | grep -i proxy || echo "(无 proxy 变量)"

echo ""
echo "=== 取消代理，改用 PyPI 官方源 ==="
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

if command -v conda &>/dev/null; then
  echo "尝试 conda 安装 opencv（推荐，不依赖 pip 镜像）..."
  if conda install -y -c conda-forge opencv 2>/dev/null; then
    python -c "import cv2; print('cv2 OK', cv2.__version__)"
    exit 0
  fi
  echo "conda 未成功，改用 pip..."
fi

pip install --default-timeout=120 opencv-python-headless -i https://pypi.org/simple
python -c "import cv2; print('cv2 OK', cv2.__version__)"
