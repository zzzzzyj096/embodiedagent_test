#!/usr/bin/env bash
# 可选：修复 habitat 环境中 numpy 2.x 与 habitat-sim 不兼容、补 Web 依赖
# 不会修改 conda 环境配置文件，仅安装/降级包
set -euo pipefail

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate habitat
else
  echo "请先: conda activate habitat"
  exit 1
fi

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY 2>/dev/null || true

echo "Python: $(which python)"
echo "修复 numpy<2（habitat-sim 需要）…"
if ! conda install -y "numpy<2" -c conda-forge 2>/dev/null; then
  pip install "numpy<2" -i https://pypi.org/simple --default-timeout=120
fi

echo "安装 Web / Agent 常用 pip 包…"
pip install fastapi uvicorn pydantic opencv-python-headless Pillow dashscope \
  -i https://pypi.org/simple --default-timeout=120

python -c "import habitat_sim; print('habitat_sim OK')"
python -c "import cv2; print('cv2 OK')"
python check_web_deps.py

echo ""
echo "完成。请用本环境启动 Web:"
echo "  conda activate habitat"
echo "  cd /autodl-tmp && python start_web.py"
