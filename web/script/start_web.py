#!/usr/bin/env python3
"""
一键启动 Web 界面。

在已 conda activate habitat 时，直接用当前 python 即可（与命令行跑 agent 相同）。

用法:
  conda activate habitat
  cd /autodl-tmp
  python start_web.py
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys

DEFAULT_PORT = int(os.environ.get("WEB_PORT", "6006"))
HOST = "0.0.0.0"
_HABITAT_IMPORT_TIMEOUT = int(os.environ.get("HABITAT_IMPORT_TIMEOUT", "180"))

_HABITAT_CANDIDATES = [
    os.environ.get("HABITAT_PYTHON"),
    os.path.expanduser("~/miniconda3/envs/habitat/bin/python"),
    "/root/miniconda3/envs/habitat/bin/python",
    os.path.join(
        os.path.dirname(sys.executable), "..", "envs", "habitat", "bin", "python"
    ),
]


def _habitat_import_check(python_exe: str) -> tuple[bool, str]:
    """
    检测能否 import habitat_sim。
    若就是当前进程的解释器，直接在本进程 import（避免子进程冷启动误判/超时）。
    """
    python_exe = os.path.abspath(python_exe)
    if python_exe == os.path.abspath(sys.executable):
        try:
            import habitat_sim  # noqa: F401

            return True, ""
        except Exception as e:
            return False, str(e)

    try:
        r = subprocess.run(
            [python_exe, "-c", "import habitat_sim"],
            capture_output=True,
            text=True,
            timeout=_HABITAT_IMPORT_TIMEOUT,
        )
        if r.returncode == 0:
            return True, ""
        err = (r.stderr or r.stdout or "").strip()
        return False, err[:1200] if err else "import 失败（无详细输出）"
    except subprocess.TimeoutExpired:
        return (
            False,
            f"子进程 import 超过 {_HABITAT_IMPORT_TIMEOUT}s（首次加载 habitat 较慢，可加大 "
            "HABITAT_IMPORT_TIMEOUT 或先运行: python -c \"import habitat_sim\"）",
        )
    except OSError as e:
        return False, str(e)


def resolve_python() -> tuple[str, bool, str]:
    """返回 (python路径, habitat_sim是否可用, 错误信息)。"""
    if os.environ.get("HABITAT_PYTHON"):
        py = os.environ["HABITAT_PYTHON"]
        ok, err = _habitat_import_check(py)
        return py, ok, err

    ok, err = _habitat_import_check(sys.executable)
    if ok:
        return sys.executable, True, ""

    for cand in _HABITAT_CANDIDATES:
        if not cand:
            continue
        cand = os.path.abspath(cand)
        if os.path.isfile(cand) and cand != os.path.abspath(sys.executable):
            ok, err = _habitat_import_check(cand)
            if ok:
                return cand, True, ""

    return sys.executable, False, err


def pick_port(start: int) -> int:
    for port in range(start, start + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((HOST, port))
                return port
            except OSError:
                continue
    return start


def main() -> None:
    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)
    py, ok, err = resolve_python()
    port = pick_port(DEFAULT_PORT)
    in_habitat_env = "envs/habitat" in py.replace("\\", "/")

    print()
    print("=" * 60)
    print("  具身导航 Agent — Web 界面")
    print("=" * 60)
    print(f"  Python: {py}")
    if py != sys.executable:
        print("  （已自动切换到 habitat 环境的 Python）")

    if not ok:
        print()
        if in_habitat_env:
            print("  当前已是 habitat 环境的 Python，但 habitat_sim 导入失败。")
            print("  （不是「没激活环境」，而是该环境内 habitat-sim / numpy 等不兼容）")
        else:
            print("  当前 Python 无法使用 habitat_sim。")
            print("  请先: conda activate habitat")
        print()
        if err:
            print("  详细错误:")
            for line in err.splitlines()[:12]:
                print(f"    {line}")
        print()
        print("  可尝试（在 habitat 环境中）:")
        print('    python -c "import habitat_sim"   # 看完整报错')
        print('    conda install -y "numpy<2" -c conda-forge')
        print("  未修改你的 conda 环境配置文件；若曾 pip 升级 numpy，按上面恢复即可。")
        print("=" * 60)
        sys.exit(1)

    print()
    print("  habitat_sim 检测通过")
    print("  浏览器打开:")
    print(f"    http://127.0.0.1:{port}/")
    print("  AutoDL: 自定义服务开放同端口后点「访问链接」")
    print("  按 Ctrl+C 停止")
    print("=" * 60)
    print()
    print("  提示: 启动后即后台加载（约 1–3 分钟）；终端见 [web] 加载阶段: …")
    print("  浏览器右上角显示「就绪」后即可导航。内存不足若出现 Killed，设 WEB_EAGER_INIT=0。\n")

    env = os.environ.copy()
    env.setdefault("AGENT_PIPELINE", "debug")
    cmd = [py, "-m", "uvicorn", "web.app:app", "--host", HOST, "--port", str(port)]
    try:
        subprocess.run(cmd, env=env)
    except KeyboardInterrupt:
        print("\n已停止 Web 服务。")


if __name__ == "__main__":
    main()
