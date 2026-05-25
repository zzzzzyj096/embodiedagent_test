"""
Embodied navigation web server — chat UI over Habitat + GroundingDINO agents.

Run from repo root:
  cd /autodl-tmp
  pip install fastapi uvicorn
  export DASHSCOPE_API_KEY=...   # optional if using keyword commands
  python -m uvicorn web.app:app --host 0.0.0.0 --port 8765

Env:
  AGENT_PIPELINE=top|v2|debug   (default: debug → embodied_agent_top_debug)
  WEB_MEDIA_ROOT=/autodl-tmp      (GIF / log files)
"""

from __future__ import annotations

import glob
import os
import re
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar

T = TypeVar("T")

# Habitat/OpenGL 必须在同一线程创建与使用（否则 Stage2 报 GL::Context::current）
_SIM_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="habitat_sim")


def run_on_sim_thread(fn: Callable[[], T]) -> T:
    """在唯一仿真线程中执行（阻塞直到完成）。"""
    return _SIM_EXECUTOR.submit(fn).result()

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Repo root on sys.path for embodied_agent_* imports
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

WEB_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(WEB_DIR, "static")
MEDIA_ROOT = os.environ.get("WEB_MEDIA_ROOT", _REPO_ROOT)
DEFAULT_PIPELINE = os.environ.get("AGENT_PIPELINE", "debug").strip().lower()
# 预探索语义拓扑（Web 默认加载，用户无需再「探索建图」）
WEB_TOPO_MAP_PATH = os.environ.get(
    "WEB_TOPO_MAP_PATH", os.path.join(_REPO_ROOT, "semantic_topo_map.json")
)
# 面向访客：仅导航，固定 top + 已有 topo；不展示建图/预加载
WEB_USER_MODE = os.environ.get("WEB_USER_MODE", "1").strip() not in ("0", "false", "False")
WEB_ALLOW_EXPLORE = os.environ.get("WEB_ALLOW_EXPLORE", "0").strip() in ("1", "true", "True")
# 默认服务启动即后台加载 sim + 检测器 + 预置 topo（内存充足时打开网页即可用）
WEB_EAGER_INIT = os.environ.get("WEB_EAGER_INIT", "1").strip() in ("1", "true", "True")

# Agent 侧常见缺包（与 habitat 环境分开装 Web 时容易漏掉）
_AGENT_IMPORT_CHECKS = [
    ("cv2", "pip install opencv-python-headless"),
    ("PIL", "pip install Pillow"),
    ("habitat_sim", "使用已安装 habitat-sim 的 conda 环境"),
    ("torch", "pip/conda install torch"),
    ("dashscope", "pip install dashscope"),
]


def check_agent_imports() -> Optional[str]:
    """返回 None 表示通过，否则为可读错误说明。"""
    missing = []
    for mod, hint in _AGENT_IMPORT_CHECKS:
        try:
            __import__(mod)
        except ImportError:
            missing.append(f"  · {mod}  →  {hint}")
    if not missing:
        return None
    py = sys.executable
    return (
        f"当前 Web 使用的 Python 缺少 Agent 依赖：\n{py}\n\n"
        + "\n".join(missing)
        + "\n\n若已在 habitat 环境中，多为 habitat_sim / numpy 不兼容，而非「环境没激活」。\n"
        "请执行: python -c \"import habitat_sim\" 查看完整报错；"
        '必要时 conda install -y "numpy<2" -c conda-forge\n'
        "然后: conda activate habitat && python start_web.py"
    )

TARGET_KEYWORDS: Dict[str, List[str]] = {
    "沙发": ["沙发", "sofa"],
    "床": ["床", "bed"],
    "椅子": ["椅子", "chair"],
    "门": ["门", "door"],
    "桌子": ["桌子", "table", "desk"],
}


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


@dataclass
class Job:
    id: str
    message: str
    pipeline: str
    status: JobStatus = JobStatus.QUEUED
    logs: List[str] = field(default_factory=list)
    success: Optional[bool] = None
    reply: str = ""
    gif_path: Optional[str] = None
    log_path: Optional[str] = None
    position: Optional[List[float]] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None

    def append_log(self, line: str) -> None:
        for part in line.splitlines():
            s = part.rstrip()
            if s:
                self.logs.append(s)
        if len(self.logs) > 800:
            self.logs = self.logs[-600:]

    def to_dict(self) -> Dict[str, Any]:
        gif_url = None
        if self.gif_path and os.path.isfile(self.gif_path):
            gif_url = f"/api/media?path={_rel_media_path(self.gif_path)}"
        log_url = None
        if self.log_path and os.path.isfile(self.log_path):
            log_url = f"/api/media?path={_rel_media_path(self.log_path)}"
        return {
            "id": self.id,
            "message": self.message,
            "pipeline": self.pipeline,
            "status": self.status.value,
            "logs": self.logs[-120:],
            "success": self.success,
            "reply": self.reply,
            "gif_url": gif_url,
            "log_url": log_url,
            "position": self.position,
            "error": self.error,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
        }


def _rel_media_path(abs_path: str) -> str:
    abs_path = os.path.abspath(abs_path)
    root = os.path.abspath(MEDIA_ROOT)
    if abs_path.startswith(root + os.sep) or abs_path == root:
        return os.path.relpath(abs_path, root)
    return os.path.basename(abs_path)


def _resolve_media_path(rel: str) -> str:
    rel = rel.replace("\\", "/").lstrip("/")
    if ".." in rel.split("/"):
        raise HTTPException(status_code=400, detail="invalid path")
    full = os.path.abspath(os.path.join(MEDIA_ROOT, rel))
    if not full.startswith(os.path.abspath(MEDIA_ROOT) + os.sep):
        raise HTTPException(status_code=400, detail="invalid path")
    if not os.path.isfile(full):
        raise HTTPException(status_code=404, detail="file not found")
    return full


def _keyword_target(text: str) -> Optional[str]:
    low = text.lower()
    for cn, keys in TARGET_KEYWORDS.items():
        for k in keys:
            if k in text or k in low:
                return cn
    return None


def _is_remap_command(text: str) -> bool:
    low = text.strip().lower()
    return any(x in text for x in ("探索", "建图", "重新建图")) or low in (
        "remap",
        "rebuild map",
        "explore",
        "map",
    )


def _unpack_navigate_result(result: Any) -> tuple:
    """
    top_debug: (success, frames, steps)
    top_final: (success, steps)
    """
    n = len(result)
    if n == 3:
        return bool(result[0]), result[1], int(result[2])
    if n == 2:
        return bool(result[0]), None, int(result[1])
    raise ValueError(f"navigate_to_target 返回值数量异常: {n}")


def _install_keyword_parser(mod) -> None:
    """无 DASHSCOPE_API_KEY 时用关键词解析，便于本地 Web 演示。"""

    def parse_command(user_input: str) -> str:
        k = _keyword_target(user_input)
        if k:
            print(f"关键词解析: '{user_input}' → '{k}'")
            return k
        if _is_remap_command(user_input):
            return "未知"
        if os.environ.get("DASHSCOPE_API_KEY"):
            return _orig_parse(user_input)
        print(f"无法解析（无 API Key）: '{user_input}' → '未知'")
        return "未知"

    _orig_parse = mod.parse_command
    mod.parse_command = parse_command


class _LogTee:
    def __init__(self, job: Job, original):
        self.job = job
        self.original = original

    def write(self, data: str) -> None:
        if data:
            self.original.write(data)
            self.job.append_log(data)

    def flush(self) -> None:
        self.original.flush()


_INIT_PHASE_LABEL = {
    "idle": "等待加载",
    "starting": "启动加载线程",
    "topo": "读取语义地图…",
    "sim": "加载 Habitat 场景…",
    "detector": "加载 GroundingDINO…",
    "ready": "就绪",
}


class AgentSession:
    """Single-threaded Habitat session (one navigation job at a time)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._job_lock = threading.Lock()
        self.pipeline: Optional[str] = None
        self.sim = None
        self.detector = None
        self.topo_map = None
        self.topo_path: Optional[str] = None
        self.topo_view_count: int = 0
        self.topo_region_count: int = 0
        self.ready = False
        self.init_phase: str = "idle"  # idle | topo | sim | detector | ready
        self.init_error: Optional[str] = None
        self._init_future: Optional[Future] = None

    def _set_phase(self, phase: str) -> None:
        with self._lock:
            self.init_phase = phase
        print(f"[web] 加载阶段: {phase}")

    def _load_module(self, pipeline: str):
        if pipeline == "v2":
            import embodied_agent_2 as mod

            return mod, "v2"
        if pipeline == "debug":
            import embodied_agent_top_debug as mod

            return mod, "debug"
        import embodied_agent_top_final as mod

        return mod, "top"

    def initialize(self, pipeline: str = DEFAULT_PIPELINE) -> None:
        pipeline = pipeline or DEFAULT_PIPELINE
        with self._lock:
            if self.ready and self.pipeline == pipeline:
                return
            if self.sim is not None:
                try:
                    self.sim.close()
                except Exception:
                    pass
                self.sim = None
            self.detector = None
            self.topo_map = None
            self.ready = False
            self.init_error = None
            self.pipeline = pipeline

        try:
            # 与 CLI run_agent 首次调用一致，不再先 check_agent_imports（会重复 import torch/habitat，更慢且易 OOM）
            mod, name = self._load_module(pipeline)
            _install_keyword_parser(mod)
            with self._lock:
                self.pipeline = name
            print(f"[web] 正在加载 Agent（{name}），流程与命令行首次导航相同…")

            topo_map = None
            topo_path = None
            view_n = region_n = 0
            if name in ("top", "debug"):
                self._set_phase("topo")
                topo_path = WEB_TOPO_MAP_PATH
                if os.path.isfile(topo_path):
                    topo_map = mod.SemanticTopoMap.load_json(topo_path)
                    view_n = len(topo_map.views)
                    region_n = len(getattr(topo_map, "regions", {}) or {})
                    print(
                        f"[web] 已加载预探索拓扑: {topo_path} "
                        f"(regions≈{region_n}, views={view_n})"
                    )
                elif WEB_USER_MODE and not WEB_ALLOW_EXPLORE:
                    raise FileNotFoundError(
                        f"未找到预探索地图: {topo_path}\n"
                        "请管理员在服务器上先完成探索并生成 semantic_topo_map.json。"
                    )

            self._set_phase("sim")
            sim = mod.make_sim(random_spawn=True)

            self._set_phase("detector")
            detector = mod.TargetDetector()

            if name in ("top", "debug") and topo_map is None:
                topo_map = mod.ensure_topo_map(sim, detector, force_rebuild=False)
                topo_path = WEB_TOPO_MAP_PATH
                view_n = len(topo_map.views)

            with self._lock:
                self.sim = sim
                self.detector = detector
                self.topo_map = topo_map
                self.topo_path = topo_path
                self.topo_view_count = view_n
                self.topo_region_count = region_n
                self.ready = True
                self.init_phase = "ready"
                self.init_error = None
            print(f"[web] Agent 就绪（{name}），与 CLI 会话已加载 sim+detector+topo 相同")
        except Exception as e:
            tb = traceback.format_exc()
            with self._lock:
                self.init_error = f"{e}\n{tb}"
                self.ready = False
            raise

    def initialize_async(self, pipeline: str = DEFAULT_PIPELINE) -> None:
        def _run():
            try:
                self.initialize(pipeline)
            except Exception as e:
                print(f"[web] 后台加载失败: {e}")

        with self._lock:
            if self.ready:
                return
            if self._init_future is not None and not self._init_future.done():
                return
            self.init_phase = "starting"
            self._init_future = _SIM_EXECUTOR.submit(_run)

    def status_dict(self) -> Dict[str, Any]:
        with self._lock:
            fut = self._init_future
            initializing = bool(fut is not None and not fut.done() and not self.ready)
            stale = None
            if not self.ready and not self.init_error and fut is not None and fut.done():
                exc = fut.exception()
                if exc is not None and self.init_phase not in ("idle", "ready"):
                    stale = (
                        f"后台加载失败: {exc}。"
                        "若终端有 Killed 则为内存不足；否则请重启 Web 后重试。"
                    )
                elif not self.ready and self.init_phase not in ("idle", "ready"):
                    stale = (
                        "后台加载被中断（终端若出现 Killed，多为内存不足）。"
                        "请释放内存后直接点击目标重试。"
                    )
            err = self.init_error or stale
            load_msg = _INIT_PHASE_LABEL.get(self.init_phase, "")
            if self.ready:
                load_msg = "系统就绪"
            return {
                "service_up": True,
                "ready": self.ready,
                "agent_ready": self.ready,
                "load_message": load_msg,
                "pipeline": self.pipeline,
                "init_error": err,
                "init_stalled": stale is not None,
                "initializing": initializing,
                "user_mode": WEB_USER_MODE,
                "topo_loaded": self.topo_map is not None,
                "topo_path": self.topo_path,
                "topo_views": self.topo_view_count,
                "topo_regions": self.topo_region_count,
                "init_phase": self.init_phase,
                "lazy_load": not WEB_EAGER_INIT,
            }

    def run_task(self, job: Job) -> None:
        with self._job_lock:
            job.status = JobStatus.RUNNING
            mod, _ = self._load_module(job.pipeline)
            if not self.ready or self.pipeline != job.pipeline:
                self.initialize(job.pipeline)

            if not self.ready:
                raise RuntimeError(self.init_error or "Agent 未初始化")

            stdout_prev = sys.stdout
            sys.stdout = _LogTee(job, stdout_prev)
            try:
                _install_keyword_parser(mod)
                text = job.message.strip()
                kw = _keyword_target(text)

                if _is_remap_command(text) and not WEB_ALLOW_EXPLORE:
                    job.success = False
                    job.reply = (
                        "场景与语义地图已由系统预置，请直接输入要去的目标，"
                        "例如：沙发、椅子、床、门（或「请到沙发旁边」）。"
                    )
                    job.status = JobStatus.DONE
                    return

                target_cn = kw or ""
                nav_steps = 0
                if job.pipeline == "v2":
                    success, frames, self.sim, self.detector = mod.run_agent(
                        text, self.sim, self.detector
                    )
                    self._attach_artifacts(job, success, frames)
                    target_cn = target_cn or _keyword_target(text) or "目标"
                else:
                    parsed = mod.parse_command(text)
                    if parsed != "未知":
                        target_cn = parsed
                    if not target_cn or target_cn not in mod.TARGETS_ZH2EN:
                        job.success = False
                        job.reply = (
                            "未能识别目标。请直接说：沙发、椅子、床、门、桌子，"
                            "或「请到沙发旁边」。"
                        )
                        job.status = JobStatus.DONE
                        return
                    if self.topo_map is None:
                        raise RuntimeError(
                            "语义拓扑未加载，请联系管理员检查 semantic_topo_map.json"
                        )
                    job.append_log(
                        f"[web] 导航目标: {target_cn}（预加载拓扑 {self.topo_path}）"
                    )
                    nav_out = mod.navigate_to_target(
                        self.sim, self.detector, self.topo_map, target_cn
                    )
                    success, nav_frames, nav_steps = _unpack_navigate_result(nav_out)
                    job.append_log(f"[web] 导航结束 success={success} steps={nav_steps}")
                    self._attach_artifacts(job, success, nav_frames)

                pos = self.sim.agents[0].state.position
                job.position = [float(pos[0]), float(pos[1]), float(pos[2])]
                tgt = target_cn or kw or "目标"

                if success:
                    job.success = True
                    steps_note = ""
                    if job.pipeline != "v2":
                        steps_note = f"（共 {nav_steps} 步）"
                    if job.gif_path:
                        job.reply = (
                            f"好的，我已经到达{tgt}旁边了{steps_note}。"
                            "导航过程见下方动图。"
                        )
                    else:
                        job.reply = f"好的，我已经到达{tgt}旁边了{steps_note}。还需要什么？"
                else:
                    job.success = False
                    job.reply = f"抱歉，我没能到达{tgt}旁边，请换种说法或稍后再试。"

                job.status = JobStatus.DONE
            except Exception as e:
                job.status = JobStatus.ERROR
                job.success = False
                job.error = str(e)
                job.reply = f"任务执行出错：{e}"
                job.append_log(traceback.format_exc())
            finally:
                sys.stdout = stdout_prev
                job.finished_at = time.time()
                self._pick_latest_log(job)

    def _attach_artifacts(self, job: Job, success: bool, frames) -> None:
        if frames:
            try:
                tgt = _keyword_target(job.message) or "nav"
                gif_path = os.path.join(
                    MEDIA_ROOT, f"navigation_{tgt}_{job.id[:8]}.gif"
                )
                frames[0].save(
                    gif_path,
                    save_all=True,
                    append_images=frames[1:],
                    duration=400,
                    loop=0,
                )
                job.gif_path = gif_path
            except Exception as e:
                job.append_log(f"[web] GIF 保存失败: {e}")

        if not job.gif_path:
            pattern = os.path.join(MEDIA_ROOT, "navigation_*.gif")
            gifs = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
            if gifs:
                job.gif_path = gifs[0]

    def _pick_latest_log(self, job: Job) -> None:
        pattern = os.path.join(MEDIA_ROOT, "navigation_log_*.txt")
        logs = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        if logs:
            job.log_path = logs[0]


# ─── FastAPI ───────────────────────────────────────────────

app = FastAPI(title="Embodied Navigation Agent", version="1.0.0")
session = AgentSession()
jobs: Dict[str, Job] = {}
jobs_lock = threading.Lock()


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=500)
    pipeline: Optional[str] = Field(None, description="top | v2 | debug")


class InitRequest(BaseModel):
    pipeline: Optional[str] = DEFAULT_PIPELINE


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/api/health")
async def health():
    dep_err = check_agent_imports()
    return {
        "ok": dep_err is None,
        "media_root": MEDIA_ROOT,
        "deps_error": dep_err,
    }


@app.get("/api/session")
async def get_session():
    return session.status_dict()


@app.post("/api/init")
async def init_session(body: InitRequest):
    pipeline = (body.pipeline or DEFAULT_PIPELINE).lower()
    if pipeline not in ("top", "v2", "debug"):
        raise HTTPException(400, "pipeline 须为 top | v2 | debug")
    session.initialize_async(pipeline)
    return {"ok": True, "message": "正在后台加载模型与仿真器…", **session.status_dict()}


@app.post("/api/chat")
async def chat(body: ChatRequest):
    text = body.message.strip()
    if not text:
        raise HTTPException(400, "消息不能为空")

    if WEB_USER_MODE:
        pipeline = DEFAULT_PIPELINE if DEFAULT_PIPELINE in ("top", "debug") else "debug"
    else:
        pipeline = (body.pipeline or session.pipeline or DEFAULT_PIPELINE).lower()
    if pipeline not in ("top", "v2", "debug"):
        raise HTTPException(400, "pipeline 须为 top | v2 | debug")

    running = False
    with jobs_lock:
        for j in jobs.values():
            if j.status in (JobStatus.QUEUED, JobStatus.RUNNING):
                running = True
                break
    if running:
        raise HTTPException(
            409,
            "上一条任务仍在执行，请稍候完成后再发送。导航在仿真中较慢，属正常现象。",
        )

    job = Job(id=str(uuid.uuid4())[:12], message=text, pipeline=pipeline)
    with jobs_lock:
        jobs[job.id] = job

    def worker():
        try:

            def _sim_work():
                if not session.ready:
                    session.initialize(pipeline)
                session.run_task(job)

            run_on_sim_thread(_sim_work)
        except Exception as e:
            job.status = JobStatus.ERROR
            job.error = str(e)
            job.reply = f"初始化或执行失败：{e}"
            job.finished_at = time.time()
            job.append_log(traceback.format_exc())

    threading.Thread(target=worker, daemon=True).start()
    return {"job_id": job.id, "status": job.status.value}


@app.get("/api/job/{job_id}")
async def get_job(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    return job.to_dict()


@app.get("/api/media")
async def serve_media(path: str):
    full = _resolve_media_path(path)
    ext = os.path.splitext(full)[1].lower()
    media_type = "image/gif" if ext == ".gif" else "text/plain"
    return FileResponse(full, media_type=media_type)


@app.on_event("startup")
async def on_startup():
    if WEB_EAGER_INIT:
        session.initialize_async(DEFAULT_PIPELINE)


@app.post("/api/warmup")
async def warmup():
    """浏览器打开时触发/确认后台加载。"""
    st = session.status_dict()
    if session.ready:
        return {"ok": True, "message": "已就绪，可直接选择目标", **st}
    if session._init_thread and session._init_thread.is_alive():
        return {"ok": True, "message": "正在加载…", **st}
    session.initialize_async(DEFAULT_PIPELINE)
    return {"ok": True, "message": "已开始加载场景、检测模型与语义地图", **st}
