const messagesEl = document.getElementById("messages");
const logsEl = document.getElementById("logs");
const form = document.getElementById("chat-form");
const input = document.getElementById("input");
const btnSend = document.getElementById("btn-send");
const btnInit = document.getElementById("btn-init");
const pipelineEl = document.getElementById("pipeline");
const adminBar = document.getElementById("admin-bar");
const sessionBadge = document.getElementById("session-badge");
const jobStatusEl = document.getElementById("job-status");
const jobMetaEl = document.getElementById("job-meta");
const gifCard = document.getElementById("gif-card");
const gifPreview = document.getElementById("gif-preview");
const gifLink = document.getElementById("gif-link");

let pollTimer = null;
let busy = false;
let agentReady = false;
let sessionFailStreak = 0;

const PHASE_LOG = {
  topo: "正在读取预探索语义地图…",
  sim: "正在加载 Habitat 场景…",
  detector: "正在加载 GroundingDINO（最耗时）…",
  starting: "正在启动加载…",
};

function timeStr() {
  return new Date().toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" });
}

function addMessage(role, text) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.innerHTML = `${escapeHtml(text)}<span class="time">${timeStr()}</span>`;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

/** Agent 回复：文字 + 可选导航 GIF（显示在对话气泡下方） */
function addAgentReply(text, gifUrl) {
  const wrap = document.createElement("div");
  wrap.className = "msg agent";
  let html = escapeHtml(text || "");
  if (gifUrl) {
    const u = gifUrl + (gifUrl.includes("?") ? "&" : "?") + "t=" + Date.now();
    html += `<img class="chat-gif" src="${u}" alt="导航回放" loading="lazy" />`;
  }
  wrap.innerHTML = `${html}<span class="time">${timeStr()}</span>`;
  messagesEl.appendChild(wrap);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function escapeHtml(s) {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function setBusy(v) {
  busy = v;
  updateSendButton();
}

function updateSendButton() {
  btnSend.disabled = busy || (!agentReady && !allowSendFallback);
}

let allowSendFallback = false;

function applySession(data) {
  if (!data) return;
  updateSessionBadge(data);

  if (data.init_error) {
    if (!busy) {
      logsEl.textContent = data.init_error;
    }
    if (data.init_stalled) {
      allowSendFallback = true;
      updateSendButton();
    }
    return;
  }

  const wasReady = agentReady;
  agentReady = !!data.agent_ready;
  updateSendButton();

  // 导航进行中时不要用 session 轮询覆盖任务日志
  if (!busy) {
    if (agentReady) {
      logsEl.textContent =
        "系统就绪（场景 + 检测模型 + 语义地图已加载）。选择目标开始导航。";
    } else if (data.initializing && PHASE_LOG[data.init_phase]) {
      logsEl.textContent = PHASE_LOG[data.init_phase];
    }
  }

  if (agentReady && !wasReady) {
    jobMetaEl.textContent = data.topo_loaded
      ? `语义地图已载入（约 ${data.topo_views} 个视角）`
      : "";
    if (!busy) {
      setJobUi("idle", "");
    }
  }
}

function updateSessionBadge(data) {
  sessionBadge.className = "badge";
  if (data.init_error) {
    sessionBadge.textContent = data.init_stalled ? "加载中断" : "加载异常";
    sessionBadge.classList.add("error");
    return;
  }
  if (data.agent_ready) {
    sessionBadge.textContent = data.topo_loaded
      ? `就绪 · ${data.topo_views || "?"} 视角`
      : "就绪";
    sessionBadge.classList.add("ready");
    return;
  }
  if (data.initializing) {
    const labels = {
      topo: "读语义地图…",
      sim: "加载场景…",
      detector: "加载检测模型…",
    };
    sessionBadge.textContent = labels[data.init_phase] || "系统加载中…";
    sessionBadge.classList.add("busy");
    return;
  }
  sessionBadge.textContent = "等待加载";
}

async function refreshSession() {
  try {
    const res = await fetch("/api/session");
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }
    const data = await res.json();
    sessionFailStreak = 0;
    applySession(data);
    return data;
  } catch (e) {
    sessionFailStreak += 1;
    // 加载/导航时主线程忙，偶发失败；已就绪时不要误报「连接失败」
    if (sessionFailStreak >= 3 && !agentReady) {
      sessionBadge.textContent = "连接异常（稍后重试）";
      sessionBadge.classList.add("error");
    }
    return null;
  }
}

async function warmupBackend() {
  try {
    const res = await fetch("/api/warmup", { method: "POST" });
    const data = await res.json();
    if (data.message) {
      logsEl.textContent = data.message;
    }
    applySession(data);
  } catch (e) {
    console.warn("warmup", e);
  }
}

async function initAgent() {
  if (!pipelineEl) return;
  await fetch("/api/init", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pipeline: pipelineEl.value }),
  });
  addMessage("system", "已重新开始后台加载。");
  await refreshSession();
}

function setJobUi(status, meta) {
  jobStatusEl.className = `job-status ${status}`;
  const labels = {
    idle: "空闲",
    queued: "排队中",
    running: "导航中…",
    done: "已完成",
    error: "出错",
  };
  jobStatusEl.textContent = labels[status] || status;
  if (meta) jobMetaEl.textContent = meta;
}

function showGif(url) {
  if (!url) {
    gifCard.hidden = true;
    return;
  }
  gifCard.hidden = false;
  gifPreview.src = url + "&t=" + Date.now();
  gifLink.href = url;
}

async function pollJob(jobId) {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    try {
      const res = await fetch(`/api/job/${jobId}`);
      if (!res.ok) return;
      const job = await res.json();
      if (job.logs && job.logs.length > 0) {
        logsEl.textContent = job.logs.join("\n");
      } else if (job.status === "running") {
        logsEl.textContent = "导航进行中，等待仿真日志…";
      }
      logsEl.scrollTop = logsEl.scrollHeight;

      if (job.status === "running" || job.status === "queued") {
        setJobUi(job.status === "queued" ? "queued" : "running", "");
        return;
      }

      clearInterval(pollTimer);
      pollTimer = null;
      setBusy(false);
      await refreshSession();

      if (job.status === "error") {
        setJobUi("error", job.error || "");
        addAgentReply(job.reply || "执行出错。", job.gif_url || null);
      } else {
        setJobUi("done", "");
        addAgentReply(job.reply || "", job.gif_url || null);
        showGif(job.gif_url);
      }
    } catch (e) {
      console.error(e);
    }
  }, 1500);
}

async function sendMessage(text) {
  const message = (text || input.value).trim();
  if (!message || busy) return;

  const st = await refreshSession();
  if (!st?.agent_ready && !allowSendFallback) {
    addMessage("system", "系统仍在加载，请稍候右上角显示「就绪」后再试。");
    return;
  }

  addMessage("user", message);
  input.value = "";
  setBusy(true);
  setJobUi("running", "");
  logsEl.textContent = "导航任务已开始，日志输出中…";
  gifCard.hidden = true;

  const body = { message };
  if (pipelineEl && adminBar && !adminBar.hidden) {
    body.pipeline = pipelineEl.value;
  }

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) {
      setBusy(false);
      setJobUi("error", "");
      addMessage("system", data.detail || `错误 ${res.status}`);
      return;
    }
    pollJob(data.job_id);
  } catch (e) {
    setBusy(false);
    setJobUi("error", "");
    addMessage("system", `网络错误：${e.message}`);
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  sendMessage();
});

if (btnInit) btnInit.addEventListener("click", initAgent);

document.querySelectorAll(".chip").forEach((btn) => {
  btn.addEventListener("click", () => {
    if (btn.disabled || btn.classList.contains("chip--debug")) return;
    sendMessage(btn.getAttribute("data-msg"));
  });
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});

addMessage(
  "system",
  "正在后台加载预置场景与语义地图（与命令行首次导航相同）。右上角显示「就绪」后即可选择沙发、椅子等目标。"
);

setJobUi("idle", "系统加载中…");
btnSend.disabled = true;

(async function boot() {
  await warmupBackend();
  let readyMsgShown = false;

  const tick = async () => {
    const d = await refreshSession();
    if (!d) {
      setTimeout(tick, 3000);
      return;
    }
    if (d.agent_ready && !readyMsgShown) {
      readyMsgShown = true;
      addMessage(
        "system",
        d.topo_loaded
          ? `系统就绪。已载入语义地图（约 ${d.topo_views} 个视角），请选择目标。`
          : "系统就绪，请选择目标。"
      );
      return;
    }
    if (d.init_error && d.init_stalled) {
      addMessage("system", d.init_error + " 也可尝试再次点击目标。");
      return;
    }
    if (!d.agent_ready) {
      setTimeout(tick, 2000);
    }
  };
  tick();
  setInterval(refreshSession, 8000);
})();
