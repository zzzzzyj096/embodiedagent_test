import habitat_sim
import numpy as np
import torch
from PIL import Image
from groundingdino.util.inference import load_model, load_image, predict
import os
import cv2
from datetime import datetime
import dashscope
from dashscope import Generation
from collections import Counter, deque
from torchvision.ops import box_convert
import groundingdino.datasets.transforms as T

# ─── 配置 ───────────────────────────────────────────
# 通过 scene_dataset 加载，自动关联同目录下的 .navmesh / 语义资源
SCENE_DATASET = '/autodl-tmp/hm3d/scene_datasets/mp3d_example/mp3d.scene_dataset_config.json'
SCENE_ID = '17DRP5sb8fy'
# 客厅出生点 (x, z)，snap 后可行走；
LIVING_ROOM_SPAWN_XZ = (-0.69, 1.96)
GDINO_CONFIG = '/autodl-tmp/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
GDINO_WEIGHTS = '/autodl-tmp/GroundingDINO/weights/groundingdino_swint_ogc.pth'

TARGETS_ZH2EN = {
    "沙发": "sofa",
    "床":   "bed",
    "桌子": "table",
    "椅子": "chair seat",
    "门":   "door"
}

HFOV_DEG = 79.0
ARRIVE_DIST = 0.85          # 到目标旁：检测框区域深度须小于此值（米）
STUCK_DIST_EPS = 0.04       # 距离变化小于此值视为没动
STUCK_STEPS_THRESH = 10     # 连续多少步判定卡住
STUCK_ARRIVE_DIST = 1.0     # 卡住脱困时允许判到达的上限（仍须对准目标）
CONF_ARRIVE = 0.45          # 判到达时最低检测置信度
ARRIVE_CENTER_LO = 0.4     # 目标须在画面中部（与伺服一致）
ARRIVE_CENTER_HI = 0.6
PLAN_HORIZON = 2.0          # 规划时目标点最大前向距离（米）
DETECT_EVERY = 1            # 每 N 步检测一次，减轻左右抖动（沙发等大目标尤甚）
TURN_LOCK_STEPS = 2         # 转向锁定步数，避免左/右来回摆
SMOOTH_ALPHA = 0.35         # 检测框中心平滑系数
TARGET_DEPTH_PERCENTILE = 15  # bbox 深度取低分位，避免大框 median 读到背景墙
FAR_TARGET_M = 2.5          # 超过此距离视为「远目标」，放宽对准、优先靠近
APPROACH_BLOCK_FRONT_M = 1.0  # 前方距障碍小于此值：侧向绕行，不盲目前冲
APPROACH_STUCK_STEPS = 4    # approach 受阻时更早触发 recovery
CONF_APPROACH = 0.28        # approach 模式最低置信度
# 与 make_sim() 里 move_forward 的 ActuationSpec(amount=...) 保持一致
FORWARD_STEP_M = 0.25
MIN_FORWARD_ACTIONS = 2   # 至少发出 2 次前进指令（防止第 0 步误报到达）
MIN_MOVE_DIST = FORWARD_STEP_M * 1.2  # 水平位移约 0.3m，允许撞墙时略小于 2×0.25
ARRIVE_STREAK = 3           # 到达视觉条件须连续多帧成立
MIN_TASK_ACTIONS = 2        # 本任务至少执行 2 个动作（含转向），不能「一眼」结束
MIN_BBOX_RATIO = 0.02       # 到达：目标框须足够大（占画面比例）
MAX_BBOX_RATIO_ARRIVE = 0.40
MAX_BBOX_RATIO_PATH = 0.42
MIN_BBOX_RATIO_DETECT = 0.003
WEAK_DETECT_CONF = 0.25
MAX_BBOX_RATIO_DETECT = 0.85
STABLE_DETECT_MIN =  1      # stable_detect_steps > 5 才 found_stable
BBOX_AREA_JUMP_MAX = 2.5
BBOX_AREA_JUMP_MIN = 0.4
BBOX_CENTER_JUMP_MAX = 0.20
TARGET_PROGRESS_MIN = 0.05  # 单步 target_depth 至少下降多少才算有进展
NO_PROGRESS_STEPS_LIMIT = 8
RECOVERY_STEPS = 3
SAME_TURN_STEPS_LIMIT = 12  # 连续同向转向超过此次数则强制反向
VISIT_GRID_M = 0.25           # visited (x,z) 栅格
HEADING_BINS = 16             # yaw 分桶

# PathFinder / 扫描锚点：吸附后不能与机身重合（否则会规划「零距离」路径）
SCAN_ANCHOR_MIN_BBOX_RATIO = 0.008
SCAN_MIN_GOAL_DIST_M = 0.48
SCAN_GOAL_MIN_OVER_DEPTH = 0.28   # 水平位移下限 ≈ depth * ratio，防止 snap 塌缩到脚下
PATH_WP_MAX_STEPS = 120           # 单路径点允许步数（大场景）
PATH_WP_STUCK_STEPS = 10          # 跟随路径时位姿几乎不变则强制绕行

# 最小 known places（轻量拓扑记忆，不引入完整SLAM）
KNOWN_PLACE_MAX_PER_TARGET = 3
KNOWN_PLACE_MERGE_DIST_M = 1.2
KNOWN_PLACE_FAIL_BLOCK = 2
KNOWN_PLACE_MAX_AGE = 3000
KNOWN_PLACE_MIN_CONF = 0.50
KNOWN_PLACES = {}  # {target_en: [{"pos": np.array([x,y,z]), "score": float, "seen_step": int, "fails": int}]}
FORCE_RESPAWN_EACH_TASK = True


# ─── 仿真器初始化 ────────────────────────────────────
def place_agent_on_navmesh(sim, random_spawn=False, living_room_spawn=True, agent_idx=0):
    """将 Agent 放到 navmesh 可行走区域；首次进场景默认出生在客厅。"""
    pathfinder = sim.pathfinder
    agent = sim.agents[agent_idx]
    state = agent.state
    if pathfinder.is_loaded:
        if random_spawn:
            if living_room_spawn:
                x, z = LIVING_ROOM_SPAWN_XZ
                state.position = pathfinder.snap_point(
                    np.array([x, 0.0, z], dtype=np.float32)
                )
            else:
                state.position = pathfinder.get_random_navigable_point()
        else:
            state.position = pathfinder.snap_point(state.position)
        agent.set_state(state)
        return state.position, pathfinder.is_navigable(state.position)
    return state.position, False


def make_sim(random_spawn=True, living_room_spawn=True):
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_dataset_config_file = SCENE_DATASET
    backend_cfg.scene_id = SCENE_ID
    backend_cfg.enable_physics = False

    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "color"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.resolution = [480, 640]
    rgb_spec.position = [0.0, 1.5, 0.0]

    depth_spec = habitat_sim.CameraSensorSpec()
    depth_spec.uuid = "depth"
    depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_spec.resolution = [480, 640]
    depth_spec.position = [0.0, 1.5, 0.0]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_spec, depth_spec]
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=FORWARD_STEP_M)),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=15.0)),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=15.0)),
    }

    cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)

    if sim.pathfinder.is_loaded:
        area = sim.pathfinder.navigable_area
        pos, ok = place_agent_on_navmesh(
            sim, random_spawn=random_spawn, living_room_spawn=living_room_spawn
        )
        print(f"Navmesh 已加载 | 可行走面积: {area:.1f} m² | 起点可通行: {ok}")
        spawn_hint = ''
        if random_spawn:
            spawn_hint = ' (客厅出生点)' if living_room_spawn else ' (随机 navmesh 点)'
        print(f"Agent 初始位置: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]{spawn_hint}")
    else:
        print("警告: Navmesh 未加载，导航可能穿墙或卡住")

    return sim


# ─── LLM指令解析 ─────────────────────────────────────
def parse_command(user_input: str) -> str:
    """用通义千问解析用户指令"""
    dashscope.api_key = os.environ["DASHSCOPE_API_KEY"]

    response = Generation.call(
        model='qwen-turbo',
        messages=[{
            'role': 'user',
            'content': f"""你是家用机器人指令解析器。
用户说："{user_input}"
从以下选项识别导航目标：沙发、床、桌子、椅子、门
只输出一个目标名称，不要其他文字。
无法识别则输出"未知"。"""
        }],
        max_tokens=10,
        temperature=0
    )

    target = response.output.text.strip()
    print(f"LLM解析: '{user_input}' → '{target}'")
    return target


# ─── GroundingDINO检测 + Stable Tracker ──────────────
SHAPE_PRIORS = {
    "sofa": {"min_area": 0.015, "max_area": 0.4, "aspect": (1.2, 5.0)},
    "door": {"min_area": 0.015, "max_area": 0.45, "aspect": (0.25, 0.85)},
    "table": {"min_area": 0.01, "max_area": 0.45, "aspect": (0.4, 4.0)},
    "chair": {"min_area": 0.006, "max_area": 0.35, "aspect": (0.3, 3.5)},
    "bed": {"min_area": 0.015, "max_area": 0.5, "aspect": (0.6, 4.5)},
}
LOCK_SEARCH_RADIUS = 0.30


class SofaTracker:
    def __init__(self):
        self.locked = False
        self.prev_center = None
        self.prev_bbox = None
        self.fail_count = 0
        self.max_fail = 5
        self.lock_strength = 0.8


def is_valid_candidate(x0, y0, x1, y1, w, h, prompt):
    key = prompt.lower()
    prior = SHAPE_PRIORS.get(
        key,
        {"min_area": 0.005, "max_area": 0.6, "aspect": (0.12, 6.0)},
    )
    bw, bh = x1 - x0, y1 - y0
    area = bw * bh / (w * h)
    aspect = bw / max(bh, 1)
    if area < prior["min_area"] or area > prior["max_area"]:
        return False
    ar_lo, ar_hi = prior["aspect"]
    return ar_lo < aspect < ar_hi


def compute_score(logit, bbox_ratio, center, prev_center):
    if prev_center is None:
        motion = 0.0
    else:
        motion = float(np.linalg.norm(np.array(center) - np.array(prev_center)))
    return logit - 0.4 * bbox_ratio - 0.8 * motion


class TargetDetector:
    def __init__(self):
        self.model = load_model(GDINO_CONFIG, GDINO_WEIGHTS)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tracker = SofaTracker()
        self.transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        print(f"GroundingDINO加载完成，设备: {self.device}")

    def reset(self):
        self.tracker = SofaTracker()

    def reset_target_lock(self):
        self.reset()

    def detect(self, rgb, prompt):
        img = Image.fromarray(rgb[:, :, :3]).convert("RGB")
        h, w = rgb.shape[:2]
        image_tensor, _ = self.transform(img, None)

        boxes, logits, _ = predict(
            model=self.model,
            image=image_tensor,
            caption=prompt,
            box_threshold=0.25,
            text_threshold=0.25,
            device=self.device,
        )

        if len(boxes) == 0:
            self.tracker.fail_count += 1
            return {"found": False, "center_x": 0.5, "confidence": 0.0, "bbox": None}

        boxes = boxes * torch.tensor([w, h, w, h], device=boxes.device)
        boxes = box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")

        trk = self.tracker
        candidates = []

        for i in range(len(logits)):
            x0, y0, x1, y1 = map(int, boxes[i].tolist())
            center = ((x0 + x1) / 2 / w, (y0 + y1) / 2 / h)
            bbox_ratio = ((x1 - x0) * (y1 - y0)) / (w * h)
            logit = float(logits[i])

            if trk.locked and trk.prev_center is not None:
                if np.linalg.norm(np.array(center) - np.array(trk.prev_center)) > LOCK_SEARCH_RADIUS:
                    continue
            elif not is_valid_candidate(x0, y0, x1, y1, w, h, prompt):
                continue

            score = compute_score(logit, bbox_ratio, center, trk.prev_center if trk.locked else None)
            candidates.append((score, center, (x0, y0, x1, y1), logit))

        if not candidates:
            trk.fail_count += 1
            return {"found": False, "center_x": 0.5, "confidence": 0.0, "bbox": None}

        candidates.sort(key=lambda x: x[0], reverse=True)
        _, center, bbox, conf = candidates[0]

        if not trk.locked:
            if conf > 0.40:
                trk.locked = True
                trk.prev_center = center
                trk.prev_bbox = bbox
                trk.fail_count = 0
        else:
            if conf < 0.25:
                trk.fail_count += 1
            else:
                trk.fail_count = 0
                trk.prev_center = tuple(
                    trk.lock_strength * np.array(trk.prev_center)
                    + (1.0 - trk.lock_strength) * np.array(center)
                )
                trk.prev_bbox = bbox
            if trk.fail_count > trk.max_fail:
                trk.locked = False
                trk.prev_center = None
                trk.prev_bbox = None

        x0, y0, x1, y1 = bbox
        return {
            "found": True,
            "center_x": center[0],
            "confidence": conf,
            "bbox": (x0, y0, x1, y1),
        }


# ───  ─────────────────────────

def yaw_from_rotation(rotation):
    """Habitat 使用 quaternion.quaternion，字段为 w,x,y,z（标量在前）"""
    if hasattr(rotation, "w"):
        qw, qx, qy, qz = float(rotation.w), float(rotation.x), float(rotation.y), float(rotation.z)
    else:
        qx, qy, qz, qw = rotation
    siny = 2.0 * (qw * qy + qx * qz)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(np.arctan2(siny, cosy))


def agent_pose_relative(agent, origin_x, origin_z):
    pos = agent.state.position
    rx = pos[0] - origin_x
    rz = pos[2] - origin_z
    yaw = yaw_from_rotation(agent.state.rotation)
    return rx, rz, yaw



class ActTracker:
    """统计本任务内实际执行的动作，用于到达判定（非随意步数阈值）。"""

    def __init__(self, agent):
        self._agent = agent
        self.total = 0
        self.forward = 0

    def __call__(self, action_name):
        self._agent.act(action_name)
        self.total += 1
        if action_name == "move_forward":
            self.forward += 1



def patch_dist(patch):
    valid = patch[np.isfinite(patch) & (patch > 0.05)]
    if len(valid) == 0:
        return 5.0
    return float(np.percentile(valid, 30))


def measure_depth_probes(depth):
    """左 / 前 / 右三向深度（图像中部横条）。"""
    left_patch = depth[200:280, 80:180]
    front_patch = depth[200:280, 280:360]
    right_patch = depth[200:280, 460:560]
    return patch_dist(left_patch), patch_dist(front_patch), patch_dist(right_patch)


def bbox_metrics(bbox, h, w):
    x0, y0, x1, y1 = bbox
    bbox_area = (x1 - x0) * (y1 - y0)
    bbox_ratio = bbox_area / float(w * h)
    center = ((x0 + x1) / 2.0 / w, (y0 + y1) / 2.0 / h)
    return bbox_area, bbox_ratio, center


def bbox_is_consistent(prev_area, prev_center, bbox_area, center):
    if prev_area is None or prev_center is None:
        return True
    area_ratio = bbox_area / max(prev_area, 1)
    if area_ratio > BBOX_AREA_JUMP_MAX or area_ratio < BBOX_AREA_JUMP_MIN:
        return False
    if abs(center[0] - prev_center[0]) > BBOX_CENTER_JUMP_MAX:
        return False
    if abs(center[1] - prev_center[1]) > BBOX_CENTER_JUMP_MAX:
        return False
    return True


def bbox_target_depth(depth, bbox, h, w):
    """框中心区域深度 + bbox 质量校验；劣质/全图框不信任。"""
    bbox_area, bbox_ratio, _ = bbox_metrics(bbox, h, w)
    if bbox_ratio < MIN_BBOX_RATIO_DETECT or bbox_ratio > MAX_BBOX_RATIO_DETECT:
        return None, bbox_area, bbox_ratio

    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0:
        return None, bbox_area, bbox_ratio

    bw, bh = x1 - x0, y1 - y0
    mx = max(1, int(bw * 0.2))
    my = max(1, int(bh * 0.2))
    cx0, cx1 = x0 + mx, x1 - mx
    cy0, cy1 = y0 + my, y1 - my
    if cx1 <= cx0 or cy1 <= cy0:
        cx0, cy0, cx1, cy1 = x0, y0, x1, y1

    patch = depth[cy0:cy1, cx0:cx1]
    valid = patch[np.isfinite(patch) & (patch > 0.05) & (patch < 10.0)]
    if len(valid) < 8:
        valid = patch[np.isfinite(patch) & (patch > 0.05)]
    if len(valid) == 0:
        return None, bbox_area, bbox_ratio
    return float(np.percentile(valid, TARGET_DEPTH_PERCENTILE)), bbox_area, bbox_ratio


def arrival_conditions_met(found_stable, cx, target_depth, confidence, bbox_ratio):
    """到达视觉条件：稳定可见 + 深度近 + 框足够大 + 对准。"""
    if not found_stable:
        return False
    if confidence < CONF_ARRIVE:
        return False
    if not (ARRIVE_CENTER_LO <= cx <= ARRIVE_CENTER_HI):
        return False
    if not (target_depth < ARRIVE_DIST and bbox_ratio > MIN_BBOX_RATIO):
        return False
    return True


def can_declare_arrival(
    mode, arrive_streak, move_dist, forward_actions, total_actions,
    found_stable, cx, target_depth, confidence, bbox_ratio,
):
    """
    真正宣告到达（防第 0 步误报）：
    - 视觉条件满足，且处于 approach；
    - 本任务已执行若干动作，且至少 MIN_FORWARD_ACTIONS 次前进；
    - 水平位移不少于 MIN_MOVE_DIST（与前进步长挂钩，非随意常数）。
    """
    if not arrival_conditions_met(found_stable, cx, target_depth, confidence, bbox_ratio):
        return False
    if mode != "approach":
        return False
    if total_actions < MIN_TASK_ACTIONS:
        return False
    if forward_actions < MIN_FORWARD_ACTIONS:
        return False
    if move_dist < MIN_MOVE_DIST:
        return False
    if arrive_streak < ARRIVE_STREAK:
        return False
    return True


def pos_visit_key(x, z):
    return (int(round(x / VISIT_GRID_M)), int(round(z / VISIT_GRID_M)))


def visit_key(x, z, yaw):
    kx, kz = pos_visit_key(x, z)
    return (kx, kz, yaw_bin(yaw))


def recent_heading_counts(recent_headings):
    return Counter(recent_headings)


def yaw_bin(yaw):
    return int((yaw + np.pi) / (2 * np.pi) * HEADING_BINS) % HEADING_BINS


def forward_offset(yaw, step_m):
    return step_m * np.sin(yaw), step_m * np.cos(yaw)


def obstacle_bypass_plan(left_dist, front_dist, right_dist):
    """前方受阻：朝更开阔一侧转，前方稍通则前进。"""
    if front_dist > 0.75:
        return "move_forward", None, 0
    if left_dist > right_dist + 0.2:
        return "turn_left", "left", 2
    if right_dist > left_dist + 0.2:
        return "turn_right", "right", 2
    if left_dist >= right_dist:
        return "turn_left", "left", 2
    return "turn_right", "right", 2


def visual_servo_plan(
    center_x, front_dist, target_depth=None,
    left_dist=5.0, right_dist=5.0,
    turn_lock=None, lock_remain=0,
):
    """视觉伺服规划：只返回动作，不执行。返回 (action, turn_lock, lock_remain)。"""
    if lock_remain > 0:
        action = "turn_left" if turn_lock == "left" else "turn_right"
        return action, turn_lock, lock_remain - 1

    far = target_depth is not None and target_depth > FAR_TARGET_M
    if far and front_dist < APPROACH_BLOCK_FRONT_M:
        return obstacle_bypass_plan(left_dist, front_dist, right_dist)

    center_lo, center_hi = (0.26, 0.74) if far else (ARRIVE_CENTER_LO, ARRIVE_CENTER_HI)

    if far and front_dist > 0.65 and center_lo <= center_x <= center_hi:
        return "move_forward", None, 0

    if center_x < center_lo:
        return "turn_left", "left", TURN_LOCK_STEPS
    if center_x > center_hi:
        return "turn_right", "right", TURN_LOCK_STEPS
    if front_dist > 0.55:
        return "move_forward", None, 0
    return obstacle_bypass_plan(left_dist, front_dist, right_dist)


def build_recovery_escape_arc(left_dist, right_dist):
    """escape arc：加长侧向绕障弧线。"""
    turn = "turn_left" if left_dist > right_dist else "turn_right"
    return [
        turn, turn, turn,
        "move_forward", "move_forward", "move_forward", "move_forward",
        turn, turn,
        "move_forward", "move_forward",
    ]


def explore_plan(pos, yaw, left_dist, front_dist, right_dist, visited, recent_headings):
    """
    探索规划：frontier bias — 优先未访问、更开阔、少重复朝向。
    返回 (action, turn_lock, lock_remain)
    """
    x, z = float(pos[0]), float(pos[2])
    scores = {}
    heading_counts = recent_heading_counts(recent_headings)

    if front_dist > 1.0:
        fx, fz = forward_offset(yaw, FORWARD_STEP_M)
        nk = visit_key(x + fx, z + fz, yaw)
        bonus = 1.5 if nk not in visited else -1.0
        scores["move_forward"] = front_dist + bonus

    left_yaw = yaw + np.radians(15.0)
    right_yaw = yaw - np.radians(15.0)
    left_penalty = heading_counts[yaw_bin(left_yaw)] * 0.15
    right_penalty = heading_counts[yaw_bin(right_yaw)] * 0.15
    scores["turn_left"] = left_dist - left_penalty
    scores["turn_right"] = right_dist - right_penalty

    action = max(scores, key=scores.get)
    lock = "left" if action == "turn_left" else ("right" if action == "turn_right" else None)
    return action, lock, 0


# ─── 导航控制（视觉伺服 + 简单探索）────────────────
def navigate(sim, detector, target_cn: str, max_steps=300):
    """看见目标 → visual servo；未见 → 前方通畅则前进否则扫描。"""
    target_en = TARGETS_ZH2EN.get(target_cn, target_cn)
    agent = sim.agents[0]

    frames = []
    prev_front_dist = None
    prev_pos = None
    stuck_steps = 0
    smooth_cx = None
    lost_streak = 0
    stable_detect_steps = 0
    prev_bbox_area = None
    prev_bbox_center = None
    bbox_area = 0
    bbox_ratio = 0.0
    target_progress = 0.0
    turn_lock = None
    lock_remain = 0
    last_found = False
    last_center_x = 0.5
    last_confidence = 0.0
    last_bbox = None
    task_start_pos = np.array(agent.state.position, copy=True)
    arrive_streak = 0
    act = ActTracker(agent)
    prev_smooth_target_depth = None
    smooth_target_depth = None
    no_progress_steps = 0
    nav_mode = "explore"
    recovery_steps = 0
    recovery_arc = None
    recovery_arc_idx = 0
    last_turn_direction = None
    same_turn_steps = 0
    visited = set()
    recent_headings = deque(maxlen=50)

    def record_turn(direction):
        nonlocal last_turn_direction, same_turn_steps
        if direction == last_turn_direction:
            same_turn_steps += 1
        else:
            same_turn_steps = 1
            last_turn_direction = direction

    def act_nav(action_name):
        act(action_name)
        if action_name == "turn_left":
            record_turn("left")
        elif action_name == "turn_right":
            record_turn("right")

    detector.reset()

    log_path = f"/autodl-tmp/navigation_log_{target_cn}_{datetime.now():%Y%m%d_%H%M%S}.txt"
    log_file = open(log_path, "w", encoding="utf-8")

    def nav_log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    nav_log(f"\n开始导航，目标: {target_cn} ({target_en}) [Visual Servo Navigation]")
    nav_log(f"任务起点: [{task_start_pos[0]:.3f}, {task_start_pos[1]:.3f}, {task_start_pos[2]:.3f}]")
    nav_log(f"日志文件: {log_path}")

    for steps in range(max_steps):
        obs = sim.get_sensor_observations()
        rgb = obs["color"]
        depth = obs["depth"]

        if steps % DETECT_EVERY == 0:
            det = detector.detect(rgb, target_en)
            found = det["found"]
            center_x = det["center_x"]
            confidence = det["confidence"]
            bbox = det["bbox"]
            
            last_found = found
            last_center_x = center_x
            last_confidence = confidence
            last_bbox = bbox
        else:
            found = last_found
            center_x = last_center_x
            confidence = last_confidence
            bbox = last_bbox

        h, w = depth.shape[:2]
        bbox_area = 0
        bbox_ratio = 0.0
        bbox_center = None
        if bbox is not None:
            bbox_area, bbox_ratio, bbox_center = bbox_metrics(bbox, h, w)

        candidate_found = found and confidence > WEAK_DETECT_CONF

        if steps % DETECT_EVERY == 0:
            if candidate_found:
                stable_detect_steps += 1
                smooth_cx = center_x if smooth_cx is None else (
                    (1 - SMOOTH_ALPHA) * smooth_cx + SMOOTH_ALPHA * center_x
                )
                if bbox is not None:
                    prev_bbox_area = bbox_area
                    prev_bbox_center = bbox_center
                lost_streak = 0
            else:
                stable_detect_steps = 0
                lost_streak += 1
                if not candidate_found:
                    smooth_cx = None
                    prev_bbox_area = None
                    prev_bbox_center = None
                    turn_lock, lock_remain = None, 0
        else:
            lost_streak += 1
            if lost_streak > 3:
                stable_detect_steps = max(0, stable_detect_steps - 1)

        found_stable = stable_detect_steps > STABLE_DETECT_MIN
        cx = smooth_cx if found_stable and smooth_cx is not None else center_x

        left_dist, front_dist, right_dist = measure_depth_probes(depth)

        if found_stable and bbox is not None:
            td, bbox_area, bbox_ratio = bbox_target_depth(depth, bbox, h, w)
            target_depth = td if td is not None else front_dist
        else:
            target_depth = front_dist

        target_progress = 0.0
        if found_stable:
            if smooth_target_depth is None:
                smooth_target_depth = target_depth
            else:
                smooth_target_depth = 0.8 * smooth_target_depth + 0.2 * target_depth
            if prev_smooth_target_depth is not None:
                target_progress = prev_smooth_target_depth - smooth_target_depth
                if target_progress < TARGET_PROGRESS_MIN:
                    no_progress_steps += 1
                else:
                    no_progress_steps = 0
            prev_smooth_target_depth = smooth_target_depth
        else:
            smooth_target_depth = None
            prev_smooth_target_depth = None
            no_progress_steps = 0

        if found_stable and no_progress_steps > NO_PROGRESS_STEPS_LIMIT and nav_mode != "recovery":
            nav_mode = "recovery"
            recovery_steps = RECOVERY_STEPS
            recovery_arc = build_recovery_escape_arc(left_dist, right_dist)
            recovery_arc_idx = 0

        approach_dist = target_depth if found_stable else front_dist
        dist_for_stuck = front_dist if (found_stable and target_depth > FAR_TARGET_M) else approach_dist
        pos = agent.state.position
        yaw = yaw_from_rotation(agent.state.rotation)
        visited.add(visit_key(pos[0], pos[2], yaw))
        recent_headings.append(yaw_bin(yaw))
        move_dist = float(np.linalg.norm(pos[[0, 2]] - task_start_pos[[0, 2]]))

        if nav_mode == "recovery":
            mode_this = "recovery"
        elif found_stable:
            mode_this = "approach"
        else:
            mode_this = "explore"

        looks_arrived = arrival_conditions_met(
            found_stable, cx, target_depth, confidence, bbox_ratio
        )
        if looks_arrived and mode_this == "approach":
            arrive_streak += 1
        else:
            arrive_streak = 0
        arrived = can_declare_arrival(
            mode_this, arrive_streak, move_dist, act.forward, act.total,
            found_stable, cx, target_depth, confidence, bbox_ratio,
        )

        is_stuck_depth = (
            prev_front_dist is not None
            and abs(dist_for_stuck - prev_front_dist) < STUCK_DIST_EPS
        )
        is_stuck_pose = (
            prev_pos is not None
            and np.linalg.norm(pos - prev_pos) < 0.02
        )
        if is_stuck_depth and is_stuck_pose:
            stuck_steps += 1
        else:
            stuck_steps = 0
        prev_front_dist = dist_for_stuck
        prev_pos = np.array(pos, copy=True)

        nav_log(
            f"[{steps:03d}] "
            f"mode:{mode_this} "
            f"cx:{cx:.2f} "
            f"conf:{confidence:.2f} "
            f"front:{front_dist:.2f} "
            f"left:{left_dist:.2f} "
            f"right:{right_dist:.2f} "
            f"target:{target_depth:.2f} "
            f"bbox_area:{bbox_area} "
            f"bbox_ratio:{bbox_ratio:.3f} "
            f"stable:{stable_detect_steps} "
            f"tprog:{target_progress:.3f} "
            f"move:{move_dist:.2f} "
            f"stuck:{stuck_steps} "
            f"lock:{turn_lock}:{lock_remain}"
        )

        if steps % 10 == 0:
            vis = rgb[:, :, :3].copy()
            if bbox is not None:
                x0, y0, x1, y1 = bbox
                color = (0, 255, 0) if found_stable else (255, 0, 0)
                cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)
                cv2.putText(
                    vis,
                    f"{confidence:.2f}",
                    (x0, y0 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )
            frames.append(Image.fromarray(vis))

        if arrived:
            nav_log(f"\n✓ 已成功到达{target_cn}旁边！"
                    f"（目标深度 {target_depth:.2f}m，本任务移动 {move_dist:.2f}m，共 {steps + 1} 步）")
            frames.append(Image.fromarray(rgb[:, :, :3]))
            log_file.close()
            return True, frames, steps

        approach_blocked = (
            found_stable
            and target_depth > FAR_TARGET_M
            and front_dist < APPROACH_BLOCK_FRONT_M
        )

        if found_stable and stuck_steps >= APPROACH_STUCK_STEPS and mode_this == "approach":
            if (act.forward >= 1 and move_dist >= FORWARD_STEP_M
                    and target_depth < STUCK_ARRIVE_DIST
                    and bbox_ratio > MIN_BBOX_RATIO
                    and ARRIVE_CENTER_LO <= cx <= ARRIVE_CENTER_HI
                    and confidence >= CONF_ARRIVE
                    and stable_detect_steps > STABLE_DETECT_MIN):
                nav_log(f"\n✓ 已接近{target_cn}（卡住但已对准且目标深度 {target_depth:.2f}m）")
                frames.append(Image.fromarray(rgb[:, :, :3]))
                log_file.close()
                return True, frames, steps
            nav_log(f"  [防卡] 进入 recovery escape arc ({recovery_steps} 轮)")
            nav_mode = "recovery"
            recovery_steps = RECOVERY_STEPS
            recovery_arc = build_recovery_escape_arc(left_dist, right_dist)
            recovery_arc_idx = 0
            stuck_steps = 0

        if (same_turn_steps > SAME_TURN_STEPS_LIMIT
                and last_turn_direction == "left"):
            nav_log(f"  [探索记忆] 连续左转 {same_turn_steps} 次，强制右转")
            next_action = "turn_right"
            turn_lock, lock_remain = "right", 0
        elif nav_mode == "recovery":
            if not found_stable:
                nav_mode = "explore"
                recovery_steps = 0
                recovery_arc = None
                recovery_arc_idx = 0
                next_action, turn_lock, lock_remain = explore_plan(
                    pos, yaw, left_dist, front_dist, right_dist, visited, recent_headings,
                )
            else:
                if recovery_arc is None:
                    recovery_arc = build_recovery_escape_arc(left_dist, right_dist)
                    recovery_arc_idx = 0
                next_action = recovery_arc[recovery_arc_idx]
                recovery_arc_idx += 1
                turn_lock = (
                    "left" if next_action == "turn_left"
                    else ("right" if next_action == "turn_right" else None)
                )
                lock_remain = 0
                if recovery_arc_idx >= len(recovery_arc):
                    recovery_arc = None
                    recovery_arc_idx = 0
                    recovery_steps -= 1
                    if recovery_steps <= 0:
                        nav_mode = "explore"
                        no_progress_steps = 0
                        stuck_steps = 0
        elif found_stable:
            if approach_blocked or (
                stuck_steps >= 3 and target_depth > ARRIVE_DIST
            ):
                next_action, turn_lock, lock_remain = obstacle_bypass_plan(
                    left_dist, front_dist, right_dist,
                )
            else:
                next_action, turn_lock, lock_remain = visual_servo_plan(
                    cx, front_dist, target_depth,
                    left_dist, right_dist, turn_lock, lock_remain,
                )
        else:
            if confidence > WEAK_DETECT_CONF and front_dist < APPROACH_BLOCK_FRONT_M:
                next_action, turn_lock, lock_remain = obstacle_bypass_plan(
                    left_dist, front_dist, right_dist,
                )
            elif confidence > WEAK_DETECT_CONF:
                if center_x < 0.5:
                    next_action = "turn_left"
                else:
                    next_action = "turn_right"
                turn_lock, lock_remain = None, 0
            else:
                next_action, turn_lock, lock_remain = explore_plan(
                    pos, yaw, left_dist, front_dist, right_dist, visited, recent_headings,
                )

        act_nav(next_action)

    nav_log(f"\n✗ 达到最大步数({max_steps})，未找到{target_cn}")
    log_file.close()
    return False, frames, steps


def estimate_target_nav_point(sim, depth, det):
    """
    用 bbox + 深度估计前方地上一点并 snap 到 navmesh。
    若射线终点落在墙/槛外，snap_point 常会把目标吸回机器人脚下 → 必须拒绝并重试缩短射线。
    """
    if not det["found"] or det["confidence"] <= 0.45 or det["bbox"] is None:
        return None, None
    h, w = depth.shape[:2]
    td, _, bbox_ratio = bbox_target_depth(depth, det["bbox"], h, w)
    if td is None or td <= 0.15 or td > 8.0:
        return None, None
    if not (SCAN_ANCHOR_MIN_BBOX_RATIO < bbox_ratio <= MAX_BBOX_RATIO_PATH):
        return None, None

    agent = sim.agents[0]
    pos = np.array(agent.state.position, dtype=np.float64)
    yaw = yaw_from_rotation(agent.state.rotation)
    cx = float(det["center_x"])
    angle_offset = (0.5 - cx) * np.radians(HFOV_DEG)
    target_yaw = yaw + angle_offset
    dx_dir = float(np.sin(target_yaw))
    dz_dir = float(np.cos(target_yaw))
    raw_tx = float(pos[0] + td * dx_dir)
    raw_tz = float(pos[2] + td * dz_dir)

    pf = sim.pathfinder
    min_horiz = max(SCAN_MIN_GOAL_DIST_M, SCAN_GOAL_MIN_OVER_DEPTH * td)

    for alpha in (1.0, 0.92, 0.84, 0.76, 0.68, 0.58, 0.48, 0.38, 0.28):
        ix = float(pos[0] + alpha * (raw_tx - pos[0]))
        iz = float(pos[2] + alpha * (raw_tz - pos[2]))
        candidate = pf.snap_point(np.array([ix, float(pos[1]), iz], dtype=np.float64))
        if not np.isfinite(candidate[0]) or not pf.is_navigable(candidate):
            continue
        dh = float(
            np.hypot(float(candidate[0]) - float(pos[0]), float(candidate[2]) - float(pos[2]))
        )
        if dh >= min_horiz:
            return np.array(candidate, dtype=np.float64), float(td * alpha)

    return None, None


def path_follow_steering_action(sim, waypoint_xyz, pos, yaw, depth):
    """沿 navmesh 朝路径点走一步（比单次 atan2 更不易贴墙死转）。"""
    pf = sim.pathfinder
    wp = np.array(waypoint_xyz, dtype=np.float64)
    path = habitat_sim.ShortestPath()
    path.requested_start = np.array(pos, dtype=np.float64)
    path.requested_end = wp
    left_dist, front_dist, right_dist = measure_depth_probes(depth)
    if pf.find_path(path) and len(path.points) >= 2:
        nxt = path.points[1]
        dx = float(nxt[0] - pos[0])
        dz = float(nxt[2] - pos[2])
        target_yaw = float(np.arctan2(dx, dz))
        yaw_diff = (target_yaw - yaw + np.pi) % (2 * np.pi) - np.pi
        if abs(yaw_diff) > np.radians(18):
            return "turn_left" if yaw_diff > 0 else "turn_right"
        if front_dist > 0.38:
            return "move_forward"
        return "turn_left" if left_dist > right_dist else "turn_right"

    dx = float(wp[0] - pos[0])
    dz = float(wp[2] - pos[2])
    target_yaw = float(np.arctan2(dx, dz))
    yaw_diff = (target_yaw - yaw + np.pi) % (2 * np.pi) - np.pi
    if abs(yaw_diff) > np.radians(20):
        return "turn_left" if yaw_diff > 0 else "turn_right"
    if front_dist > 0.4:
        return "move_forward"
    return "turn_left" if left_dist > right_dist else "turn_right"


def _dist_xz(a, b):
    return float(np.linalg.norm(np.array([a[0], a[2]]) - np.array([b[0], b[2]])))


def get_known_place_goal(target_en, agent_pos, step_count):
    items = KNOWN_PLACES.get(target_en, [])
    if not items:
        return None
    valid = []
    for p in items:
        if p["fails"] >= KNOWN_PLACE_FAIL_BLOCK:
            continue
        if step_count - p["seen_step"] > KNOWN_PLACE_MAX_AGE:
            continue
        valid.append(p)
    if not valid:
        return None
    valid.sort(key=lambda p: (_dist_xz(agent_pos, p["pos"]), -p["score"]))
    return np.array(valid[0]["pos"], dtype=np.float64)


def update_known_place(target_en, pos, conf, step_count):
    if conf < KNOWN_PLACE_MIN_CONF:
        return
    items = KNOWN_PLACES.setdefault(target_en, [])
    pos = np.array(pos, dtype=np.float64)
    for p in items:
        if _dist_xz(pos, p["pos"]) < KNOWN_PLACE_MERGE_DIST_M:
            p["pos"] = 0.7 * p["pos"] + 0.3 * pos
            p["score"] = 0.8 * p["score"] + 0.2 * float(conf)
            p["seen_step"] = step_count
            p["fails"] = max(0, p["fails"] - 1)
            return
    items.append({
        "pos": pos,
        "score": float(conf),
        "seen_step": step_count,
        "fails": 0,
    })
    items.sort(key=lambda x: (x["fails"], -x["score"], -x["seen_step"]))
    if len(items) > KNOWN_PLACE_MAX_PER_TARGET:
        del items[KNOWN_PLACE_MAX_PER_TARGET:]


def mark_known_place_fail(target_en, pos):
    items = KNOWN_PLACES.get(target_en, [])
    if not items:
        return
    pos = np.array(pos, dtype=np.float64)
    nearest = min(items, key=lambda p: _dist_xz(pos, p["pos"]))
    nearest["fails"] += 1


def navigate_with_pathfinder(sim, detector, target_cn: str, max_steps=520):
    """扫描估点 -> navmesh路径 -> 视觉精对准（无GT）。"""
    target_en = TARGETS_ZH2EN.get(target_cn, target_cn)
    agent = sim.agents[0]
    frames = []
    act = ActTracker(agent)

    log_path = f"/autodl-tmp/navigation_log_{target_cn}_{datetime.now():%Y%m%d_%H%M%S}.txt"
    log_file = open(log_path, "w", encoding="utf-8")

    def nav_log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    detector.reset()
    nav_log(f"\n开始导航(PathFinder)，目标: {target_cn} ({target_en})")
    nav_log(f"任务起点: [{agent.state.position[0]:.3f}, {agent.state.position[1]:.3f}, {agent.state.position[2]:.3f}]")
    nav_log(f"日志文件: {log_path}")

    pf = sim.pathfinder
    target_pos = None

    known_goal = get_known_place_goal(target_en, agent.state.position, act.total)
    if known_goal is not None:
        target_pos = known_goal
        nav_log(f"使用 known_place 作为初始目标点: {target_pos}")

    # 阶段1：旋转扫描 + 小步探索重定位，再扫描
    if target_pos is None:
        scan_blocks = 3
        per_scan_turns = 24
        relocate_moves = 10
        for block in range(scan_blocks):
            nav_log(f"扫描阶段 {block+1}/{scan_blocks}")
            for i in range(per_scan_turns):
                if act.total >= max_steps:
                    break
                obs = sim.get_sensor_observations()
                rgb = obs["color"]
                depth = obs["depth"]
                det = detector.detect(rgb, target_en)
                est_pos, est_depth = estimate_target_nav_point(sim, depth, det)

                if i % 3 == 0:
                    vis = rgb[:, :, :3].copy()
                    if det["bbox"] is not None:
                        x0, y0, x1, y1 = det["bbox"]
                        cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 255, 0), 2)
                    frames.append(Image.fromarray(vis))

                if est_pos is not None:
                    target_pos = est_pos
                    update_known_place(target_en, est_pos, det["confidence"], act.total)
                    nav_log(
                        f"扫描命中(锚点有效): conf={det['confidence']:.2f} "
                        f"depth={est_depth:.2f}m target_pos={target_pos}"
                    )
                    break

                act("turn_right")

            if target_pos is not None or act.total >= max_steps:
                break

            nav_log("本轮扫描未命中，执行小步重定位后继续扫描")
            for _ in range(relocate_moves):
                if act.total >= max_steps:
                    break
                obs = sim.get_sensor_observations()
                depth = obs["depth"]
                left_dist, front_dist, right_dist = measure_depth_probes(depth)
                if front_dist > 0.65:
                    act("move_forward")
                else:
                    act("turn_left" if left_dist > right_dist else "turn_right")

    if target_pos is None:
        nav_log("扫描+重定位后仍未找到可靠目标点，结束本轮")
        log_file.close()
        return False, frames, act.total

    # 阶段2：全局路径规划
    path = habitat_sim.ShortestPath()
    path.requested_start = np.array(agent.state.position, dtype=np.float64)
    path.requested_end = target_pos
    found_path = pf.find_path(path)
    if (not found_path) or len(path.points) < 2:
        mark_known_place_fail(target_en, target_pos)
        nav_log("路径规划失败，结束本轮（不回退到视觉伺服）")
        log_file.close()
        return False, frames, act.total
    nav_log(f"路径规划成功，共 {len(path.points)} 个路径点")

    # 阶段3：沿路径点前进（局部避障）
    reached_end = False
    for waypoint_idx, waypoint in enumerate(path.points[1:], 1):
        nav_log(f"前往路径点 {waypoint_idx}/{len(path.points)-1}: {np.array(waypoint)}")
        wp_prev_pose = None
        wp_stuck = 0
        for _ in range(PATH_WP_MAX_STEPS):
            if act.total >= max_steps:
                break
            obs = sim.get_sensor_observations()
            rgb = obs["color"]
            depth = obs["depth"]
            pos = np.array(agent.state.position, dtype=np.float64)
            yaw = yaw_from_rotation(agent.state.rotation)
            dist_to_wp = float(np.linalg.norm(pos[[0, 2]] - np.array([waypoint[0], waypoint[2]], dtype=np.float64)))
            if dist_to_wp < 0.4:
                break

            if wp_prev_pose is not None and np.linalg.norm(pos[[0, 2]] - wp_prev_pose) < 0.025:
                wp_stuck += 1
            else:
                wp_stuck = 0
            wp_prev_pose = np.array(pos[[0, 2]], copy=True)

            action = path_follow_steering_action(sim, waypoint, pos, yaw, depth)
            if wp_stuck >= PATH_WP_STUCK_STEPS:
                left_dist, front_dist, right_dist = measure_depth_probes(depth)
                action = obstacle_bypass_plan(left_dist, front_dist, right_dist)[0]

            act(action)

            if act.total % 5 == 0:
                frames.append(Image.fromarray(rgb[:, :, :3]))

        if act.total >= max_steps:
            break
        if waypoint_idx == len(path.points) - 1:
            reached_end = True

    # 阶段4：视觉精对准
    nav_log("进入精对准阶段")
    fine_budget = min(72, max_steps - act.total)
    for fine_step in range(max(0, fine_budget)):
        obs = sim.get_sensor_observations()
        rgb = obs["color"]
        depth = obs["depth"]
        det = detector.detect(rgb, target_en)
        left_dist, front_dist, right_dist = measure_depth_probes(depth)
        frames.append(Image.fromarray(rgb[:, :, :3]))

        if det["found"] and det["confidence"] > 0.40 and det["bbox"] is not None:
            h, w = depth.shape[:2]
            td, _, bbox_ratio = bbox_target_depth(depth, det["bbox"], h, w)
            target_depth = td if td is not None else front_dist
            cx = det["center_x"]
            if (
                target_depth < ARRIVE_DIST
                and MIN_BBOX_RATIO < bbox_ratio <= MAX_BBOX_RATIO_ARRIVE
            ):
                update_known_place(target_en, target_pos, det["confidence"], act.total)
                nav_log(f"\n✓ 已成功到达{target_cn}旁边！(depth={target_depth:.2f}, bbox_ratio={bbox_ratio:.3f})")
                log_file.close()
                return True, frames, act.total + fine_step + 1

            if cx < 0.4:
                act("turn_left")
            elif cx > 0.6:
                act("turn_right")
            elif front_dist > 0.4:
                act("move_forward")
            else:
                act("turn_left" if left_dist > right_dist else "turn_right")
        else:
            if front_dist > 0.52:
                act("move_forward")
            elif left_dist > right_dist + 0.08:
                act("turn_left")
            else:
                act("turn_right")

        if act.total >= max_steps:
            break

    if reached_end:
        update_known_place(target_en, target_pos, 0.65, act.total)
        nav_log(f"✗ 已到路径终点附近，但未完成精对准到达 {target_cn}")
    else:
        mark_known_place_fail(target_en, target_pos)
        nav_log(f"✗ 路径跟随未完成且未到达 {target_cn}")
    log_file.close()
    return False, frames, act.total


# ─── 主程序 ──────────────────────────────────────────
def run_agent(user_input: str, sim=None, detector=None):
    """
    多轮对话时传入同一 sim/detector，Agent 会留在上一轮结束位置。
    返回 (success, frames, sim, detector)
    """
    print(f"\n{'='*50}")
    print(f"用户指令: {user_input}")
    print('='*50)

    if sim is None:
        sim = make_sim(random_spawn=True)
    else:
        if FORCE_RESPAWN_EACH_TASK:
            pos, ok = place_agent_on_navmesh(
                sim, random_spawn=True, living_room_spawn=True
            )
            print(
                f"每轮强制重生到起点 | 当前位置: "
                f"[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] | 可通行: {ok}"
            )
        else:
            pos = sim.agents[0].state.position
            print(f"沿用上一轮位置继续 | 当前: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

    if detector is None:
        detector = TargetDetector()

    target_cn = parse_command(user_input)
    if target_cn == "未知" or target_cn not in TARGETS_ZH2EN:
        print(f"无法识别目标: {target_cn}")
        return False, [], sim, detector

    success, frames, steps = navigate_with_pathfinder(sim, detector, target_cn)

    if frames:
        gif_path = f"/autodl-tmp/navigation_{target_cn}.gif"
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=500,
            loop=0
        )
        print(f"导航过程已保存到 {gif_path}")

    end_pos = sim.agents[0].state.position
    print(f"本任务结束位置: [{end_pos[0]:.3f}, {end_pos[1]:.3f}, {end_pos[2]:.3f}]")

    if success:
        reply = f"好的，我已经到达{target_cn}旁边了。还需要什么？"
    else:
        reply = f"抱歉，我没能找到{target_cn}。还需要什么？"

    print(f"\nAgent: {reply}")
    return success, frames, sim, detector


if __name__ == "__main__":
    print("具身导航 Agent（输入 q 退出）")
    print("示例：请到沙发旁边 / 去床边 / 走到椅子那里")
    print("多轮任务共用同一场景，从上一轮结束位置继续出发。")
    sim, detector = None, None
    try:
        while True:
            try:
                user_input = input("\n你: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见。")
                break
            if not user_input:
                continue
            if user_input.lower() in ("q", "quit", "exit", "退出"):
                print("再见。")
                break
            _, _, sim, detector = run_agent(user_input, sim, detector)
    finally:
        if sim is not None:
            sim.close()
            print("仿真已关闭。")
