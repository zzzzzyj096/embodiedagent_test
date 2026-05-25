"""
轻量预探索 + 语义拓扑记忆 + 按指令导航

流程:
  1) systematic_explore: navmesh 采样全覆盖 → SemanticView 记忆 + 区域语义（带衰减）
  2) navigate 三阶段: coarse(邻域) → viewpoint recovery → visual servo

地图可导出: /autodl-tmp/semantic_topo_map.json
"""

import json
import math
import os
import random
import sys
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime

import cv2
import habitat_sim
import numpy as np
import torch
from PIL import Image
import dashscope
from dashscope import Generation
from groundingdino.util.inference import load_model, predict
import groundingdino.datasets.transforms as T
from torchvision.ops import box_convert

try:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from llm_judge import Rejury

# ─── 场景与目标 ───────────────────────────────────────────
SCENE_DATASET = "/autodl-tmp/hm3d/scene_datasets/mp3d_example/mp3d.scene_dataset_config.json"
SCENE_ID = "17DRP5sb8fy"
LIVING_ROOM_SPAWN_XZ = (-2.4, -1.96)
GDINO_CONFIG = "/autodl-tmp/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GDINO_WEIGHTS = "/autodl-tmp/GroundingDINO/weights/groundingdino_swint_ogc.pth"

TARGETS_ZH2EN = {
    "沙发": "sofa",
    "床": "bed",
    "桌子": "table",
    "椅子": "chair seat",
    "门": "door",
}
DETECT_LABELS = list(TARGETS_ZH2EN.values())

# ─── 拓扑 / 预探索 ─────────────────────────────────────────
REGION_SIZE = 2.5
TOPO_MAP_PATH = "/autodl-tmp/semantic_topo_map.json"
PRE_EXPLORE_MAX_STEPS = 600
PRE_DETECT_EVERY = 8
NO_NEW_REGION_PATIENCE = 250
FORWARD_STEP_M = 0.25
SEM_CAP = 5.0
FRONTIER_UNSEEN_BONUS = 2.0
FRONTIER_SEM_DIV_WEIGHT = 0.15
FRONTIER_VISIT_PENALTY = 0.8
FRONTIER_FAIL_PENALTY = 0.5
FRONTIER_DIST_PENALTY = 0.12
VISUAL_SERVO_CENTER_LO = 0.42
VISUAL_SERVO_CENTER_HI = 0.58
VISUAL_SERVO_FORWARD_DEPTH = 0.55
CAMERA_HFOV_DEG = 90.0
CAMERA_HFOV_RAD = math.radians(CAMERA_HFOV_DEG)
TURN_STEP_DEG = 15.0
TURN_STEP_RAD = math.radians(TURN_STEP_DEG)

# ─── 按指令导航 ───────────────────────────────────────────
NAV_MAX_STEPS = 1800
NAV_PER_VIEW_BUDGET = 700
# Stage1 全局（沙发/门/椅子等共用 run_persistent_path_follow + navigate_coarse_to_xz）
STAGE1_NO_PROGRESS_STEPS = 50
STAGE1_REGRESS_ABORT_M = 1.0
STAGE1_IMPROVE_EPS_M = 0.08
STAGE1_WP_STUCK_STEPS = 80
STAGE1_MIN_CANDIDATE_BUDGET = 120
# Stage1 path-follow：FOLLOW 阶段关闭 recovery/振荡（与 stage1_motion_controller 一致）
STAGE1_DISABLE_RECOVERY = True
STAGE1_DISABLE_OSCILLATION = True
STAGE1_HEADING_ALIGN_DEG = 60.0
STAGE1_WP_PROGRESS_M = 0.55
STAGE1_SPAWN_ALIGN_YAW = True
STAGE1_SPAWN_YAW_TARGET_XZ = (-3.172, -0.792)
STAGE1_PREALIGN_MAX_TURNS = 12
# Stage1 path 终点：语义区域中心（非 view 观测位姿）；view.yaw 仅 S2/S3
STAGE1_GOAL_FROM_REGION_CENTER = True
STAGE1_VIEW_YAW_IN_PATH = False
STAGE1_SKIP_BAD_WP = True
STAGE1_SKIP_BEHIND_WP_DEG = 120.0
STAGE1_SKIP_BEHIND_WP_DIST_M = 0.55
STAGE1_SKIP_DETOUR_WP_DIST_M = 3.5
STAGE1_SKIP_DETOUR_BEARING_DEG = 42.0
STAGE1_SKIP_DETOUR_DIST_EPS_M = 0.10
# Stage1 近邻成功：dist≤1.2m；或 dist≤NEAR_OPEN_MAX_D 且局部空旷（对齐绕墙最近 ~1.8m）
STAGE1_NEAR_OPEN_AREA_GATE = True
STAGE1_NEAR_OPEN_MAX_D = 2.0
# path_finished / 网格误差：绕墙最近常卡在 ~2.00–2.01m，给空旷近邻判定留小余量
STAGE1_NEAR_OPEN_DIST_SLACK_M = 0.05
STAGE1_NEAR_OPEN_FRONT_M = 0.50
STAGE1_NEAR_OPEN_MIN_FREE_M = 0.45
STAGE1_NEAR_OPEN_STREAK = 2
# 空旷近邻须 navmesh 可达够近（防岛台后直线~2m、测地仍~5m 就切视觉）
STAGE1_NEAR_OPEN_MAX_GEODESIC_M = 3.2
STAGE1_NEAR_OPEN_GEO_EUCL_RATIO_MAX = 1.45
STAGE1_NEAR_OPEN_LOS_FRONT_OVER_M = 1.25
STAGE1_NEAR_BLOCKED_MAX_REPLANS = 3
STAGE1_NEAR_BLOCKED_MAX_REWINDS = 10
# bug2_rejoin 专用：近目标且开阔则直接结束 Stage1，不进 MLINE_TRANSIT
STAGE1_BUG2_REJOIN_OPEN_FINISH = True
STAGE1_BUG2_REJOIN_OPEN_FINISH_D = 2.0
# Stage1 会话内防回绕：曾离开起点后又贴回起点且 target 无进展则退出
STAGE1_ANTI_BACKTRACK = True
STAGE1_BACKTRACK_LEAVE_M = 0.45
STAGE1_BACKTRACK_RETURN_M = 0.32
# Stage1 航向：远处 mix(wp, bearing→target)；近处 mix(wp, view.yaw+rel_angle)（仅 VIEW_YAW_IN_PATH）
STAGE1_VIEW_YAW_BLEND_DIST_M = 2.0
STAGE1_SEMANTIC_ALPHA_FAR = 0.2
STAGE1_SEMANTIC_ALPHA_NEAR = 0.7
STAGE1_SEMANTIC_ALPHA_DIST_FAR = 4.0
STAGE1_SEMANTIC_ALPHA_DIST_NEAR = 1.2
STAGE1_FORWARD_GUARD_STEPS = 3
STAGE1_FORWARD_GUARD_EPS_M = 0.04
NAV_USE_GEODESIC_VIEW_SCORE = True
NAV_GEODESIC_DISTANCE_PENALTY = 0.15
NAV_RESET_ON_S1_FAIL = True
NAV_RESET_DRIFT_MIN_M = 0.35
# Stage3 统一策略：锁定 + 迟滞 + 分级到达（避免 conf/单帧抖动）
STAGE3_LOCK_CONF = 0.65
STAGE3_LOCK_BBOX = 0.12
STAGE3_LOCK_FRAMES = 2
STAGE3_LOST_USE_LAST_MAX = 8
STAGE3_LOST_USE_LAST_TINY_BBOX = 3
STAGE3_LOCK_RELEASE_LOST = 15
STAGE3_LOCK_RELEASE_LOST_PERSIST = 45
STAGE3_LOST_FORWARD_MAX = 0
STAGE3_LOST_FORWARD_MIN_FRONT_M = 0.40
STAGE3_TRACK_MIN_BBOX = 0.08
STAGE3_TRACK_MIN_CONF = 0.35
STAGE3_HYSTERESIS_MIN_BBOX = 0.06
STAGE3_HYSTERESIS_MIN_CONF = 0.28
STAGE3_PERIPHERAL_CX_LO = 0.38
STAGE3_PERIPHERAL_CX_HI = 0.55
STAGE3_SERVO_FORWARD_CX_LO = 0.40
STAGE3_SERVO_FORWARD_CX_HI = 0.55
STAGE3_ARRIVE_STREAK = 2
# 到达 = 近 depth OR 大物体贴边（大沙发贴边时 depth 可能仍 ~1.1m）
STAGE3_ARRIVE_DEPTH_CLOSE = 0.92
STAGE3_ARRIVE_EDGE_BBOX = 0.28
STAGE3_ARRIVE_EDGE_HUGE_BBOX = 0.36
STAGE3_ARRIVE_EDGE_CONF = 0.55
STAGE3_ARRIVE_EDGE_CENTER_LO = 0.28
STAGE3_ARRIVE_EDGE_CENTER_HI = 0.72
STAGE3_ARRIVE_MIN_FORWARDS = 3
STAGE3_SUCCESS_MIN_CONF = 0.50
# Stage3 到达策略类别（显式映射 + SHAPE_PRIORS 兜底，避免“门/沙发”硬编码二义）
STAGE3_ARRIVAL_KIND_COMPACT = "compact"
STAGE3_ARRIVAL_KIND_BULKY = "bulky"
STAGE3_ARRIVAL_KIND_DEFAULT = "default"
STAGE3_ARRIVAL_KIND_BY_TARGET = {
    "door": STAGE3_ARRIVAL_KIND_COMPACT,
    "chair seat": STAGE3_ARRIVAL_KIND_COMPACT,
    "sofa": STAGE3_ARRIVAL_KIND_BULKY,
    "bed": STAGE3_ARRIVAL_KIND_BULKY,
}
STAGE3_COMPACT_MIN_CONF = 0.65
# 门在远处框很小：单独降低锁定阈值；到达仍要求小框+已锁定
STAGE3_LOCK_CONF_BY_TARGET = {"door": 0.52, "chair seat": 0.58, "bed": 0.58}
STAGE3_LOCK_BBOX_BY_TARGET = {"door": 0.05, "chair seat": 0.08, "bed": 0.10}
# 床：锁定与到达单独收紧（bulky 默认 conf≥0.5 且无 lock 也能 depth_near，易浴室误报）
STAGE3_BED_ARRIVE_MIN_CONF = 0.68
STAGE3_BED_ARRIVE_MIN_BBOX = 0.08
STAGE3_BED_ARRIVE_MAX_BBOX = 0.38
STAGE3_BED_LOCK_MIN_BBOX = 0.08
STAGE3_BED_LOCK_MAX_BBOX = 0.40
STAGE3_BED_MIN_LOCKED_STEPS = 3
STAGE3_LOCK_FAST_CONF = 0.72
STAGE3_COMPACT_ARRIVE_MAX_BBOX = 0.32
STAGE3_DOOR_ARRIVE_MAX_BBOX = 0.20
STAGE3_DOOR_LOCK_AT_ARRIVE_MAX_BBOX = 0.18
STAGE3_DOOR_LOCK_MAX_BBOX = 0.22
STAGE3_DOOR_MIN_LOCKED_STEPS = 4
STAGE3_DOOR_MIN_ASPECT = 0.35
STAGE3_DOOR_MAX_ASPECT = 1.15
STAGE3_DOOR_AIM_YAW_MAX_ERR_DEG = 42.0
STAGE3_DOOR_RECOVERY_FRONT_GAP = 0.85
STAGE3_DEPTH_FRONT_MAX_GAP = 0.40
STAGE3_BULKY_DEPTH_MAX = 1.68
STAGE3_BULKY_MIN_CONF = 0.50
STAGE3_BULKY_EDGE_BBOX_LOCKED = 0.18
STAGE3_BULKY_MIN_MAX_BBOX = 0.18
STAGE3_ARRIVE_REJECT_SYNTHETIC = True
STAGE3_ARRIVE_EDGE_MIN_MAX_BBOX = 0.12
STAGE3_FUSE_FRONT_MAX_BBOX = 0.14
STAGE3_SERVO_LOCK_CENTER_LO = 0.38
STAGE3_SERVO_LOCK_CENTER_HI = 0.62
STAGE3_DISABLE_RECOVERY_WHEN_LOCKED = True
# 锁定后逼近：先对准(cx)再前进，直到融合深度进范围或 bbox 贴边
STAGE3_LOCK_APPROACH_UNTIL_DEPTH = True
STAGE3_APPROACH_TARGET_DEPTH = 0.92
STAGE3_APPROACH_EDGE_BBOX = 0.28
STAGE3_APPROACH_MIN_FRONT_M = 0.35
STAGE3_APPROACH_CX_LO = 0.36
STAGE3_APPROACH_CX_HI = 0.58
MAX_STEPS_PER_PATH = 150
PERSISTENT_WP_ADVANCE_M = 0.8
PATH_FOLLOW_TURN_THRESH = np.radians(30.0)
# Stage1 path-follow 微避障：前向深度不足时侧转，不用 recovery
PATH_FOLLOW_FRONT_MIN_M = 0.40
PATH_FOLLOW_TURN_COMMIT_STEPS = 4
# u_exit doorway_seek 后：先左转进门，再跟 path（避免立刻 turn_right + 直行远离沙发）
STAGE1_DOORWAY_TRANSIT_TURN_STEPS = 12
STAGE1_DOORWAY_TRANSIT_ALIGN_DEG = 35.0
# 可行航向：在 desired_heading 附近 ±θ 选语义对齐 + 空闲深度最大者
STAGE1_FEASIBLE_YAW_OFFSETS_DEG = (-45, -30, -15, 0, 15, 30, 45)
STAGE1_FEASIBLE_FORWARD_OFFSET_DEG = 15.0
STAGE1_FEASIBLE_SEM_WEIGHT = 1.0
STAGE1_FEASIBLE_FREE_WEIGHT = 0.05
# Stage1 Locomotion 状态机（持续 commitment，禁止 escape 期间每帧重判 left/right）
STAGE1_LOCOMOTION_ENABLE = True
NAV_MODE_NORMAL = "NORMAL_FOLLOW"
NAV_MODE_ESCAPE_LEFT = "ESCAPE_LEFT"
NAV_MODE_ESCAPE_RIGHT = "ESCAPE_RIGHT"
NAV_MODE_ESCAPE_CORNER = "ESCAPE_CORNER"
NAV_MODE_RECOVERY_ROTATE = "RECOVERY_ROTATE"
STAGE1_ESCAPE_ENTER_NO_MOVE_STEPS = 12
STAGE1_ESCAPE_ENTER_MOVE_THRESH_M = 0.02
STAGE1_ESCAPE_ENTER_FRONT_M = 0.30
STAGE1_ESCAPE_GOAL_STALL_STEPS = 8
STAGE1_ESCAPE_GOAL_STALL_EPS_M = 0.04
STAGE1_ESCAPE_COMMIT_STEPS = 28
STAGE1_ESCAPE_TURN_BURST = 5
STAGE1_ESCAPE_FORWARD_BURST = 16
STAGE1_ESCAPE_RECOVERY_TURN_BURST = 8
STAGE1_ESCAPE_SIDE_MARGIN_M = 0.08
STAGE1_ESCAPE_FORWARD_MIN_M = 0.26
STAGE1_ESCAPE_FORWARD_CREEP_M = 0.20
STAGE1_ESCAPE_EXIT_DIST_IMPROVE_M = 0.80
STAGE1_ESCAPE_EXIT_FRONT_M = 1.20
STAGE1_ESCAPE_EXIT_HEADING_TOL_DEG = 50.0
STAGE1_ESCAPE_DEAD_END_M = 0.22
STAGE1_ESCAPE_MAX_STEPS = 220
STAGE1_ESCAPE_COOLDOWN_STEPS = 18
STAGE1_ESCAPE_STEER_ANGLES_DEG = (-90, -60, -30, 0, 30, 60, 90)
STAGE1_ESCAPE_SCORE_FREE_W = 0.06
STAGE1_ESCAPE_SCORE_PROGRESS_W = 0.45
# Bug2：撞障点→目标的 M 线；沿墙直到重新穿过 M 线且比撞障时更近目标
STAGE1_BUG2_ENABLE = True
STAGE1_BUG2_MLINE_TOL_M = 0.45
STAGE1_BUG2_LEAVE_IMPROVE_M = 0.12
STAGE1_BUG2_LEAVE_MIN_WALL_STEPS = 12
STAGE1_BUG2_LEAVE_MIN_WALL_MOVE_M = 0.25
STAGE1_BUG2_LEAVE_FRONT_M = 0.28
STAGE1_BUG2_LEAVE_MLINE_CROSS_FROM_M = 0.55
STAGE1_BUG2_LEAVE_SESSION_BEST_EPS_M = 0.06
STAGE1_BUG2_WALL_DIST_ABOVE_HIT_M = 0.05
STAGE1_BUG2_STRICT_REJOIN = True
# Stage1 协调器：path 主导；墙=escape 子程序；退出后 MLINE_TRANSIT 再回 path
STAGE1_COORD_FOLLOW_PATH = "FOLLOW_PATH"
STAGE1_COORD_GOAL_SEEK = "GOAL_SEEK"
STAGE1_COORD_FOLLOW_WALL = "FOLLOW_WALL"
STAGE1_COORD_MLINE_TRANSIT = "MLINE_TRANSIT"
# WALL 内语义重新捕获：目标可见且射线可达 → 退出绕墙回 GOAL_SEEK
STAGE1_WALL_SEMANTIC_REACQUIRE = True
STAGE1_WALL_REACQUIRE_MIN_CONF = 0.6
STAGE1_WALL_REACQUIRE_FRONT_MIN_M = 0.5
STAGE1_WALL_REACQUIRE_RAY_MARGIN_M = 0.35
STAGE1_WALL_REACQUIRE_MAX_RAY_OVER_EST_M = 0.35
STAGE1_WALL_REACQUIRE_MAX_GEODESIC_M = 3.2
STAGE1_WALL_SIDE_COMMIT = True
# 门口/目标方向开阔时禁止 enter WALL（避免 U 型出口左转进门时误绕墙）
STAGE1_WALL_ENTER_DOORWAY_GATE = True
STAGE1_WALL_ENTER_BLOCK_TOWARD_FREE_M = 1.2
STAGE1_WALL_DOORWAY_LEFT_OPEN_M = 2.5
STAGE1_WALL_DOORWAY_BEARING_MAX_DEG = -25.0
STAGE1_WALL_DOORWAY_TOWARD_FREE_M = 1.0
STAGE1_MLINE_TRANSIT_MAX_STEPS = 24
STAGE1_MLINE_TRANSIT_GOOD_STREAK = 3
STAGE1_MLINE_TRANSIT_IMPROVE_EPS_M = 0.03
STAGE1_MLINE_TRANSIT_FRONT_M = 0.28
STAGE1_MLINE_TRANSIT_PROGRESS_MIN_M = 0.02
STAGE1_REJOIN_REPLAN_PATH = True
STAGE1_ESCAPE_OPEN_ARMS_MARGIN_M = 0.05
STAGE1_ESCAPE_PEAK_DEPTH_WINDOW = 24
# 预探索 / local obstacle：保留 recovery + 较保守前向深度
LOCAL_OBSTACLE_FRONT_MIN_M = 0.38
NAV_DETECT_EVERY = 5
NAV_ARRIVE_CONF = 0.52
NAV_ARRIVE_DIST = 0.88
NAV_ARRIVE_STREAK = 3
NAV_ARRIVE_CENTER_LO = 0.35
NAV_ARRIVE_CENTER_HI = 0.65
NAV_MIN_SEMANTIC_SCORE = 0.35
NAV_MAX_CANDIDATE_REGIONS = 6
NAV_MAX_CANDIDATE_VIEWS = 8
NAV_VISUAL_SERVO_MAX_STEPS = 140
NAV_COARSE_SUCCESS_M = 1.2
NAV_VIEW_DISTANCE_PENALTY = 0.12
NAV_VIEW_REVISIT_PENALTY = 2.0
REGION_CENTER_REACH_M = 1.1
PRE_REGION_STUCK_STEPS = 36

# SemanticView 记忆 + 语义衰减
VIEW_CONF_THRESH = 0.38
VIEW_MAX_PER_LABEL = 64
VIEW_MERGE_DIST_M = 0.8
LABEL_CONFLICT_GROUPS = (frozenset({"bed", "sofa"}),)
LABEL_CONFLICT_WIN_MARGIN = 1.06
STAGE3_DISAMBIG_MARGIN = 1.04
STAGE3_BED_BULKY_FRONT_GAP = 0.55
VIEW_MERGE_YAW_RAD = math.radians(25.0)
RECOVER_VIEW_MAX_TURNS = 24
RECOVER_YAW_TOL_RAD = math.radians(12.0)
RECOVER_REL_ANGLE_TOL_RAD = math.radians(8.0)
SEM_DECAY_TAU_STEPS = 800.0
SEM_FAIL_PENALTY = 0.45
SEM_FORGET_FAIL_COUNT = 3

# ─── 系统探索（navmesh 采样全覆盖）────────────────────────────
SYSTEMATIC_N_SAMPLES = 80
SYSTEMATIC_SAMPLE_TRIES = 400
SYSTEMATIC_MIN_SAMPLE_DIST_M = 1.8
SYSTEMATIC_LEG_MAX_STEPS = 2400
SYSTEMATIC_DETECT_EVERY = 4
SYSTEMATIC_SPIN_TURNS = 24
SYSTEMATIC_SEM_MIN_WEIGHT = 0.15
SYSTEMATIC_GIF_EVERY_N = 10

# 仅系统探索/建图结束时输出（不含任务导航阶段）
TOPO_VIZ_EXPLORE_PATH = "/autodl-tmp/topo_region_graph_explore.png"

# ─── Action Memory / 短期局部记忆 ───────────────────────────
ACTION_MEMORY_ACTIONS = 12
ACTION_MEMORY_POSITIONS = 20
LOCAL_REVISIT_RADIUS_M = 2.0
LOCAL_REVISIT_PENALTY_MAX = 4.0
STUCK_MOVE_THRESH_M = 0.03
STUCK_TRIGGER_STEPS = 10
RECOVERY_HARD_TURN_STEPS = 6
RECOVERY_FORWARD_BURST = 3
OSCILLATION_SWITCH_RATE_THRESH = 0.7
OSCILLATION_HOLD_TURN_STEPS = 3
ANCHOR_NEAR_M = 1.25
ANCHOR_FAR_M = 2.0
ANCHOR_MERGE_DIST_M = 0.6
ANCHOR_MAX_PER_TYPE = 48
PLACE_TYPES = ("doorway", "corridor", "open_area", "room_center")
FRONTIER_ANCHOR_TYPES = ("doorway", "corridor")

# 检测形状先验（预探索多类检测用，不做 tracker）
SHAPE_PRIORS = {
    "sofa": {"min_area": 0.012, "max_area": 0.45, "aspect": (1.0, 5.5)},
    "door": {"min_area": 0.012, "max_area": 0.48, "aspect": (0.2, 0.9)},
    "table": {"min_area": 0.008, "max_area": 0.48, "aspect": (0.35, 4.5)},
    "chair seat": {"min_area": 0.005, "max_area": 0.38, "aspect": (0.25, 3.5)},
    "bed": {"min_area": 0.012, "max_area": 0.52, "aspect": (0.5, 4.8)},
}

# ─── 仿真 ─────────────────────────────────────────────────
def place_agent_on_navmesh(sim, random_spawn=False, living_room_spawn=True, agent_idx=0):
    pathfinder = sim.pathfinder
    agent = sim.agents[agent_idx]
    state = agent.state
    if pathfinder.is_loaded:
        if random_spawn:
            if living_room_spawn:
                x, z = LIVING_ROOM_SPAWN_XZ
                raw = np.array([x, 0.0, z], dtype=np.float32)
                state.position = pathfinder.snap_point(raw)
                drift = float(
                    np.hypot(state.position[0] - x, state.position[2] - z)
                )
                if drift > 0.05:
                    print(
                        f"[SPAWN] snap 漂移 {drift:.2f}m: "
                        f"请求=[{x:.2f},{z:.2f}] → "
                        f"实际=[{state.position[0]:.2f},{state.position[2]:.2f}]"
                    )
            else:
                state.position = pathfinder.get_random_navigable_point()
        else:
            state.position = pathfinder.snap_point(state.position)
        agent.set_state(state)
        if STAGE1_SPAWN_ALIGN_YAW and pathfinder.is_navigable(state.position):
            tx, tz = STAGE1_SPAWN_YAW_TARGET_XZ
            n = align_agent_yaw_toward_xz(agent, tx, tz, max_turns=STAGE1_PREALIGN_MAX_TURNS)
            yaw = yaw_from_rotation(agent.state.rotation)
            err, _ = heading_err_deg(yaw, yaw_toward_xz(agent.state.position, tx, tz))
            print(
                f"[SPAWN] 朝向对齐 → ({tx:.2f},{tz:.2f}) "
                f"turns={n} residual={err:.1f}°"
            )
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
            "move_forward", habitat_sim.agent.ActuationSpec(amount=FORWARD_STEP_M)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=15.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=15.0)
        ),
    }

    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_cfg]))
    if sim.pathfinder.is_loaded:
        pos, ok = place_agent_on_navmesh(sim, random_spawn=random_spawn, living_room_spawn=living_room_spawn)
        print(f"Navmesh 已加载 | 面积 {sim.pathfinder.navigable_area:.1f} m² | 起点可通行: {ok}")
        print(f"Agent 初始位置: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
    return sim

# ─── 工具 ─────────────────────────────────────────────────
def yaw_from_rotation(rotation):
    if hasattr(rotation, "w"):
        qw, qx, qy, qz = float(rotation.w), float(rotation.x), float(rotation.y), float(rotation.z)
    else:
        qx, qy, qz, qw = rotation
    siny = 2.0 * (qw * qy + qx * qz)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(np.arctan2(siny, cosy))

def fill_depth(depth):
    depth_vis = depth.copy()
    mask = (~np.isfinite(depth_vis) | (depth_vis <= 0.05)).astype(np.uint8)
    if mask.sum() == 0:
        return depth_vis
    valid = depth_vis[np.isfinite(depth_vis) & (depth_vis > 0)]
    if len(valid) == 0:
        return depth_vis
    max_d = float(np.percentile(valid, 95))
    depth_norm = np.clip(np.nan_to_num(depth_vis / max_d, nan=0.0), 0, 1)
    depth_uint8 = (depth_norm * 255).astype(np.uint8)
    filled_uint8 = cv2.inpaint(depth_uint8, mask, inpaintRadius=5, flags=cv2.INPAINT_NS)
    filled = filled_uint8.astype(np.float32) / 255.0 * max_d
    result = depth_vis.copy()
    result[mask.astype(bool)] = filled[mask.astype(bool)]
    return result

def measure_depth_probes(depth):
    left = depth[200:280, 80:180]
    front = depth[200:280, 280:360]
    right = depth[200:280, 460:560]

    def _p(patch):
        v = patch[np.isfinite(patch) & (patch > 0.05)]
        return float(np.percentile(v, 30)) if len(v) else 5.0

    return _p(left), _p(front), _p(right)

def depth_free_space_at_rel_deg(left_d, front_d, right_d, rel_deg):
    """相对当前朝向的方位角（度）：0=正前，负=左，正=右。"""
    rel_deg = float(rel_deg)
    if abs(rel_deg) <= 7.5:
        return float(front_d)
    if rel_deg < -7.5:
        if rel_deg <= -45.0:
            return float(left_d)
        t = (abs(rel_deg) - 7.5) / 37.5
        return (1.0 - t) * float(front_d) + t * float(left_d)
    if rel_deg >= 45.0:
        return float(right_d)
    t = (rel_deg - 7.5) / 37.5
    return (1.0 - t) * float(front_d) + t * float(right_d)

def stage1_score_heading_candidate(mix_yaw, offset_deg, yaw, left_d, front_d, right_d):
    """候选航向 = mix_yaw + offset；返回 (score, free_d, rel_to_current_deg, candidate_yaw)。"""
    candidate_yaw = float(mix_yaw) + math.radians(float(offset_deg))
    rel_to_current_deg = math.degrees(
        (candidate_yaw - float(yaw) + np.pi) % (2 * np.pi) - np.pi
    )
    free_d = depth_free_space_at_rel_deg(left_d, front_d, right_d, rel_to_current_deg)
    sem = max(0.0, 1.0 - abs(float(offset_deg)) / 45.0)
    score = STAGE1_FEASIBLE_SEM_WEIGHT * sem + STAGE1_FEASIBLE_FREE_WEIGHT * free_d
    return score, free_d, rel_to_current_deg, candidate_yaw

def stage1_pick_best_feasible_heading(yaw, mix_yaw, left_d, front_d, right_d):
    """在 desired_heading 邻域内选 score 最高的可行航向。"""
    best = None
    for off in STAGE1_FEASIBLE_YAW_OFFSETS_DEG:
        sc, free_d, rel_cur, cand_yaw = stage1_score_heading_candidate(
            mix_yaw, off, yaw, left_d, front_d, right_d
        )
        if best is None or sc > best[0]:
            best = (sc, off, free_d, rel_cur, cand_yaw)
    return best

def stage1_micro_obstacle_propose(
    yaw,
    mix_yaw,
    left_d,
    front_d,
    right_d,
    state,
    front_min_m=None,
):
    """
    Stage1 微避障（非 recovery）：
      1) |heading_err| 大 → 转向 mix_yaw
      2) 对准且前向通畅 → forward
      3) 前向受阻 → 可行航向搜索；仍不足 → 选更空的一侧
    返回 (action, intent, meta_extra)。
    """
    if front_min_m is None:
        front_min_m = PATH_FOLLOW_FRONT_MIN_M
    diff = (float(mix_yaw) - float(yaw) + np.pi) % (2 * np.pi) - np.pi
    meta_extra = {
        "left_d": round(left_d, 3),
        "right_d": round(right_d, 3),
    }
    desired_yaw = float(mix_yaw)
    if abs(diff) > PATH_FOLLOW_TURN_THRESH:
        action = state.start_turn_commit("turn_left" if diff > 0 else "turn_right")
        meta_extra["feasible_offset_deg"] = 0.0
        meta_extra["chosen_yaw_rad"] = desired_yaw
        return action, "turn_heading", meta_extra
    if float(front_d) >= float(front_min_m):
        state.clear_commit()
        meta_extra["feasible_offset_deg"] = 0.0
        meta_extra["feasible_free_d"] = round(front_d, 3)
        meta_extra["chosen_yaw_rad"] = desired_yaw
        return "move_forward", "forward", meta_extra

    best = stage1_pick_best_feasible_heading(yaw, mix_yaw, left_d, front_d, right_d)
    _sc, off_deg, free_d, rel_cur, cand_yaw = best
    meta_extra["feasible_offset_deg"] = round(float(off_deg), 1)
    meta_extra["feasible_free_d"] = round(float(free_d), 3)
    meta_extra["feasible_rel_deg"] = round(float(rel_cur), 1)
    chosen_yaw = float(cand_yaw)

    if (
        float(free_d) >= float(front_min_m)
        and abs(float(off_deg)) <= STAGE1_FEASIBLE_FORWARD_OFFSET_DEG
    ):
        state.clear_commit()
        meta_extra["chosen_yaw_rad"] = chosen_yaw
        return "move_forward", "feasible_forward", meta_extra

    if float(free_d) > float(front_d) + 0.03:
        turn_diff = (chosen_yaw - float(yaw) + np.pi) % (2 * np.pi) - np.pi
        action = state.start_turn_commit(
            "turn_left" if turn_diff > 0 else "turn_right"
        )
        meta_extra["chosen_yaw_rad"] = chosen_yaw
        return action, "feasible_turn", meta_extra

    if left_d > right_d:
        chosen_yaw = float(yaw) + math.radians(30.0)
    else:
        chosen_yaw = float(yaw) - math.radians(30.0)
    action = state.start_turn_commit(
        "turn_left" if left_d > right_d else "turn_right"
    )
    meta_extra["chosen_yaw_rad"] = chosen_yaw
    return action, "blocked_side", meta_extra

def stage1_bug2_dist_to_mline(pos, hit_xz, goal_xz):
    """当前位置到 M 线（撞障点 hit → 目标 goal）线段的垂直距离（xz 平面）。"""
    px, pz = float(pos[0]), float(pos[2])
    ax, az = float(hit_xz[0]), float(hit_xz[1])
    bx, bz = float(goal_xz[0]), float(goal_xz[1])
    abx, abz = bx - ax, bz - az
    ab2 = abx * abx + abz * abz
    if ab2 < 1e-8:
        return float(np.hypot(px - ax, pz - az))
    t = ((px - ax) * abx + (pz - az) * abz) / ab2
    t = max(0.0, min(1.0, t))
    cx, cz = ax + t * abx, az + t * abz
    return float(np.hypot(px - cx, pz - cz))

def stage1_bug2_should_leave(pos, dist_goal, hit_xz, dist_at_hit, goal_xz):
    """
    Bug2 沿墙离开条件：重新靠近 M 线，且到目标距离比进入沿墙时更近（≥ STAGE1_BUG2_LEAVE_IMPROVE_M）。
    旧接口保留；escape 退出请用 stage1_bug2_can_rejoin。
    """
    ok, _ = stage1_bug2_can_rejoin(
        pos,
        dist_goal,
        hit_xz,
        dist_at_hit,
        goal_xz,
        front_d=STAGE1_BUG2_LEAVE_FRONT_M,
        escape_steps=STAGE1_BUG2_LEAVE_MIN_WALL_STEPS,
        escape_cumulative_move=STAGE1_BUG2_LEAVE_MIN_WALL_MOVE_M,
        session_best_dist=dist_goal,
        saw_mline_far=True,
    )
    return ok

def stage1_bug2_progress_along_yaw(pos, yaw, goal_xz, step_m=None):
    """预测沿 yaw 前进 step_m 后距 goal_xz 的改善量（>0 更近）。"""
    if goal_xz is None:
        return 0.0
    if step_m is None:
        step_m = float(FORWARD_STEP_M)
    px, pz = float(pos[0]), float(pos[2])
    gx, gz = float(goal_xz[0]), float(goal_xz[1])
    cur = float(np.hypot(gx - px, gz - pz))
    y = float(yaw)
    pred_x = px + math.sin(y) * float(step_m)
    pred_z = pz + math.cos(y) * float(step_m)
    pred = float(np.hypot(gx - pred_x, gz - pred_z))
    return cur - pred

def stage1_bug2_can_rejoin(
    pos,
    dist_target,
    hit_xz,
    dist_at_hit,
    goal_xz,
    front_d,
    escape_steps,
    escape_cumulative_move,
    session_best_dist=None,
    saw_mline_far=False,
):
    """
    沿墙结束、进入 MLINE_TRANSIT 的闸门（防 spawn 假阳性、防 dist 变差仍退出）。
    返回 (ok, reason)。
    """
    if pos is None or hit_xz is None or goal_xz is None:
        return False, "missing_geom"
    if dist_target is None or dist_at_hit is None:
        return False, "missing_dist"
    if int(escape_steps) < int(STAGE1_BUG2_LEAVE_MIN_WALL_STEPS):
        return False, "min_steps"
    if float(escape_cumulative_move) < float(STAGE1_BUG2_LEAVE_MIN_WALL_MOVE_M):
        return False, "min_move"
    if float(front_d) < float(STAGE1_BUG2_LEAVE_FRONT_M):
        return False, "front_low"
    if float(dist_target) > float(dist_at_hit) + STAGE1_BUG2_WALL_DIST_ABOVE_HIT_M:
        return False, "above_hit"
    if session_best_dist is not None:
        if float(dist_target) > float(session_best_dist) + STAGE1_BUG2_LEAVE_SESSION_BEST_EPS_M:
            return False, "above_best"
    if not saw_mline_far:
        return False, "no_mline_depart"
    mline_d = stage1_bug2_dist_to_mline(pos, hit_xz, goal_xz)
    if mline_d > float(STAGE1_BUG2_MLINE_TOL_M):
        return False, "mline_far"
    improve = float(dist_at_hit) - float(dist_target)
    if improve < float(STAGE1_BUG2_LEAVE_IMPROVE_M):
        return False, "improve_low"
    return True, "bug2_rejoin"

class Stage1Coordinator:
    """Stage1 顶层：GOAL_SEEK(path)；escape=FOLLOW_WALL；退出墙后 MLINE_TRANSIT。"""

    def __init__(self):
        self.mode = STAGE1_COORD_GOAL_SEEK
        self.transit_steps = 0
        self.transit_improve_streak = 0
        self.last_rejoin_reason = ""
        self.doorway_transit_rem = 0

    def in_mline_transit(self):
        return self.mode == STAGE1_COORD_MLINE_TRANSIT

    def in_doorway_transit(self):
        return int(self.doorway_transit_rem) > 0

    def begin_doorway_transit(self, steps=None):
        self.doorway_transit_rem = int(
            steps if steps is not None else STAGE1_DOORWAY_TRANSIT_TURN_STEPS
        )

    def tick_doorway_transit(self):
        if self.doorway_transit_rem > 0:
            self.doorway_transit_rem -= 1

    def end_doorway_transit(self):
        self.doorway_transit_rem = 0

    def begin_goal_seek(self):
        self.mode = STAGE1_COORD_GOAL_SEEK

    def begin_wall_follow(self):
        self.mode = STAGE1_COORD_FOLLOW_WALL

    def begin_mline_transit(self, reason="bug2_rejoin"):
        self.mode = STAGE1_COORD_MLINE_TRANSIT
        self.transit_steps = 0
        self.transit_improve_streak = 0
        self.last_rejoin_reason = str(reason)

    def end_mline_transit(self):
        self.mode = STAGE1_COORD_GOAL_SEEK
        self.transit_steps = 0
        self.transit_improve_streak = 0

    def tick_transit_improve(self, dist_target, prev_dist_target):
        if (
            dist_target is not None
            and prev_dist_target is not None
            and float(dist_target) < float(prev_dist_target) - STAGE1_MLINE_TRANSIT_IMPROVE_EPS_M
        ):
            self.transit_improve_streak += 1
        else:
            self.transit_improve_streak = 0

    def transit_should_finish(self):
        return self.transit_improve_streak >= int(STAGE1_MLINE_TRANSIT_GOOD_STREAK)

    def transit_expired(self):
        return self.transit_steps >= int(STAGE1_MLINE_TRANSIT_MAX_STEPS)

class Stage1LocomotionState:
    """
    Locomotion 状态机：NORMAL_FOLLOW | ESCAPE_LEFT/RIGHT | ESCAPE_CORNER | RECOVERY_ROTATE。
    Escape 内用 burst commitment（连续 turn / forward），禁止每帧重判 left/right。
    """

    def __init__(self):
        self.nav_mode = NAV_MODE_NORMAL
        self.frozen_desired_yaw = None
        self.no_move_streak = 0
        self.goal_stall_streak = 0
        self.escape_start_dist = None
        self.escape_total_steps = 0
        self.commitment_remaining = 0
        self.phase = ""
        self.phase_remaining = 0
        self.wall_side = None
        self.phase_action_bias = None
        self.escape_cooldown = 0
        self.escape_cumulative_move = 0.0
        self.bug2_hit_xz = None
        self.bug2_mline_goal_xz = None
        self.bug2_boundary_min_dist = None
        self.peak_left_d = 0.0
        self.peak_right_d = 0.0
        self.last_exit_reason = ""
        self.bug2_peak_mline_d = 0.0
        self.bug2_saw_mline_far = False
        self.wall_side_session = None

    @property
    def in_escape(self):
        return self.nav_mode != NAV_MODE_NORMAL

    def update_streaks(self, moved_last, dist_goal, prev_dist_goal, left_d=0.0, right_d=0.0):
        if self.escape_cooldown > 0:
            self.escape_cooldown -= 1
        if not self.in_escape:
            self.peak_left_d = max(float(self.peak_left_d), float(left_d))
            self.peak_right_d = max(float(self.peak_right_d), float(right_d))
        if float(moved_last) < STAGE1_ESCAPE_ENTER_MOVE_THRESH_M:
            self.no_move_streak += 1
        else:
            self.no_move_streak = 0
            self.peak_left_d = float(left_d)
            self.peak_right_d = float(right_d)
        if prev_dist_goal is not None and dist_goal is not None:
            if float(dist_goal) >= float(prev_dist_goal) - STAGE1_ESCAPE_GOAL_STALL_EPS_M:
                self.goal_stall_streak += 1
            else:
                self.goal_stall_streak = 0
        if self.in_escape:
            self.escape_cumulative_move += float(moved_last)

    def should_trigger_escape(
        self,
        front_d,
        pos=None,
        target_xz=None,
        yaw=None,
        left_d=0.0,
        right_d=0.0,
    ):
        if not STAGE1_LOCOMOTION_ENABLE:
            return False
        if self.in_escape or self.escape_cooldown > 0:
            return False
        no_move = self.no_move_streak >= STAGE1_ESCAPE_ENTER_NO_MOVE_STEPS
        front_stall = (
            float(front_d) < STAGE1_ESCAPE_ENTER_FRONT_M
            and self.goal_stall_streak >= STAGE1_ESCAPE_GOAL_STALL_STEPS
        )
        return no_move or front_stall

    def _pick_escape_side(
        self,
        left_d,
        right_d,
        front_d,
        yaw=None,
        goal_bearing_yaw=None,
        pos=None,
        goal_xz=None,
    ):
        # 进入瞬间深度可能已因转向变小：用 streak 窗口内峰值
        left_u = max(float(left_d), float(self.peak_left_d))
        right_u = max(float(right_d), float(self.peak_right_d))
        arms_max = max(left_u, right_u)
        if arms_max > float(front_d) + STAGE1_ESCAPE_OPEN_ARMS_MARGIN_M:
            if left_u >= right_u:
                return NAV_MODE_ESCAPE_LEFT, "left"
            return NAV_MODE_ESCAPE_RIGHT, "right"
        if pos is not None and goal_xz is not None:
            px, pz = float(pos[0]), float(pos[2])
            gx, gz = float(goal_xz[0]), float(goal_xz[1])
            dx, dz = gx - px, gz - pz
            dist_pg = float(np.hypot(dx, dz))
            if dist_pg > 0.05:
                bearing = math.atan2(dx, dz)
                to_gx, to_gz = dx / dist_pg, dz / dist_pg
                along_l = (
                    math.sin(bearing + math.pi / 2),
                    math.cos(bearing + math.pi / 2),
                )
                along_r = (
                    math.sin(bearing - math.pi / 2),
                    math.cos(bearing - math.pi / 2),
                )
                dot_l = along_l[0] * to_gx + along_l[1] * to_gz
                dot_r = along_r[0] * to_gx + along_r[1] * to_gz
                if dot_l >= dot_r + 0.05:
                    return NAV_MODE_ESCAPE_LEFT, "left"
                if dot_r >= dot_l + 0.05:
                    return NAV_MODE_ESCAPE_RIGHT, "right"
        if yaw is not None and goal_bearing_yaw is not None:
            best_score = -1e9
            best_angle = 0
            for a in STAGE1_ESCAPE_STEER_ANGLES_DEG:
                free = depth_free_space_at_rel_deg(left_d, front_d, right_d, a)
                cand_yaw = float(yaw) + math.radians(float(a))
                progress = math.cos(cand_yaw - float(goal_bearing_yaw))
                if progress < 0.15:
                    continue
                score = (
                    STAGE1_ESCAPE_SCORE_FREE_W * free
                    + STAGE1_ESCAPE_SCORE_PROGRESS_W * progress
                )
                if score > best_score:
                    best_score = score
                    best_angle = int(a)
            if best_score > -1e8:
                if best_angle <= 0:
                    return NAV_MODE_ESCAPE_LEFT, "left"
                return NAV_MODE_ESCAPE_RIGHT, "right"
        if float(left_d) > float(right_d) + STAGE1_ESCAPE_SIDE_MARGIN_M:
            return NAV_MODE_ESCAPE_LEFT, "left"
        if float(right_d) > float(left_d) + STAGE1_ESCAPE_SIDE_MARGIN_M:
            return NAV_MODE_ESCAPE_RIGHT, "right"
        if float(left_d) >= float(right_d):
            return NAV_MODE_ESCAPE_LEFT, "left"
        return NAV_MODE_ESCAPE_RIGHT, "right"

    def _start_burst(self, phase, n):
        self.phase = phase
        self.phase_remaining = max(0, int(n))

    def enter_escape(
        self,
        desired_yaw_rad,
        left_d,
        right_d,
        front_d,
        dist_goal,
        yaw=None,
        goal_bearing_yaw=None,
        pos=None,
        goal_xz=None,
    ):
        if STAGE1_WALL_SIDE_COMMIT and self.wall_side_session is not None:
            wall = self.wall_side_session
            mode = (
                NAV_MODE_ESCAPE_LEFT
                if wall == "left"
                else NAV_MODE_ESCAPE_RIGHT
            )
        else:
            mode, wall = self._pick_escape_side(
                left_d,
                right_d,
                front_d,
                yaw=yaw,
                goal_bearing_yaw=goal_bearing_yaw,
                pos=pos,
                goal_xz=goal_xz,
            )
            if STAGE1_WALL_SIDE_COMMIT:
                self.wall_side_session = wall
        self.nav_mode = mode
        self.wall_side = wall
        self.frozen_desired_yaw = float(desired_yaw_rad)
        self.escape_start_dist = float(dist_goal) if dist_goal is not None else None
        self.bug2_boundary_min_dist = self.escape_start_dist
        if STAGE1_BUG2_ENABLE and pos is not None:
            self.bug2_hit_xz = (float(pos[0]), float(pos[2]))
        else:
            self.bug2_hit_xz = None
        if STAGE1_BUG2_ENABLE and goal_xz is not None:
            self.bug2_mline_goal_xz = (float(goal_xz[0]), float(goal_xz[1]))
        else:
            self.bug2_mline_goal_xz = None
        self.bug2_peak_mline_d = 0.0
        self.bug2_saw_mline_far = False
        if (
            STAGE1_BUG2_ENABLE
            and pos is not None
            and self.bug2_hit_xz is not None
            and self.bug2_mline_goal_xz is not None
        ):
            md0 = stage1_bug2_dist_to_mline(
                pos, self.bug2_hit_xz, self.bug2_mline_goal_xz
            )
            self.bug2_peak_mline_d = float(md0)
            if md0 > float(STAGE1_BUG2_LEAVE_MLINE_CROSS_FROM_M):
                self.bug2_saw_mline_far = True
        self.escape_total_steps = 0
        self.escape_cumulative_move = 0.0
        self.commitment_remaining = int(STAGE1_ESCAPE_COMMIT_STEPS)
        self._start_burst("turn", STAGE1_ESCAPE_TURN_BURST)

    def enter_corner_recovery(self):
        """死角：沿原墙反方向转出，不重新做 left/right 判定。"""
        self.nav_mode = NAV_MODE_RECOVERY_ROTATE
        if self.wall_side == "left":
            self.phase_action_bias = "turn_right"
        else:
            self.phase_action_bias = "turn_left"
        self._start_burst("turn", STAGE1_ESCAPE_RECOVERY_TURN_BURST)

    def _resume_wall_follow_after_recovery(self):
        if self.wall_side == "left":
            self.nav_mode = NAV_MODE_ESCAPE_LEFT
        else:
            self.nav_mode = NAV_MODE_ESCAPE_RIGHT
        self._start_burst("forward", STAGE1_ESCAPE_FORWARD_BURST)

    def exit_escape(self):
        self.nav_mode = NAV_MODE_NORMAL
        self.frozen_desired_yaw = None
        self.escape_start_dist = None
        self.escape_total_steps = 0
        self.commitment_remaining = 0
        self.phase = ""
        self.phase_remaining = 0
        self.wall_side = None
        self.phase_action_bias = None
        self.escape_cumulative_move = 0.0
        self.bug2_hit_xz = None
        self.bug2_mline_goal_xz = None
        self.bug2_boundary_min_dist = None
        self.peak_left_d = 0.0
        self.peak_right_d = 0.0
        self.escape_cooldown = int(STAGE1_ESCAPE_COOLDOWN_STEPS)
        self.bug2_peak_mline_d = 0.0
        self.bug2_saw_mline_far = False
        self.wall_side_session = None

    def update_bug2_mline_track(self, pos, goal_xz):
        if (
            not self.in_escape
            or pos is None
            or self.bug2_hit_xz is None
            or goal_xz is None
        ):
            return
        md = stage1_bug2_dist_to_mline(pos, self.bug2_hit_xz, goal_xz)
        self.bug2_peak_mline_d = max(float(self.bug2_peak_mline_d), float(md))
        if md > float(STAGE1_BUG2_LEAVE_MLINE_CROSS_FROM_M):
            self.bug2_saw_mline_far = True

    def should_exit_escape(
        self,
        front_d,
        dist_goal,
        yaw,
        frozen_yaw,
        pos=None,
        goal_xz=None,
        session_best_dist=None,
    ):
        """沿墙退出：优先严格 bug2_rejoin；否则 max_steps。easy exit 在 STRICT 时关闭。"""
        if not self.in_escape:
            return False, ""
        self.escape_total_steps += 1
        if self.escape_total_steps > STAGE1_ESCAPE_MAX_STEPS:
            return True, "max_steps"
        dist_ref = dist_goal
        if dist_ref is not None and self.bug2_boundary_min_dist is not None:
            self.bug2_boundary_min_dist = min(
                float(self.bug2_boundary_min_dist), float(dist_ref)
            )
        gxz = goal_xz if goal_xz is not None else self.bug2_mline_goal_xz
        if pos is not None and gxz is not None:
            self.update_bug2_mline_track(pos, gxz)
        if (
            STAGE1_BUG2_ENABLE
            and pos is not None
            and self.bug2_hit_xz is not None
            and gxz is not None
            and self.escape_start_dist is not None
            and dist_ref is not None
        ):
            ok, why = stage1_bug2_can_rejoin(
                pos,
                dist_ref,
                self.bug2_hit_xz,
                self.escape_start_dist,
                gxz,
                front_d,
                self.escape_total_steps,
                self.escape_cumulative_move,
                session_best_dist=session_best_dist,
                saw_mline_far=self.bug2_saw_mline_far,
            )
            if ok:
                return True, why
            if float(dist_ref) > float(self.escape_start_dist) + 0.20:
                return False, ""
        if STAGE1_BUG2_STRICT_REJOIN:
            return False, ""
        if self.escape_start_dist is not None and dist_ref is not None:
            if (
                float(self.escape_start_dist) - float(dist_ref)
                >= STAGE1_ESCAPE_EXIT_DIST_IMPROVE_M
            ):
                return True, "dist_improve"
        if float(front_d) >= STAGE1_ESCAPE_EXIT_FRONT_M and frozen_yaw is not None:
            err_deg, _ = heading_err_deg(yaw, frozen_yaw)
            if abs(err_deg) <= STAGE1_ESCAPE_EXIT_HEADING_TOL_DEG:
                if self.escape_cumulative_move >= 0.35:
                    if (
                        self.escape_start_dist is not None
                        and dist_ref is not None
                        and float(dist_ref)
                        <= float(self.escape_start_dist)
                        - STAGE1_BUG2_LEAVE_IMPROVE_M
                    ):
                        return True, "front_clear"
        return False, ""

    def _advance_phase(self):
        if self.nav_mode == NAV_MODE_RECOVERY_ROTATE:
            self._resume_wall_follow_after_recovery()
            return
        if self.phase == "turn":
            self._start_burst("forward", STAGE1_ESCAPE_FORWARD_BURST)
        else:
            self._start_burst("turn", STAGE1_ESCAPE_TURN_BURST)

    def _wall_follow_turn(self):
        return (
            "turn_left"
            if self.nav_mode == NAV_MODE_ESCAPE_LEFT
            else "turn_right"
        )

    def tick_escape_action(self, left_d, front_d, right_d):
        """Commitment 内：固定 burst，禁止每帧 left/right 重判。"""
        self.commitment_remaining = max(0, self.commitment_remaining - 1)

        if stage1_is_escape_dead_end(left_d, front_d, right_d):
            if self.nav_mode != NAV_MODE_RECOVERY_ROTATE:
                self.nav_mode = NAV_MODE_ESCAPE_CORNER
                self.enter_corner_recovery()

        if self.nav_mode == NAV_MODE_RECOVERY_ROTATE:
            self.phase_remaining = max(0, self.phase_remaining - 1)
            if self.phase_remaining <= 0:
                self._resume_wall_follow_after_recovery()
                return self._wall_follow_turn(), "recovery_done_turn"
            bias = self.phase_action_bias or "turn_left"
            return bias, "recovery_rotate"

        if self.phase_remaining <= 0:
            self._advance_phase()

        if self.phase == "forward":
            self.phase_remaining = max(0, self.phase_remaining - 1)
            if float(front_d) >= STAGE1_ESCAPE_FORWARD_MIN_M:
                return "move_forward", "escape_burst_forward"
            if float(front_d) >= STAGE1_ESCAPE_FORWARD_CREEP_M:
                return "move_forward", "escape_burst_creep"
            return self._wall_follow_turn(), "escape_burst_unstick"

        self.phase_remaining = max(0, self.phase_remaining - 1)
        return self._wall_follow_turn(), "escape_burst_turn"

def stage1_is_escape_dead_end(left_d, front_d, right_d):
    return (
        float(front_d) < STAGE1_ESCAPE_DEAD_END_M
        and float(left_d) < STAGE1_ESCAPE_DEAD_END_M
        and float(right_d) < STAGE1_ESCAPE_DEAD_END_M
    )

def forward_offset(yaw, step_m):
    return step_m * np.sin(yaw), step_m * np.cos(yaw)

def quaternion_from_yaw(yaw):
    half = float(yaw) * 0.5
    return np.quaternion(np.cos(half), 0.0, np.sin(half), 0.0)

def yaw_toward_xz(pos, x, z):
    dx = float(x) - float(pos[0])
    dz = float(z) - float(pos[2])
    return float(np.arctan2(dx, dz))

def heading_err_deg(yaw, target_yaw):
    diff = (float(target_yaw) - float(yaw) + np.pi) % (2 * np.pi) - np.pi
    return float(math.degrees(diff)), diff

def set_agent_yaw(agent, yaw):
    st = agent.state
    st.rotation = quaternion_from_yaw(yaw)
    agent.set_state(st)

def align_agent_yaw_toward_xz(agent, x, z, max_turns=STAGE1_PREALIGN_MAX_TURNS, pf_act=None):
    """将 agent 朝向 (x,z)，返回执行的转向步数。"""
    pos = agent.state.position
    return align_agent_yaw_rad(
        agent, yaw_toward_xz(pos, x, z), max_turns=max_turns, pf_act=pf_act
    )

def align_agent_yaw_rad(agent, target_yaw, max_turns=STAGE1_PREALIGN_MAX_TURNS, pf_act=None):
    """将 agent 朝向给定世界 yaw (rad)，返回执行的转向步数。"""
    turns = 0
    while turns < max_turns:
        yaw = yaw_from_rotation(agent.state.rotation)
        err_deg, diff = heading_err_deg(yaw, target_yaw)
        if abs(err_deg) <= math.degrees(PATH_FOLLOW_TURN_THRESH):
            break
        action = "turn_left" if diff > 0 else "turn_right"
        if pf_act is not None:
            pf_act(action)
        else:
            agent.act(action)
        turns += 1
    return turns

def semantic_aim_yaw_from_view(view):
    """Stage2 同款：建图时「看到目标」的世界朝向。"""
    if view is None:
        return None
    return float(view.yaw) + float(getattr(view, "target_rel_angle", 0.0))

def stage1_semantic_alpha(dist_target):
    """距 target 越远 alpha 越小（更信 path waypoint）。"""
    if dist_target is None:
        return STAGE1_SEMANTIC_ALPHA_FAR
    d = float(dist_target)
    d_far = float(STAGE1_SEMANTIC_ALPHA_DIST_FAR)
    d_near = float(STAGE1_SEMANTIC_ALPHA_DIST_NEAR)
    if d >= d_far:
        return STAGE1_SEMANTIC_ALPHA_FAR
    if d <= d_near:
        return STAGE1_SEMANTIC_ALPHA_NEAR
    t = (d_far - d) / max(d_far - d_near, 1e-6)
    return STAGE1_SEMANTIC_ALPHA_FAR + t * (
        STAGE1_SEMANTIC_ALPHA_NEAR - STAGE1_SEMANTIC_ALPHA_FAR
    )

def angle_mix_yaw(wp_yaw, guide_yaw, alpha):
    """圆周插值混合 waypoint 航向与 guide 航向（bearing 或 view）。"""
    a = float(np.clip(alpha, 0.0, 1.0))
    s = (1.0 - a) * math.sin(float(wp_yaw)) + a * math.sin(float(guide_yaw))
    c = (1.0 - a) * math.cos(float(wp_yaw)) + a * math.cos(float(guide_yaw))
    return float(math.atan2(s, c))

def stage1_bearing_yaw(pos, target_xz):
    """当前位置指向 target_xz 的方位角。"""
    if target_xz is None:
        return None
    return yaw_toward_xz(pos, float(target_xz[0]), float(target_xz[1]))

def stage1_near_open_dist_cap():
    """空旷近邻距离上限 = NEAR_OPEN_MAX_D + SLACK（覆盖 ~2.00m 贴墙最近）。"""
    if not STAGE1_NEAR_OPEN_AREA_GATE:
        return float(NAV_COARSE_SUCCESS_M)
    return float(STAGE1_NEAR_OPEN_MAX_D) + float(STAGE1_NEAR_OPEN_DIST_SLACK_M)

def stage1_coarse_success(d, exit_reason):
    """Stage1 粗导航是否算进入目标邻域（含空旷放宽）。"""
    d = float(d)
    if d <= float(NAV_COARSE_SUCCESS_M):
        return True
    if (
        STAGE1_NEAR_OPEN_AREA_GATE
        and d <= stage1_near_open_dist_cap()
        and str(exit_reason)
        in (
            "neighborhood",
            "neighborhood_open",
            "already_in_neighborhood",
        )
    ):
        return True
    return False

def stage1_near_dist_limit(neighborhood_m):
    """近邻检测距离上限：空旷闸门开启时用 NEAR_OPEN_MAX_D + SLACK。"""
    if STAGE1_NEAR_OPEN_AREA_GATE:
        return stage1_near_open_dist_cap()
    return float(neighborhood_m)

def stage1_near_open_reachable_ok(sim, pos, target_xz, front_d, d_gate):
    """
    空旷近邻附加闸门：测地须够近；测地/直线比不过大；前方深度不能远超目标距离（隔障望见）。
    返回 (ok, tag)。
    """
    if target_xz is None:
        return False, "no_target"
    d_gate = float(d_gate)
    geo = geodesic_distance_xz(
        sim, float(pos[0]), float(pos[2]), float(target_xz[0]), float(target_xz[1])
    )
    if geo > float(STAGE1_NEAR_OPEN_MAX_GEODESIC_M):
        return False, f"geo={geo:.2f}>{STAGE1_NEAR_OPEN_MAX_GEODESIC_M:.2f}"
    if d_gate > 0.35 and geo / d_gate > float(STAGE1_NEAR_OPEN_GEO_EUCL_RATIO_MAX):
        return False, (
            f"geo/eucl={geo / d_gate:.2f}>{STAGE1_NEAR_OPEN_GEO_EUCL_RATIO_MAX:.2f}"
        )
    if float(STAGE1_NEAR_OPEN_LOS_FRONT_OVER_M) > 0 and front_d is not None:
        over = float(STAGE1_NEAR_OPEN_LOS_FRONT_OVER_M)
        if float(front_d) > d_gate + over:
            return False, f"front={float(front_d):.2f}>d+{over:.2f}"
    return True, f"geo={geo:.2f}"

def stage1_near_open_nav_blocked(
    sim, pos, target_xz, neighborhood_m, session_best_dist, left_d, front_d, right_d
):
    """直线近、局部开阔，但 navmesh/隔障闸门未过 → 应继续绕路而非结束 Stage1。"""
    if target_xz is None or neighborhood_m is None:
        return False, ""
    d_cur = _dist_xz_to_target(pos, target_xz)
    d_gate = float(d_cur)
    if session_best_dist is not None:
        d_gate = min(d_gate, float(session_best_dist))
    if d_gate > stage1_near_dist_limit(neighborhood_m):
        return False, ""
    if not STAGE1_NEAR_OPEN_AREA_GATE:
        return False, ""
    open_ok, open_tag = stage1_open_walkable_ok(left_d, front_d, right_d)
    if not open_ok:
        return False, ""
    reach_ok, reach_tag = stage1_near_open_reachable_ok(
        sim, pos, target_xz, front_d, d_gate
    )
    if reach_ok:
        return False, ""
    return True, reach_tag or open_tag

def stage1_open_near_finish_reason(
    sim, pos, target_xz, neighborhood_m, session_best_dist=None
):
    """
    path 终点/折线耗尽时：dist≤近邻上限且单帧开阔 → 可结束 Stage1（无需 streak）。
    session_best_dist：本段 path 内最近距离；避免末点略弹远（如 2.11m）而 best≈2.0m 仍失败。
    返回 (ok, exit_reason, open_tag)。
    """
    if target_xz is None or neighborhood_m is None:
        return False, "", ""
    d_cur = _dist_xz_to_target(pos, target_xz)
    d_gate = float(d_cur)
    if session_best_dist is not None:
        d_gate = min(d_gate, float(session_best_dist))
    near_limit = stage1_near_dist_limit(neighborhood_m)
    if d_gate > near_limit:
        return False, "", ""
    if not STAGE1_NEAR_OPEN_AREA_GATE:
        if d_gate <= float(neighborhood_m):
            return True, "neighborhood", ""
        return False, "", ""
    obs = sim.get_sensor_observations()
    depth = fill_depth(obs["depth"])
    ld, fd, rd = measure_depth_probes(depth)
    open_ok, open_tag = stage1_open_walkable_ok(ld, fd, rd)
    if not open_ok:
        return False, "", open_tag
    reach_ok, reach_tag = stage1_near_open_reachable_ok(
        sim, pos, target_xz, fd, d_gate
    )
    if not reach_ok:
        return False, "", reach_tag
    tag = f"{open_tag} {reach_tag}".strip()
    if d_gate <= float(NAV_COARSE_SUCCESS_M):
        return True, "neighborhood", tag
    return True, "neighborhood_open", tag

def _stage1_handle_path_exhausted(
    sim,
    pos,
    target_xz,
    neighborhood_m,
    session_best_dist,
    nlog=None,
):
    """
    path 折线耗尽（idx 在本轮 advance 后才到 len，或循环开头已 finished）。
    返回 (exit_reason, done, want_replan)。
    done=True 表示应结束 path-follow；want_replan=True 表示近距隔障需重规划。
    """
    fin_ok, fin_exit, fin_tag = stage1_open_near_finish_reason(
        sim,
        pos,
        target_xz,
        neighborhood_m,
        session_best_dist=session_best_dist,
    )
    if fin_ok:
        if nlog is not None and target_xz is not None:
            d_now = _dist_xz_to_target(pos, target_xz)
            best_s = (
                f" best={float(session_best_dist):.2f}m"
                if session_best_dist is not None
                else ""
            )
            nlog(
                f"  path 折线已走完 dist={d_now:.2f}m{best_s} "
                f"open={fin_tag} → 停止 path-follow，切换视觉"
            )
        return fin_exit, True, False
    obs = sim.get_sensor_observations()
    depth = fill_depth(obs["depth"])
    ld, fd, rd = measure_depth_probes(depth)
    blocked, block_tag = stage1_near_open_nav_blocked(
        sim,
        pos,
        target_xz,
        neighborhood_m,
        session_best_dist,
        ld,
        fd,
        rd,
    )
    if blocked:
        if nlog is not None and target_xz is not None:
            d_now = _dist_xz_to_target(pos, target_xz)
            best_s = (
                f" best={float(session_best_dist):.2f}m"
                if session_best_dist is not None
                else ""
            )
            nlog(
                f"  path 折线已走完 dist={d_now:.2f}m{best_s} "
                f"近距隔障 ({block_tag}) → 重规划绕开"
            )
        return "path_finished", False, True
    if nlog is not None and target_xz is not None:
        d_now = _dist_xz_to_target(pos, target_xz)
        best_s = (
            f" best={float(session_best_dist):.2f}m"
            if session_best_dist is not None
            else ""
        )
        cap = stage1_near_open_dist_cap()
        gate = fin_tag or f"dist>{cap:.2f}m"
        nlog(
            f"  path 折线已走完 dist={d_now:.2f}m{best_s} "
            f"开阔闸门未通过 ({gate}) → path_finished"
        )
    return "path_finished", True, False

def _stage1_rewind_path_for_blocked_follow(follower, nlog=None):
    """折线 idx 已到尽头但测地仍远：回退到最后段 waypoint，继续 path+绕墙。"""
    n = len(follower.path_points)
    if n < 2:
        return False
    if follower.idx < n:
        return False
    follower.idx = max(1, n - 2)
    follower._wp_enter_dist = None
    follower._wp_best_dist = None
    if nlog is not None:
        nlog(
            f"  path idx→{follower.idx}/{n}（折线已尽、测地仍远）"
            f" 继续沿墙/逼近视点"
        )
    return True

def _stage1_on_path_exhausted(
    sim,
    pos,
    target_xz,
    neighborhood_m,
    session_best_dist,
    follower,
    goal,
    replans,
    near_blocked_rewinds,
    nlog,
    after_replan_fn,
    steer_state,
    trace=None,
    loco=None,
    coord=None,
    resume_wall=None,
):
    """
    path 折线耗尽时的统一处理。
    返回 (exit_reason, should_break, replans, near_blocked_rewinds)。
    should_break=False 时主循环须 continue（勿切 Stage2/3）。
    """
    exit_reason, done, want_replan = _stage1_handle_path_exhausted(
        sim,
        pos,
        target_xz,
        neighborhood_m,
        session_best_dist,
        nlog=nlog,
    )
    if done:
        return exit_reason, True, replans, near_blocked_rewinds
    if not want_replan:
        return exit_reason, True, replans, near_blocked_rewinds
    if replans < int(STAGE1_NEAR_BLOCKED_MAX_REPLANS):
        if follower.set_goal(pos, goal, trace=trace, nlog=nlog):
            after_replan_fn()
            replans += 1
            steer_state.clear_commit()
            return exit_reason, False, replans, near_blocked_rewinds
    if near_blocked_rewinds >= int(STAGE1_NEAR_BLOCKED_MAX_REWINDS):
        if nlog is not None:
            nlog(
                f"  近距隔障绕障回退已达 {near_blocked_rewinds} 次，"
                f"结束 path-follow（exit=path_finished）"
            )
        return "path_finished", True, replans, near_blocked_rewinds
    if _stage1_rewind_path_for_blocked_follow(follower, nlog=nlog):
        near_blocked_rewinds += 1
        steer_state.clear_commit()
        if (
            resume_wall is not None
            and loco is not None
            and not loco.in_escape
        ):
            rw = resume_wall
            goal_bearing = stage1_bearing_yaw(pos, target_xz)
            loco.enter_escape(
                rw["mix_yaw"],
                rw["left_d"],
                rw["right_d"],
                rw["front_d"],
                rw["dist_ref"],
                yaw=rw["yaw"],
                goal_bearing_yaw=goal_bearing,
                pos=pos,
                goal_xz=rw["goal_xz"],
            )
            if coord is not None:
                coord.begin_wall_follow()
            if nlog is not None:
                nlog(
                    f"  近距隔障且未在绕墙 → 再入 WALL side={loco.wall_side}"
                )
        return exit_reason, False, replans, near_blocked_rewinds
    return "path_finished", True, replans, near_blocked_rewinds

def stage1_open_walkable_ok(left_d, front_d, right_d):
    """近邻到达闸门：前方与两侧须有足够可行走深度，且非窄走廊。"""
    L, F, R = float(left_d), float(front_d), float(right_d)
    if F < float(STAGE1_NEAR_OPEN_FRONT_M):
        return False, f"front={F:.2f}"
    free_m = min(L, F, R)
    if free_m < float(STAGE1_NEAR_OPEN_MIN_FREE_M):
        return False, f"min_lfr={free_m:.2f}"
    scores = depth_place_scores(L, F, R)
    if scores["corridor"] >= 0.55 and F < 1.0:
        return False, "narrow_corridor"
    place_type, _ = classify_local_place(L, F, R)
    return True, f"{place_type} free={free_m:.2f}"

def stage1_heading_mode(dist_target):
    """远处：bearing→target；近处：view.yaw+rel_angle（仅 STAGE1_VIEW_YAW_IN_PATH）。"""
    if not STAGE1_VIEW_YAW_IN_PATH:
        return "bearing"
    if dist_target is None:
        return "bearing"
    if float(dist_target) > float(STAGE1_VIEW_YAW_BLEND_DIST_M):
        return "bearing"
    return "view"

def stage1_guide_yaw(pos, target_xz, view, dist_target):
    """本步用于 mix 的 guide 航向及模式标签。"""
    mode = stage1_heading_mode(dist_target)
    if mode == "view":
        view_yaw = semantic_aim_yaw_from_view(view)
        if view_yaw is not None:
            return float(view_yaw), mode
    bearing_yaw = stage1_bearing_yaw(pos, target_xz)
    if bearing_yaw is not None:
        return bearing_yaw, "bearing"
    view_yaw = semantic_aim_yaw_from_view(view)
    if view_yaw is not None:
        return float(view_yaw), "view"
    return None, "none"

def stage1_mix_yaw(pos, wp, target_xz, view, dist_target):
    """mix(wp, guide)：guide 远处为 bearing→target，近处为 view 观测朝向。"""
    wp_yaw = yaw_toward_xz(pos, float(wp[0]), float(wp[2]))
    guide_yaw, mode = stage1_guide_yaw(pos, target_xz, view, dist_target)
    bearing_yaw = stage1_bearing_yaw(pos, target_xz)
    if guide_yaw is None:
        return wp_yaw, 0.0, wp_yaw, mode, bearing_yaw
    alpha = stage1_semantic_alpha(dist_target)
    mix_yaw = angle_mix_yaw(wp_yaw, guide_yaw, alpha)
    return mix_yaw, alpha, wp_yaw, mode, bearing_yaw

def _dist_xz(a, b):
    return float(np.linalg.norm(np.array([float(a[0]), float(a[2])]) - np.array([float(b[0]), float(b[2])])))

def _dist_xz_to_target(pos, target_xz):
    """pos 为 agent position；target_xz 为 [x, z]。"""
    return float(
        np.linalg.norm(
            np.array([float(pos[0]), float(pos[2])], dtype=np.float64)
            - np.array([float(target_xz[0]), float(target_xz[1])], dtype=np.float64)
        )
    )

def geodesic_distance_xz(sim, x0, z0, x1, z1):
    """Navmesh 测地距离；失败时退回直线距离并放大，避免不可达点排前。"""
    eucl = _dist_xz((x0, 0.0, z0), (x1, 0.0, z1))
    pf = sim.pathfinder
    p0 = pf.snap_point(np.array([float(x0), 0.0, float(z0)], dtype=np.float64))
    p1 = pf.snap_point(np.array([float(x1), 0.0, float(z1)], dtype=np.float64))
    if not (pf.is_navigable(p0) and pf.is_navigable(p1)):
        return eucl * 1.5
    path = habitat_sim.ShortestPath()
    path.requested_start = np.array(p0, dtype=np.float64)
    path.requested_end = np.array(p1, dtype=np.float64)
    if pf.find_path(path):
        return float(path.geodesic_distance)
    return eucl * 2.0

def _save_agent_pose(agent):
    p = np.array(agent.state.position, dtype=np.float64)
    r = agent.state.rotation
    return p.copy(), r

def _restore_agent_pose(agent, position, rotation):
    st = habitat_sim.AgentState()
    st.position = position
    st.rotation = rotation
    agent.set_state(st, True)

def yaw_diff(a, b):
    """最短角差 |a-b| ∈ [0, π]。"""
    return abs((float(a) - float(b) + math.pi) % (2 * math.pi) - math.pi)

def region_id_from_xz(x, z):
    return (int(np.floor(float(x) / REGION_SIZE)), int(np.floor(float(z) / REGION_SIZE)))

def region_center_xz(rid):
    return ((rid[0] + 0.5) * REGION_SIZE, (rid[1] + 0.5) * REGION_SIZE)

def grid_adjacent_regions(rid):
    """空间网格邻接（仅用于 frontier 判定，不写入拓扑边）。"""
    ix, iz = rid
    for dx in (-1, 0, 1):
        for dz in (-1, 0, 1):
            if dx == 0 and dz == 0:
                continue
            yield (ix + dx, iz + dz)

def snap_navigable(sim, x, y, z):
    pf = sim.pathfinder
    p = pf.snap_point(np.array([float(x), float(y), float(z)], dtype=np.float64))
    if np.isfinite(p[0]) and pf.is_navigable(p):
        return np.array(p, dtype=np.float64)
    return None

def depth_place_scores(left_d, front_d, right_d):
    """由深度探针估计各 place 类型得分。"""

    def near(d):
        return d < ANCHOR_NEAR_M

    def far(d):
        return d > ANCHOR_FAR_M

    L, F, R = float(left_d), float(front_d), float(right_d)
    corridor = (float(near(L)) + float(near(R)) + float(far(F))) / 3.0
    doorway = 0.0
    if near(L) and near(R) and F > max(L, R) * 0.85 and F > ANCHOR_NEAR_M:
        doorway = min(1.0, 0.45 + (F - 0.5 * (L + R)) / 1.2)
    open_area = (float(far(L)) + float(far(R)) + float(far(F))) / 3.0
    mid = 1.0 - abs(L - R) / max(L, R, 0.2)
    room_center = 0.35 + 0.25 * mid
    return {
        "corridor": corridor,
        "doorway": doorway,
        "open_area": open_area,
        "room_center": room_center,
    }

def classify_local_place(left_d, front_d, right_d):
    scores = depth_place_scores(left_d, front_d, right_d)
    best = max(scores, key=scores.get)
    if scores[best] < 0.45:
        best = "room_center"
        return best, scores["room_center"]
    return best, float(scores[best])

def local_free_space_score(left_d, front_d, right_d):
    return float(min(left_d, front_d, right_d))

# ─── SemanticView 记忆 ─────────────────────────────────────
@dataclass
class SemanticView:
    label: str
    pos: list
    yaw: float
    conf: float
    bbox_ratio: float
    timestamp: int
    view_id: int = 0
    region_id: list = field(default_factory=lambda: [0, 0])
    target_rel_angle: float = 0.0
    fail_count: int = 0
    forgotten: bool = False

    def xz(self):
        """pos 统一为 [x, z]；兼容旧数据 [x, y, z]。"""
        if len(self.pos) >= 3:
            return float(self.pos[0]), float(self.pos[2])
        return float(self.pos[0]), float(self.pos[1])

    def to_dict(self):
        vx, vz = self.xz()
        return {
            "view_id": int(self.view_id),
            "label": str(self.label),
            "pos": [round(vx, 3), round(vz, 3)],
            "yaw": round(float(self.yaw), 3),
            "conf": round(float(self.conf), 3),
            "bbox_ratio": round(float(self.bbox_ratio), 4),
            "timestamp": int(self.timestamp),
            "region_id": list(self.region_id),
            "target_rel_angle": round(float(self.target_rel_angle), 4),
            "fail_count": int(self.fail_count),
            "forgotten": bool(self.forgotten),
        }

    @classmethod
    def from_dict(cls, d):
        raw = d.get("pos", [0, 0])
        if len(raw) >= 3:
            pos = [float(raw[0]), float(raw[2])]
        elif len(raw) >= 2:
            pos = [float(raw[0]), float(raw[1])]
        else:
            pos = [0.0, 0.0]
        return cls(
            label=str(d.get("label", "")).lower(),
            pos=pos,
            yaw=float(d.get("yaw", 0)),
            conf=float(d.get("conf", 0)),
            bbox_ratio=float(d.get("bbox_ratio", 0)),
            timestamp=int(d.get("timestamp", 0)),
            view_id=int(d.get("view_id", 0)),
            region_id=list(d.get("region_id", [0, 0])),
            target_rel_angle=float(d.get("target_rel_angle", 0)),
            fail_count=int(d.get("fail_count", 0)),
            forgotten=bool(d.get("forgotten", False)),
        )

def semantic_decay_score(base_score, age_steps, tau=SEM_DECAY_TAU_STEPS):
    if base_score <= 0:
        return 0.0
    return float(base_score) * math.exp(-max(0.0, float(age_steps)) / float(tau))

# ─── 语义拓扑图 ───────────────────────────────────────────
class SemanticTopoMap:
    """region graph + SemanticView 记忆；语义带时间衰减与失败惩罚。"""

    def __init__(self):
        self.regions = {}
        self.known_places = {pt: [] for pt in PLACE_TYPES}
        self.views = []
        self._next_view_id = 1
        self._last_rid = None
        self._rid_to_serial = {}
        self._serial_to_rid = {}
        self._known_rids = set()
        self._last_new_region_step = 0

    def _ensure(self, rid):
        if rid not in self.regions:
            cx, cz = region_center_xz(rid)
            self.regions[rid] = {
                "center": [float(cx), float(cz)],
                "neighbors": [],
                "semantics": {},
                "prototype_positions": [],
                "visit_count": 0,
                "last_visit_step": -1,
                "failed_targets": set(),
            }
        return self.regions[rid]

    def _neighbor_lists(self, node):
        return [tuple(n) for n in node["neighbors"]]

    def _add_edge(self, rid_a, rid_b):
        if rid_a == rid_b:
            return
        a = self._ensure(rid_a)
        b = self._ensure(rid_b)
        if rid_b not in self._neighbor_lists(a):
            a["neighbors"].append(list(rid_b))
        if rid_a not in self._neighbor_lists(b):
            b["neighbors"].append(list(rid_a))

    def _record_prototype(self, rid, x, z):
        node = self._ensure(rid)
        node["prototype_positions"].append([float(x), float(z)])
        protos = node["prototype_positions"]
        mx = sum(p[0] for p in protos) / len(protos)
        mz = sum(p[1] for p in protos) / len(protos)
        node["center"] = [float(mx), float(mz)]

    def most_central_prototype_xz(self, rid):
        node = self.regions.get(rid)
        if not node or not node["prototype_positions"]:
            return region_center_xz(rid)
        protos = node["prototype_positions"]
        mx = sum(p[0] for p in protos) / len(protos)
        mz = sum(p[1] for p in protos) / len(protos)
        best = min(protos, key=lambda p: (p[0] - mx) ** 2 + (p[1] - mz) ** 2)
        return float(best[0]), float(best[1])

    def on_step(self, x, z, step):
        rid = region_id_from_xz(x, z)
        is_new = rid not in self._known_rids
        if is_new:
            self._known_rids.add(rid)
            self._last_new_region_step = int(step)
        node = self._ensure(rid)
        if self._last_rid != rid:
            node["visit_count"] += 1
        node["last_visit_step"] = int(step)
        self._record_prototype(rid, x, z)
        if self._last_rid is not None and self._last_rid != rid:
            self._add_edge(self._last_rid, rid)
        self._last_rid = rid
        return rid

    def steps_since_new_region(self, step):
        return int(step) - int(self._last_new_region_step)

    def _parse_semantic_entry(self, raw, fallback_step=0):
        if isinstance(raw, dict):
            return {
                "score": float(raw.get("score", 0)),
                "step": int(raw.get("step", fallback_step)),
                "fail_count": int(raw.get("fail_count", 0)),
            }
        return {"score": float(raw or 0), "step": int(fallback_step), "fail_count": 0}

    def get_region_semantic_score(self, rid, label, current_step):
        """区域语义有效分 = base × exp(-age/τ) - fail_count × penalty。"""
        node = self.regions.get(tuple(rid) if not isinstance(rid, tuple) else rid)
        if not node:
            return 0.0
        key = str(label).lower()
        ent = self._parse_semantic_entry(
            node["semantics"].get(key), int(node.get("last_visit_step", 0))
        )
        if ent["fail_count"] >= SEM_FORGET_FAIL_COUNT:
            return 0.0
        age = max(0, int(current_step) - ent["step"])
        sc = semantic_decay_score(ent["score"], age) - ent["fail_count"] * SEM_FAIL_PENALTY
        return max(0.0, sc)

    def add_semantic(self, x, z, label, confidence, step):
        if not label or confidence <= 0:
            return
        rid = region_id_from_xz(x, z)
        node = self._ensure(rid)
        key = str(label).lower()
        ent = self._parse_semantic_entry(
            node["semantics"].get(key), int(node.get("last_visit_step", 0))
        )
        age = max(0, int(step) - ent["step"])
        decayed = semantic_decay_score(ent["score"], age)
        new_score = min(max(decayed, float(confidence)), SEM_CAP)
        node["semantics"][key] = {
            "score": new_score,
            "step": int(step),
            "fail_count": ent["fail_count"],
        }
        node["last_visit_step"] = int(step)

    def _view_by_id(self, view_id):
        for v in self.views:
            if int(v.view_id) == int(view_id):
                return v
        return None

    @staticmethod
    def _view_obs_quality(conf, bbox_ratio, step):
        """用于 merge 时择优：(conf, bbox, timestamp) 字典序。"""
        return (float(conf), float(bbox_ratio), int(step))

    def _views_spatial_duplicate(self, va, vb, same_label_only=True):
        if va.forgotten or vb.forgotten:
            return False
        if same_label_only and va.label != vb.label:
            return False
        ax, az = va.xz()
        bx, bz = vb.xz()
        if _dist_xz((ax, 0, az), (bx, 0, bz)) >= VIEW_MERGE_DIST_M:
            return False
        return yaw_diff(va.yaw, vb.yaw) <= VIEW_MERGE_YAW_RAD

    def _view_confounded(self, view):
        """同位姿存在拮抗类 view 且置信度接近 → 该 bed/sofa 候选不可信。"""
        for v in self.views:
            if v.forgotten or v.view_id == view.view_id:
                continue
            if not _labels_conflict(v.label, view.label):
                continue
            if not self._views_spatial_duplicate(view, v, same_label_only=False):
                continue
            if float(v.conf) >= float(view.conf) / LABEL_CONFLICT_WIN_MARGIN:
                return True, v.label
        return False, None

    def _merge_into_view(self, v, x, z, yaw, conf, bbox_ratio, target_rel_angle, step, rid):
        """合并：conf/bbox 取 max；位姿/yaw/相对角取自更高 conf、更大 bbox 或更新鲜的观测。"""
        conf = float(conf)
        bbox_ratio = float(bbox_ratio)
        step = int(step)
        old_q = self._view_obs_quality(v.conf, v.bbox_ratio, v.timestamp)
        new_q = self._view_obs_quality(conf, bbox_ratio, step)
        v.conf = max(float(v.conf), conf)
        v.bbox_ratio = max(float(v.bbox_ratio), bbox_ratio)
        v.timestamp = max(int(v.timestamp), step)
        if new_q >= old_q:
            v.pos = [float(x), float(z)]
            v.yaw = float(yaw)
            v.target_rel_angle = float(target_rel_angle)
        v.region_id = list(rid)

    def add_view(self, label, x, y, z, yaw, conf, bbox_ratio, step, target_rel_angle=0.0):
        """检测成功且 conf > 阈值时写入 SemanticView；近距同向则 spatial merge。"""
        key = str(label).lower()
        conf = float(conf)
        if conf < VIEW_CONF_THRESH:
            return None
        bbox_ratio = float(bbox_ratio)
        target_rel_angle = float(target_rel_angle)
        rid = list(region_id_from_xz(x, z))
        for v in self.views:
            if v.forgotten or v.label != key:
                continue
            vx, vz = v.xz()
            if _dist_xz((x, 0, z), (vx, 0, vz)) >= VIEW_MERGE_DIST_M:
                continue
            if yaw_diff(yaw, v.yaw) > VIEW_MERGE_YAW_RAD:
                continue
            self._merge_into_view(v, x, z, yaw, conf, bbox_ratio, target_rel_angle, step, rid)
            self.add_semantic(x, z, key, conf, step)
            return v

        for v in list(self.views):
            if v.forgotten or not _labels_conflict(v.label, key):
                continue
            vx, vz = v.xz()
            if _dist_xz((x, 0, z), (vx, 0, vz)) >= VIEW_MERGE_DIST_M:
                continue
            if yaw_diff(yaw, v.yaw) > VIEW_MERGE_YAW_RAD:
                continue
            if conf >= float(v.conf) * LABEL_CONFLICT_WIN_MARGIN:
                v.forgotten = True
            elif float(v.conf) >= conf / LABEL_CONFLICT_WIN_MARGIN:
                return None

        if sum(1 for v in self.views if v.label == key and not v.forgotten) >= VIEW_MAX_PER_LABEL:
            self.views.sort(
                key=lambda v: (v.label != key, v.conf, v.bbox_ratio, -v.timestamp),
            )
            for i, v in enumerate(self.views):
                if v.label == key and not v.forgotten:
                    v.forgotten = True
                    break

        view = SemanticView(
            label=key,
            pos=[float(x), float(z)],
            yaw=float(yaw),
            conf=conf,
            bbox_ratio=bbox_ratio,
            timestamp=int(step),
            view_id=self._next_view_id,
            region_id=rid,
            target_rel_angle=target_rel_angle,
        )
        self._next_view_id += 1
        self.views.append(view)
        self.add_semantic(x, z, key, conf, step)
        return view

    def try_add_view_from_det(self, label, position, yaw, det, step):
        if not det.get("found"):
            return None
        conf = float(det["confidence"])
        if conf < VIEW_CONF_THRESH:
            return None
        pn = np.asarray(position, dtype=np.float64)
        bbox_ratio = float(det.get("bbox_ratio", 0.05))
        rel_angle = target_rel_angle_from_det(det)
        return self.add_view(
            label,
            float(pn[0]),
            float(pn[1]),
            float(pn[2]),
            float(yaw),
            conf,
            bbox_ratio,
            step,
            target_rel_angle=rel_angle,
        )

    def view_nav_score(self, view, agent_x, agent_z, current_step, memory=None, sim=None):
        if view.forgotten or view.fail_count >= SEM_FORGET_FAIL_COUNT:
            return -1e9
        confounded, rival = self._view_confounded(view)
        if confounded:
            return -1e9
        age = max(0, int(current_step) - int(view.timestamp))
        sem_sc = semantic_decay_score(view.conf, age) - view.fail_count * SEM_FAIL_PENALTY
        vx, vz = view.xz()
        if sim is not None and NAV_USE_GEODESIC_VIEW_SCORE:
            dist = geodesic_distance_xz(sim, agent_x, agent_z, vx, vz)
            dist_pen = NAV_GEODESIC_DISTANCE_PENALTY
        else:
            dist = _dist_xz((agent_x, 0, agent_z), (vx, 0, vz))
            dist_pen = NAV_VIEW_DISTANCE_PENALTY
        revisit = 0.0
        if memory is not None:
            revisit = min(NAV_VIEW_REVISIT_PENALTY, memory.local_revisit_penalty(vx, vz))
        return sem_sc + view.bbox_ratio - revisit - dist * dist_pen

    def retrieve_candidate_views(
        self,
        label,
        agent_x,
        agent_z,
        current_step,
        memory=None,
        max_n=NAV_MAX_CANDIDATE_VIEWS,
        sim=None,
    ):
        """score = semantic + bbox - revisit - geodesic(或直线)距离惩罚。"""
        key = str(label).lower()
        items = []
        for v in self.views:
            if v.label != key:
                continue
            sc = self.view_nav_score(v, agent_x, agent_z, current_step, memory, sim=sim)
            if sc <= -1e8:
                continue
            items.append((v.view_id, sc, v))
        items.sort(key=lambda x: -x[1])
        deduped = []
        for vid, sc, v in items:
            if any(self._views_spatial_duplicate(v, kept) for _, _, kept in deduped):
                continue
            deduped.append((vid, sc, v))
            if len(deduped) >= max_n:
                break
        return deduped

    def compact_spatial_views(self):
        """加载旧图后合并近距同向重复 view（dist<0.8m, yaw<25°）。"""
        merged = 0
        active = [v for v in self.views if not v.forgotten]
        for i, va in enumerate(active):
            if va.forgotten:
                continue
            for vb in active[i + 1 :]:
                if vb.forgotten or not self._views_spatial_duplicate(va, vb):
                    continue
                if self._view_obs_quality(vb.conf, vb.bbox_ratio, vb.timestamp) > self._view_obs_quality(
                    va.conf, va.bbox_ratio, va.timestamp
                ):
                    survivor, donor = vb, va
                else:
                    survivor, donor = va, vb
                dx, dz = donor.xz()
                self._merge_into_view(
                    survivor,
                    dx,
                    dz,
                    donor.yaw,
                    donor.conf,
                    donor.bbox_ratio,
                    donor.target_rel_angle,
                    donor.timestamp,
                    donor.region_id,
                )
                donor.forgotten = True
                merged += 1
                if donor is va:
                    break
        if merged:
            print(f"  compact_spatial_views: 合并 {merged} 个重复 view")
        return merged

    def mark_view_failed(self, view_id, label, reason=""):
        key = str(label).lower()
        v = self._view_by_id(view_id)
        if v is None:
            return
        v.fail_count += 1
        if v.fail_count >= SEM_FORGET_FAIL_COUNT:
            v.forgotten = True
        rid = tuple(v.region_id)
        if rid in self.regions:
            node = self.regions[rid]
            if key in node.get("failed_targets", set()):
                pass
            ent = self._parse_semantic_entry(
                node["semantics"].get(key), int(node.get("last_visit_step", 0))
            )
            ent["fail_count"] = min(ent["fail_count"] + 1, SEM_FORGET_FAIL_COUNT)
            node["semantics"][key] = ent
            if ent["fail_count"] >= SEM_FORGET_FAIL_COUNT:
                node["failed_targets"].add(key)
        if reason:
            print(f"  view {view_id} 标记失败 [{key}] x{ v.fail_count }: {reason}")

    def navigable_goal_from_view(self, sim, view):
        x, z = view.xz()
        y = float(sim.agents[0].state.position[1])
        return snap_navigable(sim, x, y, z)

    def view_region_id_resolved(self, view):
        """view 的 region_id；无效时用观测位姿反推网格。"""
        try:
            rid = (int(view.region_id[0]), int(view.region_id[1]))
        except (TypeError, IndexError, ValueError):
            rid = None
        if rid is None or rid == (-1, -1):
            x, z = view.xz()
            rid = region_id_from_xz(x, z)
        return rid

    def stage1_approach_target(self, sim, view):
        """
        Stage1 path 终点与 dist 参考点：view 所在语义区域中心（prototype/网格），
        非预探索时 agent 站立位姿。返回 (goal, target_xz, rid, source) 或 None。
        """
        rid = self.view_region_id_resolved(view)
        tx, tz = self.most_central_prototype_xz(rid)
        source = "region_prototype" if (
            rid in self.regions and self.regions[rid].get("prototype_positions")
        ) else "region_grid"
        y = float(sim.agents[0].state.position[1])
        goal = snap_navigable(sim, tx, y, tz)
        if goal is not None:
            return goal, [float(tx), float(tz)], rid, source
        vx, vz = view.xz()
        goal = snap_navigable(sim, vx, y, vz)
        if goal is None:
            return None
        return goal, [float(vx), float(vz)], rid, "view_pose_fallback"

    def _region_unvisited(self, rid):
        if rid not in self.regions:
            return True
        return int(self.regions[rid]["visit_count"]) == 0

    def register_local_anchor(self, x, z, yaw, left_d, front_d, right_d, step):
        place_type, conf = classify_local_place(left_d, front_d, right_d)
        rid = region_id_from_xz(x, z)
        free_s = local_free_space_score(left_d, front_d, right_d)
        anchor = {
            "place_type": place_type,
            "position": [float(x), float(z)],
            "heading": float(yaw),
            "local_free_space_score": round(float(max(conf, free_s * 0.15)), 3),
            "region_id": list(rid),
            "visit_count": 0,
            "last_visit_step": int(step),
        }
        bucket = self.known_places[place_type]
        for ex in bucket:
            if _dist_xz((x, 0, z), (ex["position"][0], 0, ex["position"][1])) < ANCHOR_MERGE_DIST_M:
                if anchor["local_free_space_score"] >= ex["local_free_space_score"]:
                    ex.update(anchor)
                    ex["visit_count"] = int(ex.get("visit_count", 0)) + 1
                ex["last_visit_step"] = int(step)
                return ex
        if len(bucket) >= ANCHOR_MAX_PER_TYPE:
            bucket.sort(key=lambda a: a.get("local_free_space_score", 0))
            bucket.pop(0)
        bucket.append(anchor)
        return anchor

    def anchors_in_region(self, rid, place_types=None):
        rid_t = tuple(rid) if not isinstance(rid, tuple) else rid
        types = place_types or PLACE_TYPES
        out = []
        for pt in types:
            for a in self.known_places.get(pt, []):
                if tuple(a.get("region_id", [])) == rid_t:
                    out.append(a)
        return out

    def best_anchor_in_region(self, rid, prefer_types=("doorway", "corridor", "room_center", "open_area")):
        cands = self.anchors_in_region(rid, prefer_types)
        if cands:
            return max(cands, key=lambda a: float(a.get("local_free_space_score", 0)))
        node = self.regions.get(tuple(rid) if isinstance(rid, tuple) else rid)
        if node and node.get("prototype_positions"):
            px, pz = self.most_central_prototype_xz(rid)
            return {
                "place_type": "room_center",
                "position": [px, pz],
                "heading": 0.0,
                "local_free_space_score": 0.3,
                "region_id": list(rid) if isinstance(rid, tuple) else list(rid),
            }
        cx, cz = region_center_xz(rid)
        return {
            "place_type": "room_center",
            "position": [cx, cz],
            "heading": 0.0,
            "local_free_space_score": 0.2,
            "region_id": list(rid) if isinstance(rid, tuple) else [rid[0], rid[1]],
        }

    def _anchor_is_frontier(self, anchor):
        rid = tuple(anchor.get("region_id", []))
        if len(rid) != 2:
            return False
        for nb in grid_adjacent_regions(rid):
            if self._region_unvisited(nb):
                return True
        return False

    def compute_frontier_anchors(self, place_types=FRONTIER_ANCHOR_TYPES):
        pool = []
        for pt in place_types:
            for a in self.known_places.get(pt, []):
                if self._anchor_is_frontier(a):
                    pool.append(a)
        return pool

    def frontier_anchor_score(self, anchor, agent_x, agent_z, memory=None):
        rid = tuple(anchor.get("region_id", []))
        unseen_bonus = FRONTIER_UNSEEN_BONUS if int(anchor.get("visit_count", 0)) <= 1 else 0.0
        if rid in self.regions:
            n = self.regions[rid]
            semantic_diversity = len(n["semantics"]) * FRONTIER_SEM_DIV_WEIGHT
            visit_penalty = n["visit_count"] * FRONTIER_VISIT_PENALTY
        else:
            semantic_diversity = 0.0
            visit_penalty = 0.0
        ax, az = anchor["position"][0], anchor["position"][1]
        distance_penalty = _dist_xz((agent_x, 0, agent_z), (ax, 0, az)) * FRONTIER_DIST_PENALTY
        type_bonus = 0.6 if anchor.get("place_type") == "doorway" else 0.2
        score = (
            unseen_bonus
            + semantic_diversity
            + type_bonus
            + float(anchor.get("local_free_space_score", 0))
            - visit_penalty
            - distance_penalty
        )
        if memory is not None:
            score -= memory.local_revisit_penalty(ax, az)
        return score

    def select_next_frontier_anchor(self, x, z, step, memory=None):
        """Frontier = doorway（及 fallback corridor）锚点，全局排序。"""
        pool = self.compute_frontier_anchors(("doorway",))
        if not pool:
            pool = self.compute_frontier_anchors(("corridor", "open_area"))
        if pool:
            return max(pool, key=lambda a: self.frontier_anchor_score(a, x, z, memory))
        return self._fallback_explore_anchor(x, z, memory)

    def _fallback_explore_anchor(self, x, z, memory=None):
        rid = region_id_from_xz(x, z)
        seeds = {rid}
        if self._last_rid is not None:
            seeds.add(self._last_rid)
            for nb in grid_adjacent_regions(self._last_rid):
                seeds.add(nb)
        best_rid = None
        best_sc = -1e9
        for s in seeds:
            for nb in grid_adjacent_regions(s):
                if not self._region_unvisited(nb):
                    continue
                sc = FRONTIER_UNSEEN_BONUS
                if memory is not None:
                    gx, gz = region_center_xz(nb)
                    sc -= memory.local_revisit_penalty(gx, gz)
                if sc > best_sc:
                    best_sc = sc
                    best_rid = nb
        if best_rid is None:
            return None
        return self.best_anchor_in_region(best_rid, ("doorway", "corridor", "room_center"))

    def navigable_goal_from_anchor(self, sim, anchor):
        ax, az = anchor["position"][0], anchor["position"][1]
        y = float(sim.agents[0].state.position[1])
        return snap_navigable(sim, ax, y, az)

    def navigable_goal(self, sim, rid, prefer_types=("doorway", "corridor", "room_center")):
        anchor = self.best_anchor_in_region(rid, prefer_types)
        return self.navigable_goal_from_anchor(sim, anchor)

    navigable_center = navigable_goal

    select_next_region = select_next_frontier_anchor
    pick_frontier_region = select_next_frontier_anchor

    def retrieve_candidate_regions(
        self, label, min_score=NAV_MIN_SEMANTIC_SCORE, max_n=NAV_MAX_CANDIDATE_REGIONS,
        current_step=0,
    ):
        """按衰减后区域语义分排序（无 view 时的 fallback）。"""
        key = str(label).lower()
        items = []
        for rid, node in self.regions.items():
            failed = node.get("failed_targets") or set()
            if key in failed:
                continue
            sc = self.get_region_semantic_score(rid, key, current_step)
            if sc >= min_score:
                items.append((rid, sc))
        items.sort(key=lambda x: -x[1])
        return items[:max_n]

    def mark_region_failed(self, rid, label, reason=""):
        node = self._ensure(rid)
        key = str(label).lower()
        node["failed_targets"].add(key)
        if reason:
            print(f"  region {rid} 标记失败 [{key}]: {reason}")

    def best_region_for_label(self, label):
        cands = self.retrieve_candidate_regions(label, min_score=0.0, max_n=1)
        if not cands:
            return None, 0.0
        return cands[0]

    def _assign_serial_ids(self):
        rids = sorted(self.regions.keys())
        self._rid_to_serial = {rid: i for i, rid in enumerate(rids)}
        self._serial_to_rid = {i: rid for rid, i in self._rid_to_serial.items()}

    def to_export_dict(self):
        self._assign_serial_ids()
        out = {}
        for rid, node in self.regions.items():
            sid = self._rid_to_serial[rid]
            neigh_serial = []
            for nb in node["neighbors"]:
                t = tuple(nb)
                if t in self._rid_to_serial:
                    neigh_serial.append(self._rid_to_serial[t])
            sem = {}
            for k, v in node["semantics"].items():
                if isinstance(v, dict):
                    sem[k] = {
                        "score": round(float(v.get("score", 0)), 3),
                        "step": int(v.get("step", 0)),
                        "fail_count": int(v.get("fail_count", 0)),
                    }
                else:
                    sem[k] = round(float(v), 3)
            failed = sorted(node.get("failed_targets") or [])
            protos = [[round(p[0], 3), round(p[1], 3)] for p in node.get("prototype_positions", [])[-80:]]
            ba = self.best_anchor_in_region(rid)
            nav_xz = ba["position"]
            region_anchors = self.anchors_in_region(rid)
            out[f"region_{sid}"] = {
                "grid_id": list(rid),
                "center": [round(node["center"][0], 3), round(node["center"][1], 3)],
                "nav_goal": [round(nav_xz[0], 3), round(nav_xz[1], 3)],
                "best_anchor": {
                    "place_type": ba.get("place_type"),
                    "position": [round(nav_xz[0], 3), round(nav_xz[1], 3)],
                    "heading": round(float(ba.get("heading", 0)), 3),
                    "local_free_space_score": float(ba.get("local_free_space_score", 0)),
                },
                "anchors": [
                    {
                        "place_type": a.get("place_type"),
                        "position": [round(a["position"][0], 3), round(a["position"][1], 3)],
                        "heading": round(float(a.get("heading", 0)), 3),
                        "local_free_space_score": float(a.get("local_free_space_score", 0)),
                    }
                    for a in region_anchors[:12]
                ],
                "prototype_positions": protos,
                "neighbors": neigh_serial,
                "semantics": sem,
                "failed_targets": failed,
                "visit_count": int(node["visit_count"]),
                "last_visit_step": int(node["last_visit_step"]),
            }
        places_out = {}
        for pt in PLACE_TYPES:
            places_out[pt] = [
                {
                    "position": [round(a["position"][0], 3), round(a["position"][1], 3)],
                    "heading": round(float(a.get("heading", 0)), 3),
                    "local_free_space_score": float(a.get("local_free_space_score", 0)),
                    "region_id": a.get("region_id"),
                    "visit_count": int(a.get("visit_count", 0)),
                }
                for a in self.known_places.get(pt, [])[:ANCHOR_MAX_PER_TYPE]
            ]
        views_out = [v.to_dict() for v in self.views if not v.forgotten][:VIEW_MAX_PER_LABEL * len(DETECT_LABELS)]
        return {
            "scene_id": SCENE_ID,
            "region_size_m": REGION_SIZE,
            "detect_labels": DETECT_LABELS,
            "region_count": len(out),
            "view_count": len(views_out),
            "semantic_views": views_out,
            "known_places": places_out,
            "topo_map": out,
        }

    def save_json(self, path=TOPO_MAP_PATH):
        data = self.to_export_dict()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"语义拓扑图已导出: {path} ({data['region_count']} regions)")
        return path

    @classmethod
    def load_json(cls, path=TOPO_MAP_PATH):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        m = cls()
        raw = data.get("topo_map", data)
        keys_sorted = sorted(raw.keys(), key=lambda k: int(k.split("_")[1]))
        serial_to_rid = {}
        for key in keys_sorted:
            sid = int(key.split("_")[1])
            entry = raw[key]
            rid = tuple(entry.get("grid_id", [0, 0]))
            serial_to_rid[sid] = rid
            protos = entry.get("prototype_positions", [])
            if not protos and "nav_goal" in entry:
                protos = [list(entry["nav_goal"])]
            m.regions[rid] = {
                "center": list(entry["center"]),
                "neighbors": [],
                "semantics": {
                    k: (
                        v if isinstance(v, dict)
                        else {"score": float(v), "step": int(entry.get("last_visit_step", 0)), "fail_count": 0}
                    )
                    for k, v in entry.get("semantics", {}).items()
                },
                "prototype_positions": [list(p) for p in protos],
                "visit_count": int(entry.get("visit_count", 0)),
                "last_visit_step": int(entry.get("last_visit_step", -1)),
                "failed_targets": set(entry.get("failed_targets", [])),
            }
            m._known_rids.add(rid)
        for key in keys_sorted:
            sid = int(key.split("_")[1])
            rid = serial_to_rid[sid]
            for nb_sid in raw[key].get("neighbors", []):
                if nb_sid in serial_to_rid:
                    m.regions[rid]["neighbors"].append(list(serial_to_rid[nb_sid]))
        kp = data.get("known_places", {})
        for pt in PLACE_TYPES:
            m.known_places[pt] = []
            for a in kp.get(pt, []):
                m.known_places[pt].append({
                    "place_type": pt,
                    "position": list(a["position"]),
                    "heading": float(a.get("heading", 0)),
                    "local_free_space_score": float(a.get("local_free_space_score", 0)),
                    "region_id": list(a.get("region_id", [0, 0])),
                    "visit_count": int(a.get("visit_count", 0)),
                    "last_visit_step": -1,
                })
        for vd in data.get("semantic_views", []):
            v = SemanticView.from_dict(vd)
            m.views.append(v)
            m._next_view_id = max(m._next_view_id, int(v.view_id) + 1)
        m.compact_spatial_views()
        m._assign_serial_ids()
        print(
            f"已加载语义拓扑图: {path} ({len(m.regions)} regions, "
            f"views={len(m.views)}, anchors={sum(len(v) for v in m.known_places.values())})"
        )
        return m

    def summary_top(self, n=5):
        items = []
        for rid, node in self.regions.items():
            total = 0.0
            for v in node["semantics"].values():
                total += float(v.get("score", v) if isinstance(v, dict) else v)
            items.append((total, rid, node))
        items.sort(reverse=True)
        lines = [f"regions={len(self.regions)} views={len(self.views)}"]
        for total, rid, node in items[:n]:
            sem_items = []
            for k, v in node["semantics"].items():
                if isinstance(v, dict):
                    sem_items.append((k, float(v.get("score", 0))))
                else:
                    sem_items.append((k, float(v)))
            top = sorted(sem_items, key=lambda x: -x[1])[:3]
            top_s = ", ".join(f"{k}:{v:.1f}" for k, v in top)
            lines.append(f"  {rid} center={node['center']} visits={node['visit_count']} | {top_s}")
        return "\n".join(lines)

    def semantic_discriminative_score(self, rid, label, current_step=0):
        node = self.regions.get(tuple(rid) if not isinstance(rid, tuple) else rid)
        if not node:
            return 0.0
        key = str(label).lower()
        target_sc = self.get_region_semantic_score(rid, key, current_step)
        other = [
            self.get_region_semantic_score(rid, k, current_step)
            for k in node["semantics"]
            if str(k).lower() != key
        ]
        return target_sc - (max(other) if other else 0.0)

    def best_region_discriminative(self, label, min_score=NAV_MIN_SEMANTIC_SCORE, current_step=0):
        key = str(label).lower()
        best_rid, best_disc = None, -1e9
        for rid, node in self.regions.items():
            if self.get_region_semantic_score(rid, key, current_step) < min_score:
                continue
            disc = self.semantic_discriminative_score(rid, key, current_step)
            if disc > best_disc:
                best_disc = disc
                best_rid = rid
        return best_rid, best_disc

def _append_trajectory(trajectory, x, z, step):
    if trajectory is not None:
        trajectory.append((float(x), float(z), int(step)))

def visualize_region_graph(
    topo_map,
    trajectory=None,
    output_path=TOPO_VIZ_EXPLORE_PATH,
    title="Semantic Topo Map",
    highlight_target_en=None,
):
    """
    region graph: centers, neighbor edges, semantic labels,
    anchor types, frontier nodes, traversed trajectory.
    """
    if not HAS_MATPLOTLIB:
        print("未安装 matplotlib，跳过拓扑可视化。可执行: pip install matplotlib")
        return None

    fig, ax = plt.subplots(figsize=(14, 11))
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)

    centers = {rid: (node["center"][0], node["center"][1]) for rid, node in topo_map.regions.items()}

    for rid, node in topo_map.regions.items():
        x0, z0 = centers[rid]
        for nb in node["neighbors"]:
            nb_t = tuple(nb)
            if nb_t in centers:
                x1, z1 = centers[nb_t]
                ax.plot([x0, x1], [z0, z1], color="#888888", linewidth=1.2, alpha=0.65, zorder=1)

    anchor_colors = {
        "doorway": "#e74c3c",
        "corridor": "#3498db",
        "open_area": "#2ecc71",
        "room_center": "#9b59b6",
    }
    for pt in PLACE_TYPES:
        for a in topo_map.known_places.get(pt, []):
            ax.scatter(
                a["position"][0], a["position"][1],
                c=anchor_colors.get(pt, "#555555"),
                s=42, marker="^", edgecolors="white", linewidths=0.4, zorder=4,
            )

    for a in topo_map.compute_frontier_anchors():
        ax.scatter(
            a["position"][0], a["position"][1],
            s=220, facecolors="none", edgecolors="#f39c12", linewidths=2.2, zorder=5,
        )

    if trajectory and len(trajectory) > 0 and len(trajectory[0]) == 2:
        trajectory = [(p[0], p[1], i) for i, p in enumerate(trajectory)]
    if trajectory:
        tx = [p[0] for p in trajectory]
        tz = [p[1] for p in trajectory]
        ax.plot(tx, tz, color="#1abc9c", linewidth=1.5, alpha=0.85, label="trajectory", zorder=2)
        ax.scatter(tx[0], tz[0], c="#27ae60", s=80, marker="o", zorder=6, label="start")
        ax.scatter(tx[-1], tz[-1], c="#c0392b", s=90, marker="*", zorder=6, label="end")

    best_rid = None
    if highlight_target_en:
        best_rid, _ = topo_map.best_region_discriminative(highlight_target_en)

    def _sem_val(raw):
        return float(raw.get("score", raw) if isinstance(raw, dict) else raw)

    for v in getattr(topo_map, "views", []):
        if not v.forgotten:
            vx, vz = v.xz()
            ax.scatter(
                vx, vz, c="#e056fd", s=28, marker="D",
                edgecolors="white", linewidths=0.3, zorder=6, alpha=0.85,
            )

    for rid, node in topo_map.regions.items():
        cx, cz = centers[rid]
        sem = node.get("semantics", {})
        label_lines = [str(rid)]
        if highlight_target_en:
            key = highlight_target_en.lower()
            if key in sem:
                label_lines.append(
                    f"{key}:{_sem_val(sem[key]):.1f} "
                    f"d={topo_map.semantic_discriminative_score(rid, key):.1f}"
                )
            for k, v in sorted(sem.items(), key=lambda x: -_sem_val(x[1]))[:2]:
                if k != key:
                    label_lines.append(f"{k}:{_sem_val(v):.1f}")
        else:
            label_lines.extend(
                f"{k}:{_sem_val(v):.1f}" for k, v in sorted(sem.items(), key=lambda x: -_sem_val(x[1]))[:3]
            )

        face = "#fdebd0" if rid == best_rid else "#ebf5fb"
        edge = "#e67e22" if rid == best_rid else "#2980b9"
        ax.scatter(cx, cz, s=320, c=face, edgecolors=edge, linewidths=2.0, zorder=3)
        ax.annotate("\n".join(label_lines), (cx, cz), fontsize=7, ha="center", va="center", zorder=7)

    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#ebf5fb", markeredgecolor="#2980b9", markersize=10, label="region center"),
        Line2D([0], [0], color="#888888", linewidth=1.5, label="neighbor edge"),
        Line2D([0], [0], color="#1abc9c", linewidth=1.5, label="trajectory"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#e74c3c", markersize=8, label="anchor"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none", markeredgecolor="#f39c12", markersize=12, markeredgewidth=2, label="frontier"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#e056fd", markersize=6, label="semantic view"),
    ]
    ax.legend(handles=legend_elems, loc="upper left", fontsize=8)
    ax.set_title(f"{title} | regions={len(topo_map.regions)}", fontsize=12)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"拓扑图已保存: {output_path}")
    return output_path

# ─── 检测器（预探索无 tracker；导航单类可开 tracker）────────
def _labels_conflict(a, b):
    sa, sb = str(a or "").lower(), str(b or "").lower()
    if sa == sb:
        return False
    for group in LABEL_CONFLICT_GROUPS:
        if sa in group and sb in group:
            return True
    return False

def _confusable_label(target_en):
    t = str(target_en or "").lower()
    for group in LABEL_CONFLICT_GROUPS:
        if t in group:
            for other in group:
                if other != t:
                    return other
    return None

def detect_target_disambiguated(detector, rgb, target_en):
    """导航单类检测 + 拮抗类（床/沙发）置信度对比，抑制 open-vocab 互误。"""
    det = detector.detect(rgb, target_en)
    rival = _confusable_label(target_en)
    if rival is None or not det.get("found"):
        return det
    alt = detector.detect(rgb, rival)
    if not alt.get("found"):
        return det
    if float(alt["confidence"]) >= float(det["confidence"]) * STAGE3_DISAMBIG_MARGIN:
        out = _empty_det()
        out["disambig_rejected"] = rival
        out["disambig_target_conf"] = float(det["confidence"])
        out["disambig_rival_conf"] = float(alt["confidence"])
        return out
    return det

def is_valid_candidate(x0, y0, x1, y1, w, h, prompt):
    prior = SHAPE_PRIORS.get(
        prompt.lower(),
        {"min_area": 0.004, "max_area": 0.65, "aspect": (0.1, 6.0)},
    )
    bw, bh = x1 - x0, y1 - y0
    area = bw * bh / (w * h)
    aspect = bw / max(bh, 1)
    if area < prior["min_area"] or area > prior["max_area"]:
        return False
    lo, hi = prior["aspect"]
    return lo < aspect < hi

class TargetDetector:
    def __init__(self):
        self.model = load_model(GDINO_CONFIG, GDINO_WEIGHTS)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        print(f"GroundingDINO 就绪 ({self.device})")

    def reset_tracker(self):
        pass

    def _infer(self, rgb, prompt):
        img = Image.fromarray(rgb[:, :, :3]).convert("RGB")
        image_tensor, _ = self.transform(img, None)
        h, w = rgb.shape[:2]
        boxes, logits, _ = predict(
            model=self.model,
            image=image_tensor,
            caption=prompt,
            box_threshold=0.22,
            text_threshold=0.22,
            device=self.device,
        )
        empty = {"found": False, "confidence": 0.0, "bbox": None, "center_x": 0.5, "bbox_ratio": 0.0}
        if len(boxes) == 0:
            return empty
        boxes = boxes * torch.tensor([w, h, w, h], device=boxes.device)
        boxes = box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
        best_conf = -1.0
        best = empty
        for i in range(len(logits)):
            x0, y0, x1, y1 = map(int, boxes[i].tolist())
            if not is_valid_candidate(x0, y0, x1, y1, w, h, prompt):
                continue
            conf = float(logits[i])
            if conf > best_conf:
                best_conf = conf
                best = {
                    "found": True,
                    "confidence": conf,
                    "bbox": (x0, y0, x1, y1),
                    "center_x": (x0 + x1) / 2 / w,
                    "bbox_ratio": (x1 - x0) * (y1 - y0) / (w * h),
                }
        return best

    def detect(self, rgb, prompt):
        return self._infer(rgb, prompt)

    def detect_all_labels(self, rgb):
        out = {}
        for label in DETECT_LABELS:
            out[label] = self._infer(rgb, label)
        return out

# ─── Action Memory / Recovery ───────────────────────────────
class ActionMemory:
    """短期动作与位置记忆：振荡抑制、局部重访惩罚、卡住计数。"""

    def __init__(self):
        self.recent_actions = deque(maxlen=ACTION_MEMORY_ACTIONS)
        self.recent_positions = deque(maxlen=ACTION_MEMORY_POSITIONS)
        self.stuck_steps = 0
        self._last_pos = None
        self._osc_action = None
        self._osc_remaining = 0

    def update_stuck(self, pos):
        if self._last_pos is not None:
            moved = _dist_xz(pos, self._last_pos)
            if moved < STUCK_MOVE_THRESH_M:
                self.stuck_steps += 1
            else:
                self.stuck_steps = 0
        self._last_pos = np.array(pos, dtype=np.float64)
        return self.stuck_steps

    def record(self, action, x, z):
        self.recent_actions.append(action)
        self.recent_positions.append((float(x), float(z)))

    def local_revisit_penalty(self, x, z, radius=LOCAL_REVISIT_RADIUS_M):
        penalty = 0.0
        for px, pz in self.recent_positions:
            d = float(np.hypot(x - px, z - pz))
            if d < radius:
                penalty += 1.5 * (1.0 - d / radius)
        return min(penalty, LOCAL_REVISIT_PENALTY_MAX)

    def _recent_turns(self):
        return [a for a in self.recent_actions if a in ("turn_left", "turn_right")]

    def turn_switch_rate(self):
        turns = self._recent_turns()
        if len(turns) < 3:
            return 0.0
        switches = sum(1 for i in range(1, len(turns)) if turns[i] != turns[i - 1])
        return switches / (len(turns) - 1)

    def detect_turn_oscillation(self):
        return self.turn_switch_rate() >= OSCILLATION_SWITCH_RATE_THRESH

    def _start_oscillation_override(self):
        turns = self._recent_turns()
        hold = turns[-1] if turns else random.choice(["turn_left", "turn_right"])
        if random.random() < 0.5:
            self._osc_action = "move_forward"
            self._osc_remaining = 1
        else:
            self._osc_action = hold
            self._osc_remaining = OSCILLATION_HOLD_TURN_STEPS

    def apply_oscillation_override(self, proposed):
        if self._osc_remaining > 0:
            self._osc_remaining -= 1
            return self._osc_action
        if self.detect_turn_oscillation():
            self._start_oscillation_override()
            return self._osc_action
        return proposed

class RecoveryState:
    """卡住恢复: phase1 hard_turn × N → phase2 move_forward × burst（不重新决策）。"""

    def __init__(self):
        self.phase = None
        self.remaining = 0
        self.action = None

    def active(self):
        return self.phase is not None

    def try_trigger(self, stuck_steps):
        if stuck_steps > STUCK_TRIGGER_STEPS and not self.active():
            self.phase = "hard_turn"
            self.remaining = RECOVERY_HARD_TURN_STEPS
            self.action = random.choice(["turn_left", "turn_right"])
            return True
        return False

    def tick(self):
        if not self.active():
            return None
        act = self.action
        self.remaining -= 1
        if self.remaining <= 0:
            if self.phase == "hard_turn":
                self.phase = "forward_burst"
                self.remaining = RECOVERY_FORWARD_BURST
                self.action = "move_forward"
            else:
                self.phase = None
                self.action = None
        return act

    def clear(self):
        self.phase = None
        self.remaining = 0
        self.action = None

def motion_controller(
    agent,
    depth,
    memory,
    recovery,
    propose_action,
    enable_recovery=True,
    allow_oscillation=True,
    allow_recovery=None,
):
    """
    控制器优先级（最高在前）:
      1) RecoveryState — 固定 turn 序列，不重新决策
      2) 常规划法 propose_action
      3) ActionMemory 振荡覆盖
    """
    if allow_recovery is not None:
        enable_recovery = bool(allow_recovery)
    pos = agent.state.position
    memory.update_stuck(pos)
    if enable_recovery:
        recovery.try_trigger(memory.stuck_steps)
    rec = recovery.tick() if enable_recovery else None
    if rec is not None:
        action = rec
    else:
        proposed = propose_action()
        if allow_oscillation:
            action = memory.apply_oscillation_override(proposed)
        else:
            action = proposed
    memory.record(action, float(pos[0]), float(pos[2]))
    return action

# ─── Persistent Path（一次规划，不每步重算）──────────────────
class PersistentPathFollower:
    """缓存 ShortestPath，按 waypoint 推进；advance 使用滞后阈值。"""

    def __init__(self, sim):
        self.sim = sim
        self.path_points = []
        self.idx = 0
        self._wp_enter_dist = None
        self._wp_best_dist = None

    def set_goal(self, start, goal, trace=None, nlog=None):
        path = habitat_sim.ShortestPath()
        path.requested_start = np.array(start, dtype=np.float64)
        path.requested_end = np.array(goal, dtype=np.float64)
        pf = self.sim.pathfinder
        if not pf.find_path(path):
            return False
        self.path_points = list(path.points)
        self.idx = 1
        self._wp_enter_dist = None
        self._wp_best_dist = None
        self.last_geodesic = float(getattr(path, "geodesic_distance", 0.0))
        if trace is not None:
            trace.add_waypoint_path(self.path_points)
        return len(self.path_points) >= 2

    def finished(self):
        return self.idx >= len(self.path_points)

    def current_waypoint(self):
        if self.finished():
            return None
        return self.path_points[self.idx]

    def _dist_to_current_wp(self, pos):
        wp = self.current_waypoint()
        if wp is None:
            return None
        return float(
            np.linalg.norm(
                np.array([float(pos[0]), float(pos[2])], dtype=np.float64)
                - np.array([float(wp[0]), float(wp[2])], dtype=np.float64)
            )
        )

    def advance_if_needed(
        self,
        pos,
        thresh=PERSISTENT_WP_ADVANCE_M,
        progress_m=STAGE1_WP_PROGRESS_M,
    ):
        """推进 waypoint：到达 thresh 内，或朝 wp 靠近 progress_m 以上。"""
        if self.finished():
            return
        d = self._dist_to_current_wp(pos)
        if d is None:
            return
        if self._wp_enter_dist is None:
            self._wp_enter_dist = d
            self._wp_best_dist = d
        else:
            self._wp_best_dist = min(float(self._wp_best_dist), d)
        advance = d < thresh
        if (
            not advance
            and progress_m > 0
            and self._wp_enter_dist - self._wp_best_dist >= progress_m
        ):
            advance = True
        if advance:
            self.idx += 1
            self._wp_enter_dist = None
            self._wp_best_dist = None

class PathFollowSteeringState:
    """path-follow 转向滞后：避免 left/right 每步翻转。"""

    def __init__(self):
        self.turn_commit = None
        self.turn_commit_steps = 0

    def consume_commit(self):
        if self.turn_commit_steps > 0:
            self.turn_commit_steps -= 1
            return self.turn_commit
        return None

    def start_turn_commit(self, action, steps=None):
        self.turn_commit = action
        n = int(steps if steps is not None else PATH_FOLLOW_TURN_COMMIT_STEPS)
        self.turn_commit_steps = max(0, n - 1)
        return action

    def clear_commit(self):
        self.turn_commit = None
        self.turn_commit_steps = 0

def path_follow_steering_persistent(
    waypoint,
    pos,
    yaw,
    depth,
    state=None,
    target_xz=None,
    view=None,
    dist_target=None,
    return_meta=False,
    nlog=None,
):
    """沿持久 path：mix(wp, bearing|view) + 微避障（depth 可行航向，非 recovery）。"""
    if state is None:
        state = PathFollowSteeringState()

    left_d, front_d, right_d = measure_depth_probes(depth)
    wp = np.asarray(waypoint, dtype=np.float64)
    mix_yaw, alpha, wp_yaw, h_mode, bearing_yaw = stage1_mix_yaw(
        pos, wp, target_xz, view, dist_target
    )
    guide_yaw, _ = stage1_guide_yaw(pos, target_xz, view, dist_target)
    diff = (mix_yaw - yaw + np.pi) % (2 * np.pi) - np.pi
    action, intent, micro_meta = stage1_micro_obstacle_propose(
        yaw, mix_yaw, left_d, front_d, right_d, state
    )
    chosen_yaw = float(micro_meta.get("chosen_yaw_rad", mix_yaw))
    meta = {
        "front_d": round(front_d, 3),
        "heading_err_deg": round(math.degrees(diff), 1),
        "mix_alpha": round(alpha, 2),
        "wp_yaw_deg": round(math.degrees(wp_yaw), 1),
        "heading_mode": h_mode,
        "mix_yaw_deg": round(math.degrees(mix_yaw), 1),
        "intent": intent,
    }
    meta.update(micro_meta)
    meta["chosen_yaw_deg"] = round(math.degrees(chosen_yaw), 1)
    if guide_yaw is not None:
        meta["guide_yaw_deg"] = round(math.degrees(guide_yaw), 1)
    if bearing_yaw is not None:
        meta["bearing_yaw_deg"] = round(math.degrees(bearing_yaw), 1)
    if return_meta:
        return action, meta
    return action

def stage1_motion_controller(agent, depth, memory, recovery, propose_action):
    """Stage1 path-follow（FOLLOW）：由 STAGE1_DISABLE_* 控制 recovery/振荡。"""
    return motion_controller(
        agent,
        depth,
        memory,
        recovery,
        propose_action,
        enable_recovery=not STAGE1_DISABLE_RECOVERY,
        allow_oscillation=not STAGE1_DISABLE_OSCILLATION,
    )

def stage1_skip_bad_waypoints(follower, pos, yaw, target_xz=None, nlog=None):
    """
    跳过有害 path 折点：
    1) 贴身且在身后（大 heading_err）；
    2) 折点方向明显偏离 target bearing，且到达折点不会更接近 target。
    """
    if not STAGE1_SKIP_BAD_WP:
        return 0
    skipped = 0
    while not follower.finished():
        wp = follower.current_waypoint()
        if wp is None:
            break
        wx, wz = float(wp[0]), float(wp[2])
        d_wp = follower._dist_to_current_wp(pos)
        if d_wp is None:
            break
        wp_yaw = yaw_toward_xz(pos, wx, wz)
        err_body, _ = heading_err_deg(yaw, wp_yaw)
        reason = None
        if (
            d_wp <= float(STAGE1_SKIP_BEHIND_WP_DIST_M)
            and abs(err_body) > float(STAGE1_SKIP_BEHIND_WP_DEG)
        ):
            reason = f"behind err={err_body:.0f}°"
        elif target_xz is not None and d_wp <= float(STAGE1_SKIP_DETOUR_WP_DIST_M):
            bearing = stage1_bearing_yaw(pos, target_xz)
            if bearing is not None:
                err_det, _ = heading_err_deg(bearing, wp_yaw)
                d_now = _dist_xz_to_target(pos, target_xz)
                d_at_wp = float(
                    np.hypot(
                        wx - float(target_xz[0]),
                        wz - float(target_xz[1]),
                    )
                )
                if abs(err_det) >= float(STAGE1_SKIP_DETOUR_BEARING_DEG):
                    if d_at_wp > d_now + float(STAGE1_SKIP_DETOUR_DIST_EPS_M):
                        reason = (
                            f"detour err={err_det:.0f}° "
                            f"d_wp_pt={d_at_wp:.2f}>{d_now:.2f}"
                        )
                    else:
                        reason = f"misaligned err={err_det:.0f}°"
        if reason is None:
            break
        follower.idx += 1
        follower._wp_enter_dist = None
        follower._wp_best_dist = None
        skipped += 1
        if nlog is not None:
            nlog(
                f"  [Stage1] 跳过 waypoint[{follower.idx - 1}] "
                f"[{wx:.2f},{wz:.2f}] reason={reason}"
            )
    if skipped and nlog is not None:
        nlog(
            f"  [Stage1] 累计跳过有害 waypoint ×{skipped} "
            f"→ wp_idx={follower.idx}/{len(follower.path_points)}"
        )
    return skipped

def _stage1_log_path_start(agent, goal, target_xz, follower, view=None, nlog=None):
    pos = agent.state.position
    yaw = yaw_from_rotation(agent.state.rotation)
    lines = [
        f"  [Stage1 start] agent=[{pos[0]:.2f},{pos[2]:.2f}] yaw={math.degrees(yaw):.1f}° "
        f"coordinator=path+escape+transit bug2_rejoin=strict",
        f"    goal=[{goal[0]:.2f},{goal[2]:.2f}]",
    ]
    d_tgt = None
    if target_xz is not None:
        tx, tz = float(target_xz[0]), float(target_xz[1])
        d_tgt = _dist_xz_to_target(pos, [tx, tz])
        bearing = stage1_bearing_yaw(pos, [tx, tz])
        err_b, _ = heading_err_deg(yaw, bearing)
        h_mode = stage1_heading_mode(d_tgt)
        lines.append(
            f"    target_approach=[{tx:.2f},{tz:.2f}] dist={d_tgt:.2f}m "
            f"bearing_err={err_b:.1f}° mode={h_mode}"
        )
    if view is not None and STAGE1_VIEW_YAW_IN_PATH:
        view_yaw = semantic_aim_yaw_from_view(view)
        if view_yaw is not None and d_tgt is not None:
            err_v, _ = heading_err_deg(yaw, view_yaw)
            lines.append(
                f"    view_aim_yaw={math.degrees(view_yaw):.1f}° "
                f"err_view={err_v:.1f}° (近距≤{STAGE1_VIEW_YAW_BLEND_DIST_M}m 时启用)"
            )
    elif view is not None:
        vx, vz = view.xz()
        lines.append(
            f"    view.pose=[{vx:.2f},{vz:.2f}] (仅 S2/S3，path 不混 view 航向)"
        )
    wp = follower.current_waypoint()
    if wp is not None:
        d_wp = follower._dist_to_current_wp(pos)
        err_w, _ = heading_err_deg(yaw, yaw_toward_xz(pos, float(wp[0]), float(wp[2])))
        lines.append(
            f"    wp[1]=[{float(wp[0]):.2f},{float(wp[2]):.2f}] "
            f"dist_wp={d_wp:.2f}m heading_err={err_w:.1f}° "
            f"path_len={len(follower.path_points)}"
        )
    for line in lines:
        if nlog is not None:
            nlog(line)

def stage1_mline_transit_propose(
    pos, yaw, goal_xz, front_d, dist_target, prev_dist_target, steer_state
):
    """退出墙后缓冲：只朝 target bearing，前进需 Δdist 改善。"""
    desired = stage1_bearing_yaw(pos, goal_xz)
    if desired is None:
        return "turn_left", "transit_no_bearing"
    err_deg, diff = heading_err_deg(yaw, desired)
    if abs(err_deg) > STAGE1_HEADING_ALIGN_DEG:
        turn = "turn_left" if diff > 0 else "turn_right"
        return steer_state.start_turn_commit(turn), "transit_align"
    if float(front_d) < float(STAGE1_MLINE_TRANSIT_FRONT_M):
        turn = "turn_left" if diff > 0 else "turn_right"
        return steer_state.start_turn_commit(turn), "transit_front_low"
    if (
        dist_target is not None
        and prev_dist_target is not None
        and float(dist_target)
        > float(prev_dist_target) + STAGE1_MLINE_TRANSIT_IMPROVE_EPS_M
    ):
        turn = "turn_left" if diff > 0 else "turn_right"
        return steer_state.start_turn_commit(turn), "transit_dist_worse"
    imp = stage1_bug2_progress_along_yaw(pos, desired, goal_xz)
    if imp <= float(STAGE1_MLINE_TRANSIT_PROGRESS_MIN_M):
        turn = "turn_left" if diff > 0 else "turn_right"
        return steer_state.start_turn_commit(turn), "transit_no_progress"
    return "move_forward", "transit_forward"

def _candidate_step_end(
    act_total,
    per_view_budget=NAV_PER_VIEW_BUDGET,
    global_max=NAV_MAX_STEPS,
    rank=1,
    n_candidates=1,
):
    """本候选预算：不超过 per_view，且与剩余全局步数、剩余候选数均分。"""
    act_total = int(act_total)
    remaining = max(0, int(global_max) - act_total)
    n_left = max(1, int(n_candidates) - int(rank) + 1)
    fair = max(STAGE1_MIN_CANDIDATE_BUDGET, remaining // n_left)
    budget = min(int(per_view_budget), fair)
    return min(act_total + budget, int(global_max))

def run_persistent_path_follow(
    sim,
    agent,
    goal,
    act,
    pf_act,
    step_end,
    memory=None,
    recovery=None,
    target_xz=None,
    view=None,
    neighborhood_m=None,
    nlog=None,
    on_step=None,
    trace=None,
    detector=None,
    target_en=None,
):
    """
    持久 path-follow：整条 path 最多 MAX_STEPS_PER_PATH 步后重新规划；
    dist(target_xz) < neighborhood_m 时立即停止（切换纯视觉）。
    step_end: act.total 绝对上限（每候选独立预算）。
    """
    if memory is None:
        memory = ActionMemory()
    if recovery is None:
        recovery = RecoveryState()
    follower = PersistentPathFollower(sim)
    pos = np.array(agent.state.position, dtype=np.float64)
    if not follower.set_goal(pos, goal, trace=trace, nlog=nlog):
        return False, "path_plan_failed"

    def _after_replan():
        y0 = yaw_from_rotation(agent.state.rotation)
        p0 = np.array(agent.state.position, dtype=np.float64)
        stage1_skip_bad_waypoints(
            follower, p0, y0, target_xz=target_xz, nlog=nlog
        )

    _after_replan()
    s1_start_xz = [float(pos[0]), float(pos[2])]
    s1_max_leave = 0.0
    s1_near_open_streak = 0

    _stage1_log_path_start(
        agent, goal, target_xz, follower, view=view, nlog=nlog
    )
    if target_xz is not None:
        bearing = stage1_bearing_yaw(agent.state.position, target_xz)
        if bearing is not None:
            n_pre = align_agent_yaw_rad(
                agent,
                bearing,
                max_turns=STAGE1_PREALIGN_MAX_TURNS,
                pf_act=pf_act,
            )
            if n_pre > 0 and nlog is not None:
                nlog(f"  [Stage1 prealign] 转向 bearing→target turns={n_pre}")
    elif STAGE1_SPAWN_ALIGN_YAW and target_xz is not None:
        n_pre = align_agent_yaw_toward_xz(
            agent,
            float(target_xz[0]),
            float(target_xz[1]),
            max_turns=STAGE1_PREALIGN_MAX_TURNS,
            pf_act=pf_act,
        )
        if n_pre > 0 and nlog is not None:
            nlog(f"  [Stage1 prealign] 转向 target_xz turns={n_pre}")
    path_steps = 0
    replans = 0
    near_blocked_rewinds = 0
    wp_idx_prev = follower.idx
    exit_reason = "budget_exhausted"
    prev_dist = None
    prev_dist_target = None
    forward_guard_steps = 0
    prev_pos = pos.copy()
    s1_no_progress = 0
    s1_best_dist = None
    wp_stuck_steps = 0
    wp_idx_loop = follower.idx
    steer_state = PathFollowSteeringState()
    loco = Stage1LocomotionState()
    coord = Stage1Coordinator()
    prev_moved = 0.0
    last_action = None
    prev_dist_goal = None
    prev_dist_for_streak = None

    while act.total < step_end:
        pos = np.array(agent.state.position, dtype=np.float64)
        dist_goal = _dist_xz(pos, goal)
        if prev_dist is not None:
            delta = dist_goal - prev_dist
        else:
            delta = 0.0

        if target_xz is not None and neighborhood_m is not None:
            d = _dist_xz_to_target(pos, target_xz)
            if STAGE1_ANTI_BACKTRACK:
                leave = float(
                    np.hypot(
                        float(pos[0]) - s1_start_xz[0],
                        float(pos[2]) - s1_start_xz[1],
                    )
                )
                s1_max_leave = max(s1_max_leave, leave)
                if (
                    s1_max_leave >= float(STAGE1_BACKTRACK_LEAVE_M)
                    and leave <= float(STAGE1_BACKTRACK_RETURN_M)
                    and s1_best_dist is not None
                    and d > s1_best_dist + STAGE1_IMPROVE_EPS_M
                    and s1_no_progress >= max(8, STAGE1_NO_PROGRESS_STEPS // 4)
                ):
                    exit_reason = "backtrack_to_start"
                    if nlog is not None:
                        nlog(
                            f"  Stage1 提前退出: 回绕起点 "
                            f"leave_max={s1_max_leave:.2f}m now={leave:.2f}m "
                            f"dist_target={d:.2f}m best={s1_best_dist:.2f}m"
                        )
                    break
            if s1_best_dist is not None:
                if d < s1_best_dist - STAGE1_IMPROVE_EPS_M:
                    s1_best_dist = d
                    s1_no_progress = 0
                else:
                    s1_no_progress += 1
                if (
                    s1_no_progress >= STAGE1_NO_PROGRESS_STEPS
                    and d > s1_best_dist + STAGE1_REGRESS_ABORT_M
                ):
                    exit_reason = "stuck_regress"
                    if nlog is not None:
                        nlog(
                            f"  Stage1 提前退出: {s1_no_progress} 步无进展 "
                            f"dist={d:.2f}m > best+{STAGE1_REGRESS_ABORT_M} "
                            f"(best={s1_best_dist:.2f}m)"
                        )
                    break
            if follower.idx == wp_idx_loop:
                wp_stuck_steps += 1
            else:
                wp_stuck_steps = 0
                wp_idx_loop = follower.idx
            if (
                target_xz is not None
                and wp_stuck_steps >= STAGE1_WP_STUCK_STEPS
                and s1_no_progress >= STAGE1_NO_PROGRESS_STEPS // 2
            ):
                exit_reason = "wp_stuck"
                if nlog is not None:
                    nlog(
                        f"  Stage1 提前退出: waypoint 卡住 idx={follower.idx} "
                        f"{wp_stuck_steps} 步"
                    )
                break
            near_limit = stage1_near_dist_limit(neighborhood_m)
            if d < near_limit:
                can_finish = True
                open_tag = ""
                if STAGE1_NEAR_OPEN_AREA_GATE:
                    obs_near = sim.get_sensor_observations()
                    depth_near = fill_depth(obs_near["depth"])
                    ld, fd, rd = measure_depth_probes(depth_near)
                    open_ok, open_tag = stage1_open_walkable_ok(ld, fd, rd)
                    if open_ok:
                        reach_ok, reach_tag = stage1_near_open_reachable_ok(
                            sim, pos, target_xz, fd, d
                        )
                        if not reach_ok:
                            open_ok = False
                            open_tag = reach_tag
                    if open_ok:
                        s1_near_open_streak += 1
                    else:
                        s1_near_open_streak = 0
                    can_finish = (
                        s1_near_open_streak >= int(STAGE1_NEAR_OPEN_STREAK)
                    )
                if can_finish:
                    exit_reason = (
                        "neighborhood_open"
                        if (
                            STAGE1_NEAR_OPEN_AREA_GATE
                            and d > float(NAV_COARSE_SUCCESS_M)
                        )
                        else "neighborhood"
                    )
                    if nlog is not None:
                        extra = (
                            f" open={open_tag}"
                            if STAGE1_NEAR_OPEN_AREA_GATE
                            else ""
                        )
                        nlog(
                            f"  dist={d:.2f}m < {near_limit:.2f}m{extra} "
                            f"→ 停止 path-follow，切换视觉"
                        )
                    break

        if follower.finished():
            exit_reason, should_break, replans, near_blocked_rewinds = (
                _stage1_on_path_exhausted(
                    sim,
                    pos,
                    target_xz,
                    neighborhood_m,
                    s1_best_dist,
                    follower,
                    goal,
                    replans,
                    near_blocked_rewinds,
                    nlog,
                    _after_replan,
                    steer_state,
                    trace=trace,
                    loco=loco,
                    coord=coord,
                )
            )
            if should_break:
                break
            continue

        follower.advance_if_needed(pos, thresh=PERSISTENT_WP_ADVANCE_M)
        if follower.idx > wp_idx_prev:
            wp_idx_prev = follower.idx

        if path_steps >= MAX_STEPS_PER_PATH:
            if not follower.set_goal(pos, goal, trace=trace, nlog=nlog):
                exit_reason = "replan_failed"
                if nlog is not None:
                    nlog("  path 重规划失败")
                break
            _after_replan()
            path_steps = 0
            replans += 1
            steer_state.clear_commit()
            if nlog is not None:
                nlog(f"  path 步数达 {MAX_STEPS_PER_PATH}，重新规划 (#{replans})")
            continue

        wp = follower.current_waypoint()
        if wp is None:
            exit_reason, should_break, replans, near_blocked_rewinds = (
                _stage1_on_path_exhausted(
                    sim,
                    pos,
                    target_xz,
                    neighborhood_m,
                    s1_best_dist,
                    follower,
                    goal,
                    replans,
                    near_blocked_rewinds,
                    nlog,
                    _after_replan,
                    steer_state,
                    trace=trace,
                    loco=loco,
                    coord=coord,
                )
            )
            if should_break:
                break
            continue

        dist_wp = float(
            np.linalg.norm(
                pos[[0, 2]] - np.array([float(wp[0]), float(wp[2])], dtype=np.float64)
            )
        )
        dist_target = (
            _dist_xz_to_target(pos, target_xz) if target_xz is not None else None
        )

        obs = sim.get_sensor_observations()
        depth = fill_depth(obs["depth"])
        yaw = yaw_from_rotation(agent.state.rotation)
        if on_step is not None:
            on_step(obs, depth, pos, yaw)

        mix_yaw, alpha, _wp_yaw, h_mode, _bearing_yaw = stage1_mix_yaw(
            pos, wp, target_xz, view, dist_target
        )
        yaw_err_deg, yaw_diff_rad = heading_err_deg(yaw, mix_yaw)
        committed = steer_state.consume_commit()
        left_d, front_d, right_d = measure_depth_probes(depth)
        goal_xz = (
            (float(target_xz[0]), float(target_xz[1]))
            if target_xz is not None
            else (float(goal[0]), float(goal[2]))
        )

        dist_for_streak = (
            dist_target if dist_target is not None else dist_goal
        )
        loco.update_streaks(
            prev_moved,
            dist_for_streak,
            prev_dist_for_streak,
            left_d=left_d,
            right_d=right_d,
        )
        bug2_mline_d = None
        if (
            loco.in_escape
            and STAGE1_BUG2_ENABLE
            and loco.bug2_hit_xz is not None
            and loco.bug2_mline_goal_xz is not None
        ):
            bug2_mline_d = stage1_bug2_dist_to_mline(
                pos, loco.bug2_hit_xz, loco.bug2_mline_goal_xz
            )
        mode_label = (
            loco.nav_mode
            if loco.in_escape
            else coord.mode
        )

        dist_ref = dist_target if dist_target is not None else dist_goal
        steer_override = None

        if loco.in_escape:
            past_ok, past_tag = Rejury.should_past_doorway_exit(
                pos,
                loco.bug2_hit_xz,
                bug2_mline_d,
                dist_ref,
                loco.escape_start_dist,
                escape_steps=loco.escape_total_steps,
                boundary_min_dist=loco.bug2_boundary_min_dist,
            )
            door_ok, door_tag = Rejury.should_doorway_seek(
                pos,
                target_xz,
                yaw,
                left_d,
                front_d,
                right_d,
                hit_xz=loco.bug2_hit_xz,
                mline_d=bug2_mline_d,
            )
            if past_ok:
                door_ok = True
                door_tag = past_tag
                loco.last_exit_reason = "past_doorway"
            elif door_ok:
                loco.last_exit_reason = "doorway_seek"
            if door_ok:
                wall_used = loco.wall_side
                if not loco.last_exit_reason:
                    loco.last_exit_reason = "doorway_seek"
                loco.exit_escape()
                coord.begin_goal_seek()
                steer_state.clear_commit()
                u_exit = "u_exit" in str(door_tag)
                if u_exit:
                    coord.begin_doorway_transit()
                    steer_state.start_turn_commit(
                        "turn_left", steps=STAGE1_DOORWAY_TRANSIT_TURN_STEPS
                    )
                    if STAGE1_REJOIN_REPLAN_PATH and follower.set_goal(
                        pos, goal, trace=trace, nlog=nlog
                    ):
                        _after_replan()
                        path_steps = 0
                        if nlog is not None:
                            nlog("  [COORD] u_exit doorway → path 重规划 + 左转进门")
                    steer_override = (
                        "turn_left",
                        {
                            "intent": "doorway_u_exit_turn",
                            "controller_mode": STAGE1_COORD_GOAL_SEEK,
                            "heading_err_deg": round(yaw_err_deg, 1),
                            "mix_alpha": round(alpha, 2),
                            "front_d": round(front_d, 3),
                            "doorway_transit_rem": coord.doorway_transit_rem,
                        },
                    )
                msg = (
                    f"  [COORD] WALL doorway_seek {door_tag} "
                    f"wall_side={wall_used} → {STAGE1_COORD_GOAL_SEEK}"
                    + (" +doorway_transit" if u_exit else "")
                )
                if nlog is not None:
                    nlog(msg)
            elif (
                STAGE1_WALL_SEMANTIC_REACQUIRE
                and detector is not None
                and target_en
            ):
                raw_det = detect_target_disambiguated(detector, obs["color"], target_en)
                geo_dist = None
                if sim is not None and target_xz is not None:
                    geo_dist = geodesic_distance_xz(
                        sim,
                        float(pos[0]),
                        float(pos[2]),
                        float(target_xz[0]),
                        float(target_xz[1]),
                    )
                reacq_ok, reacq_tag, reacq_info, _rejury = Rejury.evaluate_reacquire(
                    raw_det,
                    depth,
                    yaw,
                    left_d,
                    front_d,
                    right_d,
                    pos,
                    target_xz,
                    geo_dist=geo_dist,
                )
                if reacq_ok:
                    wall_used = loco.wall_side
                    loco.last_exit_reason = "semantic_reacquire"
                    loco.exit_escape()
                    coord.begin_goal_seek()
                    steer_state.clear_commit()
                    msg = (
                        f"  [COORD] WALL Rejury reacquire {reacq_tag} "
                        f"wall_side={wall_used} → {STAGE1_COORD_GOAL_SEEK}"
                    )
                    if nlog is not None:
                        nlog(msg)
            if loco.in_escape:
                frozen_yaw = loco.frozen_desired_yaw
                should_exit, exit_reason_esc = loco.should_exit_escape(
                    front_d,
                    dist_ref,
                    yaw,
                    frozen_yaw,
                    pos=pos,
                    goal_xz=goal_xz,
                    session_best_dist=s1_best_dist,
                )
                if not should_exit:
                    exit_reason_esc = ""
            if loco.in_escape and should_exit:
                loco.last_exit_reason = exit_reason_esc
                rejoin_open_finish = False
                open_tag = ""
                if (
                    STAGE1_BUG2_REJOIN_OPEN_FINISH
                    and exit_reason_esc == "bug2_rejoin"
                    and dist_ref is not None
                    and float(dist_ref) <= float(STAGE1_BUG2_REJOIN_OPEN_FINISH_D)
                    and target_xz is not None
                    and neighborhood_m is not None
                ):
                    fin_ok, _fin_exit, fin_tag = stage1_open_near_finish_reason(
                        sim,
                        pos,
                        target_xz,
                        neighborhood_m,
                        session_best_dist=s1_best_dist,
                    )
                    open_tag = fin_tag
                    rejoin_open_finish = bool(fin_ok)
                if rejoin_open_finish:
                    msg = (
                        f"  [COORD] bug2_rejoin @ dist={float(dist_ref):.2f}m "
                        f"open={open_tag} → 直接结束 Stage1（跳过 MLINE_TRANSIT）"
                    )
                    if nlog is not None:
                        nlog(msg)
                    loco.exit_escape()
                    coord.begin_goal_seek()
                    steer_state.clear_commit()
                    exit_reason = "neighborhood_open"
                    break
                if exit_reason_esc == "bug2_rejoin":
                    coord.begin_mline_transit(exit_reason_esc)
                    next_mode = STAGE1_COORD_MLINE_TRANSIT
                else:
                    coord.begin_goal_seek()
                    next_mode = STAGE1_COORD_GOAL_SEEK
                msg = (
                    f"  [COORD] exit WALL reason={exit_reason_esc} "
                    f"steps={loco.escape_total_steps} "
                    f"front={front_d:.2f} dist_target={dist_ref:.2f} "
                    f"dist_at_hit={loco.escape_start_dist} "
                    f"mline_peak={loco.bug2_peak_mline_d:.2f} "
                    f"saw_far={loco.bug2_saw_mline_far} → {next_mode}"
                )
                if nlog is not None:
                    nlog(msg)
                loco.exit_escape()
                coord.begin_goal_seek()
                steer_state.clear_commit()
                if (
                    exit_reason_esc == "bug2_rejoin"
                    and STAGE1_REJOIN_REPLAN_PATH
                ):
                    if follower.set_goal(pos, goal, trace=trace, nlog=nlog):
                        _after_replan()
                        path_steps = 0
                        if nlog is not None:
                            nlog("  [COORD] bug2_rejoin → path 重规划")
        elif coord.in_mline_transit():
            coord.transit_steps += 1
            coord.tick_transit_improve(dist_target, prev_dist_target)
            if coord.transit_should_finish():
                coord.end_mline_transit()
                if STAGE1_REJOIN_REPLAN_PATH and follower.set_goal(
                    pos, goal, trace=trace, nlog=nlog
                ):
                    _after_replan()
                    path_steps = 0
                msg = (
                    f"  [COORD] MLINE_TRANSIT done streak={coord.transit_improve_streak} "
                    f"→ {STAGE1_COORD_GOAL_SEEK}"
                )
                if nlog is not None:
                    nlog(msg)
            elif coord.transit_expired():
                coord.end_mline_transit()
                if nlog is not None:
                    nlog(
                        f"  [COORD] MLINE_TRANSIT timeout {coord.transit_steps} "
                        f"→ {STAGE1_COORD_GOAL_SEEK}"
                    )
            elif (
                float(front_d) < STAGE1_MLINE_TRANSIT_FRONT_M
                and loco.no_move_streak >= 6
            ):
                coord.end_mline_transit()
                mline_wall_blocked, _mw_tag = Rejury.should_block_wall_enter(
                    pos,
                    target_xz,
                    yaw,
                    left_d,
                    front_d,
                    right_d,
                    hit_xz=loco.bug2_hit_xz,
                )
                if not mline_wall_blocked:
                    goal_bearing = (
                        stage1_bearing_yaw(pos, target_xz)
                        if target_xz is not None
                        else mix_yaw
                    )
                    loco.enter_escape(
                        mix_yaw,
                        left_d,
                        right_d,
                        front_d,
                        dist_ref,
                        yaw=yaw,
                        goal_bearing_yaw=goal_bearing,
                        pos=pos,
                        goal_xz=goal_xz,
                    )
                    coord.begin_wall_follow()
                    steer_state.clear_commit()
                    if nlog is not None:
                        nlog(
                            f"  [COORD] MLINE_TRANSIT 受阻 → 再入 WALL "
                            f"side={loco.wall_side}"
                        )
                elif nlog is not None:
                    nlog(
                        f"  [COORD] MLINE 受阻但门口开阔，保持 {STAGE1_COORD_GOAL_SEEK}"
                    )
                    coord.begin_goal_seek()
        elif loco.should_trigger_escape(
            front_d,
            pos=pos,
            target_xz=target_xz,
            yaw=yaw,
            left_d=left_d,
            right_d=right_d,
        ):
            goal_bearing = (
                stage1_bearing_yaw(pos, target_xz)
                if target_xz is not None
                else mix_yaw
            )
            loco.enter_escape(
                mix_yaw,
                left_d,
                right_d,
                front_d,
                dist_ref,
                yaw=yaw,
                goal_bearing_yaw=goal_bearing,
                pos=pos,
                goal_xz=goal_xz,
            )
            coord.begin_wall_follow()
            steer_state.clear_commit()
            msg = (
                f"  [COORD] enter WALL {loco.nav_mode} "
                f"wall_side={loco.wall_side} (committed) "
                f"no_move={loco.no_move_streak} goal_stall={loco.goal_stall_streak} "
                f"dist_at_hit={loco.escape_start_dist:.2f} "
                f"L/F/R={left_d:.2f}/{front_d:.2f}/{right_d:.2f} "
                f"bug2_hit={[round(loco.bug2_hit_xz[0],2), round(loco.bug2_hit_xz[1],2)] if loco.bug2_hit_xz else None}"
            )
            if nlog is not None:
                nlog(msg)
        elif (
            STAGE1_WALL_ENTER_DOORWAY_GATE
            and target_xz is not None
            and (
                loco.no_move_streak >= STAGE1_ESCAPE_ENTER_NO_MOVE_STEPS
                or (
                    float(front_d) < STAGE1_ESCAPE_ENTER_FRONT_M
                    and loco.goal_stall_streak >= STAGE1_ESCAPE_GOAL_STALL_STEPS
                )
            )
        ):
            _skip, skip_tag = Rejury.should_block_wall_enter(
                pos,
                target_xz,
                yaw,
                left_d,
                front_d,
                right_d,
                hit_xz=loco.bug2_hit_xz,
            )
            if _skip:
                if (
                    "doorway" in str(skip_tag)
                    or "toward_free" in str(skip_tag)
                    or "u_exit" in str(skip_tag)
                ):
                    steer_state.start_turn_commit("turn_left")
        if steer_override is not None:
            proposed, meta = steer_override[0], steer_override[1]
        elif loco.in_escape:
            frozen_yaw = float(loco.frozen_desired_yaw)
            proposed, intent = loco.tick_escape_action(left_d, front_d, right_d)
            steer_state.clear_commit()
            meta = {
                "intent": intent,
                "controller_mode": loco.nav_mode,
                "heading_err_deg": round(
                    heading_err_deg(yaw, frozen_yaw)[0], 1
                ),
                "mix_alpha": 0.0,
                "heading_mode": "frozen",
                "front_d": round(front_d, 3),
                "frozen_desired_deg": round(math.degrees(frozen_yaw), 1),
                "commit_rem": loco.commitment_remaining,
                "escape_phase": loco.phase,
                "escape_phase_rem": loco.phase_remaining,
            }
        elif coord.in_mline_transit():
            proposed, intent = stage1_mline_transit_propose(
                pos,
                yaw,
                goal_xz,
                front_d,
                dist_target,
                prev_dist_target,
                steer_state,
            )
            bearing = stage1_bearing_yaw(pos, goal_xz)
            tr_err = (
                heading_err_deg(yaw, bearing)[0]
                if bearing is not None
                else yaw_err_deg
            )
            meta = {
                "intent": intent,
                "controller_mode": STAGE1_COORD_MLINE_TRANSIT,
                "heading_err_deg": round(tr_err, 1),
                "mix_alpha": round(
                    stage1_semantic_alpha(dist_target) if dist_target else 0.5,
                    2,
                ),
                "heading_mode": "transit_bearing",
                "front_d": round(front_d, 3),
                "transit_steps": coord.transit_steps,
                "transit_streak": coord.transit_improve_streak,
            }
        elif coord.in_doorway_transit():
            guide_yaw, _ = stage1_guide_yaw(pos, target_xz, view, dist_target)
            align_yaw = (
                float(guide_yaw) if guide_yaw is not None else float(mix_yaw)
            )
            align_err_deg, align_diff = heading_err_deg(yaw, align_yaw)
            left_open = float(left_d) >= float(right_d) + 0.4
            if (
                left_open
                and abs(align_err_deg) > float(STAGE1_DOORWAY_TRANSIT_ALIGN_DEG)
            ):
                proposed = steer_state.start_turn_commit(
                    "turn_left", steps=STAGE1_DOORWAY_TRANSIT_TURN_STEPS
                )
                intent = "doorway_transit_turn"
            elif (
                float(front_d) >= PATH_FOLLOW_FRONT_MIN_M
                and abs(align_err_deg) <= float(STAGE1_DOORWAY_TRANSIT_ALIGN_DEG)
            ):
                proposed = "move_forward"
                intent = "doorway_transit_forward"
                coord.end_doorway_transit()
            else:
                proposed = steer_state.start_turn_commit(
                    "turn_left", steps=STAGE1_DOORWAY_TRANSIT_TURN_STEPS
                )
                intent = "doorway_transit_turn"
            coord.tick_doorway_transit()
            meta = {
                "intent": intent,
                "controller_mode": STAGE1_COORD_GOAL_SEEK,
                "heading_err_deg": round(align_err_deg, 1),
                "mix_alpha": round(alpha, 2),
                "front_d": round(front_d, 3),
                "doorway_transit_rem": coord.doorway_transit_rem,
            }
        elif committed is not None:
            proposed = committed
            meta = {
                "intent": "turn_commit",
                "controller_mode": STAGE1_COORD_GOAL_SEEK,
                "heading_err_deg": round(yaw_err_deg, 1),
                "mix_alpha": round(alpha, 2),
                "front_d": round(front_d, 3),
            }
        elif abs(yaw_err_deg) > STAGE1_HEADING_ALIGN_DEG:
            proposed = steer_state.start_turn_commit(
                "turn_left" if yaw_diff_rad > 0 else "turn_right"
            )
            meta = {
                "intent": "heading_align",
                "controller_mode": STAGE1_COORD_GOAL_SEEK,
                "heading_err_deg": round(yaw_err_deg, 1),
                "mix_alpha": round(alpha, 2),
                "heading_mode": h_mode,
                "front_d": round(front_d, 3),
            }
        else:
            proposed, meta = path_follow_steering_persistent(
                wp,
                pos,
                yaw,
                depth,
                state=steer_state,
                target_xz=target_xz,
                view=view,
                dist_target=dist_target,
                return_meta=True,
                nlog=nlog,
            )

        if (
            not loco.in_escape
            and not coord.in_mline_transit()
            and not coord.in_doorway_transit()
            and proposed == "move_forward"
            and dist_target is not None
            and prev_dist_target is not None
            and dist_target > prev_dist_target + STAGE1_FORWARD_GUARD_EPS_M
        ):
            forward_guard_steps += 1
        else:
            forward_guard_steps = 0

        if forward_guard_steps >= STAGE1_FORWARD_GUARD_STEPS:
            guard_alpha = max(alpha, STAGE1_SEMANTIC_ALPHA_NEAR)
            guide_yaw, _guard_mode = stage1_guide_yaw(
                pos, target_xz, view, dist_target
            )
            if guide_yaw is None:
                guide_yaw = _bearing_yaw if _bearing_yaw is not None else _wp_yaw
            guard_yaw = angle_mix_yaw(_wp_yaw, guide_yaw, guard_alpha)
            _, guard_diff = heading_err_deg(yaw, guard_yaw)
            proposed = steer_state.start_turn_commit(
                "turn_left" if guard_diff > 0 else "turn_right"
            )
            meta = {
                "intent": "forward_guard",
                "heading_err_deg": round(math.degrees(guard_diff), 1),
                "mix_alpha": round(guard_alpha, 2),
                "front_d": None,
            }
            forward_guard_steps = 0

        def _propose():
            return proposed

        action = stage1_motion_controller(
            agent, depth, memory, recovery, _propose
        )
        yaw_diff_deg = float(
            meta.get("heading_err_deg") if meta.get("heading_err_deg") is not None else yaw_err_deg
        )
        target_s = (
            f" dist_target={dist_target:.2f}" if dist_target is not None else ""
        )
        alpha_s = f" α={meta.get('mix_alpha', alpha):.2f} mode={meta.get('heading_mode', h_mode)}"
        dist_real_s = ""
        if prev_dist_target is not None and dist_target is not None:
            dist_real_s = f" Δdist_real={float(prev_dist_target) - float(dist_target):+.2f}"
        pf_act(action)
        new_pos = np.array(agent.state.position, dtype=np.float64)
        moved_dist = float(
            np.linalg.norm(new_pos[[0, 2]] - prev_pos[[0, 2]])
        )
        prev_pos = new_pos
        prev_moved = moved_dist
        last_action = action
        prev_dist = dist_goal
        prev_dist_goal = dist_goal
        prev_dist_for_streak = dist_for_streak
        if dist_target is not None:
            prev_dist_target = dist_target
        path_steps += 1
    if exit_reason == "path_finished" and target_xz is not None:
        best_d = s1_best_dist
        fin_ok, fin_exit, _fin_tag = stage1_open_near_finish_reason(
            sim,
            agent.state.position,
            target_xz,
            neighborhood_m,
            session_best_dist=best_d,
        )
        if fin_ok:
            exit_reason = fin_exit
            if nlog is not None:
                nlog(
                    f"  path 收尾复检通过 exit={exit_reason} "
                    f"(session_best={best_d:.2f}m)"
                )
    return (
        exit_reason in ("neighborhood", "neighborhood_open"),
        exit_reason,
    )

@dataclass
class NavTraceRecorder:
    """导航 path 记录：waypoint path 供 path 规划。"""

    waypoint_paths: list = field(default_factory=list)

    def add_waypoint_path(self, path_points):
        if not path_points:
            return
        self.waypoint_paths.append(
            [(float(p[0]), float(p[2])) for p in path_points]
        )


def make_tracked_pf_act(agent, act, recovery, trace=None):
    """包装 pf_act：执行动作并累计步数。"""

    def pf_act(name):
        agent.act(name)
        act.total += 1
        if name == "move_forward":
            act.forward += 1

    return pf_act

# ─── 局部执行层（障碍避让 / 路径跟随）──────────────────────
def local_obstacle_action(agent, depth):
    """局部避障（预探索漫游）：recovery 开启；较保守前向深度阈值。"""
    left_d, front_d, right_d = measure_depth_probes(depth)
    if front_d >= LOCAL_OBSTACLE_FRONT_MIN_M:
        return "move_forward"
    return "turn_left" if left_d > right_d else "turn_right"

class ExploreRegionState:
    """预探索：select_next_frontier_anchor → navigate_to(best_anchor)。"""

    def __init__(self):
        self.target_anchor = None
        self.goal = None
        self.steps_toward = 0

def _explore_propose_action(sim, agent, topo_map, depth, step, state, memory):
    pos = agent.state.position
    x, z = float(pos[0]), float(pos[2])
    yaw = yaw_from_rotation(agent.state.rotation)
    left_d, front_d, right_d = measure_depth_probes(depth)
    topo_map.register_local_anchor(x, z, yaw, left_d, front_d, right_d, step)

    need_new = state.target_anchor is None
    if state.target_anchor is not None and state.goal is not None:
        if _dist_xz(pos, state.goal) < REGION_CENTER_REACH_M:
            need_new = True
        elif (
            state.steps_toward >= PRE_REGION_STUCK_STEPS
            and memory.stuck_steps >= STUCK_TRIGGER_STEPS
        ):
            need_new = True

    if need_new:
        state.target_anchor = topo_map.select_next_frontier_anchor(x, z, step, memory)
        state.goal = None
        state.steps_toward = 0
        if state.target_anchor is not None:
            state.goal = topo_map.navigable_goal_from_anchor(sim, state.target_anchor)

    if state.goal is not None:
        state.steps_toward += 1
        return steer_toward(sim, state.goal, pos, yaw, depth, front_d, left_d, right_d)
    return local_obstacle_action(agent, depth)

def explore_region_policy_step(sim, agent, topo_map, depth, step, state, memory, recovery):
    return motion_controller(
        agent,
        depth,
        memory,
        recovery,
        lambda: _explore_propose_action(sim, agent, topo_map, depth, step, state, memory),
        enable_recovery=True,
    )

def steer_toward(sim, goal, pos, yaw, depth, front_d, left_d, right_d):
    dx = float(goal[0] - pos[0])
    dz = float(goal[2] - pos[2])
    target_yaw = float(np.arctan2(dx, dz))
    diff = (target_yaw - yaw + np.pi) % (2 * np.pi) - np.pi
    if abs(diff) > PATH_FOLLOW_TURN_THRESH:
        return "turn_left" if diff > 0 else "turn_right"
    if front_d < LOCAL_OBSTACLE_FRONT_MIN_M:
        return "turn_left" if left_d > right_d else "turn_right"
    return "move_forward"

def bbox_front_depth(depth, bbox, h, w):
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    patch = depth[max(0, y0):y1, max(0, x0):x1]
    valid = patch[np.isfinite(patch) & (patch > 0.05) & (patch < 10.0)]
    if len(valid) < 4:
        return None
    return float(np.percentile(valid, 15))

def _empty_det():
    return {"found": False, "confidence": 0.0, "bbox": None, "center_x": 0.5, "bbox_ratio": 0.0}

def _det_depth(det, depth):
    if not det.get("found"):
        return None
    h, w = depth.shape[:2]
    td = bbox_front_depth(depth, det.get("bbox"), h, w)
    if td is None:
        front = depth[160:320, 240:400]
        v = front[np.isfinite(front) & (front > 0.05)]
        td = float(np.percentile(v, 20)) if len(v) else 5.0
    return float(td)

def _det_depth_stage3(det, depth, front_d=None, compact=False):
    """
    bbox 很小时（贴近沙发/只看到局部），bbox 深度常偏大；
    与前方探针 front_d 取 min 作为融合深度。
    门/椅等小目标：始终与 front_d 取 min，避免大 bbox 墙/框误判 depth_near。
    """
    td_bbox = _det_depth(det, depth)
    if not det.get("found"):
        return td_bbox
    br = float(det.get("bbox_ratio", 0.0))
    if front_d is None:
        return td_bbox
    fd = float(front_d)
    if td_bbox is None:
        return fd
    if compact or br < STAGE3_FUSE_FRONT_MAX_BBOX:
        return min(float(td_bbox), fd)
    return float(td_bbox)

def stage3_arrival_kind(target_en):
    """compact=门/椅；bulky=沙发/床；未知类用 SHAPE_PRIORS.max_area 推断。"""
    key = str(target_en or "")
    kind = STAGE3_ARRIVAL_KIND_BY_TARGET.get(key)
    if kind:
        return kind
    priors = SHAPE_PRIORS.get(key, {})
    max_area = float(priors.get("max_area", 0.4))
    if max_area >= 0.45:
        return STAGE3_ARRIVAL_KIND_BULKY
    if max_area <= 0.40:
        return STAGE3_ARRIVAL_KIND_COMPACT
    return STAGE3_ARRIVAL_KIND_DEFAULT

def stage3_is_compact_target(target_en):
    return stage3_arrival_kind(target_en) == STAGE3_ARRIVAL_KIND_COMPACT

def stage3_is_bulky_target(target_en):
    return stage3_arrival_kind(target_en) == STAGE3_ARRIVAL_KIND_BULKY

def stage3_arrive_min_conf(target_en):
    if str(target_en or "") == "bed":
        return float(STAGE3_BED_ARRIVE_MIN_CONF)
    if stage3_is_compact_target(target_en):
        return STAGE3_COMPACT_MIN_CONF
    if stage3_is_bulky_target(target_en):
        return STAGE3_BULKY_MIN_CONF
    return STAGE3_SUCCESS_MIN_CONF

def stage3_is_bed_target(target_en):
    return str(target_en or "") == "bed"

def stage3_lock_conf(target_en):
    return float(STAGE3_LOCK_CONF_BY_TARGET.get(str(target_en or ""), STAGE3_LOCK_CONF))

def stage3_lock_bbox(target_en):
    return float(STAGE3_LOCK_BBOX_BY_TARGET.get(str(target_en or ""), STAGE3_LOCK_BBOX))

def stage3_compact_arrive_max_bbox(target_en):
    if str(target_en or "") == "door":
        return float(STAGE3_DOOR_ARRIVE_MAX_BBOX)
    return float(STAGE3_COMPACT_ARRIVE_MAX_BBOX)

def stage3_lock_frames_needed(target_en, conf, br):
    need = int(STAGE3_LOCK_FRAMES)
    if str(target_en or "") == "door":
        if conf >= STAGE3_LOCK_FAST_CONF and stage3_lock_bbox(target_en) <= br <= STAGE3_DOOR_LOCK_MAX_BBOX:
            return 1
    return need

def stage3_door_bbox_aspect_ok(det):
    """门框应偏高窄；壁画/整墙多为大框+近方形。"""
    if not det.get("found") or not det.get("bbox"):
        return False
    x0, y0, x1, y1 = det["bbox"]
    bw = max(1, int(x1) - int(x0))
    bh = max(1, int(y1) - int(y0))
    aspect = bw / float(bh)
    return STAGE3_DOOR_MIN_ASPECT <= aspect <= STAGE3_DOOR_MAX_ASPECT

def stage3_yaw_err_rad(yaw_a, yaw_b):
    return abs((float(yaw_a) - float(yaw_b) + math.pi) % (2 * math.pi) - math.pi)

def stage3_recovery_allowed(front_d, det, target_en, agent_yaw=None, aim_yaw=None):
    """门已检出但前方极开阔：禁止 recovery 直行冲过门框/走廊。"""
    if not det.get("found") or not stage3_is_compact_target(target_en):
        return True
    if str(target_en or "") != "door":
        return True
    br = float(det.get("bbox_ratio", 0.0))
    conf = float(det.get("confidence", 0.0))
    fd = float(front_d) if front_d is not None else 99.0
    if conf >= 0.45 and br < 0.22 and fd > STAGE3_DOOR_RECOVERY_FRONT_GAP + 2.0:
        return False
    if agent_yaw is not None and aim_yaw is not None:
        err_deg = math.degrees(stage3_yaw_err_rad(agent_yaw, aim_yaw))
        if err_deg > STAGE3_DOOR_AIM_YAW_MAX_ERR_DEG and br >= 0.22:
            return False
    return True

def near_arrival_check_stage3(
    det, depth, front_d=None, locked=False, max_bbox=0.0, target_en=None
):
    """
    到达判据（按 arrival_kind，见 stage3_arrival_kind）：
      compact: depth_near(严) + 须锁定；融合 depth 与 front 一致性检查
      bulky:   depth_near OR bulky_locked(锁定+历史大框+depth<1.68) OR 贴边
      default: 同 bulky 宽松度，无 bulky_locked
    """
    out = {
        "ok": False,
        "tier": None,
        "td": None,
        "td_bbox": None,
        "front_d": front_d,
        "conf": 0.0,
        "cx": 0.5,
        "bbox_ratio": 0.0,
        "found": bool(det.get("found")),
        "fails": [],
    }
    if not out["found"]:
        out["fails"].append("no_detect")
        return out
    out["conf"] = float(det["confidence"])
    out["cx"] = float(det.get("center_x", 0.5))
    out["bbox_ratio"] = float(det.get("bbox_ratio", 0.0))
    out["td_bbox"] = _det_depth(det, depth)
    kind = stage3_arrival_kind(target_en)
    compact = kind == STAGE3_ARRIVAL_KIND_COMPACT
    bulky = kind == STAGE3_ARRIVAL_KIND_BULKY
    min_conf = stage3_arrive_min_conf(target_en)
    out["arrival_kind"] = kind
    td = _det_depth_stage3(det, depth, front_d, compact=compact)
    out["td"] = td
    if td is None:
        out["fails"].append("depth")
        return out
    br = out["bbox_ratio"]
    conf = out["conf"]
    cx = out["cx"]
    if compact:
        saw_target = locked
    else:
        saw_target = locked or float(max_bbox) >= STAGE3_ARRIVE_EDGE_MIN_MAX_BBOX
    cx_ok_edge = STAGE3_ARRIVE_EDGE_CENTER_LO <= cx <= STAGE3_ARRIVE_EDGE_CENTER_HI
    cx_ok_huge = 0.30 <= cx <= 0.70

    depth_near = td < STAGE3_ARRIVE_DEPTH_CLOSE and conf >= min_conf
    if compact:
        if not locked:
            depth_near = False
        max_br = stage3_compact_arrive_max_bbox(target_en)
        if br > max_br:
            depth_near = False
        if str(target_en or "") == "door" and depth_near and not stage3_door_bbox_aspect_ok(det):
            depth_near = False
    if stage3_is_bed_target(target_en):
        if not locked:
            depth_near = False
        if br < STAGE3_BED_ARRIVE_MIN_BBOX or br > STAGE3_BED_ARRIVE_MAX_BBOX:
            depth_near = False
    if (
        compact
        and depth_near
        and front_d is not None
        and out["td_bbox"] is not None
        and float(front_d) > STAGE3_ARRIVE_DEPTH_CLOSE + STAGE3_DEPTH_FRONT_MAX_GAP
    ):
        depth_near = False
    bulky_locked = (
        bulky
        and locked
        and float(max_bbox) >= STAGE3_BULKY_MIN_MAX_BBOX
        and td < STAGE3_BULKY_DEPTH_MAX
        and conf >= STAGE3_BULKY_MIN_CONF
        and cx_ok_edge
    )
    edge_locked = (
        bulky
        and locked
        and br >= STAGE3_BULKY_EDGE_BBOX_LOCKED
        and conf >= STAGE3_ARRIVE_EDGE_CONF
        and cx_ok_edge
    )
    edge_at = (
        not compact
        and br >= STAGE3_ARRIVE_EDGE_BBOX
        and conf >= STAGE3_ARRIVE_EDGE_CONF
        and cx_ok_edge
        and saw_target
    )
    edge_huge = (
        not compact
        and br >= STAGE3_ARRIVE_EDGE_HUGE_BBOX
        and conf >= (min_conf if compact else NAV_ARRIVE_CONF)
        and cx_ok_huge
        and (
            locked
            if compact
            else (locked or float(max_bbox) >= STAGE3_LOCK_BBOX)
        )
    )

    if depth_near:
        out["ok"] = True
        out["tier"] = "depth_near"
    elif bulky_locked:
        out["ok"] = True
        out["tier"] = "bulky_locked"
    elif edge_huge:
        out["ok"] = True
        out["tier"] = "edge_huge"
    elif edge_at:
        out["ok"] = True
        out["tier"] = "edge_at"
    elif edge_locked:
        out["ok"] = True
        out["tier"] = "edge_locked"
    else:
        if conf < min_conf:
            out["fails"].append("conf")
        if not depth_near and not bulky_locked:
            out["fails"].append("depth")
        if not edge_at and not edge_huge and not edge_locked:
            out["fails"].append("edge")
        if not saw_target:
            out["fails"].append("no_target_history")
    return out

def stage3_confirm_success(
    check, effective, track, arrive_streak, forwards_in_cand, target_en=None
):
    """最终到达闸门：沙发/床须融合深度足够近；贴边档仅作逼近中的提示，不能远距伪成功。"""
    if arrive_streak < STAGE3_ARRIVE_STREAK:
        return False, "streak"
    if forwards_in_cand < STAGE3_ARRIVE_MIN_FORWARDS:
        return False, "forwards"
    if STAGE3_ARRIVE_REJECT_SYNTHETIC and effective.get("synthetic"):
        return False, "synthetic"
    compact = stage3_is_compact_target(target_en)
    if compact:
        if not track.locked:
            return False, "compact_need_lock"
    elif stage3_is_bed_target(target_en):
        if not track.locked:
            return False, "bed_need_lock"
    elif not (track.locked or track.max_bbox >= STAGE3_LOCK_BBOX):
        return False, "no_lock_history"
    if not check.get("ok"):
        return False, "check"
    conf = float(effective.get("confidence", 0.0))
    min_conf = stage3_arrive_min_conf(target_en)
    if conf < min_conf:
        return False, f"conf({conf:.2f}<{min_conf})"
    tier = check.get("tier") or ""
    if tier == "depth_near":
        td = check.get("td")
        if td is None or float(td) >= STAGE3_ARRIVE_DEPTH_CLOSE:
            td_s = "n/a" if td is None else f"{float(td):.2f}"
            return False, f"depth_near({td_s}>={STAGE3_ARRIVE_DEPTH_CLOSE})"
        if compact:
            br = float(effective.get("bbox_ratio", 0.0))
            max_br = stage3_compact_arrive_max_bbox(target_en)
            if br > max_br:
                return False, f"compact_bbox({br:.3f}>{max_br:.2f})"
            lock_cap = max_br
            if str(target_en or "") == "door":
                lock_cap = float(STAGE3_DOOR_LOCK_AT_ARRIVE_MAX_BBOX)
                if track.locked_steps < STAGE3_DOOR_MIN_LOCKED_STEPS:
                    return False, f"door_lock_steps({track.locked_steps}<{STAGE3_DOOR_MIN_LOCKED_STEPS})"
                if not stage3_door_bbox_aspect_ok(effective):
                    return False, "door_aspect"
            if track.lock_at_bbox > lock_cap + 0.02:
                return (
                    False,
                    f"compact_lock_bbox({track.lock_at_bbox:.3f}>{lock_cap:.2f})",
                )
        if stage3_is_bed_target(target_en):
            if not track.locked:
                return False, "bed_need_lock"
            br = float(effective.get("bbox_ratio", 0.0))
            if br < STAGE3_BED_ARRIVE_MIN_BBOX or br > STAGE3_BED_ARRIVE_MAX_BBOX:
                return False, f"bed_bbox({br:.3f})"
            if track.locked_steps < STAGE3_BED_MIN_LOCKED_STEPS:
                return (
                    False,
                    f"bed_lock_steps({track.locked_steps}<{STAGE3_BED_MIN_LOCKED_STEPS})",
                )
            if track.lock_at_bbox < STAGE3_BED_ARRIVE_MIN_BBOX:
                return False, f"bed_lock_bbox({track.lock_at_bbox:.3f})"
    elif tier == "bulky_locked":
        if not track.locked:
            return False, "bulky_need_lock"
        td = check.get("td")
        if td is None or float(td) >= STAGE3_BULKY_DEPTH_MAX:
            td_s = "n/a" if td is None else f"{float(td):.2f}"
            return False, f"bulky_depth({td_s}>={STAGE3_BULKY_DEPTH_MAX})"
        if track.max_bbox < STAGE3_BULKY_MIN_MAX_BBOX:
            return False, f"bulky_max_bbox({track.max_bbox:.3f})"
        if str(target_en or "") == "bed":
            br = float(effective.get("bbox_ratio", 0.0))
            fd = check.get("front_d")
            if fd is not None and float(fd) > float(td) + STAGE3_BED_BULKY_FRONT_GAP:
                return False, f"bed_open_front({float(fd):.2f}>{float(td):.2f})"
            if br >= 0.42:
                return False, f"bed_bbox_wide({br:.3f})"
    elif tier in ("edge_at", "edge_huge", "edge_locked"):
        br = float(effective.get("bbox_ratio", 0.0))
        if tier == "edge_at" and br < STAGE3_ARRIVE_EDGE_BBOX:
            return False, f"edge_bbox({br:.3f})"
        if tier == "edge_huge" and br < STAGE3_ARRIVE_EDGE_HUGE_BBOX:
            return False, f"edge_huge({br:.3f})"
        if tier == "edge_locked" and br < STAGE3_BULKY_EDGE_BBOX_LOCKED:
            return False, f"edge_locked({br:.3f})"
        if tier == "edge_locked" and not track.locked:
            return False, "edge_locked_need_lock"
        td = check.get("td")
        if td is None:
            return False, "edge_no_depth"
        if float(td) > float(STAGE3_BULKY_DEPTH_MAX):
            return False, f"edge_far({float(td):.2f}>={STAGE3_BULKY_DEPTH_MAX:.2f})"
        if stage3_arrival_kind(target_en) == STAGE3_ARRIVAL_KIND_BULKY:
            return (
                False,
                f"bulky_edge_preview({tier}) need depth<={STAGE3_BULKY_DEPTH_MAX:.2f}",
            )
    else:
        return False, f"tier({tier})"
    return True, tier

@dataclass
class Stage3TrackState:
    """Stage3 目标锁定 + 丢检迟滞。"""

    locked: bool = False
    lock_streak: int = 0
    lost_count: int = 0
    lost_forward_used: int = 0
    last_det: dict = field(default_factory=_empty_det)
    max_bbox: float = 0.0
    lock_at_bbox: float = 0.0
    locked_steps: int = 0

    def _track_quality_ok(self, det):
        if not det.get("found"):
            return False
        br = float(det.get("bbox_ratio", 0.0))
        conf = float(det.get("confidence", 0.0))
        return br >= STAGE3_TRACK_MIN_BBOX or conf >= STAGE3_TRACK_MIN_CONF

    def _hysteresis_ok(self, det):
        if not det.get("found"):
            return False
        br = float(det.get("bbox_ratio", 0.0))
        conf = float(det.get("confidence", 0.0))
        return br >= STAGE3_HYSTERESIS_MIN_BBOX and conf >= STAGE3_HYSTERESIS_MIN_CONF

    def _lost_hysteresis_limit(self):
        if not self.last_det.get("found"):
            return 0
        br = float(self.last_det.get("bbox_ratio", 0.0))
        if br < STAGE3_TRACK_MIN_BBOX:
            return STAGE3_LOST_USE_LAST_TINY_BBOX
        return STAGE3_LOST_USE_LAST_MAX

    def update_raw(self, det, target_en=None):
        if det.get("found"):
            self.lost_count = 0
            self.lost_forward_used = 0
            br = float(det.get("bbox_ratio", 0.0))
            conf = float(det.get("confidence", 0.0))
            lock_conf = stage3_lock_conf(target_en)
            lock_bbox = stage3_lock_bbox(target_en)
            if self._track_quality_ok(det):
                if self.last_det.get("found"):
                    last_cx = float(self.last_det.get("center_x", 0.5))
                    new_cx = float(det.get("center_x", 0.5))
                    if (
                        abs(new_cx - last_cx) > 0.35
                        and br < 0.10
                        and conf < 0.50
                        and self.max_bbox >= lock_bbox
                    ):
                        pass
                    else:
                        self.last_det = dict(det)
                else:
                    self.last_det = dict(det)
                self.max_bbox = max(self.max_bbox, br)
            door_huge = (
                str(target_en or "") == "door" and br > STAGE3_DOOR_LOCK_MAX_BBOX
            )
            door_shape_bad = (
                str(target_en or "") == "door" and not stage3_door_bbox_aspect_ok(det)
            )
            bed_bad_lock = stage3_is_bed_target(target_en) and (
                br < STAGE3_BED_LOCK_MIN_BBOX or br > STAGE3_BED_LOCK_MAX_BBOX
            )
            if (
                not door_huge
                and not door_shape_bad
                and not bed_bad_lock
                and conf >= lock_conf
                and br >= lock_bbox
            ):
                self.lock_streak += 1
            else:
                self.lock_streak = max(0, self.lock_streak - 1)
            need = stage3_lock_frames_needed(target_en, conf, br)
            if self.lock_streak >= need:
                if not self.locked:
                    self.lock_at_bbox = br
                self.locked = True
                self.locked_steps += 1
            elif self.locked:
                self.locked_steps += 1
        else:
            self.lost_count += 1
            self.lock_streak = 0
            release_at = STAGE3_LOCK_RELEASE_LOST
            if self.locked and self.max_bbox >= STAGE3_ARRIVE_EDGE_MIN_MAX_BBOX:
                release_at = STAGE3_LOCK_RELEASE_LOST_PERSIST
            if self.lost_count >= release_at:
                self.locked = False
                self.lock_at_bbox = 0.0
                self.locked_steps = 0
                self.lost_forward_used = 0

    def effective_det(self, raw_det):
        if raw_det.get("found"):
            return raw_det
        limit = self._lost_hysteresis_limit()
        if (
            self.lost_count <= limit
            and self.last_det.get("found")
            and self._hysteresis_ok(self.last_det)
        ):
            eff = dict(self.last_det)
            eff["synthetic"] = True
            return eff
        return _empty_det()

def stage3_needs_approach_forward(check, track, target_en=None):
    """已锁定但融合深度仍远：须前进靠近，不能因 edge/bbox 提前 hold。"""
    if not track.locked or not STAGE3_LOCK_APPROACH_UNTIL_DEPTH:
        return False
    td = check.get("td")
    if td is None:
        return True
    if stage3_arrival_kind(target_en) == STAGE3_ARRIVAL_KIND_BULKY:
        cap = float(STAGE3_BULKY_DEPTH_MAX)
    else:
        cap = float(STAGE3_APPROACH_TARGET_DEPTH)
    return float(td) > cap

def visual_servo_locked_approach(det, depth, front_d, track, target_en=None):
    """
    目标锁定后：先对准画面中心，再持续前进直到融合深度 < APPROACH_TARGET
    或 bbox 贴边（占屏过大），随后交给到达判据 / hold。
    """
    check = near_arrival_check_stage3(
        det,
        depth,
        front_d=front_d,
        locked=True,
        max_bbox=track.max_bbox,
        target_en=target_en,
    )
    still_far = stage3_needs_approach_forward(check, track, target_en=target_en)
    if check["ok"] and not still_far:
        return None, "lock_approach_arrived"
    cx = float(det.get("center_x", 0.5))
    td = check["td"]
    br = float(det.get("bbox_ratio", 0.0))
    fd = float(front_d) if front_d is not None else 99.0

    # bbox 贴边：仅当已足够近才 hold；远距时继续对准+前进（避免 2.2m 就 lock_edge_hold）
    if not still_far and (
        br >= STAGE3_APPROACH_EDGE_BBOX
        or (td is not None and td <= VISUAL_SERVO_FORWARD_DEPTH)
    ):
        if cx < STAGE3_SERVO_LOCK_CENTER_LO:
            return "turn_left", "lock_edge_align"
        if cx > STAGE3_SERVO_LOCK_CENTER_HI:
            return "turn_right", "lock_edge_align"
        return None, "lock_edge_hold"

    if cx < STAGE3_APPROACH_CX_LO:
        return "turn_left", "lock_approach_align"
    if cx > STAGE3_APPROACH_CX_HI:
        return "turn_right", "lock_approach_align"

    if td is None or td > STAGE3_APPROACH_TARGET_DEPTH:
        if fd >= STAGE3_APPROACH_MIN_FRONT_M:
            return "move_forward", "lock_approach_forward"
        return ("turn_left" if cx < 0.5 else "turn_right"), "lock_approach_blocked"

    if cx < STAGE3_SERVO_LOCK_CENTER_LO:
        return "turn_left", "lock_approach_fine"
    if cx > STAGE3_SERVO_LOCK_CENTER_HI:
        return "turn_right", "lock_approach_fine"
    return "move_forward", "lock_approach_fine"

def visual_servo_action_stage3(
    det, depth, locked=False, front_d=None, max_bbox=0.0, target_en=None
):
    if not det.get("found"):
        return None
    check = near_arrival_check_stage3(
        det,
        depth,
        front_d=front_d,
        locked=locked,
        max_bbox=max_bbox,
        target_en=target_en,
    )
    if check["ok"]:
        return None
    cx = float(det.get("center_x", 0.5))
    td = check["td"]
    if cx < STAGE3_PERIPHERAL_CX_LO:
        return "turn_left"
    if cx > STAGE3_PERIPHERAL_CX_HI:
        return "turn_right"
    if cx < STAGE3_SERVO_FORWARD_CX_LO or cx > STAGE3_SERVO_FORWARD_CX_HI:
        return "turn_left" if cx < 0.5 else "turn_right"
    if td is None or td > VISUAL_SERVO_FORWARD_DEPTH:
        return "move_forward"
    lo = STAGE3_SERVO_LOCK_CENTER_LO if locked else VISUAL_SERVO_CENTER_LO
    hi = STAGE3_SERVO_LOCK_CENTER_HI if locked else VISUAL_SERVO_CENTER_HI
    if cx < lo:
        return "turn_left"
    if cx > hi:
        return "turn_right"
    return "move_forward"

def stage3_hold_action(det):
    """到达判据已满足：宽中心带内保持，仅微调。"""
    cx = float(det.get("center_x", 0.5))
    if cx < 0.40:
        return "turn_left"
    if cx > 0.60:
        return "turn_right"
    return None

def stage3_propose_action(
    effective,
    depth,
    track,
    scan_idx,
    front_d,
    target_en=None,
    agent_yaw=None,
    aim_yaw=None,
):
    """
    返回 (action, source)。action=None 表示本步 hold（不移动，只计步）。
    """
    if (
        str(target_en or "") == "door"
        and not track.locked
        and agent_yaw is not None
        and aim_yaw is not None
    ):
        err = stage3_yaw_err_rad(agent_yaw, aim_yaw)
        br = float(effective.get("bbox_ratio", 0.0))
        if math.degrees(err) > STAGE3_DOOR_AIM_YAW_MAX_ERR_DEG and (
            not effective.get("found") or br >= 0.20
        ):
            turn = "turn_left" if err > 0 else "turn_right"
            return turn, "door_aim_turn"

    if not effective.get("found"):
        if track.locked and track.last_det.get("found"):
            if track.lost_count <= STAGE3_LOST_USE_LAST_MAX:
                cx = float(track.last_det.get("center_x", 0.5))
                if cx < STAGE3_APPROACH_CX_LO:
                    return "turn_left", "lock_lost_recenter"
                if cx > STAGE3_APPROACH_CX_HI:
                    return "turn_right", "lock_lost_recenter"
                if front_d >= STAGE3_APPROACH_MIN_FRONT_M:
                    return "move_forward", "lock_lost_seek"
        if (
            track.locked
            and STAGE3_LOST_FORWARD_MAX > 0
            and track.lost_count <= STAGE3_LOST_USE_LAST_MAX
        ):
            if (
                track.lost_forward_used < STAGE3_LOST_FORWARD_MAX
                and front_d >= STAGE3_LOST_FORWARD_MIN_FRONT_M
            ):
                track.lost_forward_used += 1
                return "move_forward", "lost_creep"
        if track.locked and track.lost_count <= STAGE3_LOST_USE_LAST_MAX * 2:
            return small_scan_action(scan_idx), "lost_scan"
        return small_scan_action(scan_idx), "search_scan"

    check = near_arrival_check_stage3(
        effective,
        depth,
        front_d=front_d,
        locked=track.locked,
        max_bbox=track.max_bbox,
        target_en=target_en,
    )
    if check["ok"] and not stage3_needs_approach_forward(
        check, track, target_en=target_en
    ):
        hold = stage3_hold_action(effective)
        if hold is None:
            return None, "hold"
        return hold, "hold_align"

    if track.locked and STAGE3_LOCK_APPROACH_UNTIL_DEPTH:
        act, src = visual_servo_locked_approach(
            effective, depth, front_d, track, target_en=target_en
        )
        return act, src

    act = visual_servo_action_stage3(
        effective,
        depth,
        locked=track.locked,
        front_d=front_d,
        max_bbox=track.max_bbox,
        target_en=target_en,
    )
    if act is not None:
        return act, "servo"
    return small_scan_action(scan_idx), "small_scan"

def motion_controller_stage3(
    agent,
    depth,
    memory,
    recovery,
    propose_fn,
    track=None,
    recovery_allowed=True,
):
    """Stage3：锁定/近距时不触发 recovery，避免贴墙乱转。"""
    pos = agent.state.position
    memory.update_stuck(pos)
    use_recovery = bool(recovery_allowed)
    if track is not None and STAGE3_DISABLE_RECOVERY_WHEN_LOCKED:
        if track.locked and track.lost_count <= STAGE3_LOST_USE_LAST_MAX:
            use_recovery = False
    if use_recovery:
        recovery.try_trigger(memory.stuck_steps)
    rec = recovery.tick() if use_recovery else None
    if rec is not None:
        action = rec
        source = "recovery"
    else:
        proposed, psrc = propose_fn()
        if proposed is None:
            action = None
            source = psrc
        else:
            overridden = memory.apply_oscillation_override(proposed)
            if overridden != proposed:
                action = overridden
                source = "oscillation"
            else:
                action = proposed
                source = psrc
    return action, source

# ─── LLM ──────────────────────────────────────────────────
def parse_command(user_input: str) -> str:
    dashscope.api_key = os.environ["DASHSCOPE_API_KEY"]
    response = Generation.call(
        model="qwen-turbo",
        messages=[{
            "role": "user",
            "content": f"""你是家用机器人指令解析器。
用户说："{user_input}"
从以下选项识别导航目标：沙发、床、桌子、椅子、门
只输出一个目标名称，不要其他文字。
无法识别则输出"未知"。""",
        }],
        max_tokens=10,
        temperature=0,
    )
    target = response.output.text.strip()
    print(f"LLM: '{user_input}' → '{target}'")
    return target

def semantic_weight_from_det(det):
    """检测置信度 × bbox 占比，过滤远景误检。"""
    if not det.get("found"):
        return 0.0
    bbox_ratio = float(det.get("bbox_ratio", 0.05))
    return float(det["confidence"]) * min(bbox_ratio * 8.0, 1.0)

def write_semantics_from_dets(topo, pn, dets, step, yaw=0.0):
    for label, det in dets.items():
        if det.get("found") and float(det["confidence"]) >= VIEW_CONF_THRESH:
            topo.try_add_view_from_det(label, pn, yaw, det, step)
        w = semantic_weight_from_det(det)
        if w >= SYSTEMATIC_SEM_MIN_WEIGHT:
            topo.add_semantic(float(pn[0]), float(pn[2]), label, w, step)

def sample_navmesh_goals(
    pathfinder,
    n_samples=SYSTEMATIC_N_SAMPLES,
    min_dist_m=SYSTEMATIC_MIN_SAMPLE_DIST_M,
    max_tries=SYSTEMATIC_SAMPLE_TRIES,
):
    """从 navmesh 随机采样，并按最小间距筛选，提高空间覆盖率。"""
    points = []
    for _ in range(max_tries):
        if len(points) >= n_samples:
            break
        p = pathfinder.get_random_navigable_point()
        if not np.isfinite(p[0]):
            continue
        p = np.array(p, dtype=np.float64)
        if all(_dist_xz(p, q) >= min_dist_m for q in points):
            points.append(p)
    return points

def spin_detect_at_region(sim, agent, topo, detector, act, turns=10, use_weight=True):
    """到达采样点后转圈写语义 + 注册局部拓扑锚点。"""
    for _ in range(turns):
        agent.act("turn_right")
        act.total += 1
        obs = sim.get_sensor_observations()
        depth = fill_depth(obs["depth"])
        pn = agent.state.position
        topo.on_step(float(pn[0]), float(pn[2]), act.total)
        yaw = yaw_from_rotation(agent.state.rotation)
        ld, fd, rd = measure_depth_probes(depth)
        topo.register_local_anchor(float(pn[0]), float(pn[2]), yaw, ld, fd, rd, act.total)
        dets = detector.detect_all_labels(obs["color"])
        if use_weight:
            write_semantics_from_dets(topo, pn, dets, act.total, yaw=yaw)
        else:
            for label, det in dets.items():
                if det["found"]:
                    topo.try_add_view_from_det(label, pn, yaw, det, act.total)
                    topo.add_semantic(float(pn[0]), float(pn[2]), label, det["confidence"], act.total)

def systematic_explore(
    sim,
    detector,
    n_samples=SYSTEMATIC_N_SAMPLES,
    map_path=TOPO_MAP_PATH,
    save_gif=True,
    save_viz=True,
    trajectory=None,
):
    """
    从 navmesh 采样若干导航点，逐一前往并检测，尽量覆盖全场景后生成拓扑语义地图。
    """
    agent = sim.agents[0]
    pf = sim.pathfinder
    topo = SemanticTopoMap()
    act = type("Act", (), {})()
    act.total = 0
    act.forward = 0
    memory = ActionMemory()
    recovery = RecoveryState()
    frames = []
    if trajectory is None:
        trajectory = []

    def pf_act(name):
        agent.act(name)
        act.total += 1
        if name == "move_forward":
            act.forward += 1
        pn = agent.state.position
        topo.on_step(float(pn[0]), float(pn[2]), act.total)
        _append_trajectory(trajectory, float(pn[0]), float(pn[2]), act.total)

    log_path = f"/autodl-tmp/systematic_explore_log_{datetime.now():%Y%m%d_%H%M%S}.txt"
    log = open(log_path, "w", encoding="utf-8")

    def plog(msg):
        print(msg)
        log.write(msg + "\n")
        log.flush()

    sample_points = sample_navmesh_goals(pf, n_samples=n_samples)
    plog(
        f"\n=== 系统探索开始 n_samples={n_samples} "
        f"实际采样={len(sample_points)} REGION_SIZE={REGION_SIZE}m ==="
    )
    plog(f"检测类别: {DETECT_LABELS}")

    for idx, goal in enumerate(sample_points):
        plog(f"前往采样点 {idx + 1}/{len(sample_points)}: [{goal[0]:.2f}, {goal[2]:.2f}]")
        leg_start = act.total

        def _on_path_step(obs, depth, pos, yaw):
            pn = agent.state.position
            if act.total % SYSTEMATIC_DETECT_EVERY == 0:
                write_semantics_from_dets(
                    topo, pn, detector.detect_all_labels(obs["color"]), act.total, yaw=yaw
                )
            ld, fd, rd = measure_depth_probes(depth)
            topo.register_local_anchor(float(pos[0]), float(pos[2]), yaw, ld, fd, rd, act.total)

        reached, _exit = run_persistent_path_follow(
            sim,
            agent,
            goal,
            act,
            pf_act,
            leg_start + SYSTEMATIC_LEG_MAX_STEPS,
            memory=memory,
            recovery=recovery,
            target_xz=[float(goal[0]), float(goal[2])],
            neighborhood_m=0.6,
            on_step=_on_path_step,
        )
        if not reached:
            plog("  路径跟随未完成，仍环视")
            continue

        spin_detect_at_region(
            sim, agent, topo, detector, act, turns=SYSTEMATIC_SPIN_TURNS, use_weight=True
        )
        plog(f"  到达并环视完成 step={act.total} regions={len(topo.regions)}")

        if save_gif and idx % SYSTEMATIC_GIF_EVERY_N == 0:
            obs = sim.get_sensor_observations()
            vis = obs["color"][:, :, :3].copy()
            cv2.putText(
                vis,
                f"sys {idx + 1}/{len(sample_points)} r={len(topo.regions)}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )
            frames.append(Image.fromarray(vis))

    plog(f"\n系统探索结束 总步数={act.total} regions={len(topo.regions)}")
    plog(topo.summary_top(16))
    topo.save_json(map_path)
    log.close()

    if save_viz:
        visualize_region_graph(
            topo,
            trajectory=trajectory,
            output_path=TOPO_VIZ_EXPLORE_PATH,
            title="Topo Map — Explore",
        )

    if save_gif and frames:
        gif_path = "/autodl-tmp/systematic_explore.gif"
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=200, loop=0)
        print(f"系统探索 GIF: {gif_path}")
        pre_gif = "/autodl-tmp/pre_explore.gif"
        frames[0].save(pre_gif, save_all=True, append_images=frames[1:], duration=200, loop=0)

    return topo, map_path

# ─── 阶段 1: 预探索建图（frontier，保留作备用）────────────────
def pre_explore(
    sim, detector, max_steps=PRE_EXPLORE_MAX_STEPS, map_path=TOPO_MAP_PATH,
    save_gif=True, save_viz=True, trajectory=None,
):
    """
    高层: frontier doorway anchor → navigate_to(best_anchor)
    局部: depth 避障 + 到 region 后 spin 写语义
    """
    agent = sim.agents[0]
    topo = SemanticTopoMap()
    explore_state = ExploreRegionState()
    action_memory = ActionMemory()
    recovery = RecoveryState()
    last_spin_rid = None
    frames = []
    act = type("Act", (), {})()
    act.total = 0
    if trajectory is None:
        trajectory = []

    def pf_act(name):
        agent.act(name)
        act.total += 1
        pn = agent.state.position
        topo.on_step(float(pn[0]), float(pn[2]), act.total)
        _append_trajectory(trajectory, float(pn[0]), float(pn[2]), act.total)

    log_path = f"/autodl-tmp/pre_explore_log_{datetime.now():%Y%m%d_%H%M%S}.txt"
    log = open(log_path, "w", encoding="utf-8")

    def plog(msg):
        print(msg)
        log.write(msg + "\n")
        log.flush()

    plog(f"\n=== 预探索开始 max_steps={max_steps} REGION_SIZE={REGION_SIZE}m ===")
    plog(f"检测类别: {DETECT_LABELS} | 覆盖终止 patience={NO_NEW_REGION_PATIENCE}")

    t = 0
    while t < max_steps:
        obs = sim.get_sensor_observations()
        rgb = obs["color"]
        depth = fill_depth(obs["depth"])

        if t % PRE_DETECT_EVERY == 0:
            dets = detector.detect_all_labels(rgb)
            pn = agent.state.position
            for label, det in dets.items():
                if det["found"]:
                    topo.try_add_view_from_det(
                        label, pn, yaw_from_rotation(agent.state.rotation), det, act.total
                    )
                    topo.add_semantic(float(pn[0]), float(pn[2]), label, det["confidence"], act.total)
            if t % (PRE_DETECT_EVERY * 3) == 0 and dets:
                tops = sorted(
                    ((lb, d["confidence"]) for lb, d in dets.items() if d["found"]),
                    key=lambda x: -x[1],
                )[:3]
                plog(f"  step={t} pos=[{pn[0]:.2f},{pn[2]:.2f}] top={tops}")

        cur_rid = region_id_from_xz(float(agent.state.position[0]), float(agent.state.position[2]))
        if (
            explore_state.goal is not None
            and _dist_xz(agent.state.position, explore_state.goal) < REGION_CENTER_REACH_M
            and cur_rid != last_spin_rid
        ):
            spin_detect_at_region(sim, agent, topo, detector, act, turns=10)
            last_spin_rid = cur_rid
            explore_state.target_anchor = None
            explore_state.goal = None

        action = explore_region_policy_step(
            sim, agent, topo, depth, act.total, explore_state, action_memory, recovery
        )
        pf_act(action)

        if topo.steps_since_new_region(act.total) >= NO_NEW_REGION_PATIENCE:
            plog(f"  覆盖终止: {NO_NEW_REGION_PATIENCE} 步无新 region (step={act.total})")
            break

        t += 1
        if save_gif and t % 4 == 0:
            vis = rgb[:, :, :3].copy()
            rid = region_id_from_xz(float(agent.state.position[0]), float(agent.state.position[2]))
            cv2.putText(
                vis, f"pre {rid} s={t}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2
            )
            frames.append(Image.fromarray(vis))

    plog(f"\n预探索结束 steps={act.total}")
    plog(topo.summary_top(12))
    topo.save_json(map_path)
    log.close()

    if save_viz:
        visualize_region_graph(
            topo,
            trajectory=trajectory,
            output_path=TOPO_VIZ_EXPLORE_PATH,
            title="Topo Map — Explore",
        )

    if save_gif and frames:
        gif_path = "/autodl-tmp/pre_explore.gif"
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=120, loop=0)
        print(f"预探索 GIF: {gif_path}")

    return topo, map_path

def navigate_coarse_to_xz(
    sim,
    agent,
    goal,
    target_xz,
    act,
    pf_act,
    step_end,
    memory=None,
    recovery=None,
    view=None,
    nlog=None,
    trace=None,
    detector=None,
    target_en=None,
):
    """
    Stage 1 coarse：PersistentPathFollower 逼近 target_xz（通常为语义区域中心）。
    dist < NAV_COARSE_SUCCESS_M (1.2m) 即停；view 仅当 STAGE1_VIEW_YAW_IN_PATH 时参与航向 mix。
    """
    target_xz = [float(target_xz[0]), float(target_xz[1])]
    d0 = _dist_xz_to_target(agent.state.position, target_xz)
    h_mode0 = stage1_heading_mode(d0)
    bearing0 = stage1_bearing_yaw(agent.state.position, target_xz)
    view_yaw0 = semantic_aim_yaw_from_view(view)
    if nlog is not None:
        pos = agent.state.position
        yaw = yaw_from_rotation(agent.state.rotation)
        err_b = 0.0
        if bearing0 is not None:
            err_b, _ = heading_err_deg(yaw, bearing0)
        extra = (
            f" mode={h_mode0} bearing_err={err_b:.1f}° α={stage1_semantic_alpha(d0):.2f}"
        )
        if view_yaw0 is not None:
            err_v, _ = heading_err_deg(yaw, view_yaw0)
            extra += (
                f" view_yaw={math.degrees(view_yaw0):.1f}° err_view={err_v:.1f}°"
                f"(≤{STAGE1_VIEW_YAW_BLEND_DIST_M}m)"
            )
        nlog(
            f"  Stage1 入口: dist={d0:.2f}m{extra} "
            f"(recovery={'off' if STAGE1_DISABLE_RECOVERY else 'on'})"
        )
    if stage1_coarse_success(d0, "already_in_neighborhood"):
        if nlog is not None:
            nlog(
                f"  Stage1: 已在邻域 dist={d0:.2f}m "
                f"(跳过 path-follow，steps=0)"
            )
        return True, d0

    path_ok, exit_r = run_persistent_path_follow(
        sim,
        agent,
        goal,
        act,
        pf_act,
        step_end,
        memory=memory,
        recovery=recovery,
        target_xz=target_xz,
        view=view,
        neighborhood_m=NAV_COARSE_SUCCESS_M,
        nlog=nlog,
        trace=trace,
        detector=detector,
        target_en=target_en,
    )

    d = _dist_xz_to_target(agent.state.position, target_xz)
    ok = bool(path_ok) and stage1_coarse_success(d, exit_r)
    if nlog is not None:
        nlog(
            f"  Stage1 coarse 结束: dist={d:.2f}m ok={ok} "
            f"(邻域 ≤{NAV_COARSE_SUCCESS_M}m 或空旷≤{stage1_near_open_dist_cap():.2f}m, "
            f"exit={exit_r})"
        )
    return ok, d

def navigate_near_saved_view(
    sim,
    agent,
    view,
    act,
    pf_act,
    step_end,
    memory=None,
    recovery=None,
    topo_map=None,
    nlog=None,
    trace=None,
    detector=None,
    target_en=None,
):
    """
    Stage 1：逼近 view 所属语义区域中心邻域（<NAV_COARSE_SUCCESS_M）。
    view 观测位姿/yaw 仅用于 Stage2/Stage3，不参与 path 终点与 mix 航向（可配置）。
    """
    vx, vz = view.xz()
    path_view = view if STAGE1_VIEW_YAW_IN_PATH else None
    y = float(agent.state.position[1])
    if STAGE1_GOAL_FROM_REGION_CENTER and topo_map is not None:
        pack = topo_map.stage1_approach_target(sim, view)
        if pack is None:
            return False, None
        goal, target_xz, rid, src = pack
        if nlog is not None:
            nlog(
                f"  Stage1 区域逼近: region={rid} target={target_xz} "
                f"src={src} | view.pose=[{vx:.2f},{vz:.2f}] (S2/S3)"
            )
    else:
        goal = snap_navigable(sim, vx, y, vz)
        if goal is None:
            return False, None
        target_xz = [float(vx), float(vz)]
    return navigate_coarse_to_xz(
        sim,
        agent,
        goal,
        target_xz,
        act,
        pf_act,
        step_end,
        memory,
        recovery,
        view=path_view,
        nlog=nlog,
        trace=trace,
        detector=detector,
        target_en=target_en,
    )

def navigate_to_region_center(
    sim,
    agent,
    goal,
    act,
    pf_act,
    step_end,
    memory=None,
    recovery=None,
    nlog=None,
    trace=None,
):
    """Stage 1 fallback：粗导航到 region 中心邻域。"""
    target_xz = [float(goal[0]), float(goal[2])]
    ok, _ = navigate_coarse_to_xz(
        sim,
        agent,
        goal,
        target_xz,
        act,
        pf_act,
        step_end,
        memory,
        recovery,
        nlog=nlog,
        trace=trace,
    )
    return ok

def recover_view_direction(
    sim,
    agent,
    view,
    act,
    pf_act,
    max_turns=RECOVER_VIEW_MAX_TURNS,
    nlog=None,
    memory=None,
    recovery=None,
):
    """
    Stage 2 viewpoint recovery：转向 saved_yaw + target_rel_angle，再进入 visual servo。
    """
    if memory is None:
        memory = ActionMemory()
    if recovery is None:
        recovery = RecoveryState()

    target_yaw = float(view.yaw)
    rel_angle = float(getattr(view, "target_rel_angle", 0.0))
    aim_yaw = target_yaw + rel_angle
    tol = (
        RECOVER_REL_ANGLE_TOL_RAD
        if abs(rel_angle) > RECOVER_REL_ANGLE_TOL_RAD
        else RECOVER_YAW_TOL_RAD
    )
    turns = 0

    def _turn(action):
        nonlocal turns
        if turns >= max_turns:
            return False
        obs = sim.get_sensor_observations()
        depth = fill_depth(obs["depth"])

        def _propose():
            return action

        pf_act(
            motion_controller(
                agent,
                depth,
                memory,
                recovery,
                _propose,
                enable_recovery=not STAGE1_DISABLE_RECOVERY,
            )
        )
        turns += 1
        return True

    exit_reason = "aligned"
    while turns < max_turns:
        yaw = yaw_from_rotation(agent.state.rotation)
        diff = (aim_yaw - yaw + math.pi) % (2 * math.pi) - math.pi
        if abs(diff) <= tol:
            break
        if not _turn("turn_left" if diff > 0 else "turn_right"):
            exit_reason = "turn_budget"
            break
    else:
        exit_reason = "turn_budget"

    yaw_now = yaw_from_rotation(agent.state.rotation)
    diff_end = (aim_yaw - yaw_now + math.pi) % (2 * np.pi) - np.pi
    aligned = abs(diff_end) <= tol
    if nlog is not None:
        nlog(
            f"  Stage2 viewpoint recovery: aim_yaw={aim_yaw:.2f} "
            f"(saved={target_yaw:.2f}+rel={rel_angle:.2f}) now_yaw={yaw_now:.2f} "
            f"turns={turns} aligned={aligned}"
        )
    return turns

def small_scan_action(scan_idx):
    """Stage 3 无检测时：左右小幅扫描，不用 path-follow / 避障巡航。"""
    return "turn_left" if (int(scan_idx) % 2) == 0 else "turn_right"

def _stage3_aim_yaw(view, det, yaw):
    """Stage2 保存朝向；无 view 时用检测 cx 推算视觉 aim。"""
    if view is not None:
        return float(view.yaw) + float(getattr(view, "target_rel_angle", 0.0))
    if det.get("found"):
        cx = float(det.get("center_x", 0.5))
        return float(yaw) + (cx - 0.5) * CAMERA_HFOV_RAD
    return float(yaw)

def visual_servo_recovery(
    sim,
    detector,
    target_en,
    act,
    pf_act,
    step_end,
    nlog,
    memory=None,
    recovery=None,
    cand_forward_begin=0,
    view=None,
):
    """
    Stage3：锁定 + 迟滞 + 分级到达 + hold/丢目标有限前进。
    step_end: 本候选 act.total 绝对上限。
    """
    agent = sim.agents[0]
    if memory is None:
        memory = ActionMemory()
    if recovery is None:
        recovery = RecoveryState()
    track = Stage3TrackState()
    arrive_streak = 0
    scan_idx = 0
    last_arrive_tier = None

    while act.total < step_end:
        obs = sim.get_sensor_observations()
        rgb = obs["color"]
        depth = fill_depth(obs["depth"])
        raw_det = detect_target_disambiguated(detector, rgb, target_en)
        track.update_raw(raw_det, target_en=target_en)
        effective = track.effective_det(raw_det)

        _left, front_d, _right = measure_depth_probes(depth)
        check = near_arrival_check_stage3(
            effective,
            depth,
            front_d=front_d,
            locked=track.locked,
            max_bbox=track.max_bbox,
            target_en=target_en,
        )
        ok = bool(check["ok"])
        if STAGE3_ARRIVE_REJECT_SYNTHETIC and effective.get("synthetic"):
            ok = False
        td = check["td"]
        arrive_streak = arrive_streak + 1 if ok else 0
        if ok:
            last_arrive_tier = check.get("tier")
        forwards_in_cand = int(act.forward) - int(cand_forward_begin)
        success_ok, _success_reason = stage3_confirm_success(
            check, effective, track, arrive_streak, forwards_in_cand, target_en=target_en
        )
        if success_ok:
            nlog(
                f"  Stage3 到达 tier={last_arrive_tier} "
                f"conf={effective.get('confidence', 0):.2f} depth={td:.2f}m "
                f"bbox={effective.get('bbox_ratio', 0):.3f} forwards={forwards_in_cand}"
            )
            return True
        yaw_now = yaw_from_rotation(agent.state.rotation)
        aim_yaw_step = _stage3_aim_yaw(view, effective, yaw_now)

        def _propose():
            nonlocal scan_idx
            act_s, src = stage3_propose_action(
                effective,
                depth,
                track,
                scan_idx,
                front_d,
                target_en=target_en,
                agent_yaw=yaw_now,
                aim_yaw=aim_yaw_step,
            )
            if src in ("search_scan", "lost_scan", "small_scan"):
                scan_idx += 1
            return act_s, src

        rec_ok = stage3_recovery_allowed(
            front_d,
            effective,
            target_en,
            agent_yaw=yaw_now,
            aim_yaw=aim_yaw_step,
        )
        action, _src = motion_controller_stage3(
            agent,
            depth,
            memory,
            recovery,
            _propose,
            track=track,
            recovery_allowed=rec_ok,
        )

        if action is None:
            act.total += 1
        else:
            pf_act(action)

    return False


# ─── 阶段 2: 按语义地图导航 ───────────────────────────────
def navigate_to_target(sim, detector, topo_map, target_cn, max_steps=NAV_MAX_STEPS):
    """
    retrieve_candidate_views → 三阶段导航：
      Stage1 coarse(邻域 0.8~1.2m) → Stage2 viewpoint recovery → Stage3 visual servo。
    无 view 时 fallback 到 region 候选。
    """
    target_en = TARGETS_ZH2EN.get(target_cn, target_cn)
    agent = sim.agents[0]
    pos0 = agent.state.position
    ax, az = float(pos0[0]), float(pos0[2])

    act = type("Act", (), {"total": 0, "forward": 0})
    action_memory = ActionMemory()
    recovery = RecoveryState()
    nav_trace = NavTraceRecorder()
    ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_path = f"/autodl-tmp/navigation_log_{target_cn}_{ts_tag}.txt"
    log = open(log_path, "w", encoding="utf-8")

    def nlog(msg):
        print(msg)
        log.write(msg + "\n")
        log.flush()

    pf_act = make_tracked_pf_act(agent, act, recovery, trace=nav_trace)

    nav_start_pos, nav_start_rot = _save_agent_pose(agent)

    candidates = topo_map.retrieve_candidate_views(
        target_en, ax, az, act.total, action_memory, max_n=NAV_MAX_CANDIDATE_VIEWS, sim=sim
    )
    use_views = len(candidates) > 0
    region_cands = []
    if not use_views:
        region_cands = topo_map.retrieve_candidate_regions(
            target_en, current_step=act.total, max_n=NAV_MAX_CANDIDATE_REGIONS
        )
        nlog(f"导航 {target_cn}({target_en}) 无 SemanticView，fallback regions: {region_cands}")
    else:
        nlog(
            f"导航 {target_cn}({target_en}) 候选 views: "
            f"{[(vid, round(sc, 2)) for vid, sc, _ in candidates]}"
        )

    if not use_views and not region_cands:
        print(f"拓扑图中无 {target_en} 候选 view/region")
        log.close()
        return False, 0

    if use_views:
        n_cand = len(candidates)
        for rank, (view_id, nav_score, view) in enumerate(candidates, 1):
            if act.total >= max_steps:
                break
            approach = topo_map.stage1_approach_target(sim, view)
            if approach is None:
                topo_map.mark_view_failed(view_id, target_en, "approach goal not navigable")
                continue
            _goal, approach_xz, approach_rid, approach_src = approach
            cand_step_begin = act.total
            cand_forward_begin = act.forward
            step_end = _candidate_step_end(
                cand_step_begin,
                NAV_PER_VIEW_BUDGET,
                max_steps,
                rank=rank,
                n_candidates=n_cand,
            )
            nlog(
                f"\n[{rank}/{n_cand}] view#{view_id} score={nav_score:.2f} "
                f"conf={view.conf:.2f} bbox={view.bbox_ratio:.3f} "
                f"pos={view.pos} yaw={view.yaw:.2f} rel_angle={view.target_rel_angle:.2f} "
                f"region={view.region_id}"
            )
            agent_pos = agent.state.position
            vx, vz = view.xz()
            ax, az = float(approach_xz[0]), float(approach_xz[1])
            geo_d = geodesic_distance_xz(
                sim, float(agent_pos[0]), float(agent_pos[2]), ax, az
            )
            nlog(
                f"  Stage1→region{approach_rid} target={approach_xz} src={approach_src} "
                f"| view.pose=[{vx:.2f},{vz:.2f}] (S2/S3)"
            )
            nlog(
                f"  起点距区域中心 直线={_dist_xz_to_target(agent_pos, approach_xz):.2f}m "
                f"测地={geo_d:.2f}m agent=[{agent_pos[0]:.2f},{agent_pos[2]:.2f}]"
            )
            nlog(
                f"  本候选独立预算 per_view={NAV_PER_VIEW_BUDGET} "
                f"step_begin={cand_step_begin} step_end={step_end}"
            )
            nlog(
                f"  Stage1 coarse navigation (persistent path, 邻域 <{NAV_COARSE_SUCCESS_M}m 即切视觉)"
            )
            reached, dist_m = navigate_near_saved_view(
                sim,
                agent,
                view,
                act,
                pf_act,
                step_end,
                action_memory,
                recovery,
                topo_map=topo_map,
                nlog=nlog,
                trace=nav_trace,
                detector=detector,
                target_en=target_en,
            )
            if not reached:
                dist_s = f"{dist_m:.2f}" if dist_m is not None else "n/a"
                nlog(f"  Stage1 失败原因: 未进入邻域 dist={dist_s}m → 下一候选")
                topo_map.mark_view_failed(view_id, target_en, "coarse nav neighborhood failed")
                if NAV_RESET_ON_S1_FAIL:
                    pn = agent.state.position
                    drift = _dist_xz_to_target(pn, [nav_start_pos[0], nav_start_pos[2]])
                    if drift >= NAV_RESET_DRIFT_MIN_M:
                        _restore_agent_pose(agent, nav_start_pos, nav_start_rot)
                        recovery.clear()
                        nlog(
                            f"  Stage1 失败后复位到导航起点 "
                            f"(漂移 {drift:.2f}m >= {NAV_RESET_DRIFT_MIN_M}m)"
                        )
                continue
            recover_budget = min(RECOVER_VIEW_MAX_TURNS, step_end - act.total)
            nlog(f"  Stage2 viewpoint recovery (max {recover_budget} 步)")
            recover_view_direction(
                sim,
                agent,
                view,
                act,
                pf_act,
                max_turns=recover_budget,
                nlog=nlog,
                memory=action_memory,
                recovery=recovery,
            )
            nlog(f"  Stage3 visual servo (step_end={step_end}, detect→forward / 否则 small scan)")
            if visual_servo_recovery(
                sim,
                detector,
                target_en,
                act,
                pf_act,
                step_end,
                nlog,
                action_memory,
                recovery,
                cand_forward_begin=cand_forward_begin,
                view=view,
            ):
                nlog(f"✓ 到达 {target_cn} @ view#{view_id} steps={act.total}")
                log.close()
                return True, act.total
            topo_map.mark_view_failed(view_id, target_en, "no visual confirm at view")
            nlog(f"  Stage3 失败: 无视觉确认 view#{view_id} → 下一候选")
    else:
        n_reg = len(region_cands)
        for rank, (rid, sem_score) in enumerate(region_cands, 1):
            if act.total >= max_steps:
                break
            goal = topo_map.navigable_goal(sim, rid)
            if goal is None:
                topo_map.mark_region_failed(rid, target_en, "center not navigable")
                continue
            ba = topo_map.best_anchor_in_region(rid)
            nlog(
                f"\n[{rank}/{n_reg}] region {rid} score={sem_score:.2f} "
                f"anchor={ba.get('place_type')} pos={ba.get('position')}"
            )
            cand_step_begin = act.total
            cand_forward_begin = act.forward
            step_end = _candidate_step_end(
                cand_step_begin,
                NAV_PER_VIEW_BUDGET,
                max_steps,
                rank=rank,
                n_candidates=n_reg,
            )
            nlog(
                f"  本候选独立预算 per_view={NAV_PER_VIEW_BUDGET} "
                f"step_begin={cand_step_begin} step_end={step_end}"
            )
            nlog(f"  Stage1 coarse → region 中心")
            reached = navigate_to_region_center(
                sim,
                agent,
                goal,
                act,
                pf_act,
                step_end,
                action_memory,
                recovery,
                nlog=nlog,
                trace=nav_trace,
            )
            if not reached:
                nlog(f"  Stage1 失败: 未进入 region {rid} 邻域 → 下一候选")
                topo_map.mark_region_failed(rid, target_en, "coarse nav to region failed")
                if NAV_RESET_ON_S1_FAIL:
                    pn = agent.state.position
                    drift = _dist_xz_to_target(pn, [nav_start_pos[0], nav_start_pos[2]])
                    if drift >= NAV_RESET_DRIFT_MIN_M:
                        _restore_agent_pose(agent, nav_start_pos, nav_start_rot)
                        recovery.clear()
                        nlog(
                            f"  Stage1 失败后复位到导航起点 "
                            f"(漂移 {drift:.2f}m >= {NAV_RESET_DRIFT_MIN_M}m)"
                        )
                continue
            nlog(f"  Stage3 visual servo (step_end={step_end})")
            if visual_servo_recovery(
                sim,
                detector,
                target_en,
                act,
                pf_act,
                step_end,
                nlog,
                action_memory,
                recovery,
                cand_forward_begin=cand_forward_begin,
                view=None,
            ):
                nlog(f"✓ 到达 {target_cn} @ region {rid} steps={act.total}")
                log.close()
                return True, act.total
            topo_map.mark_region_failed(rid, target_en, "no visual confirm at region")
            nlog(f"  region {rid} 无视觉确认 → 下一候选")

    nlog(f"所有候选均失败 steps={act.total}")
    log.close()
    return False, act.total


# ─── 会话入口 ─────────────────────────────────────────────
def ensure_topo_map(sim, detector, force_rebuild=False, map_path=TOPO_MAP_PATH):
    if force_rebuild or not os.path.isfile(map_path):
        print("开始系统探索建图（navmesh 采样）…")
        return systematic_explore(sim, detector, map_path=map_path)[0]
    return SemanticTopoMap.load_json(map_path)


def run_agent(user_input: str, sim=None, detector=None, topo_map=None, force_remap=False):
    print(f"\n{'=' * 50}\n用户: {user_input}\n{'=' * 50}")

    if sim is None:
        sim = make_sim(random_spawn=True)
    if detector is None:
        detector = TargetDetector()

    target_cn = parse_command(user_input)
    if target_cn == "未知" or target_cn not in TARGETS_ZH2EN:
        print(f"无法识别: {target_cn}")
        return False, sim, detector, topo_map

    low = user_input.strip().lower()
    trajectory = []
    remap_cmd = (
        "探索" in user_input or "建图" in user_input
        or low in ("remap", "rebuild map", "重新建图", "explore", "map")
    )
    if force_remap or remap_cmd:
        topo_map, _ = systematic_explore(sim, detector, trajectory=trajectory)
    else:
        topo_map = ensure_topo_map(sim, detector, force_rebuild=force_remap)

    if remap_cmd and target_cn == "未知":
        print(f"系统探索建图完成。拓扑图: {TOPO_VIZ_EXPLORE_PATH}")
        return True, sim, detector, topo_map

    success, steps = navigate_to_target(sim, detector, topo_map, target_cn)
    pos = sim.agents[0].state.position
    print(f"结束位置: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
    if success:
        print(f"Agent: 好的，我已经到达{target_cn}旁边了。")
    else:
        print(f"Agent: 抱歉，我没能到达{target_cn}。")
    return success, sim, detector, topo_map


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--viz-map":
        map_path = sys.argv[2] if len(sys.argv) > 2 else TOPO_MAP_PATH
        target_cn = sys.argv[3] if len(sys.argv) > 3 else "沙发"
        topo = SemanticTopoMap.load_json(map_path)
        visualize_region_graph(
            topo,
            output_path=TOPO_VIZ_EXPLORE_PATH,
            title="Topo Map (from JSON)",
            highlight_target_en=TARGETS_ZH2EN.get(target_cn, target_cn),
        )
        return

    print("语义拓扑导航 Agent")
    print("  首次运行或输入「探索/建图」会 navmesh 系统探索并保存 semantic_topo_map.json")
    print(f"  探索建图拓扑图: {TOPO_VIZ_EXPLORE_PATH}（仅探索结束生成，导航不画图）")
    print(f"  仅从 JSON 出图: python {os.path.basename(__file__)} --viz-map [json] [沙发|门]")
    print("  之后输入：请到沙发旁边 / 去床边 等")
    print("  输入 q 退出\n")

    sim = None
    detector = None
    topo_map = None
    try:
        while True:
            try:
                text = input("\n你: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见。")
                break
            if not text:
                continue
            if text.lower() in ("q", "quit", "exit", "退出"):
                break
            success, sim, detector, topo_map = run_agent(text, sim, detector, topo_map)
    finally:
        if sim is not None:
            sim.close()
            print("仿真已关闭。")

if __name__ == "__main__":
    main()
