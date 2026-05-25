"""
Stage1 WALL 语义评判（对应 EDMX 新模型 Rejury）。

规则层：visible / reachable / doorway / reacquire；
可选 geo_dist（navmesh 测地）由调用方传入，避免本模块依赖 Habitat。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

# ─── Rejury / WALL 闸门常量 ─────────────────────────────────
REJURY_MODEL_NAME = "Rejury"

STAGE1_WALL_SEMANTIC_REACQUIRE = True
STAGE1_WALL_REACQUIRE_MIN_CONF = 0.6
STAGE1_WALL_REACQUIRE_FRONT_MIN_M = 0.5
STAGE1_WALL_REACQUIRE_RAY_MARGIN_M = 0.35
STAGE1_WALL_REACQUIRE_MAX_RAY_OVER_EST_M = 0.35
STAGE1_WALL_REACQUIRE_MAX_GEODESIC_M = 3.2

STAGE1_WALL_ENTER_DOORWAY_GATE = True
STAGE1_WALL_ENTER_BLOCK_TOWARD_FREE_M = 1.2
STAGE1_WALL_DOORWAY_LEFT_OPEN_M = 2.5
STAGE1_WALL_DOORWAY_BEARING_MAX_DEG = -25.0
STAGE1_WALL_DOORWAY_TOWARD_FREE_M = 1.0
# 左侧通道明显宽于正前/右侧（仅 U 型出口贴 hit 时视为门口，非客厅大空间）
STAGE1_WALL_DOORWAY_LEFT_DOM_FRONT_M = 0.6
STAGE1_WALL_DOORWAY_LEFT_DOM_RIGHT_M = 0.8
STAGE1_WALL_DOORWAY_NEAR_HIT_M = 1.65
STAGE1_WALL_PAST_DOORWAY_HIT_M = 2.0
STAGE1_WALL_PAST_DOORWAY_MLINE_M = 1.45
STAGE1_WALL_PAST_DOORWAY_REGRESS_M = 0.15

CAMERA_HFOV_RAD = math.radians(90.0)


def _dist_xz_to_target(pos, target_xz) -> float:
    return float(
        np.linalg.norm(
            np.array([float(pos[0]), float(pos[2])], dtype=np.float64)
            - np.array([float(target_xz[0]), float(target_xz[1])], dtype=np.float64)
        )
    )


def _dist_xz_points(ax, az, bx, bz) -> float:
    return float(
        np.hypot(float(ax) - float(bx), float(az) - float(bz))
    )


def stage1_dist_from_wall_hit(pos, hit_xz: Optional[Tuple[float, float]]) -> Optional[float]:
    if pos is None or hit_xz is None:
        return None
    return _dist_xz_points(
        float(pos[0]), float(pos[2]), float(hit_xz[0]), float(hit_xz[1])
    )


def stage1_near_wall_hit(
    pos, hit_xz: Optional[Tuple[float, float]], max_m: Optional[float] = None
) -> bool:
    d = stage1_dist_from_wall_hit(pos, hit_xz)
    if d is None:
        return False
    cap = float(max_m if max_m is not None else STAGE1_WALL_DOORWAY_NEAR_HIT_M)
    return float(d) <= cap


def depth_free_space_at_rel_deg(left_d, front_d, right_d, rel_deg: float) -> float:
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


def stage1_bearing_yaw(pos, target_xz) -> Optional[float]:
    if target_xz is None:
        return None
    dx = float(target_xz[0]) - float(pos[0])
    dz = float(target_xz[1]) - float(pos[2])
    if abs(dx) + abs(dz) < 1e-6:
        return None
    return float(math.atan2(dx, dz))


def target_rel_angle_from_det(det) -> float:
    cx = float(det.get("center_x", 0.5))
    return (cx - 0.5) * CAMERA_HFOV_RAD


def stage1_target_bearing_yaw(yaw, det) -> float:
    return float(yaw) + target_rel_angle_from_det(det)


def stage1_left_passage_open(left_d, front_d, right_d) -> bool:
    """左侧门口/通道开阔：深度上 L 明显大于 F、R（与当前朝向无关）。"""
    L, F, R = float(left_d), float(front_d), float(right_d)
    if L < float(STAGE1_WALL_DOORWAY_LEFT_OPEN_M):
        return False
    return (
        L >= F + float(STAGE1_WALL_DOORWAY_LEFT_DOM_FRONT_M)
        and L >= R + float(STAGE1_WALL_DOORWAY_LEFT_DOM_RIGHT_M)
    )


def stage1_bearing_ray_depth(yaw, bearing_yaw, left_d, front_d, right_d) -> Tuple[float, float]:
    rel_deg = math.degrees(
        (float(bearing_yaw) - float(yaw) + math.pi) % (2 * math.pi) - math.pi
    )
    ray_d = depth_free_space_at_rel_deg(left_d, front_d, right_d, rel_deg)
    return float(ray_d), float(rel_deg)


def bbox_front_depth(depth, bbox, h: int, w: int) -> Optional[float]:
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    patch = depth[max(0, y0) : y1, max(0, x0) : x1]
    valid = patch[np.isfinite(patch) & (patch > 0.05) & (patch < 10.0)]
    if len(valid) < 4:
        return None
    return float(np.percentile(valid, 15))


def stage1_wall_enter_blocked(
    pos,
    target_xz,
    yaw,
    left_d,
    front_d,
    right_d,
    hit_xz: Optional[Tuple[float, float]] = None,
) -> Tuple[bool, str]:
    """禁止 enter WALL：目标/门口方向开阔（贴 bug2_hit 穿出 U 时仍允许 enter）。"""
    if stage1_near_wall_hit(pos, hit_xz):
        return False, ""
    bearing = stage1_bearing_yaw(pos, target_xz)
    if bearing is None:
        return False, ""
    toward_free, rel_deg = stage1_bearing_ray_depth(
        yaw, bearing, left_d, front_d, right_d
    )
    if toward_free >= float(STAGE1_WALL_ENTER_BLOCK_TOWARD_FREE_M):
        return True, f"toward_free={toward_free:.2f}°={rel_deg:.0f}"
    if (
        float(left_d) >= float(STAGE1_WALL_DOORWAY_LEFT_OPEN_M)
        and rel_deg <= float(STAGE1_WALL_DOORWAY_BEARING_MAX_DEG)
        and toward_free >= float(STAGE1_WALL_DOORWAY_TOWARD_FREE_M)
    ):
        return True, f"doorway_left L={left_d:.2f} bear={rel_deg:.0f}°"
    return False, ""


def stage1_wall_doorway_seek_ok(
    pos,
    target_xz,
    yaw,
    left_d,
    front_d,
    right_d,
    hit_xz: Optional[Tuple[float, float]] = None,
    mline_d: Optional[float] = None,
) -> Tuple[bool, str]:
    """U 型出口贴 hit 左转进门（~60）；已过门口后不用此规则。"""
    if stage1_near_wall_hit(pos, hit_xz) and stage1_left_passage_open(
        left_d, front_d, right_d
    ):
        d_hit = stage1_dist_from_wall_hit(pos, hit_xz)
        return (
            True,
            f"u_exit_left L={float(left_d):.2f} d_hit={float(d_hit):.2f}",
        )
    if not stage1_near_wall_hit(pos, hit_xz):
        return False, ""
    if target_xz is None:
        return False, ""
    bearing = stage1_bearing_yaw(pos, target_xz)
    if bearing is None:
        return False, ""
    toward_free, rel_deg = stage1_bearing_ray_depth(
        yaw, bearing, left_d, front_d, right_d
    )
    if float(left_d) < float(STAGE1_WALL_DOORWAY_LEFT_OPEN_M):
        return False, ""
    if rel_deg > float(STAGE1_WALL_DOORWAY_BEARING_MAX_DEG):
        return False, ""
    if toward_free < float(STAGE1_WALL_DOORWAY_TOWARD_FREE_M):
        return False, ""
    return True, f"doorway L={left_d:.2f} toward={toward_free:.2f}°={rel_deg:.0f}"


def stage1_wall_past_doorway(
    pos,
    hit_xz: Optional[Tuple[float, float]],
    mline_d: Optional[float],
    dist_goal: Optional[float],
    dist_at_hit: Optional[float],
    escape_steps: int = 0,
    boundary_min_dist: Optional[float] = None,
) -> Tuple[bool, str]:
    """
    已穿过门口进入新结构区（~73+）：不宜继续 FOLLOW_WALL / frozen 贴墙。
    """
    d_hit = stage1_dist_from_wall_hit(pos, hit_xz)
    if d_hit is None:
        return False, ""
    if float(d_hit) >= float(STAGE1_WALL_PAST_DOORWAY_HIT_M):
        if mline_d is not None and float(mline_d) >= float(STAGE1_WALL_PAST_DOORWAY_MLINE_M):
            return (
                True,
                f"past_doorway d_hit={float(d_hit):.2f} mline={float(mline_d):.2f}",
            )
        if (
            boundary_min_dist is not None
            and dist_goal is not None
            and int(escape_steps) >= 10
            and float(dist_goal)
            > float(boundary_min_dist) + float(STAGE1_WALL_PAST_DOORWAY_REGRESS_M)
        ):
            return (
                True,
                f"past_doorway regress dist={float(dist_goal):.2f} "
                f"best={float(boundary_min_dist):.2f} d_hit={float(d_hit):.2f}",
            )
    return False, ""


def stage1_wall_semantic_reacquire(
    det,
    depth,
    yaw,
    left_d,
    front_d,
    right_d,
    pos,
    target_xz,
    geo_dist: Optional[float] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """WALL 内语义重新捕获：visible vs reachable。"""
    info: Dict[str, Any] = {"visible": False, "reachable": False, "model": REJURY_MODEL_NAME}
    if not det.get("found"):
        return False, "no_visible", info
    info["visible"] = True
    conf = float(det.get("confidence", 0.0))
    info["conf"] = conf
    if conf < float(STAGE1_WALL_REACQUIRE_MIN_CONF):
        return False, f"conf={conf:.2f}", info
    bearing_yaw = stage1_target_bearing_yaw(yaw, det)
    ray_d, rel_deg = stage1_bearing_ray_depth(yaw, bearing_yaw, left_d, front_d, right_d)
    info["bearing_deg"] = round(rel_deg, 1)
    info["bearing_ray"] = round(ray_d, 2)
    if ray_d < float(STAGE1_WALL_REACQUIRE_FRONT_MIN_M):
        return False, f"bearing_blocked ray={ray_d:.2f}", info
    h, w = depth.shape[:2]
    target_ray_depth = bbox_front_depth(depth, det.get("bbox"), h, w)
    if target_ray_depth is None:
        return False, "no_ray_depth", info
    target_ray_depth = float(target_ray_depth)
    info["target_ray_depth"] = round(target_ray_depth, 2)
    est_dist = (
        _dist_xz_to_target(pos, target_xz)
        if target_xz is not None
        else target_ray_depth
    )
    est_dist = float(est_dist)
    info["est_dist"] = round(est_dist, 2)
    over = float(STAGE1_WALL_REACQUIRE_MAX_RAY_OVER_EST_M)
    if target_ray_depth > est_dist + over:
        return (
            False,
            f"visible_not_reachable ray={target_ray_depth:.2f}>est+{over:.2f}",
            info,
        )
    margin = float(STAGE1_WALL_REACQUIRE_RAY_MARGIN_M)
    if target_ray_depth < est_dist - margin:
        return (
            False,
            f"visible_not_reachable ray={target_ray_depth:.2f}<est-{margin:.2f}",
            info,
        )
    if geo_dist is not None:
        geo = float(geo_dist)
        info["geo"] = round(geo, 2)
        if geo > float(STAGE1_WALL_REACQUIRE_MAX_GEODESIC_M):
            return (
                False,
                f"geo={geo:.2f}>{STAGE1_WALL_REACQUIRE_MAX_GEODESIC_M:.2f}",
                info,
            )
    info["reachable"] = True
    return (
        True,
        f"conf={conf:.2f} ray={target_ray_depth:.2f} est={est_dist:.2f}",
        info,
    )


@dataclass
class RejuryVerdict:
    """EDMX Rejury 评判结果（可序列化回写）。"""

    model: str = REJURY_MODEL_NAME
    visible: bool = False
    reachable: bool = False
    recommend_goal_seek: bool = False
    tag: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


class Rejury:
    """
    EDMX Rejury 新模型入口。

    - evaluate_reacquire: 是否退出 WALL → GOAL_SEEK
    - should_block_wall_enter: 是否禁止 enter WALL
    - should_doorway_seek: 已在 WALL 但应改走门口
    """

    @staticmethod
    def evaluate_reacquire(
        det,
        depth,
        yaw,
        left_d,
        front_d,
        right_d,
        pos,
        target_xz,
        geo_dist: Optional[float] = None,
    ) -> Tuple[bool, str, Dict[str, Any], RejuryVerdict]:
        ok, tag, info = stage1_wall_semantic_reacquire(
            det,
            depth,
            yaw,
            left_d,
            front_d,
            right_d,
            pos,
            target_xz,
            geo_dist=geo_dist,
        )
        verdict = RejuryVerdict(
            visible=bool(info.get("visible")),
            reachable=bool(info.get("reachable")),
            recommend_goal_seek=bool(ok),
            tag=tag,
            extra=dict(info),
        )
        return ok, tag, info, verdict

    @staticmethod
    def should_block_wall_enter(
        pos,
        target_xz,
        yaw,
        left_d,
        front_d,
        right_d,
        hit_xz: Optional[Tuple[float, float]] = None,
    ) -> Tuple[bool, str]:
        if not STAGE1_WALL_ENTER_DOORWAY_GATE:
            return False, ""
        return stage1_wall_enter_blocked(
            pos, target_xz, yaw, left_d, front_d, right_d, hit_xz=hit_xz
        )

    @staticmethod
    def should_doorway_seek(
        pos,
        target_xz,
        yaw,
        left_d,
        front_d,
        right_d,
        hit_xz: Optional[Tuple[float, float]] = None,
        mline_d: Optional[float] = None,
    ) -> Tuple[bool, str]:
        return stage1_wall_doorway_seek_ok(
            pos, target_xz, yaw, left_d, front_d, right_d, hit_xz=hit_xz, mline_d=mline_d
        )

    @staticmethod
    def should_past_doorway_exit(
        pos,
        hit_xz: Optional[Tuple[float, float]],
        mline_d: Optional[float],
        dist_goal: Optional[float],
        dist_at_hit: Optional[float],
        escape_steps: int = 0,
        boundary_min_dist: Optional[float] = None,
    ) -> Tuple[bool, str]:
        return stage1_wall_past_doorway(
            pos,
            hit_xz,
            mline_d,
            dist_goal,
            dist_at_hit,
            escape_steps=escape_steps,
            boundary_min_dist=boundary_min_dist,
        )

    @staticmethod
    def left_passage_open(left_d, front_d, right_d) -> bool:
        return stage1_left_passage_open(left_d, front_d, right_d)
