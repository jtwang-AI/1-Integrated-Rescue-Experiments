from __future__ import annotations

import json
import math
import random
import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Circle, Polygon
import numpy as np

from GMRACCR import GMRACCR


@dataclass(frozen=True)
class AgentProfile:
    name: str
    speed: float
    maneuver: float
    communication: float
    payload: float
    endurance: float
    stability: float


@dataclass(frozen=True)
class RoleProfile:
    name: str
    radius: float
    phase: float
    weights: Tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class SituationIndicators:
    bearing_rate: float
    acceleration: float
    rescue_speed: float
    distance: float
    bearing: float


@dataclass(frozen=True)
class OrbitTarget:
    semi_major: float
    semi_minor: float
    phase: float


@dataclass
class SimulationMetrics:
    mean_radial_error: float
    mean_phase_error: float
    mean_role_error: float
    mean_control_ratio: float
    role_coverage: float
    assignment_utility: float
    compatible_utility: float
    mean_current_magnitude: float
    min_separation: float
    collision_violations: int


AGENTS: List[AgentProfile] = [
    AgentProfile("UUV1", 0.40, 0.55, 0.58, 0.48, 0.60, 0.72),
    AgentProfile("UUV2", 0.78, 0.92, 0.70, 0.62, 0.69, 0.68),
    AgentProfile("UUV3", 0.46, 0.68, 0.45, 0.72, 0.63, 0.76),
    AgentProfile("UUV4", 0.72, 0.82, 0.96, 0.56, 0.74, 0.70),
    AgentProfile("UUV5", 0.38, 0.48, 0.64, 0.88, 0.86, 0.84),
    AgentProfile("UUV6", 0.57, 0.66, 0.73, 0.66, 0.80, 0.78),
]

ROLES: List[RoleProfile] = [
    RoleProfile("Approach guidance", 2.4, 0.0, (0.18, 0.32, 0.10, 0.05, 0.10, 0.25)),
    RoleProfile("High-mobility interception", 3.0, 1.1, (0.42, 0.33, 0.05, 0.05, 0.05, 0.10)),
    RoleProfile("Communication relay", 4.8, 2.3, (0.18, 0.10, 0.42, 0.00, 0.18, 0.12)),
    RoleProfile("Close-range support", 3.7, 3.5, (0.15, 0.20, 0.10, 0.32, 0.08, 0.15)),
    RoleProfile("Safety backup", 4.3, 4.7, (0.08, 0.08, 0.15, 0.17, 0.28, 0.24)),
]

FEATURE_KEYS = ("speed", "maneuver", "communication", "payload", "endurance", "stability")

AHP_WEIGHTS = (0.03, 0.06, 0.15, 0.46, 0.30)

SITUATION_ROLE_TEMPLATES: List[Tuple[float, float, float, float, float]] = [
    (0.16, 0.08, 0.20, 0.26, 0.30),  # approach guidance
    (0.22, 0.24, 0.26, 0.14, 0.14),  # high-mobility interception
    (0.08, 0.05, 0.16, 0.34, 0.37),  # communication relay
    (0.10, 0.12, 0.18, 0.38, 0.22),  # close-range support
    (0.06, 0.08, 0.12, 0.36, 0.38),  # safety backup
]

SITUATION_SNAPSHOTS: Dict[str, List[SituationIndicators]] = {
    "baseline": [
        SituationIndicators(4.0, 0.8, 0.8, 1800.0, 80.0),
        SituationIndicators(13.0, 4.0, 2.0, 4000.0, 100.0),
        SituationIndicators(6.0, 2.0, 0.5, 2400.0, 70.0),
        SituationIndicators(10.0, 2.0, 3.0, 3800.0, 80.0),
        SituationIndicators(2.0, 1.0, 1.0, 1200.0, 100.0),
        SituationIndicators(5.0, 0.2, 1.5, 3000.0, 70.0),
    ],
    "bearing_shift": [
        SituationIndicators(5.0, 1.0, 0.9, 2600.0, 112.0),
        SituationIndicators(9.0, 3.2, 1.8, 3300.0, 104.0),
        SituationIndicators(11.0, 2.7, 1.1, 1300.0, 74.0),
        SituationIndicators(7.0, 1.8, 2.7, 4100.0, 96.0),
        SituationIndicators(3.0, 1.2, 1.2, 1900.0, 84.0),
        SituationIndicators(8.0, 1.5, 2.1, 1500.0, 76.0),
    ],
}

RCC = [
    [0.00, 0.45, -0.20, 0.05, 0.00],
    [0.45, 0.00, -0.30, 0.05, -0.10],
    [-0.20, -0.30, 0.00, -0.10, 0.35],
    [0.05, 0.05, -0.10, 0.00, 0.25],
    [0.00, -0.10, 0.35, 0.25, 0.00],
]

PLOT_COLORS = ["#2f7fbd", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def set_ocean_plot_style() -> None:
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.45,
        "grid.alpha": 0.25,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def draw_usv_marker(ax, x: float, y: float, angle: float, color: str, scale: float = 0.22) -> None:
    hull = np.array([[1.25, 0.0], [-0.7, 0.45], [-1.0, 0.0], [-0.7, -0.45]]) * scale
    rot = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    pts = hull @ rot.T + np.array([x, y])
    ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor="white", linewidth=0.5, zorder=5))


def plot_rescue_trajectory(ax, history: Dict[int, List[Tuple[float, float]]], title: str,
                           orbit_mode: str = "circle", show_ylabel: bool = True) -> None:
    ax.set_facecolor("#f7fbff")
    theta = np.linspace(0.0, 2.0 * np.pi, 360)
    for role in ROLES:
        if orbit_mode == "ellipse":
            x_orbit = 1.20 * role.radius * np.cos(theta)
            y_orbit = 0.82 * role.radius * np.sin(theta)
        else:
            x_orbit = role.radius * np.cos(theta)
            y_orbit = role.radius * np.sin(theta)
        ax.plot(x_orbit, y_orbit, "--", linewidth=0.8, color="#b8b8b8", zorder=1)
    ax.add_patch(Circle((0, 0), 0.18, facecolor="#e41a1c", edgecolor="white", linewidth=0.6, zorder=6))
    ax.text(0.22, 0.18, "Target", fontsize=7, color="#b2182b")
    for agent_idx, trajectory in history.items():
        if not trajectory:
            continue
        xs = [p[0] for p in trajectory]
        ys = [p[1] for p in trajectory]
        color = PLOT_COLORS[agent_idx % len(PLOT_COLORS)]
        ax.plot(xs, ys, linewidth=1.8, color=color, label=AGENTS[agent_idx].name, zorder=3)
        mid = len(trajectory) // 2
        ax.scatter([xs[0], xs[mid], xs[-1]], [ys[0], ys[mid], ys[-1]],
                   marker="o", s=[14, 20, 28], color=color, edgecolor="white", linewidth=0.35, zorder=4)
        if len(trajectory) > 2:
            angle = math.atan2(ys[-1] - ys[-2], xs[-1] - xs[-2])
        else:
            angle = 0.0
        draw_usv_marker(ax, xs[-1], ys[-1], angle, color)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-5.7, 5.9)
    ax.set_ylim(-5.4, 5.8)
    ax.set_xlabel("x (m)")
    if show_ylabel:
        ax.set_ylabel("y (m)")
    else:
        ax.set_ylabel("")
    ax.set_title(title, pad=4)
    ax.grid(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def circular_mean(angles: Iterable[float], weights: Optional[Iterable[float]] = None) -> float:
    angles = list(angles)
    if weights is None:
        weights = [1.0] * len(angles)
    weights = list(weights)
    x = sum(w * math.cos(a) for a, w in zip(angles, weights))
    y = sum(w * math.sin(a) for a, w in zip(angles, weights))
    return math.atan2(y, x)


def build_qualification_matrix(agent_indices: Sequence[int], role_indices: Sequence[int]) -> List[List[float]]:
    matrix: List[List[float]] = []
    for agent_idx in agent_indices:
        agent = AGENTS[agent_idx]
        features = [getattr(agent, key) for key in FEATURE_KEYS]
        row = []
        for role_idx in role_indices:
            weights = ROLES[role_idx].weights
            row.append(sum(f * w for f, w in zip(features, weights)))
        matrix.append(row)
    max_value = max(max(row) for row in matrix)
    return [[value / max_value for value in row] for row in matrix]


def build_situation_qualification_matrix(agent_indices: Sequence[int], role_indices: Sequence[int],
                                         indicators: Sequence[SituationIndicators],
                                         reference_distance: float = 4500.0,
                                         preferred_bearing: float = 90.0) -> List[List[float]]:
    raw_scores: List[List[float]] = []
    for item in indicators:
        distance_score = reference_distance / item.distance if item.distance <= reference_distance else 0.1
        bearing_score = 1.0 / (1.0 + abs(abs(item.bearing) - preferred_bearing) / 20.0)
        raw_scores.append([
            abs(item.bearing_rate),
            item.acceleration,
            item.rescue_speed,
            distance_score,
            bearing_score,
        ])

    columns = list(zip(*raw_scores))
    normalized_columns: List[List[float]] = []
    for column in columns:
        c_min = min(column)
        c_max = max(column)
        if abs(c_max - c_min) < 1e-9:
            normalized_columns.append([1.0 for _ in column])
        else:
            normalized_columns.append([(value - c_min) / (c_max - c_min) for value in column])
    normalized_scores = [list(row) for row in zip(*normalized_columns)]

    matrix: List[List[float]] = []
    for agent_idx in agent_indices:
        weighted = [score * weight for score, weight in zip(normalized_scores[agent_idx], AHP_WEIGHTS)]
        row = []
        for role_idx in role_indices:
            template = SITUATION_ROLE_TEMPLATES[role_idx]
            row.append(sum(value * role_weight for value, role_weight in zip(weighted, template)))
        matrix.append(row)
    max_value = max(max(row) for row in matrix)
    return [[value / max_value if max_value > 0 else 0.0 for value in row] for row in matrix]


def build_assignment_from_matrix(assignment_matrix: Sequence[Sequence[int]], agent_indices: Sequence[int],
                                 role_indices: Sequence[int]) -> Dict[int, List[int]]:
    role_map: Dict[int, List[int]] = {agent_idx: [] for agent_idx in agent_indices}
    for local_agent_idx, row in enumerate(assignment_matrix):
        for local_role_idx, flag in enumerate(row):
            if flag:
                role_map[agent_indices[local_agent_idx]].append(role_indices[local_role_idx])
    return {agent_idx: roles for agent_idx, roles in role_map.items() if roles}


def greedy_unique_assignment(agent_indices: Sequence[int], role_indices: Sequence[int]) -> Dict[int, List[int]]:
    q = build_qualification_matrix(agent_indices, role_indices)
    remaining_agents = set(range(len(agent_indices)))
    assignment: Dict[int, List[int]] = {}
    for role_local_idx in range(len(role_indices)):
        best_local_agent = max(remaining_agents, key=lambda idx: q[idx][role_local_idx])
        assignment[agent_indices[best_local_agent]] = [role_indices[role_local_idx]]
        remaining_agents.remove(best_local_agent)
    return assignment


def average_capability_greedy(agent_indices: Sequence[int], role_indices: Sequence[int]) -> Dict[int, List[int]]:
    q = build_qualification_matrix(agent_indices, role_indices)
    averages = [sum(row) / len(row) for row in q]
    ranked_local_agents = sorted(range(len(agent_indices)), key=lambda idx: averages[idx], reverse=True)
    remaining_roles = set(range(len(role_indices)))
    assignment: Dict[int, List[int]] = {}
    for local_agent_idx in ranked_local_agents[: len(role_indices)]:
        best_role_local = max(remaining_roles, key=lambda role_local: q[local_agent_idx][role_local])
        assignment[agent_indices[local_agent_idx]] = [role_indices[best_role_local]]
        remaining_roles.remove(best_role_local)
    return assignment


def random_feasible_assignment(agent_indices: Sequence[int], role_indices: Sequence[int], seed: int = 7) -> Dict[int, List[int]]:
    rng = random.Random(seed)
    shuffled_agents = list(agent_indices)
    rng.shuffle(shuffled_agents)
    assignment: Dict[int, List[int]] = {}
    for role_idx, agent_idx in zip(role_indices, shuffled_agents):
        assignment.setdefault(agent_idx, []).append(role_idx)
    return assignment


def sequential_auction_assignment(agent_indices: Sequence[int], role_indices: Sequence[int],
                                  capacities: Sequence[int],
                                  qualification_matrix: Optional[Sequence[Sequence[float]]] = None,
                                  rcc_matrix: Optional[Sequence[Sequence[float]]] = None) -> Dict[int, List[int]]:
    q = [list(row) for row in qualification_matrix] if qualification_matrix is not None else build_qualification_matrix(agent_indices, role_indices)
    rcc = [list(row) for row in rcc_matrix] if rcc_matrix is not None else RCC
    remaining_capacity = {agent_idx: capacities[local_idx] for local_idx, agent_idx in enumerate(agent_indices)}
    assignment: Dict[int, List[int]] = {}
    for local_role_idx, role_idx in enumerate(role_indices):
        best_agent: Optional[int] = None
        best_bid = float("-inf")
        for local_agent_idx, agent_idx in enumerate(agent_indices):
            if remaining_capacity[agent_idx] <= 0:
                continue
            held_roles = assignment.get(agent_idx, [])
            marginal_compat = 0.0
            for held_role_idx in held_roles:
                local_held = role_indices.index(held_role_idx)
                marginal_compat += rcc[local_role_idx][local_held] + rcc[local_held][local_role_idx]
            bid = q[local_agent_idx][local_role_idx] + marginal_compat
            if bid > best_bid:
                best_bid = bid
                best_agent = agent_idx
        if best_agent is None:
            raise RuntimeError("No feasible sequential-auction assignment found")
        assignment.setdefault(best_agent, []).append(role_idx)
        remaining_capacity[best_agent] -= 1
    return assignment


def optimized_assignment(agent_indices: Sequence[int], role_indices: Sequence[int], capacities: Sequence[int],
                         use_compatibility: bool = True, compatibility_scale: float = 1.0,
                         qualification_matrix: Optional[Sequence[Sequence[float]]] = None,
                         rcc_matrix: Optional[Sequence[Sequence[float]]] = None) -> Tuple[Dict[int, List[int]], float, float]:
    q = [list(row) for row in qualification_matrix] if qualification_matrix is not None else build_qualification_matrix(agent_indices, role_indices)
    if use_compatibility:
        source_rcc = rcc_matrix if rcc_matrix is not None else RCC
        rcc = [[compatibility_scale * value for value in row] for row in source_rcc]
    else:
        rcc = [[0.0 for _ in role_indices] for _ in role_indices]
    solver = GMRACCR(
        nagent=len(agent_indices),
        nrole=len(role_indices),
        QM=q,
        RA=[1] * len(role_indices),
        LA=capacities,
        RCC=rcc,
    )
    result = solver.solve(prefer_pulp=True)
    assignment = build_assignment_from_matrix(result.assignment, agent_indices, role_indices)
    return assignment, result.base_score, result.compatibility_score


def assignment_objective(assignment: Dict[int, List[int]], agent_indices: Sequence[int], role_indices: Sequence[int],
                         qualification_matrix: Optional[Sequence[Sequence[float]]] = None,
                         rcc_matrix: Optional[Sequence[Sequence[float]]] = None) -> Tuple[float, float]:
    q = [list(row) for row in qualification_matrix] if qualification_matrix is not None else build_qualification_matrix(agent_indices, role_indices)
    rcc = [list(row) for row in rcc_matrix] if rcc_matrix is not None else RCC
    local_agent_lookup = {agent_idx: i for i, agent_idx in enumerate(agent_indices)}
    local_role_lookup = {role_idx: j for j, role_idx in enumerate(role_indices)}
    base = 0.0
    compat = 0.0
    for agent_idx, roles in assignment.items():
        local_i = local_agent_lookup[agent_idx]
        for role_idx in roles:
            base += q[local_i][local_role_lookup[role_idx]]
        for role_i in roles:
            for role_j in roles:
                compat += rcc[local_role_lookup[role_i]][local_role_lookup[role_j]]
    return base, compat


def composite_target_for_roles(role_indices: Sequence[int], orbit_mode: str = "circle") -> OrbitTarget:
    radii = [ROLES[idx].radius for idx in role_indices]
    phases = [ROLES[idx].phase for idx in role_indices]
    mean_radius = float(np.mean(radii))
    if orbit_mode == "ellipse":
        semi_major = 1.20 * mean_radius
        semi_minor = 0.82 * mean_radius
    else:
        semi_major = mean_radius
        semi_minor = mean_radius
    return OrbitTarget(semi_major=semi_major, semi_minor=semi_minor, phase=circular_mean(phases))


def current_field(x: float, y: float, step: int, dt: float, scale: float = 1.0) -> np.ndarray:
    t = step * dt
    r = math.hypot(x, y)
    vortex = 0.06 / (1.0 + r)
    surge_x = 0.10 * math.sin(0.35 * t) + vortex * y
    surge_y = 0.08 * math.cos(0.23 * t) - vortex * x
    return scale * np.array([surge_x, surge_y], dtype=float)


def simulate_assignment(assignment: Dict[int, List[int]], total_steps: int = 360, dt: float = 0.08,
                        failure_step: Optional[int] = None, failed_agent: Optional[int] = None,
                        reassignment: Optional[Dict[int, List[int]]] = None,
                        output_prefix: Optional[Path] = None, enable_current: bool = True,
                        enable_collision_avoidance: bool = True, current_scale: float = 1.0,
                        safe_distance: float = 0.85, avoidance_gain: float = 0.22,
                        compact_start: bool = False, orbit_mode: str = "circle",
                        initial_phase_noise: float = 0.0, initial_radius_noise: float = 0.0,
                        seed: Optional[int] = None,
                        qualification_matrix: Optional[Sequence[Sequence[float]]] = None,
                        rcc_matrix: Optional[Sequence[Sequence[float]]] = None) -> SimulationMetrics:
    standby_radius = 5.6
    base_omega = 0.18
    k_radial = 0.85
    k_phase = 0.55
    multitask_penalty = 0.18

    states: Dict[int, Dict[str, float]] = {}
    current_assignment = {agent_idx: list(roles) for agent_idx, roles in assignment.items()}
    all_agent_ids = sorted(set(current_assignment) | ({failed_agent} if failed_agent is not None else set()) | ({idx for idx in range(len(AGENTS))}))

    rng = random.Random(seed)

    for agent_idx in range(len(AGENTS)):
        roles = current_assignment.get(agent_idx, [])
        if roles:
            target = composite_target_for_roles(roles, orbit_mode=orbit_mode)
            orbit_scale = 0.5 * (target.semi_major + target.semi_minor)
            phase = target.phase
            if compact_start:
                orbit_scale = 2.2 + 0.08 * agent_idx
                phase = 0.30 + 0.16 * agent_idx
            else:
                orbit_scale *= 1.15
        else:
            orbit_scale = standby_radius
            phase = 5.4 + 0.3 * agent_idx
        if initial_phase_noise > 0.0:
            phase += rng.uniform(-initial_phase_noise, initial_phase_noise)
        if initial_radius_noise > 0.0:
            orbit_scale *= 1.0 + rng.uniform(-initial_radius_noise, initial_radius_noise)
        states[agent_idx] = {
            "x": orbit_scale * math.cos(phase),
            "y": orbit_scale * math.sin(phase),
            "alive": 1.0,
            "theta": phase,
        }

    radial_errors: List[float] = []
    phase_errors: List[float] = []
    role_errors: List[float] = []
    control_ratios: List[float] = []
    role_coverage: List[float] = []
    current_magnitudes: List[float] = []
    history: Dict[int, List[Tuple[float, float]]] = {agent_idx: [] for agent_idx in range(len(AGENTS))}
    min_separation = float("inf")
    collision_violations = 0

    for step in range(total_steps):
        if failure_step is not None and step == failure_step and failed_agent is not None:
            states[failed_agent]["alive"] = 0.0
            current_assignment.pop(failed_agent, None)
        if failure_step is not None and step == failure_step and reassignment is not None:
            current_assignment = {agent_idx: list(roles) for agent_idx, roles in reassignment.items()}
            if failed_agent is not None:
                current_assignment.pop(failed_agent, None)

        active_agents = [agent_idx for agent_idx, roles in current_assignment.items() if roles and states[agent_idx]["alive"] > 0.5]
        role_to_agent = {}
        for agent_idx in active_agents:
            for role_idx in current_assignment[agent_idx]:
                role_to_agent[role_idx] = agent_idx

        for agent_idx in range(len(AGENTS)):
            x = states[agent_idx]["x"]
            y = states[agent_idx]["y"]
            history[agent_idx].append((x, y))

        role_coverage.append(len(role_to_agent) / len(ROLES))

        thetas = {agent_idx: math.atan2(states[agent_idx]["y"], states[agent_idx]["x"]) for agent_idx in active_agents}
        for idx_a, agent_a in enumerate(active_agents):
            for agent_b in active_agents[idx_a + 1:]:
                dx = states[agent_a]["x"] - states[agent_b]["x"]
                dy = states[agent_a]["y"] - states[agent_b]["y"]
                distance = math.hypot(dx, dy)
                min_separation = min(min_separation, distance)
                if distance < safe_distance:
                    collision_violations += 1

        for agent_idx in range(len(AGENTS)):
            if states[agent_idx]["alive"] < 0.5:
                continue
            assigned_roles = current_assignment.get(agent_idx, [])
            if assigned_roles:
                target = composite_target_for_roles(assigned_roles, orbit_mode=orbit_mode)
                semi_major = target.semi_major
                semi_minor = target.semi_minor
                phase_target = target.phase
                phase_error = 0.0
                neighbor_count = 0
                theta_i = math.atan2(states[agent_idx]["y"] / max(semi_minor, 1e-6),
                                     states[agent_idx]["x"] / max(semi_major, 1e-6))
                for other_idx in active_agents:
                    if other_idx == agent_idx:
                        continue
                    other_target = composite_target_for_roles(current_assignment[other_idx], orbit_mode=orbit_mode)
                    phase_other = other_target.phase
                    theta_j = math.atan2(
                        states[other_idx]["y"] / max(other_target.semi_minor, 1e-6),
                        states[other_idx]["x"] / max(other_target.semi_major, 1e-6),
                    )
                    phase_error += wrap_angle((theta_i - theta_j) - (phase_target - phase_other))
                    neighbor_count += 1
                if neighbor_count:
                    phase_error /= neighbor_count
            else:
                semi_major = standby_radius
                semi_minor = standby_radius
                phase_target = math.atan2(states[agent_idx]["y"], states[agent_idx]["x"])
                phase_error = 0.0

            x = states[agent_idx]["x"]
            y = states[agent_idx]["y"]
            r = max(math.hypot(x, y), 1e-6)
            normalized_radius = math.sqrt((x / max(semi_major, 1e-6)) ** 2 + (y / max(semi_minor, 1e-6)) ** 2)
            grad = np.array([
                x / max(semi_major ** 2, 1e-6),
                y / max(semi_minor ** 2, 1e-6),
            ], dtype=float)
            grad_norm = max(float(np.linalg.norm(grad)), 1e-6)
            n_vec = grad / grad_norm
            t_vec = np.array([-y / r, x / r])
            if orbit_mode == "ellipse":
                t_vec = np.array([-n_vec[1], n_vec[0]])
            radial_error = (normalized_radius - 1.0) * (0.5 * (semi_major + semi_minor))
            theta = math.atan2(y / max(semi_minor, 1e-6), x / max(semi_major, 1e-6))
            own_phase_error = wrap_angle(theta - phase_target)

            speed_limit = AGENTS[agent_idx].speed
            if assigned_roles:
                speed_limit /= (1.0 + multitask_penalty * (len(assigned_roles) - 1))

            v_n = -k_radial * radial_error
            tangential_scale = 0.5 * (semi_major + semi_minor)
            v_t = base_omega * tangential_scale - k_phase * tangential_scale * phase_error
            u = v_n * n_vec + v_t * t_vec
            if enable_collision_avoidance and assigned_roles:
                repulsion = np.zeros(2, dtype=float)
                for other_idx in active_agents:
                    if other_idx == agent_idx:
                        continue
                    delta = np.array([
                        states[agent_idx]["x"] - states[other_idx]["x"],
                        states[agent_idx]["y"] - states[other_idx]["y"],
                    ], dtype=float)
                    distance = float(np.linalg.norm(delta))
                    if 1e-6 < distance < safe_distance:
                        repulsion += avoidance_gain * ((1.0 / distance) - (1.0 / safe_distance)) * (delta / (distance ** 3))
                u = u + repulsion
            current = current_field(x, y, step, dt, scale=current_scale) if enable_current else np.zeros(2, dtype=float)
            current_magnitudes.append(float(np.linalg.norm(current)))
            u = u + current
            speed = float(np.linalg.norm(u))
            if speed > speed_limit:
                u = u * (speed_limit / speed)
                speed = speed_limit

            states[agent_idx]["x"] = x + dt * float(u[0])
            states[agent_idx]["y"] = y + dt * float(u[1])
            states[agent_idx]["theta"] = math.atan2(states[agent_idx]["y"], states[agent_idx]["x"])

            if assigned_roles:
                radial_errors.append(abs(radial_error))
                phase_errors.append(abs(phase_error))
                control_ratios.append(speed / max(speed_limit, 1e-6))
                for role_idx in assigned_roles:
                    role = ROLES[role_idx]
                    role_phase_error = abs(wrap_angle(theta - role.phase))
                    if orbit_mode == "ellipse":
                        role_major = 1.20 * role.radius
                        role_minor = 0.82 * role.radius
                    else:
                        role_major = role.radius
                        role_minor = role.radius
                    role_norm = math.sqrt((x / max(role_major, 1e-6)) ** 2 + (y / max(role_minor, 1e-6)) ** 2)
                    role_radius_error = abs(role_norm - 1.0)
                    role_errors.append(role_phase_error + role_radius_error)

    if output_prefix is not None:
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        set_ocean_plot_style()
        fig, ax = plt.subplots(figsize=(5.2, 4.8))
        plot_rescue_trajectory(ax, history, output_prefix.name.replace("_", " ").title(), orbit_mode=orbit_mode)
        ax.legend(frameon=True, ncol=2, loc="lower right", borderpad=0.35, handlelength=1.6)
        fig.tight_layout()
        fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
        plt.close()
        with output_prefix.with_suffix(".json").open("w", encoding="utf-8") as f:
            json.dump({str(agent_idx): trajectory for agent_idx, trajectory in history.items()}, f)

    base, compat = assignment_objective(current_assignment, list(range(len(AGENTS))), list(range(len(ROLES))), qualification_matrix, rcc_matrix)
    return SimulationMetrics(
        mean_radial_error=float(np.mean(radial_errors)) if radial_errors else 0.0,
        mean_phase_error=float(np.mean(phase_errors)) if phase_errors else 0.0,
        mean_role_error=float(np.mean(role_errors)) if role_errors else 0.0,
        mean_control_ratio=float(np.mean(control_ratios)) if control_ratios else 0.0,
        role_coverage=float(np.mean(role_coverage)) if role_coverage else 0.0,
        assignment_utility=base,
        compatible_utility=compat,
        mean_current_magnitude=float(np.mean(current_magnitudes)) if current_magnitudes else 0.0,
        min_separation=min_separation if min_separation != float("inf") else 0.0,
        collision_violations=collision_violations,
    )


def run_beta_sweep(output_dir: Path, beta_values: Sequence[float]) -> List[Dict[str, float]]:
    scarce_agents = [0, 1, 3, 5]
    role_indices = list(range(len(ROLES)))
    scarce_capacities = [1, 1, 2, 1]
    rows: List[Dict[str, float]] = []
    for beta in beta_values:
        assignment, _, _ = optimized_assignment(
            scarce_agents,
            role_indices,
            scarce_capacities,
            use_compatibility=beta > 0.0,
            compatibility_scale=beta,
        )
        metrics = simulate_assignment(assignment)
        rows.append({
            "beta": beta,
            "assignment_utility": metrics.assignment_utility,
            "compatible_utility": metrics.compatible_utility,
            "mean_role_error": metrics.mean_role_error,
            "mean_phase_error": metrics.mean_phase_error,
            "mean_control_ratio": metrics.mean_control_ratio,
            "min_separation": metrics.min_separation,
            "collision_violations": float(metrics.collision_violations),
        })

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9))
    fig.patch.set_facecolor("white")
    betas = [row["beta"] for row in rows]

    axes[0].plot(
        betas,
        [row["assignment_utility"] for row in rows],
        marker="o",
        markersize=4.5,
        linewidth=2.1,
        color="#153b50",
        label="Base utility",
    )
    axes[0].plot(
        betas,
        [row["compatible_utility"] for row in rows],
        marker="s",
        markersize=4.2,
        linewidth=2.0,
        color="#c84c09",
        label="Compatibility utility",
    )
    axes[0].plot(
        betas,
        [row["mean_role_error"] for row in rows],
        marker="^",
        markersize=4.4,
        linewidth=1.9,
        color="#6f7d1c",
        label="Mean role error",
    )
    axes[0].set_xlabel(r"Compatibility weight $\beta$")
    axes[0].set_ylabel("Utility / error")
    axes[0].set_xlim(min(betas) - 0.03, max(betas) + 0.03)
    axes[0].grid(axis="y", alpha=0.18, linewidth=0.7)
    axes[0].legend(frameon=False, loc="lower right")
    axes[0].text(0.02, 0.96, "(a)", transform=axes[0].transAxes, ha="left", va="top", fontweight="bold")

    ax2 = axes[1]
    ax2.plot(
        betas,
        [row["min_separation"] for row in rows],
        marker="o",
        markersize=4.5,
        linewidth=2.1,
        color="#153b50",
        label="Min separation",
    )
    ax2.set_xlabel(r"Compatibility weight $\beta$")
    ax2.set_ylabel("Min separation")
    ax2.set_xlim(min(betas) - 0.03, max(betas) + 0.03)
    ax2.grid(axis="y", alpha=0.18, linewidth=0.7)
    ax2b = ax2.twinx()
    ax2b.plot(
        betas,
        [row["collision_violations"] for row in rows],
        marker="s",
        markersize=4.2,
        linewidth=2.0,
        linestyle="--",
        color="#b33a3a",
        label="Collisions",
    )
    ax2b.set_ylabel("Collision violations")
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper right")
    ax2.text(0.02, 0.96, "(b)", transform=ax2.transAxes, ha="left", va="top", fontweight="bold")

    for axis in (axes[0], ax2, ax2b):
        axis.spines["top"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / "beta_sweep.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return rows


def run_disturbance_sweep(output_dir: Path, current_scales: Sequence[float]) -> List[Dict[str, float]]:
    agent_indices = list(range(len(AGENTS)))
    role_indices = list(range(len(ROLES)))
    opt_assignment, _, _ = optimized_assignment(agent_indices, role_indices, [1] * len(AGENTS), use_compatibility=True)
    rows: List[Dict[str, float]] = []
    for current_scale in current_scales:
        for enable_collision_avoidance, label in ((False, "Current-only"), (True, "Collision-aware")):
            metrics = simulate_assignment(
                opt_assignment,
                enable_current=True,
                enable_collision_avoidance=enable_collision_avoidance,
                compact_start=True,
                current_scale=current_scale,
                safe_distance=0.9,
                avoidance_gain=2.5 if enable_collision_avoidance else 0.22,
            )
            rows.append({
                "current_scale": current_scale,
                "mode": label,
                "mean_role_error": metrics.mean_role_error,
                "mean_phase_error": metrics.mean_phase_error,
                "mean_control_ratio": metrics.mean_control_ratio,
                "min_separation": metrics.min_separation,
                "collision_violations": float(metrics.collision_violations),
            })

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9))
    fig.patch.set_facecolor("white")
    for label, color in (("Current-only", "#c84c09"), ("Collision-aware", "#1f6f5f")):
        subset = [row for row in rows if row["mode"] == label]
        scales = [row["current_scale"] for row in subset]
        axes[0].plot(
            scales,
            [row["mean_role_error"] for row in subset],
            marker="o",
            markersize=4.5,
            linewidth=2.0,
            color=color,
            label=label,
        )
        axes[1].plot(
            scales,
            [row["min_separation"] for row in subset],
            marker="o",
            markersize=4.5,
            linewidth=2.0,
            color=color,
            label=label,
        )

    axes[0].set_xlabel("Current scale")
    axes[0].set_ylabel("Mean role error")
    axes[0].grid(axis="y", alpha=0.18, linewidth=0.7)
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].text(0.02, 0.96, "(a)", transform=axes[0].transAxes, ha="left", va="top", fontweight="bold")

    axes[1].set_xlabel("Current scale")
    axes[1].set_ylabel("Min separation")
    axes[1].grid(axis="y", alpha=0.18, linewidth=0.7)
    ax1b = axes[1].twinx()
    for label, color in (("Current-only", "#b33a3a"), ("Collision-aware", "#4c4f8f")):
        subset = [row for row in rows if row["mode"] == label]
        scales = [row["current_scale"] for row in subset]
        ax1b.plot(
            scales,
            [row["collision_violations"] for row in subset],
            marker="s",
            markersize=4.0,
            linestyle="--",
            linewidth=1.8,
            color=color,
            label=f"{label} collisions",
        )
    ax1b.set_ylabel("Collision violations")
    lines1, labels1 = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    axes[1].legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper left")
    axes[1].text(0.02, 0.96, "(b)", transform=axes[1].transAxes, ha="left", va="top", fontweight="bold")

    for axis in (axes[0], axes[1], ax1b):
        axis.spines["top"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / "disturbance_sweep.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return rows


def run_situation_aware_end_to_end(output_dir: Path) -> List[Dict[str, object]]:
    agent_indices = list(range(len(AGENTS)))
    role_indices = list(range(len(ROLES)))
    capacities = [1] * len(AGENTS)
    rows: List[Dict[str, object]] = []

    baseline_q = build_situation_qualification_matrix(agent_indices, role_indices, SITUATION_SNAPSHOTS["baseline"])
    shifted_q = build_situation_qualification_matrix(agent_indices, role_indices, SITUATION_SNAPSHOTS["bearing_shift"])
    static_q = build_qualification_matrix(agent_indices, role_indices)

    baseline_assignment, _, _ = optimized_assignment(
        agent_indices,
        role_indices,
        capacities,
        use_compatibility=True,
        qualification_matrix=baseline_q,
    )
    shifted_assignment, _, _ = optimized_assignment(
        agent_indices,
        role_indices,
        capacities,
        use_compatibility=True,
        qualification_matrix=shifted_q,
    )
    static_assignment, _, _ = optimized_assignment(
        agent_indices,
        role_indices,
        capacities,
        use_compatibility=True,
        qualification_matrix=static_q,
    )

    scenarios = [
        ("Baseline situation", "baseline", baseline_q, baseline_assignment, output_dir / "situation_baseline_ahp"),
        ("Shifted situation, stale baseline assignment", "bearing_shift_stale", shifted_q, baseline_assignment, output_dir / "situation_shift_stale"),
        ("Shifted situation, AHP reassignment", "bearing_shift_reassign", shifted_q, shifted_assignment, output_dir / "situation_shift_reassign"),
        ("Shifted situation, capability-template assignment", "bearing_shift_template", shifted_q, static_assignment, output_dir / "situation_shift_template"),
    ]

    for label, key, q, assignment, prefix in scenarios:
        metrics = simulate_assignment(assignment, output_prefix=prefix, qualification_matrix=q)
        rows.append({
            "scenario": key,
            "method": label,
            "assignment": {AGENTS[a].name: [ROLES[r].name for r in roles] for a, roles in assignment.items()},
            "assignment_utility": metrics.assignment_utility,
            "compatible_utility": metrics.compatible_utility,
            "mean_role_error": metrics.mean_role_error,
            "mean_phase_error": metrics.mean_phase_error,
            "mean_control_ratio": metrics.mean_control_ratio,
            "min_separation": metrics.min_separation,
            "collision_violations": float(metrics.collision_violations),
        })

    set_ocean_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.6))
    fig.patch.set_facecolor("white")
    plot_specs = [
        (output_dir / "situation_baseline_ahp.json", "(a) Baseline AHP", True),
        (output_dir / "situation_shift_stale.json", "(b) Shift, stale", False),
        (output_dir / "situation_shift_reassign.json", "(c) Shift, reassigned", False),
    ]
    for ax, (path, title, show_ylabel) in zip(axes, plot_specs):
        plot_rescue_trajectory(ax, load_history(path), title, show_ylabel=show_ylabel)
        if ax.get_legend():
            ax.legend_.remove()
    fig.tight_layout(w_pad=0.4)
    fig.savefig(output_dir / "situation_aware_panel.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return rows


def perturb_rcc(seed: int, noise_scale: float = 0.18) -> List[List[float]]:
    rng = random.Random(seed)
    n_roles = len(RCC)
    perturbed = [[0.0 for _ in range(n_roles)] for _ in range(n_roles)]
    for i in range(n_roles):
        for j in range(i + 1, n_roles):
            value = RCC[i][j] + rng.uniform(-noise_scale, noise_scale)
            value = max(-1.0, min(1.0, value))
            perturbed[i][j] = value
            perturbed[j][i] = value
    return perturbed


def run_rcc_perturbation_robustness(output_dir: Path, seed_count: int = 51,
                                    noise_scale: float = 0.18) -> Dict[str, Dict[str, float]]:
    scarce_agents = [0, 1, 3, 5]
    role_indices = list(range(len(ROLES)))
    scarce_capacities = [1, 1, 2, 1]

    results: Dict[str, List[SimulationMetrics]] = {
        "Sequential auction perturbed-RCC": [],
        "GMRA compatibility-blind": [],
        "GMRACCR perturbed-RCC": [],
    }
    base_assignments: Dict[str, List[str]] = {
        "Sequential auction perturbed-RCC": [],
        "GMRA compatibility-blind": [],
        "GMRACCR perturbed-RCC": [],
    }

    gmra_assignment, _, _ = optimized_assignment(
        scarce_agents,
        role_indices,
        scarce_capacities,
        use_compatibility=False,
    )
    for seed in range(seed_count):
        rcc = perturb_rcc(3000 + seed, noise_scale=noise_scale)
        auction_assignment = sequential_auction_assignment(
            scarce_agents,
            role_indices,
            scarce_capacities,
            rcc_matrix=rcc,
        )
        gmraccr_assignment, _, _ = optimized_assignment(
            scarce_agents,
            role_indices,
            scarce_capacities,
            use_compatibility=True,
            rcc_matrix=rcc,
        )
        sim_kwargs = {
            "initial_phase_noise": 0.06,
            "initial_radius_noise": 0.03,
            "current_scale": 1.0 + random.Random(4000 + seed).uniform(-0.06, 0.06),
            "seed": 5000 + seed,
        }
        results["Sequential auction perturbed-RCC"].append(
            simulate_assignment(auction_assignment, rcc_matrix=rcc, **sim_kwargs)
        )
        results["GMRA compatibility-blind"].append(
            simulate_assignment(gmra_assignment, rcc_matrix=rcc, **sim_kwargs)
        )
        results["GMRACCR perturbed-RCC"].append(
            simulate_assignment(gmraccr_assignment, rcc_matrix=rcc, **sim_kwargs)
        )
        for label, assignment in (
            ("Sequential auction perturbed-RCC", auction_assignment),
            ("GMRA compatibility-blind", gmra_assignment),
            ("GMRACCR perturbed-RCC", gmraccr_assignment),
        ):
            compact = "; ".join(
                f"{AGENTS[a].name}:{'+'.join(ROLES[r].name.split()[0] for r in roles)}"
                for a, roles in sorted(assignment.items())
            )
            base_assignments[label].append(compact)

    summary = {name: summarize_metrics(metrics) for name, metrics in results.items()}
    for name, assignments in base_assignments.items():
        summary[name]["unique_assignments"] = float(len(set(assignments)))
        summary[name]["zero_collision_rate"] = float(np.mean([m.collision_violations == 0 for m in results[name]]))

    labels = ["Auction", "GMRA", "GMRACCR"]
    source_names = ["Sequential auction perturbed-RCC", "GMRA compatibility-blind", "GMRACCR perturbed-RCC"]
    colors = ["#6f7d1c", "#c84c09", "#1f6f5f"]
    x = np.arange(len(labels))

    set_ocean_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.25))
    sep_mean = [summary[name]["min_separation_mean"] for name in source_names]
    sep_std = [summary[name]["min_separation_std"] for name in source_names]
    axes[0].bar(x, sep_mean, yerr=sep_std, color=colors, capsize=3, width=0.58)
    axes[0].axhline(0.85, color="#d62728", linestyle=":", linewidth=1.1)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, ha="right")
    axes[0].set_ylabel("Min separation (m)")
    axes[0].grid(axis="y")
    axes[0].text(0.02, 0.96, "(a)", transform=axes[0].transAxes, ha="left", va="top", fontweight="bold")

    coll_mean = [summary[name]["collision_violations_mean"] for name in source_names]
    coll_std = [summary[name]["collision_violations_std"] for name in source_names]
    axes[1].bar(x, coll_mean, yerr=coll_std, color=colors, capsize=3, width=0.58)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15, ha="right")
    axes[1].set_ylabel("Collision violations")
    axes[1].grid(axis="y")
    axes[1].text(0.02, 0.96, "(b)", transform=axes[1].transAxes, ha="left", va="top", fontweight="bold")

    zero_rates = [100.0 * summary[name]["zero_collision_rate"] for name in source_names]
    axes[2].bar(x, zero_rates, color=colors, width=0.58)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=15, ha="right")
    axes[2].set_ylabel("Zero-collision trials (%)")
    axes[2].set_ylim(0, 105)
    axes[2].grid(axis="y")
    axes[2].text(0.02, 0.96, "(c)", transform=axes[2].transAxes, ha="left", va="top", fontweight="bold")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout(w_pad=0.55)
    fig.savefig(output_dir / "rcc_perturbation_robustness.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return summary


def summarize_metrics(metrics: Sequence[SimulationMetrics]) -> Dict[str, float]:
    return {
        "mean_role_error_mean": float(np.mean([m.mean_role_error for m in metrics])),
        "mean_role_error_std": float(np.std([m.mean_role_error for m in metrics], ddof=0)),
        "min_separation_mean": float(np.mean([m.min_separation for m in metrics])),
        "min_separation_std": float(np.std([m.min_separation for m in metrics], ddof=0)),
        "collision_violations_mean": float(np.mean([m.collision_violations for m in metrics])),
        "collision_violations_std": float(np.std([m.collision_violations for m in metrics], ddof=0)),
    }


def run_stochastic_scarce_trials(output_dir: Path, seed_count: int = 21) -> Dict[str, Dict[str, float]]:
    scarce_agents = [0, 1, 3, 5]
    role_indices = list(range(len(ROLES)))
    scarce_capacities = [1, 1, 2, 1]
    gmra_assignment, _, _ = optimized_assignment(scarce_agents, role_indices, scarce_capacities, use_compatibility=False)
    auction_assignment = sequential_auction_assignment(scarce_agents, role_indices, scarce_capacities)
    gmraccr_assignment, _, _ = optimized_assignment(scarce_agents, role_indices, scarce_capacities, use_compatibility=True)

    results: Dict[str, List[SimulationMetrics]] = {
        "Sequential auction multi-role": [],
        "GMRA multi-role": [],
        "GMRACCR multi-role": [],
    }
    for seed in range(seed_count):
        current_scale = 1.0 + random.Random(1000 + seed).uniform(-0.08, 0.08)
        sim_kwargs = {
            "current_scale": current_scale,
            "initial_phase_noise": 0.08,
            "initial_radius_noise": 0.04,
            "seed": 2000 + seed,
        }
        results["Sequential auction multi-role"].append(simulate_assignment(auction_assignment, **sim_kwargs))
        results["GMRA multi-role"].append(simulate_assignment(gmra_assignment, **sim_kwargs))
        results["GMRACCR multi-role"].append(simulate_assignment(gmraccr_assignment, **sim_kwargs))

    labels = list(results.keys())
    means_sep = [np.mean([m.min_separation for m in results[label]]) for label in labels]
    std_sep = [np.std([m.min_separation for m in results[label]], ddof=0) for label in labels]
    means_coll = [np.mean([m.collision_violations for m in results[label]]) for label in labels]
    std_coll = [np.std([m.collision_violations for m in results[label]], ddof=0) for label in labels]

    short_labels = ["Auction", "GMRA", "GMRACCR"]
    x = np.arange(len(labels))
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    fig.patch.set_facecolor("white")
    axes[0].bar(x, means_sep, yerr=std_sep, capsize=4, color=["#6f7d1c", "#c84c09", "#1f6f5f"], width=0.62)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(short_labels)
    axes[0].set_ylabel("Min separation")
    axes[0].grid(axis="y", alpha=0.18, linewidth=0.7)
    axes[0].text(0.02, 0.96, "(a)", transform=axes[0].transAxes, ha="left", va="top", fontweight="bold")

    axes[1].bar(x, means_coll, yerr=std_coll, capsize=4, color=["#6f7d1c", "#c84c09", "#1f6f5f"], width=0.62)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short_labels)
    axes[1].set_ylabel("Collision violations")
    axes[1].grid(axis="y", alpha=0.18, linewidth=0.7)
    axes[1].text(0.02, 0.96, "(b)", transform=axes[1].transAxes, ha="left", va="top", fontweight="bold")

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "stochastic_scarce.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {name: summarize_metrics(metrics) for name, metrics in results.items()}


def synthetic_assignment_inputs(n_agents: int, n_roles: int) -> Tuple[List[List[float]], List[List[float]], List[int], List[int]]:
    base_q = build_qualification_matrix(list(range(len(AGENTS))), list(range(len(ROLES))))
    q: List[List[float]] = []
    for i in range(n_agents):
        row = []
        for j in range(n_roles):
            value = base_q[i % len(AGENTS)][j % len(ROLES)]
            value += 0.015 * ((i * 3 + j * 5) % 7)
            row.append(min(1.0, value))
        q.append(row)

    rcc: List[List[float]] = []
    for j in range(n_roles):
        row = []
        for k in range(n_roles):
            if j == k:
                row.append(0.0)
            else:
                row.append(RCC[j % len(ROLES)][k % len(ROLES)])
        rcc.append(row)

    role_demands = [1] * n_roles
    capacities = [1] * n_agents
    extra_capacity = max(0, n_roles - n_agents)
    for idx in range(extra_capacity):
        capacities[idx % n_agents] += 1
    return q, rcc, role_demands, capacities


def sequential_auction_matrix(q: Sequence[Sequence[float]], rcc: Sequence[Sequence[float]],
                              capacities: Sequence[int]) -> List[List[int]]:
    n_agents = len(q)
    n_roles = len(q[0])
    remaining = list(capacities)
    assignment = [[0 for _ in range(n_roles)] for _ in range(n_agents)]
    held_roles: List[List[int]] = [[] for _ in range(n_agents)]
    for role_idx in range(n_roles):
        best_agent = None
        best_bid = float("-inf")
        for agent_idx in range(n_agents):
            if remaining[agent_idx] <= 0:
                continue
            marginal_compat = sum(rcc[role_idx][held] + rcc[held][role_idx] for held in held_roles[agent_idx])
            bid = q[agent_idx][role_idx] + marginal_compat
            if bid > best_bid:
                best_bid = bid
                best_agent = agent_idx
        if best_agent is None:
            raise RuntimeError("No feasible synthetic auction assignment found")
        assignment[best_agent][role_idx] = 1
        held_roles[best_agent].append(role_idx)
        remaining[best_agent] -= 1
    return assignment


def time_call(fn, repeats: int) -> Tuple[float, float]:
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        values.append((time.perf_counter() - start) * 1000.0)
    return float(np.mean(values)), float(np.std(values, ddof=0))


def run_assignment_scaling(output_dir: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for n_agents, n_roles, repeats in ((6, 5, 50), (8, 6, 20), (10, 8, 5)):
        q, rcc, role_demands, capacities = synthetic_assignment_inputs(n_agents, n_roles)

        def solve_gmraccr() -> None:
            GMRACCR(n_agents, n_roles, q, role_demands, capacities, rcc).solve(prefer_pulp=True)

        def solve_auction() -> None:
            sequential_auction_matrix(q, rcc, capacities)

        g_mean, g_std = time_call(solve_gmraccr, repeats)
        a_mean, a_std = time_call(solve_auction, repeats)
        rows.append({
            "agents": float(n_agents),
            "roles": float(n_roles),
            "gmraccr_ms_mean": g_mean,
            "gmraccr_ms_std": g_std,
            "auction_ms_mean": a_mean,
            "auction_ms_std": a_std,
        })
    labels = [f"{int(row['agents'])}/{int(row['roles'])}" for row in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    fig.patch.set_facecolor("white")
    ax.errorbar(x, [row["gmraccr_ms_mean"] for row in rows], yerr=[row["gmraccr_ms_std"] for row in rows],
                marker="o", linewidth=2.0, capsize=4, color="#153b50", label="GMRACCR")
    ax.errorbar(x, [row["auction_ms_mean"] for row in rows], yerr=[row["auction_ms_std"] for row in rows],
                marker="s", linewidth=2.0, capsize=4, color="#c84c09", label="Sequential auction")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Agents / roles")
    ax.set_ylabel("Runtime (ms, log scale)")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.18, linewidth=0.7)
    ax.legend(frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "assignment_scaling.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return rows


def load_history(path: Path) -> Dict[int, List[Tuple[float, float]]]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(agent_idx): [tuple(point) for point in points] for agent_idx, points in raw.items()}


def make_scarce_comparison_panel(output_dir: Path, scarce_results: Dict[str, SimulationMetrics]) -> None:
    set_ocean_plot_style()
    fig = plt.figure(figsize=(10.2, 6.1))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.05, 1.0], hspace=0.34, wspace=0.18)
    histories = [
        load_history(output_dir / "scarce_auction.json"),
        load_history(output_dir / "scarce_gmra.json"),
        load_history(output_dir / "scarce_gmraccr.json"),
    ]
    labels = ["(a) Sequential auction", "(b) GMRA", "(c) GMRACCR"]
    for idx, (history, label) in enumerate(zip(histories, labels)):
        ax = fig.add_subplot(gs[0, idx])
        plot_rescue_trajectory(ax, history, label, show_ylabel=(idx == 0))
        ax.legend_.remove() if ax.get_legend() else None

    methods = ["Auction", "GMRA", "GMRACCR"]
    source_names = ["Sequential auction multi-role", "GMRA-multi", "GMRACCR-multi"]
    colors = ["#6f7d1c", "#c84c09", "#1f6f5f"]
    x = np.arange(len(methods))
    ax_sep = fig.add_subplot(gs[1, :2])
    sep_values = [scarce_results[name].min_separation for name in source_names]
    bars = ax_sep.bar(x, sep_values, color=colors, width=0.58)
    ax_sep.axhline(0.85, color="#d62728", linestyle=":", linewidth=1.2, label="Safety threshold")
    ax_sep.set_xticks(x)
    ax_sep.set_xticklabels(methods)
    ax_sep.set_ylabel("Minimum separation (m)")
    ax_sep.grid(axis="y")
    ax_sep.legend(frameon=False, loc="upper left")
    for bar, value in zip(bars, sep_values):
        ax_sep.text(bar.get_x() + bar.get_width() / 2, value + 0.05, f"{value:.2f}",
                    ha="center", va="bottom", fontsize=8)
    ax_sep.text(0.01, 0.95, "(d)", transform=ax_sep.transAxes, ha="left", va="top", fontweight="bold")

    ax_col = fig.add_subplot(gs[1, 2])
    collision_values = [scarce_results[name].collision_violations for name in source_names]
    bars = ax_col.bar(x, collision_values, color=colors, width=0.58)
    ax_col.set_xticks(x)
    ax_col.set_xticklabels(methods, rotation=25, ha="right")
    ax_col.set_ylabel("Collision violations")
    ax_col.grid(axis="y")
    for bar, value in zip(bars, collision_values):
        ax_col.text(bar.get_x() + bar.get_width() / 2, value + max(collision_values) * 0.02 + 1,
                    f"{int(value)}", ha="center", va="bottom", fontsize=8)
    ax_col.text(0.03, 0.95, "(e)", transform=ax_col.transAxes, ha="left", va="top", fontweight="bold")

    for ax in (ax_sep, ax_col):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.savefig(output_dir / "scarce_comparison_panel.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_robustness_scaling_panel(output_dir: Path, stochastic_scarce: Dict[str, Dict[str, float]],
                                  assignment_scaling: List[Dict[str, float]]) -> None:
    set_ocean_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.1), gridspec_kw={"width_ratios": [1.0, 1.0, 1.25]})
    methods = ["Auction", "GMRA", "GMRACCR"]
    source_names = ["Sequential auction multi-role", "GMRA multi-role", "GMRACCR multi-role"]
    colors = ["#6f7d1c", "#c84c09", "#1f6f5f"]
    x = np.arange(len(methods))

    sep_mean = [stochastic_scarce[name]["min_separation_mean"] for name in source_names]
    sep_std = [stochastic_scarce[name]["min_separation_std"] for name in source_names]
    axes[0].bar(x, sep_mean, yerr=sep_std, color=colors, capsize=3, width=0.58)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=15, ha="right")
    axes[0].set_ylabel("Min separation (m)")
    axes[0].grid(axis="y")
    axes[0].text(0.02, 0.96, "(a)", transform=axes[0].transAxes, ha="left", va="top", fontweight="bold")

    coll_mean = [stochastic_scarce[name]["collision_violations_mean"] for name in source_names]
    coll_std = [stochastic_scarce[name]["collision_violations_std"] for name in source_names]
    axes[1].bar(x, coll_mean, yerr=coll_std, color=colors, capsize=3, width=0.58)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, rotation=15, ha="right")
    axes[1].set_ylabel("Collision violations")
    axes[1].grid(axis="y")
    axes[1].text(0.02, 0.96, "(b)", transform=axes[1].transAxes, ha="left", va="top", fontweight="bold")

    labels = [f"{int(row['agents'])}/{int(row['roles'])}" for row in assignment_scaling]
    sx = np.arange(len(labels))
    axes[2].errorbar(sx, [row["gmraccr_ms_mean"] for row in assignment_scaling],
                     yerr=[row["gmraccr_ms_std"] for row in assignment_scaling],
                     marker="o", linewidth=1.8, capsize=3, color="#153b50", label="GMRACCR")
    axes[2].errorbar(sx, [row["auction_ms_mean"] for row in assignment_scaling],
                     yerr=[row["auction_ms_std"] for row in assignment_scaling],
                     marker="s", linewidth=1.8, capsize=3, color="#c84c09", label="Sequential auction")
    axes[2].set_xticks(sx)
    axes[2].set_xticklabels(labels)
    axes[2].set_xlabel("Agents / roles")
    axes[2].set_ylabel("Runtime (ms, log scale)")
    axes[2].set_yscale("log")
    axes[2].grid(axis="y")
    axes[2].legend(frameon=False, loc="upper left")
    axes[2].text(0.02, 0.96, "(c)", transform=axes[2].transAxes, ha="left", va="top", fontweight="bold")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(output_dir / "robustness_scaling_panel.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_all_experiments(output_dir: Path) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    agent_indices = list(range(len(AGENTS)))
    role_indices = list(range(len(ROLES)))

    opt_assignment, opt_base, opt_compat = optimized_assignment(agent_indices, role_indices, [1] * len(AGENTS), use_compatibility=True)
    auction_assignment = sequential_auction_assignment(agent_indices, role_indices, [1] * len(AGENTS))
    greedy_assignment = average_capability_greedy(agent_indices, role_indices)
    random_assignment = random_feasible_assignment(agent_indices, role_indices, seed=11)

    static_results = {
        "GMRACCR-opt": simulate_assignment(opt_assignment, output_prefix=output_dir / "static_gmraccr"),
        "Sequential auction": simulate_assignment(auction_assignment, output_prefix=output_dir / "static_auction"),
        "Avg-greedy": simulate_assignment(greedy_assignment, output_prefix=output_dir / "static_greedy"),
        "Random": simulate_assignment(random_assignment, output_prefix=output_dir / "static_random"),
    }

    scarce_agents = [0, 1, 3, 5]  # UUV1, UUV2, UUV4, UUV6
    scarce_capacities = [1, 1, 2, 1]
    gmra_multi_assignment, gmra_base, gmra_compat = optimized_assignment(scarce_agents, role_indices, scarce_capacities, use_compatibility=False)
    auction_multi_assignment = sequential_auction_assignment(scarce_agents, role_indices, scarce_capacities)
    gmraccr_multi_assignment, gmraccr_base, gmraccr_compat = optimized_assignment(scarce_agents, role_indices, scarce_capacities, use_compatibility=True)

    scarce_results = {
        "Sequential auction multi-role": simulate_assignment(auction_multi_assignment, output_prefix=output_dir / "scarce_auction"),
        "GMRA-multi": simulate_assignment(gmra_multi_assignment, output_prefix=output_dir / "scarce_gmra"),
        "GMRACCR-multi": simulate_assignment(gmraccr_multi_assignment, output_prefix=output_dir / "scarce_gmraccr"),
    }

    relay_role_idx = 2
    failed_agent = next(agent_idx for agent_idx, roles in opt_assignment.items() if relay_role_idx in roles)
    survivors = [idx for idx in agent_indices if idx != failed_agent]
    reassigned_assignment, reassigned_base, reassigned_compat = optimized_assignment(survivors, role_indices, [1] * len(survivors), use_compatibility=True)
    no_reassign_assignment = {agent_idx: roles for agent_idx, roles in opt_assignment.items() if agent_idx != failed_agent}

    failure_results = {
        "No-reassignment": simulate_assignment(
            opt_assignment,
            failure_step=180,
            failed_agent=failed_agent,
            reassignment=no_reassign_assignment,
            output_prefix=output_dir / "failure_no_reassign",
        ),
        "Reassignment": simulate_assignment(
            opt_assignment,
            failure_step=180,
            failed_agent=failed_agent,
            reassignment=reassigned_assignment,
            output_prefix=output_dir / "failure_reassign",
        ),
    }

    disturbance_results = {
        "Current-only": simulate_assignment(
            opt_assignment,
            enable_current=True,
            enable_collision_avoidance=False,
            compact_start=True,
            current_scale=1.2,
            safe_distance=0.9,
            output_prefix=output_dir / "disturbance_current_only",
        ),
        "Current+collision-aware": simulate_assignment(
            opt_assignment,
            enable_current=True,
            enable_collision_avoidance=True,
            compact_start=True,
            current_scale=1.0,
            safe_distance=0.9,
            avoidance_gain=2.5,
            output_prefix=output_dir / "disturbance_collision_aware",
        ),
    }

    shape_results = {
        "Ellipse-reference": simulate_assignment(
            opt_assignment,
            enable_current=True,
            enable_collision_avoidance=True,
            current_scale=1.0,
            orbit_mode="ellipse",
            output_prefix=output_dir / "shape_ellipse_reference",
        ),
        "Circular-simplification": simulate_assignment(
            opt_assignment,
            enable_current=True,
            enable_collision_avoidance=True,
            current_scale=1.0,
            orbit_mode="circle",
            output_prefix=output_dir / "shape_circular_reference",
        ),
    }

    beta_sweep = run_beta_sweep(output_dir, beta_values=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    disturbance_sweep = run_disturbance_sweep(output_dir, current_scales=[0.6, 0.8, 1.0, 1.2, 1.4, 1.6])
    situation_aware = run_situation_aware_end_to_end(output_dir)
    rcc_perturbation = run_rcc_perturbation_robustness(output_dir, seed_count=51, noise_scale=0.18)
    stochastic_scarce = run_stochastic_scarce_trials(output_dir, seed_count=21)
    assignment_scaling = run_assignment_scaling(output_dir)
    make_scarce_comparison_panel(output_dir, scarce_results)
    make_robustness_scaling_panel(output_dir, stochastic_scarce, assignment_scaling)

    summary = {
        "static_assignments": {k: {AGENTS[a].name: [ROLES[r].name for r in v] for a, v in assignment.items()}
                               for k, assignment in {
                                   "GMRACCR-opt": opt_assignment,
                                   "Sequential auction": auction_assignment,
                                   "Avg-greedy": greedy_assignment,
                                   "Random": random_assignment,
                               }.items()},
        "scarce_assignments": {k: {AGENTS[a].name: [ROLES[r].name for r in v] for a, v in assignment.items()}
                               for k, assignment in {
                                   "Sequential auction multi-role": auction_multi_assignment,
                                   "GMRA-multi": gmra_multi_assignment,
                                   "GMRACCR-multi": gmraccr_multi_assignment,
                               }.items()},
        "failure_failed_agent": AGENTS[failed_agent].name,
        "failure_reassignment": {AGENTS[a].name: [ROLES[r].name for r in v] for a, v in reassigned_assignment.items()},
        "static_results": {k: asdict(v) for k, v in static_results.items()},
        "scarce_results": {k: asdict(v) for k, v in scarce_results.items()},
        "failure_results": {k: asdict(v) for k, v in failure_results.items()},
        "disturbance_results": {k: asdict(v) for k, v in disturbance_results.items()},
        "shape_results": {k: asdict(v) for k, v in shape_results.items()},
        "beta_sweep": beta_sweep,
        "disturbance_sweep": disturbance_sweep,
        "situation_aware": situation_aware,
        "rcc_perturbation": rcc_perturbation,
        "stochastic_scarce": stochastic_scarce,
        "assignment_scaling": assignment_scaling,
        "assignment_objectives": {
            "GMRACCR-opt": {"base": opt_base, "compat": opt_compat},
            "GMRA-multi": {"base": gmra_base, "compat": gmra_compat},
            "GMRACCR-multi": {"base": gmraccr_base, "compat": gmraccr_compat},
            "Failure-reassignment": {"base": reassigned_base, "compat": reassigned_compat},
        },
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    table_rows = []
    for group_name in ("static_results", "scarce_results", "failure_results", "disturbance_results", "shape_results"):
        for method_name, metrics in summary[group_name].items():
            table_rows.append({
                "group": group_name,
                "method": method_name,
                **metrics,
            })
    for row in beta_sweep:
        table_rows.append({"group": "beta_sweep", "method": f"beta={row['beta']:.1f}", **row})
    for row in disturbance_sweep:
        table_rows.append({"group": "disturbance_sweep", "method": f"{row['mode']}-scale={row['current_scale']:.1f}", **row})
    for row in situation_aware:
        table_row = {key: value for key, value in row.items() if key != "assignment"}
        table_rows.append({"group": "situation_aware", **table_row})
    for method_name, row in rcc_perturbation.items():
        table_rows.append({"group": "rcc_perturbation", "method": method_name, **row})
    for method_name, row in stochastic_scarce.items():
        table_rows.append({"group": "stochastic_scarce", "method": method_name, **row})
    for row in assignment_scaling:
        table_rows.append({"group": "assignment_scaling", "method": f"{int(row['agents'])}/{int(row['roles'])}", **row})
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = []
        for row in table_rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table_rows)

    return summary


if __name__ == "__main__":
    output = Path(__file__).resolve().parent / "results"
    summary = run_all_experiments(output)
    print(json.dumps(summary, indent=2))
