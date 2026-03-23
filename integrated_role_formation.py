from __future__ import annotations

import json
import math
import random
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
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

RCC = [
    [0.00, 0.45, -0.20, 0.05, 0.00],
    [0.45, 0.00, -0.30, 0.05, -0.10],
    [-0.20, -0.30, 0.00, -0.10, 0.35],
    [0.05, 0.05, -0.10, 0.00, 0.25],
    [0.00, -0.10, 0.35, 0.25, 0.00],
]


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


def optimized_assignment(agent_indices: Sequence[int], role_indices: Sequence[int], capacities: Sequence[int],
                         use_compatibility: bool = True, compatibility_scale: float = 1.0) -> Tuple[Dict[int, List[int]], float, float]:
    q = build_qualification_matrix(agent_indices, role_indices)
    if use_compatibility:
        rcc = [[compatibility_scale * value for value in row] for row in RCC]
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


def assignment_objective(assignment: Dict[int, List[int]], agent_indices: Sequence[int], role_indices: Sequence[int]) -> Tuple[float, float]:
    q = build_qualification_matrix(agent_indices, role_indices)
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
                compat += RCC[local_role_lookup[role_i]][local_role_lookup[role_j]]
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
                        compact_start: bool = False, orbit_mode: str = "circle") -> SimulationMetrics:
    standby_radius = 5.6
    base_omega = 0.18
    k_radial = 0.85
    k_phase = 0.55
    multitask_penalty = 0.18

    states: Dict[int, Dict[str, float]] = {}
    current_assignment = {agent_idx: list(roles) for agent_idx, roles in assignment.items()}
    all_agent_ids = sorted(set(current_assignment) | ({failed_agent} if failed_agent is not None else set()) | ({idx for idx in range(len(AGENTS))}))

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
        plt.figure(figsize=(6, 6))
        theta = np.linspace(0.0, 2.0 * np.pi, 300)
        for role in ROLES:
            if orbit_mode == "ellipse":
                x_orbit = 1.20 * role.radius * np.cos(theta)
                y_orbit = 0.82 * role.radius * np.sin(theta)
            else:
                x_orbit = role.radius * np.cos(theta)
                y_orbit = role.radius * np.sin(theta)
            plt.plot(x_orbit, y_orbit, "--", linewidth=0.7, color="0.7")
        for agent_idx, trajectory in history.items():
            xs = [p[0] for p in trajectory]
            ys = [p[1] for p in trajectory]
            label = AGENTS[agent_idx].name
            plt.plot(xs, ys, linewidth=1.8, label=label)
        plt.axis("equal")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(output_prefix.with_suffix(".png"), dpi=200)
        plt.close()

    base, compat = assignment_objective(current_assignment, list(range(len(AGENTS))), list(range(len(ROLES))))
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


def run_all_experiments(output_dir: Path) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    agent_indices = list(range(len(AGENTS)))
    role_indices = list(range(len(ROLES)))

    opt_assignment, opt_base, opt_compat = optimized_assignment(agent_indices, role_indices, [1] * len(AGENTS), use_compatibility=True)
    greedy_assignment = average_capability_greedy(agent_indices, role_indices)
    random_assignment = random_feasible_assignment(agent_indices, role_indices, seed=11)

    static_results = {
        "GMRACCR-opt": simulate_assignment(opt_assignment, output_prefix=output_dir / "static_gmraccr"),
        "Avg-greedy": simulate_assignment(greedy_assignment, output_prefix=output_dir / "static_greedy"),
        "Random": simulate_assignment(random_assignment, output_prefix=output_dir / "static_random"),
    }

    scarce_agents = [0, 1, 3, 5]  # UUV1, UUV2, UUV4, UUV6
    scarce_capacities = [1, 1, 2, 1]
    gmra_multi_assignment, gmra_base, gmra_compat = optimized_assignment(scarce_agents, role_indices, scarce_capacities, use_compatibility=False)
    gmraccr_multi_assignment, gmraccr_base, gmraccr_compat = optimized_assignment(scarce_agents, role_indices, scarce_capacities, use_compatibility=True)

    scarce_results = {
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

    summary = {
        "static_assignments": {k: {AGENTS[a].name: [ROLES[r].name for r in v] for a, v in assignment.items()}
                               for k, assignment in {
                                   "GMRACCR-opt": opt_assignment,
                                   "Avg-greedy": greedy_assignment,
                                   "Random": random_assignment,
                               }.items()},
        "scarce_assignments": {k: {AGENTS[a].name: [ROLES[r].name for r in v] for a, v in assignment.items()}
                               for k, assignment in {
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
