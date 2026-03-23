from __future__ import annotations

import itertools
import statistics


AGENTS = ["UUV1", "UUV2", "UUV3", "UUV4", "UUV5", "UUV6"]
ROLES = [
    "Approach guidance",
    "High-mobility interception",
    "Communication relay",
    "Close-range support",
    "Safety backup",
]

# Agent-role qualification matrix used in the paper.
Q = [
    [0.21, 0.15, 0.19, 0.25, 0.38],
    [0.69, 0.79, 0.48, 0.56, 0.48],
    [0.32, 0.39, 0.12, 0.33, 0.33],
    [0.53, 0.39, 0.72, 0.53, 0.38],
    [0.10, 0.18, 0.24, 0.16, 0.48],
    [0.26, 0.03, 0.36, 0.42, 0.33],
]

# A simple compatibility setting for the multi-role experiment.
RCC = [
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, -0.4, 0.0, 0.0],
    [0.0, -0.4, 0.0, 0.3, 0.2],
    [0.0, 0.0, 0.3, 0.0, 0.1],
    [0.0, 0.0, 0.2, 0.1, 0.0],
]


def assignment_score(role_to_agent: tuple[int, ...]) -> float:
    return sum(Q[a][r] for r, a in enumerate(role_to_agent))


def exhaustive_single_role() -> tuple[float, tuple[int, ...], list[float]]:
    values = []
    best: tuple[float, tuple[int, ...]] | None = None
    for perm in itertools.permutations(range(len(AGENTS)), len(ROLES)):
        score = assignment_score(perm)
        values.append(score)
        if best is None or score > best[0]:
            best = (score, perm)
    assert best is not None
    return best[0], best[1], values


def greedy_by_role() -> tuple[float, tuple[int, ...]]:
    remaining = set(range(len(AGENTS)))
    role_to_agent = []
    for role in range(len(ROLES)):
        best_agent = max(remaining, key=lambda idx: Q[idx][role])
        role_to_agent.append(best_agent)
        remaining.remove(best_agent)
    assignment = tuple(role_to_agent)
    return assignment_score(assignment), assignment


def greedy_by_agent_average() -> tuple[float, list[int]]:
    averages = [sum(row) / len(row) for row in Q]
    ranked_agents = sorted(range(len(AGENTS)), key=lambda idx: averages[idx], reverse=True)
    remaining_roles = set(range(len(ROLES)))
    role_to_agent: list[int | None] = [None] * len(ROLES)
    for agent in ranked_agents[: len(ROLES)]:
        role = max(remaining_roles, key=lambda r: Q[agent][r])
        role_to_agent[role] = agent
        remaining_roles.remove(role)
    assert all(agent is not None for agent in role_to_agent)
    final = [int(agent) for agent in role_to_agent]
    return assignment_score(tuple(final)), final


def leave_one_out() -> list[tuple[str, float, tuple[int, ...]]]:
    results = []
    for leave_idx, name in enumerate(AGENTS):
        best: tuple[float, tuple[int, ...]] | None = None
        available = [idx for idx in range(len(AGENTS)) if idx != leave_idx]
        for perm in itertools.permutations(available, len(ROLES)):
            score = assignment_score(perm)
            if best is None or score > best[0]:
                best = (score, perm)
        assert best is not None
        results.append((name, best[0], best[1]))
    return results


def multi_role_with_compatibility(beta: float = 0.15) -> tuple[float, tuple[int, ...], float]:
    best: tuple[float, tuple[int, ...], float] | None = None
    for role_to_agent in itertools.product(range(len(AGENTS)), repeat=len(ROLES)):
        if any(role_to_agent.count(agent) > 2 for agent in range(len(AGENTS))):
            continue
        base = assignment_score(role_to_agent)
        score = base
        for agent in range(len(AGENTS)):
            roles = [role for role, assignee in enumerate(role_to_agent) if assignee == agent]
            for r1 in roles:
                for r2 in roles:
                    score += beta * Q[agent][r1] * RCC[r1][r2]
        if best is None or score > best[0]:
            best = (score, role_to_agent, base)
    assert best is not None
    return best


def format_assignment(role_to_agent: tuple[int, ...] | list[int]) -> str:
    return ", ".join(f"{role}={AGENTS[agent]}" for role, agent in zip(ROLES, role_to_agent))


def main() -> None:
    best_score, best_assignment, values = exhaustive_single_role()
    greedy_score, greedy_assignment = greedy_by_role()
    avg_score, avg_assignment = greedy_by_agent_average()
    multi_score, multi_assignment, multi_base = multi_role_with_compatibility()

    print("Single-role optimum")
    print(f"score={best_score:.2f}")
    print(format_assignment(best_assignment))
    print()

    print("Assignment distribution over all feasible single-role deployments")
    print(f"mean={statistics.mean(values):.2f}")
    print(f"median={statistics.median(values):.2f}")
    print(f"stdev={statistics.pstdev(values):.2f}")
    print(f"min={min(values):.2f}")
    print(f"max={max(values):.2f}")
    print()

    print("Heuristic baselines")
    print(f"role-greedy={greedy_score:.2f} :: {format_assignment(greedy_assignment)}")
    print(f"agent-average-greedy={avg_score:.2f} :: {format_assignment(avg_assignment)}")
    print()

    print("Leave-one-agent-out robustness")
    for name, score, assignment in leave_one_out():
        print(f"{name}: {score:.2f} :: {format_assignment(assignment)}")
    print()

    print("Multi-role compatibility experiment")
    print(f"compatibility-aware-score={multi_score:.3f}")
    print(f"base-score-without-compatibility={multi_base:.2f}")
    print(format_assignment(multi_assignment))


if __name__ == "__main__":
    main()
