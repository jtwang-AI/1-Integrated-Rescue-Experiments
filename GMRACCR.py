"""
GMRACCR solver utilities.

This file refactors the original demo script into an importable module while
preserving a standalone example. It supports a PuLP-based solver when PuLP is
available and falls back to an exact depth-first enumeration solver for the
small- and medium-sized cases used in the paper experiments.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import List, Sequence, Tuple

try:
    import pulp  # type: ignore
except Exception:  # pragma: no cover - fallback path is intentional
    pulp = None


AssignmentMatrix = List[List[int]]


@dataclass
class GMRACCRResult:
    assignment: AssignmentMatrix
    pair_assignment: List[int]
    base_score: float
    compatibility_score: float
    objective: float
    solver: str


class GMRACCR:
    def __init__(self, nagent: int, nrole: int, QM: Sequence[Sequence[float]],
                 RA: Sequence[int], LA: Sequence[int], RCC: Sequence[Sequence[float]]):
        self.m = nagent
        self.n = nrole
        self.L = list(RA)
        self.LA = list(LA)
        self.Q = [list(row) for row in QM]
        self.RCC = [list(row) for row in RCC]

    def _objective_components(self, assignment: AssignmentMatrix) -> Tuple[float, float]:
        base = 0.0
        compat = 0.0
        for i in range(self.m):
            roles_i = []
            for j in range(self.n):
                if assignment[i][j]:
                    base += self.Q[i][j]
                    roles_i.append(j)
            for j1 in roles_i:
                for j2 in roles_i:
                    compat += self.RCC[j1][j2]
        return base, compat

    def _flatten_pair_assignment(self, assignment: AssignmentMatrix) -> List[int]:
        pair_assignment = [0] * (self.m * self.n * self.n)
        for i in range(self.m):
            roles_i = [j for j in range(self.n) if assignment[i][j] == 1]
            for j1 in roles_i:
                for j2 in roles_i:
                    pair_assignment[(j2 * self.m * self.n) + (j1 * self.m) + i] = 1
        return pair_assignment

    def _solve_with_pulp(self) -> GMRACCRResult:
        if pulp is None:
            raise RuntimeError("PuLP is not available")
        agents = range(self.m)
        roles = range(self.n)
        mn = self.m * self.n
        mnn = self.m * self.n * self.n
        gra = pulp.LpProblem("GMRACCR_Model", pulp.LpMaximize)
        assignment_indices = [i * self.n + j for i in agents for j in roles]
        pair_indices = [(j2 * mn) + (j1 * self.m) + i for i in agents for j1 in roles for j2 in roles]

        x = pulp.LpVariable.dicts("Assignment", range(mn), 0, 1, pulp.LpInteger)
        z = pulp.LpVariable.dicts("Assignment1", range(mnn), 0, 1, pulp.LpInteger)

        gra += (
            pulp.lpSum(x[idx] * self.Q[idx // self.n][idx % self.n] for idx in assignment_indices)
            + pulp.lpSum(
                z[idx] * self.RCC[(idx - (idx // mn) * mn) // self.m][idx // mn]
                for idx in pair_indices
            ),
            "objective",
        )

        for j in roles:
            gra += pulp.lpSum(x[i * self.n + j] for i in agents) == self.L[j], f"role_{j}"
        for i in agents:
            gra += pulp.lpSum(x[i * self.n + j] for j in roles) <= self.LA[i], f"agent_{i}"
        for i in agents:
            for j1 in roles:
                for j2 in roles:
                    idx = (j2 * self.m * self.n) + (j1 * self.m) + i
                    gra += z[idx] * 2 <= x[i * self.n + j1] + x[i * self.n + j2]
                    gra += x[i * self.n + j1] + x[i * self.n + j2] <= z[idx] + 1

        gra.solve(pulp.PULP_CBC_CMD(msg=False))
        assignment = [[0 for _ in roles] for _ in agents]
        for i in agents:
            for j in roles:
                value = x[i * self.n + j].value()
                assignment[i][j] = 1 if value is not None and abs(value - 1) < 1e-6 else 0
        pair_assignment = self._flatten_pair_assignment(assignment)
        base, compat = self._objective_components(assignment)
        return GMRACCRResult(
            assignment=assignment,
            pair_assignment=pair_assignment,
            base_score=base,
            compatibility_score=compat,
            objective=base + compat,
            solver="pulp",
        )

    def _solve_with_enumeration(self) -> GMRACCRResult:
        role_slots = []
        for role, demand in enumerate(self.L):
            role_slots.extend([role] * demand)

        assignment = [[0 for _ in range(self.n)] for _ in range(self.m)]
        used_capacity = [0] * self.m
        best_assignment: AssignmentMatrix | None = None
        best_objective = float("-inf")

        max_q_per_role = [max(self.Q[i][j] for i in range(self.m)) for j in range(self.n)]
        optimistic_slot_gain = [max_q_per_role[role] + max(max(row) for row in self.RCC) for role in role_slots]

        def dfs(slot_idx: int, current_base: float, current_compat: float) -> None:
            nonlocal best_assignment, best_objective
            if slot_idx == len(role_slots):
                objective = current_base + current_compat
                if objective > best_objective:
                    best_objective = objective
                    best_assignment = copy.deepcopy(assignment)
                return

            remaining_upper = sum(optimistic_slot_gain[slot_idx:])
            if current_base + current_compat + remaining_upper <= best_objective:
                return

            role = role_slots[slot_idx]
            for agent in range(self.m):
                if used_capacity[agent] >= self.LA[agent]:
                    continue
                if assignment[agent][role] == 1:
                    continue

                compat_delta = 0.0
                for other_role in range(self.n):
                    if assignment[agent][other_role] == 1:
                        compat_delta += self.RCC[role][other_role] + self.RCC[other_role][role]

                assignment[agent][role] = 1
                used_capacity[agent] += 1
                dfs(
                    slot_idx + 1,
                    current_base + self.Q[agent][role],
                    current_compat + compat_delta,
                )
                used_capacity[agent] -= 1
                assignment[agent][role] = 0

        dfs(0, 0.0, 0.0)
        if best_assignment is None:
            raise RuntimeError("No feasible assignment found")
        pair_assignment = self._flatten_pair_assignment(best_assignment)
        base, compat = self._objective_components(best_assignment)
        return GMRACCRResult(
            assignment=best_assignment,
            pair_assignment=pair_assignment,
            base_score=base,
            compatibility_score=compat,
            objective=base + compat,
            solver="enumeration",
        )

    def solve(self, prefer_pulp: bool = True) -> GMRACCRResult:
        if prefer_pulp and pulp is not None:
            return self._solve_with_pulp()
        return self._solve_with_enumeration()

    @property
    def resolve(self) -> Tuple[List[int], List[int]]:
        result = self.solve()
        flat_assignment = [value for row in result.assignment for value in row]
        return flat_assignment, result.pair_assignment


def printDMatrix(x: Sequence[Sequence[float]], m: int, n: int) -> None:
    txt = "{:.2f}"
    for i in range(m):
        for j in range(n):
            print(txt.format(x[i][j]), " ", end="")
        print()


def printIMatrix(x: Sequence[Sequence[int]], m: int, n: int) -> None:
    txt = "{:2}"
    for i in range(m):
        for j in range(n):
            print(txt.format(x[i][j]), " ", end="")
        print()


def sigmaL(L: Sequence[int]) -> int:
    return sum(L)


def getWQ(m: int, n: int, Q: Sequence[Sequence[float]], W: Sequence[float]) -> List[List[float]]:
    maxQ = 1.0
    WQ = copy.deepcopy(Q)
    for i in range(m):
        for j in range(n):
            WQ[i][j] = Q[i][j] * W[j]
            if WQ[i][j] > maxQ:
                maxQ = WQ[i][j]
    for i in range(m):
        for j in range(n):
            WQ[i][j] = WQ[i][j] / maxQ
    return WQ


def _demo() -> None:
    m = 6
    n = 5
    L = [1, 1, 1, 1, 1]
    LA = [2, 2, 2, 2, 1, 2]
    Q = [
        [0.51, 0.87, 0.49, 0.66, 0.43],
        [0.76, 0.43, 0.65, 0.74, 0.19],
        [0.38, 0.21, 0.57, 0.05, 0.87],
        [0.51, 0.87, 0.49, 0.66, 0.43],
        [0.76, 0.43, 0.65, 0.74, 0.19],
        [0.38, 0.21, 0.57, 0.05, 0.87],
    ]
    RCC = [
        [0.0, 0.2, 0.3, 0.0, 0.4],
        [0.2, 0.0, 1.0, -0.2, 0.3],
        [0.3, 1.0, 0.0, -0.5, 0.4],
        [0.0, -0.2, -0.5, 0.0, 0.4],
        [0.4, 0.3, 0.4, 0.4, 0.0],
    ]

    t1 = int(round(time.time() * 1000))
    solver = GMRACCR(m, n, Q, L, LA, RCC)
    result = solver.solve(prefer_pulp=True)
    t2 = int(round(time.time() * 1000))

    print("Q=")
    printDMatrix(Q, m, n)
    print("RCC=")
    printDMatrix(RCC, n, n)
    print("T=")
    printIMatrix(result.assignment, m, n)
    print(
        f"Total objective = {result.objective:.2f} "
        f"(base={result.base_score:.2f}, compat={result.compatibility_score:.2f}) "
        f"Time = {t2 - t1} ms, solver = {result.solver}"
    )


if __name__ == "__main__":
    _demo()
