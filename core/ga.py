"""
core/ga.py
Cài đặt Thuật toán Di truyền (Genetic Algorithm - GA) giải bài toán Network Flow.
Phụ trách: Thành viên 2 (TV2)

Các toán tử sử dụng:
  - Tournament selection: chọn lọc cạnh tranh, mặc định kích thước giải đấu là 3.
  - Arithmetic crossover: lai ghép số học, mặc định xác suất 0.9.
  - Uniform mutation: đột biến đều trong miền [lower, upper], mặc định p = 1 / dim.
  - Elitism: giữ lại các cá thể tốt nhất qua thế hệ mới.
"""

from __future__ import annotations
import time
from functools import cmp_to_key
from typing import Any

import numpy as np

from core.graph import Graph
from core.fitness import Evaluation
from core.solver_base import SolverBase, Result


class GASolver(SolverBase):
    """Thuật toán Di truyền (GA) cho Network Flow với mã hóa theo đường đi."""

    def __init__(
        self,
        graph: Graph,
        mode: str = "max_flow",
        params: dict[str, Any] | None = None,
        seed: int = 42
    ):
        super().__init__(graph, mode=mode, params=params, seed=seed)
        self.pop_size = int(self.params.get("pop_size", 50))
        self.crossover_rate = float(self.params.get("crossover_rate", 0.9))
        self.tournament_size = int(self.params.get("tournament_size", 3))
        self.elitism_count = int(self.params.get("elitism_count", 2))

        # Theo đặc tả nhóm: xác suất đột biến mặc định bằng 1 / số chiều.
        default_mutation_rate = 1.0 / max(1, self.dim)
        self.mutation_rate = float(self.params.get("mutation_rate", default_mutation_rate))

        if self.pop_size < 1:
            raise ValueError("pop_size phải lớn hơn hoặc bằng 1.")
        if self.tournament_size < 1:
            raise ValueError("tournament_size phải lớn hơn hoặc bằng 1.")
        if self.elitism_count < 0:
            raise ValueError("elitism_count không được âm.")
        if not 0.0 <= self.crossover_rate <= 1.0:
            raise ValueError("crossover_rate phải nằm trong [0, 1].")
        if not 0.0 <= self.mutation_rate <= 1.0:
            raise ValueError("mutation_rate phải nằm trong [0, 1].")

    def initialize_population(self) -> np.ndarray:
        """Tạo quần thể ban đầu dùng chung seed với WOA."""
        return self.generate_initial_population(self.pop_size)

    def select_parent(self, population: np.ndarray, fitnesses: np.ndarray) -> np.ndarray:
        """Chọn một cha mẹ bằng tournament selection (mặc định 3 ứng viên)."""
        population = np.asarray(population, dtype=np.float64)
        fitnesses = np.asarray(fitnesses, dtype=np.float64)
        if population.ndim != 2 or len(population) == 0:
            raise ValueError("population phải là ma trận và không được rỗng.")
        if fitnesses.ndim != 1 or len(fitnesses) != len(population):
            raise ValueError("fitnesses phải có một giá trị cho mỗi cá thể.")

        count = min(self.tournament_size, len(population))
        contestants = self.rng.choice(len(population), size=count, replace=False)
        winner = int(contestants[np.argmax(fitnesses[contestants])])
        return population[winner].copy()

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Lai ghép số học: con = alpha * cha1 + (1-alpha) * cha2."""
        parent1 = np.asarray(parent1, dtype=np.float64)
        parent2 = np.asarray(parent2, dtype=np.float64)
        if parent1.shape != (self.dim,) or parent2.shape != (self.dim,):
            raise ValueError(f"Mỗi cá thể phải có kích thước ({self.dim},).")

        if self.rng.random() < self.crossover_rate:
            alpha = self.rng.random(self.dim)
            child1 = alpha * parent1 + (1.0 - alpha) * parent2
            child2 = alpha * parent2 + (1.0 - alpha) * parent1
        else:
            child1 = parent1.copy()
            child2 = parent2.copy()

        return self.encoding.clip(child1), self.encoding.clip(child2)

    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """Đột biến đều: mỗi gene được chọn với xác suất mutation_rate."""
        mutated = np.asarray(individual, dtype=np.float64).copy()
        if mutated.shape != (self.dim,):
            raise ValueError(f"Cá thể phải có kích thước ({self.dim},).")
        if self.dim == 0:
            return mutated

        mutation_mask = self.rng.random(self.dim) < self.mutation_rate
        number_mutated = int(np.count_nonzero(mutation_mask))
        if number_mutated:
            random_values = self.rng.uniform(
                self.lower[mutation_mask],
                self.upper[mutation_mask],
                size=number_mutated
            )
            mutated[mutation_mask] = random_values
        return self.encoding.clip(mutated)

    @staticmethod
    def _compare_evaluations(left: Evaluation, right: Evaluation) -> int:
        """Sắp xếp theo nguyên tắc: feasible trước, sau đó fitness giảm dần."""
        if left.feasible != right.feasible:
            return -1 if left.feasible else 1
        if left.fitness > right.fitness:
            return -1
        if left.fitness < right.fitness:
            return 1
        return 0

    def run(self, max_iter: int, max_evals: int | None = None) -> Result:
        """Chạy GA đến hết số thế hệ hoặc chạm ngân sách đánh giá."""
        if max_iter < 0:
            raise ValueError("max_iter không được âm.")
        if max_evals is not None and max_evals < 0:
            raise ValueError("max_evals không được âm.")

        start_time = time.perf_counter()
        self.eval_count = 0
        self.history = []
        self.history_flows = []
        details = {"selected": 0, "crossovers": 0, "mutations": 0, "mutated_genes": 0}

        best_x: np.ndarray | None = None
        best_eval: Evaluation | None = None
        best_flows: np.ndarray | None = None

        # 1. Khởi tạo và đánh giá quần thể ban đầu.
        population = self.initialize_population()
        evaluated_population: list[np.ndarray] = []
        evaluations: list[Evaluation] = []
        flows_population: list[np.ndarray] = []

        for individual in population:
            if self.budget_exhausted(max_evals):
                break
            ev, flows = self.evaluate_candidate(individual)
            evaluated_population.append(individual.copy())
            evaluations.append(ev)
            flows_population.append(flows.copy())
            if self.is_better(ev, best_eval):
                best_eval = ev
                best_flows = flows.copy()
                best_x = individual.copy()

        # max_evals=0: không gọi hàm đánh giá và vẫn trả về Result hợp lệ.
        if best_eval is None:
            best_eval = Evaluation(
                fitness=-float("inf"), total_flow=0.0, total_cost=0.0,
                violation=0.0, feasible=False
            )
            best_flows = np.zeros(self.graph.num_edges, dtype=np.float64)
            best_x = np.zeros(self.dim, dtype=np.float64)

        self.history.append(best_eval.fitness)
        self.history_flows.append(best_flows.copy())

        # 2. Tiến hóa qua từng thế hệ.
        for _generation in range(max_iter):
            if self.budget_exhausted(max_evals) or not evaluated_population:
                break

            # Xếp hạng để bảo toàn elite (cá thể ưu tú) ở đầu thế hệ mới.
            ranked_indices = sorted(
                range(len(evaluations)),
                key=cmp_to_key(lambda i, j: self._compare_evaluations(evaluations[i], evaluations[j]))
            )
            elite_count = min(self.elitism_count, self.pop_size, len(ranked_indices))
            next_population: list[np.ndarray] = [
                evaluated_population[index].copy() for index in ranked_indices[:elite_count]
            ]
            next_evaluations: list[Evaluation] = [evaluations[index] for index in ranked_indices[:elite_count]]
            next_flows: list[np.ndarray] = [flows_population[index].copy() for index in ranked_indices[:elite_count]]

            fitnesses = np.asarray([ev.fitness for ev in evaluations], dtype=np.float64)

            while len(next_population) < self.pop_size and not self.budget_exhausted(max_evals):
                parent1 = self.select_parent(np.asarray(evaluated_population), fitnesses)
                parent2 = self.select_parent(np.asarray(evaluated_population), fitnesses)
                details["selected"] += 2

                child1, child2 = self.crossover(parent1, parent2)
                if self.crossover_rate > 0.0:
                    # Bộ đếm số cặp được đưa qua toán tử lai ghép.
                    details["crossovers"] += 1

                for child in (child1, child2):
                    if len(next_population) >= self.pop_size or self.budget_exhausted(max_evals):
                        break
                    before = child.copy()
                    child = self.mutate(child)
                    if not np.array_equal(before, child):
                        details["mutations"] += 1
                        details["mutated_genes"] += int(np.count_nonzero(before != child))

                    # Đánh giá ứng viên duy nhất qua API chuẩn của SolverBase.
                    ev, flows = self.evaluate_candidate(child)
                    next_population.append(child.copy())
                    next_evaluations.append(ev)
                    next_flows.append(flows.copy())

                    if self.is_better(ev, best_eval):
                        best_eval = ev
                        best_flows = flows.copy()
                        best_x = child.copy()

            evaluated_population = next_population
            evaluations = next_evaluations
            flows_population = next_flows

            self.history.append(best_eval.fitness)
            self.history_flows.append(best_flows.copy())

        elapsed_s = time.perf_counter() - start_time
        return Result(
            best_flows=best_flows,
            best_eval=best_eval,
            history=self.history,
            evaluations=self.eval_count,
            elapsed_s=elapsed_s,
            mode=self.mode,
            params={
                **self.params,
                "pop_size": self.pop_size,
                "crossover_rate": self.crossover_rate,
                "mutation_rate": self.mutation_rate,
                "tournament_size": self.tournament_size,
                "elitism_count": self.elitism_count,
                "details": details,
            },
            algorithm_name="GA",
            best_x=best_x,
            history_flows=self.history_flows
        )
