"""
core/ga.py
Cài đặt Thuật toán Di truyền (Genetic Algorithm - GA) giải bài toán Network Flow.
Phụ trách: Thành viên 2 (TV2)
Deadline mốc giữa: 29/11 | Bản cuối: 05/12

Quy tắc bắt buộc:
  1. Kế thừa SolverBase từ core.solver_base.
  2. Làm việc trên không gian nghiệm vector x kích thước self.dim trong khoảng [self.lower, self.upper].
  3. Kẹp nghiệm hợp lệ: candidate_x = self.encoding.clip(candidate_x) sau khi lai ghép / đột biến.
  4. Đánh giá cá thể BẮT BUỘC gọi duy nhất: ev, flows = self.evaluate_candidate(x).
  5. Luôn TỐI ĐA HÓA fitness: cập nhật khi ev.fitness > best_eval.fitness.
  6. Dừng theo ngân sách: if self.budget_exhausted(max_evals): break.
"""

from __future__ import annotations
import time
from typing import Any
import numpy as np

from core.graph import Graph
from core.solver_base import SolverBase, Result


class GASolver(SolverBase):
    """
    Thuật toán Di truyền (GA) cho bài toán Network Flow (mã hóa theo đường đi).
    """

    def __init__(
        self,
        graph: Graph,
        mode: str = "max_flow",
        params: dict[str, Any] | None = None,
        seed: int = 42
    ):
        super().__init__(graph, mode=mode, params=params, seed=seed)
        # Các tham số mặc định của GA (có thể ghi đè qua params)
        self.pop_size = self.params.get("pop_size", 50)
        self.crossover_rate = self.params.get("crossover_rate", 0.8)
        self.mutation_rate = self.params.get("mutation_rate", 0.1)
        self.tournament_size = self.params.get("tournament_size", 3)
        self.elitism_count = self.params.get("elitism_count", 2)

    def initialize_population(self) -> np.ndarray:
        """
        Khởi tạo quần thể ban đầu gồm pop_size cá thể trong khoảng [self.lower, self.upper].

        Returns:
        --------
        np.ndarray:
            Ma trận quần thể kích thước (pop_size, self.dim).
        """
        # TODO (TV2): Sinh ma trận ngẫu nhiên đều trong đoạn [self.lower, self.upper] bằng self.rng
        raise NotImplementedError("TODO: Phụ trách bởi Thành viên 2 (TV2) - initialize_population")

    def select_parent(self, population: np.ndarray, fitnesses: np.ndarray) -> np.ndarray:
        """
        Chọn lọc cha mẹ (Tournament selection hoặc Roulette Wheel).
        """
        # TODO (TV2): Triển khai tournament selection
        raise NotImplementedError("TODO: Phụ trách bởi Thành viên 2 (TV2) - select_parent")

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Lai ghép 2 cá thể cha mẹ tạo 2 cá thể con (Arithmetic, BLX-alpha hoặc Crossover đa điểm).
        """
        # TODO (TV2): Triển khai lai ghép và kẹp biên self.encoding.clip(child)
        raise NotImplementedError("TODO: Phụ trách bởi Thành viên 2 (TV2) - crossover")

    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """
        Đột biến cá thể (Gaussian mutation hoặc Uniform mutation).
        """
        # TODO (TV2): Triển khai đột biến và kẹp biên self.encoding.clip(mutated)
        raise NotImplementedError("TODO: Phụ trách bởi Thành viên 2 (TV2) - mutate")

    def run(self, max_iter: int, max_evals: int | None = None) -> Result:
        """
        Chạy thuật toán GA qua max_iter thế hệ hoặc dừng sớm khi hết ngân sách max_evals.
        """
        start_time = time.perf_counter()
        self.eval_count = 0
        self.history = []

        # TODO (TV2): Hoàn thiện vòng lặp tiến hóa chính theo khung sau:
        # 1. Khởi tạo quần thể
        # 2. Đánh giá từng cá thể bằng self.evaluate_candidate(ind)
        # 3. Lưu best_eval, best_flows, best_x, ghi nhận self.history.append(best_eval.fitness)
        # 4. Lặp qua các thế hệ gen in range(max_iter):
        #    - Kiểm tra self.budget_exhausted(max_evals) -> break
        #    - Tạo thế hệ mới qua chọn lọc, lai ghép, đột biến, giữ lại elite
        #    - Đánh giá thế hệ mới, cập nhật best nghiệm
        #    - self.history.append(best_eval.fitness)
        # 5. Trả về Result(...)
        raise NotImplementedError("TODO: Phụ trách bởi Thành viên 2 (TV2) - run")
