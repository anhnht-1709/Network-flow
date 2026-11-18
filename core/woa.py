"""
core/woa.py
Cài đặt Thuật toán Tối ưu hóa Bầy Cá Voi (Whale Optimization Algorithm - WOA) giải bài toán Network Flow.
Phụ trách: Thành viên 5 (TV5) - Minh Nhân
Deadline mốc giữa: 29/11 | Bản cuối: 05/12

Quy tắc bắt buộc:
  1. Kế thừa SolverBase từ core.solver_base.
  2. Làm việc trên không gian nghiệm vector x kích thước self.dim trong khoảng [self.lower, self.upper].
  3. Kẹp nghiệm hợp lệ: candidate_x = self.encoding.clip(candidate_x) sau mỗi bước di chuyển.
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


class WOASolver(SolverBase):
    """
    Thuật toán Whale Optimization Algorithm (WOA) cho bài toán Network Flow (mã hóa theo đường đi).
    """

    def __init__(
        self,
        graph: Graph,
        mode: str = "max_flow",
        params: dict[str, Any] | None = None,
        seed: int = 42
    ):
        super().__init__(graph, mode=mode, params=params, seed=seed)
        # Các tham số mặc định của WOA
        self.num_whales = self.params.get("num_whales", 30)
        self.b = self.params.get("b", 1.0)  # Hằng số định hình đường xoắn ốc logarithmic

    def initialize_whales(self) -> np.ndarray:
        """
        Khởi tạo vị trí ban đầu của đàn cá voi trong khoảng [self.lower, self.upper].

        Returns:
        --------
        np.ndarray:
            Ma trận vị trí bầy cá voi kích thước (num_whales, self.dim).
        """
        return self.rng.uniform(self.lower, self.upper, size=(self.num_whales, self.dim))

    def run(self, max_iter: int, max_evals: int | None = None) -> Result:
        """
        Chạy thuật toán WOA qua max_iter vòng lặp hoặc dừng sớm khi hết ngân sách max_evals.
        """
        start_time = time.perf_counter()
        self.eval_count = 0
        self.history = []

        dim = self.dim
        num_whales = self.num_whales

        # 1. Khởi tạo vị trí đàn cá voi
        whales = self.initialize_whales()

        best_x: np.ndarray | None = None
        best_eval = None
        best_flows: np.ndarray | None = None

        # 2. Đánh giá bầy ban đầu để tìm con mồi tốt nhất (Leader / Best Whale)
        for i in range(num_whales):
            if self.budget_exhausted(max_evals):
                break
            ev, flows = self.evaluate_candidate(whales[i])
            if best_eval is None or ev.fitness > best_eval.fitness:
                best_eval = ev
                best_flows = flows
                best_x = whales[i].copy()

        # Trường hợp hết ngân sách ngay vòng đầu hoặc không tìm thấy ứng viên
        if best_eval is None:
            best_x = np.zeros(dim, dtype=np.float64)
            best_eval, best_flows = self.evaluate_candidate(best_x)

        self.history.append(best_eval.fitness)

        # 3. Vòng lặp tối ưu hóa chính
        for t in range(max_iter):
            if self.budget_exhausted(max_evals):
                break

            # Hệ số a giảm tuyến tính từ 2 về 0 qua các thế hệ
            a = 2.0 - 2.0 * (t / max(1, max_iter))

            for i in range(num_whales):
                if self.budget_exhausted(max_evals):
                    break

                # Sinh các hệ số ngẫu nhiên cho cá voi i
                r1 = self.rng.random()
                r2 = self.rng.random()
                A = 2.0 * a * r1 - a
                C = 2.0 * r2
                p = self.rng.random()

                if p < 0.5:
                    if abs(A) < 1.0:
                        # Nhánh 1: Bao vây con mồi quanh best_whale (Exploitation)
                        D = np.abs(C * best_x - whales[i])
                        new_whale = best_x - A * D
                    else:
                        # Nhánh 2: Tìm kiếm ngẫu nhiên quanh rand_whale (Exploration)
                        rand_idx = self.rng.integers(0, num_whales)
                        rand_whale = whales[rand_idx]
                        D = np.abs(C * rand_whale - whales[i])
                        new_whale = rand_whale - A * D
                else:
                    # Nhánh 3: Bơi xoắn ốc bọt khí quanh best_whale (Bubble-net attacking)
                    l = self.rng.uniform(-1.0, 1.0)
                    D_prime = np.abs(best_x - whales[i])
                    new_whale = best_x + D_prime * np.exp(self.b * l) * np.cos(2.0 * np.pi * l)

                # Kẹp biên vị trí mới về [self.lower, self.upper]
                new_whale = self.encoding.clip(new_whale)

                # Đánh giá cá thể qua evaluate_candidate (decode -> flows -> evaluate)
                ev, flows = self.evaluate_candidate(new_whale)
                whales[i] = new_whale

                # Cập nhật con mồi tốt nhất nếu fitness cải thiện (Tối đa hóa)
                if ev.fitness > best_eval.fitness:
                    best_eval = ev
                    best_flows = flows
                    best_x = new_whale.copy()

            # Lưu fitness tốt nhất ở cuối vòng lặp
            self.history.append(best_eval.fitness)

        elapsed_s = time.perf_counter() - start_time

        return Result(
            best_flows=best_flows,
            best_eval=best_eval,
            history=self.history,
            evaluations=self.eval_count,
            elapsed_s=elapsed_s,
            mode=self.mode,
            params={
                "num_whales": self.num_whales,
                "b": self.b,
                **self.params,
            },
            algorithm_name="WOA",
            best_x=best_x
        )
