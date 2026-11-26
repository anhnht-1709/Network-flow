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
  5. Luôn TỐI ĐA HÓA fitness & ưu tiên khả thi: self.is_better(ev, best_eval).
  6. Dừng theo ngân sách: if self.budget_exhausted(max_evals): break.
  7. Lưu history_flows song song với history để phục vụ thanh tua Replay trên giao diện GUI.
"""

from __future__ import annotations
import time
from typing import Any
import numpy as np

from core.graph import Graph
from core.fitness import Evaluation
from core.solver_base import SolverBase, Result, DEFAULT_POP_SIZE


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
        # Hỗ trợ cả 2 tên tham số: num_whales (đặc thù WOA) và pop_size (dùng chung cho form GUI & GA)
        self.num_whales = int(self.params.get("num_whales", self.params.get("pop_size", DEFAULT_POP_SIZE)))
        if self.num_whales < 1:
            raise ValueError("num_whales/pop_size phải lớn hơn hoặc bằng 1.")
        self.b = self.params.get("b", 1.0)  # Hằng số định hình đường xoắn ốc logarithmic

    def initialize_whales(self) -> np.ndarray:
        """
        Khởi tạo vị trí ban đầu của đàn cá voi trong khoảng [self.lower, self.upper].
        Đổi sang gọi hàm chung generate_initial_population của SolverBase để GA và WOA xuất phát cùng 1 quần thể khi cùng seed.

        Returns:
        --------
        np.ndarray:
            Ma trận vị trí bầy cá voi kích thước (num_whales, self.dim).
        """
        return self.generate_initial_population(self.num_whales)

    def run(self, max_iter: int, max_evals: int | None = None) -> Result:
        """
        Chạy thuật toán WOA qua max_iter vòng lặp hoặc dừng sớm khi hết ngân sách max_evals.
        """
        start_time = time.perf_counter()
        self.reset_tracking()

        # Khởi tạo bộ đếm số lần dùng 3 nhánh và kẹp biên
        details = {"encircle": 0, "explore": 0, "spiral": 0, "clipped": 0}

        dim = self.dim
        num_whales = self.num_whales

        # 1. Khởi tạo vị trí đàn cá voi từ quần thể ban đầu chung
        whales = self.initialize_whales()

        best_x: np.ndarray | None = None
        best_eval = None
        best_flows: np.ndarray | None = None

        # 2. Đánh giá bầy ban đầu để tìm con mồi tốt nhất (Leader / Best Whale)
        for i in range(num_whales):
            if self.budget_exhausted(max_evals):
                break
            ev, flows = self.evaluate_candidate(whales[i])
            if self.is_better(ev, best_eval):
                best_eval = ev
                best_flows = flows
                best_x = whales[i].copy()

        # Trường hợp hết ngân sách ngay từ đầu (ví dụ max_evals <= 0):
        # KHÔNG gọi evaluate_candidate để tránh vi phạm trần ngân sách 0 lần.
        if best_eval is None:
            best_x = np.zeros(dim, dtype=np.float64)
            best_flows = np.zeros(self.graph.num_edges, dtype=np.float64)
            best_eval = Evaluation(
                fitness=-float("inf"),
                total_flow=0.0,
                total_cost=0.0,
                violation=0.0,
                feasible=False
            )

        self.record_progress(best_eval, best_flows)

        # 3. Vòng lặp tối ưu hóa chính
        for t in range(max_iter):
            if self.budget_exhausted(max_evals):
                break

            # Lịch giảm hệ số a về đúng 0.0 ở vòng cuối: t = max_iter - 1
            a = 2.0 * (1.0 - t / max(1, max_iter - 1))

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
                        # Nhánh 1: Bao vây con mồi quanh best_whale
                        details["encircle"] += 1
                        D = np.abs(C * best_x - whales[i])
                        new_whale = best_x - A * D
                    else:
                        # Nhánh 2: Tìm kiếm ngẫu nhiên quanh rand_whale
                        details["explore"] += 1
                        rand_idx = self.rng.integers(0, num_whales)
                        rand_whale = whales[rand_idx]
                        D = np.abs(C * rand_whale - whales[i])
                        new_whale = rand_whale - A * D
                else:
                    # Nhánh 3: Bơi xoắn ốc bọt khí quanh best_whale
                    details["spiral"] += 1
                    l = self.rng.uniform(-1.0, 1.0)
                    D_prime = np.abs(best_x - whales[i])
                    new_whale = best_x + D_prime * np.exp(self.b * l) * np.cos(2.0 * np.pi * l)

                # Trước khi clip, nếu có phần tử vượt biên thì tăng details["clipped"] += 1
                if np.any((new_whale < self.lower) | (new_whale > self.upper)):
                    details["clipped"] += 1

                # Kẹp biên vị trí mới về [self.lower, self.upper]
                new_whale = self.encoding.clip(new_whale)

                # Đánh giá cá thể qua evaluate_candidate (decode -> flows -> evaluate)
                ev, flows = self.evaluate_candidate(new_whale)
                whales[i] = new_whale

                # Cập nhật con mồi tốt nhất: ưu tiên feasible và fitness cao hơn
                if self.is_better(ev, best_eval):
                    best_eval = ev
                    best_flows = flows
                    best_x = new_whale.copy()

            # Cuối mỗi vòng lặp t: lưu fitness, luồng tốt nhất và số lượt đánh giá đã dùng
            self.record_progress(best_eval, best_flows)

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
                "num_whales": self.num_whales,
                "pop_size": self.num_whales,
                "b": self.b,
                "details": details,
            },
            algorithm_name="WOA",
            best_x=best_x,
            history_flows=self.history_flows,
            history_evals=self.history_evals
        )
