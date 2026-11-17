"""
core/woa.py
Cài đặt Thuật toán Tối ưu hóa Bầy Cá Voi (Whale Optimization Algorithm - WOA) giải bài toán Network Flow.
Phụ trách: Thành viên 5 (TV5)
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
        # TODO (TV5): Khởi tạo ngẫu nhiên vị trí các cá voi bằng self.rng
        raise NotImplementedError("TODO: Phụ trách bởi Thành viên 5 (TV5) - initialize_whales")

    def run(self, max_iter: int, max_evals: int | None = None) -> Result:
        """
        Chạy thuật toán WOA qua max_iter vòng lặp hoặc dừng sớm khi hết ngân sách max_evals.
        """
        start_time = time.perf_counter()
        self.eval_count = 0
        self.history = []

        # TODO (TV5): Hoàn thiện vòng lặp chính của WOA theo khung sau:
        # 1. Khởi tạo vị trí đàn cá voi
        # 2. Đánh giá đàn cá voi bằng self.evaluate_candidate(whale_i)
        # 3. Xác định con mồi tốt nhất (best_whale / best_x)
        # 4. Lặp qua từng vòng lặp t in range(max_iter):
        #    - Kiểm tra self.budget_exhausted(max_evals) -> break
        #    - Giảm dần hệ số a từ 2 về 0: a = 2.0 - 2.0 * (t / max_iter)
        #    - Với mỗi cá voi i:
        #      + Tính vector A = 2 * a * r - a, C = 2 * r (r là số ngẫu nhiên trong [0, 1])
        #      + Sinh số ngẫu nhiên p trong [0, 1]
        #      + Nếu p < 0.5:
        #        * Nếu |A| < 1: Bao vây con mồi (Shrinking encircling) quanh best_whale
        #        * Nếu |A| >= 1: Tìm kiếm ngẫu nhiên (Exploration) quanh cá voi ngẫu nhiên (rand_whale)
        #      + Nếu p >= 0.5:
        #        * Bơi theo đường xoắn ốc bong bóng (Spiral bubble-net attacking)
        #      + Kẹp biên vị trí mới: whale_i = self.encoding.clip(whale_i)
        #      + Đánh giá cá thể: ev, flows = self.evaluate_candidate(whale_i)
        #      + Cập nhật con mồi tốt nhất nếu ev.fitness > best_eval.fitness
        #    - self.history.append(best_eval.fitness)
        # 5. Trả về Result(...)
        raise NotImplementedError("TODO: Phụ trách bởi Thành viên 5 (TV5) - run")
