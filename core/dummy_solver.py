"""
core/dummy_solver.py
Thuật toán tìm kiếm ngẫu nhiên (Random Search) đóng vai trò Solver giả định.
Áp dụng mã hóa nghiệm theo đường đi (Path-based encoding).
Dùng để kiểm thử khung giao diện, pipeline dữ liệu và làm mẫu chuẩn tham khảo cho GA & WOA.
Phụ trách: Thành viên 1 (Nhóm trưởng)
"""

from __future__ import annotations
import time
from typing import Any
import numpy as np

from core.graph import Graph
from core.solver_base import SolverBase, Result


class RandomSolver(SolverBase):
    """
    Solver tìm kiếm ngẫu nhiên kế thừa SolverBase (mã hóa theo đường đi).
    Ở mỗi vòng lặp, thuật toán thử nghiệm một số ứng viên x ngẫu nhiên trong khoảng [lower, upper]
    và tìm kiếm cục bộ xung quanh nghiệm tốt nhất hiện có.
    """

    def __init__(
        self,
        graph: Graph,
        mode: str = "max_flow",
        params: dict[str, Any] | None = None,
        seed: int = 42
    ):
        super().__init__(graph, mode=mode, params=params, seed=seed)
        # Tham số mẫu: số lượng ứng viên thử trong mỗi vòng lặp
        self.samples_per_iter = self.params.get("samples_per_iter", 30)

    def run(self, max_iter: int, max_evals: int | None = None) -> Result:
        """
        Chạy tìm kiếm ngẫu nhiên với số thế hệ tối đa max_iter hoặc ngân sách max_evals.
        """
        start_time = time.perf_counter()
        self.eval_count = 0
        self.history = []
        self.history_flows = []

        # =========================================================================
        # CÁC DÒNG GA VÀ WOA SẼ CẦN DÙNG TƯƠNG TỰ:
        # 1. Số chiều không gian nghiệm = số đường đi: self.dim
        # 2. Biên dưới nghiệm: self.lower (thường là 0)
        # 3. Biên trên nghiệm: self.upper (nút cổ chai của từng đường đi)
        # =========================================================================
        dim = self.dim
        lower = self.lower
        upper = self.upper

        # 1. Khởi tạo nghiệm ban đầu: vector 0 (chưa bơm luồng)
        best_x = np.zeros(dim, dtype=np.float64)
        best_eval, best_flows = self.evaluate_candidate(best_x)
        self.history.append(best_eval.fitness)
        self.history_flows.append(best_flows.copy())

        # 2. Vòng lặp tìm kiếm
        for _ in range(max_iter):
            # Kiểm tra ngân sách đánh giá
            if self.budget_exhausted(max_evals):
                break

            for _ in range(self.samples_per_iter):
                if self.budget_exhausted(max_evals):
                    break

                # Chiến lược kết hợp: 50% sinh ngẫu nhiên toàn cục, 50% nhiễu quanh nghiệm tốt nhất
                if self.rng.random() < 0.5 or best_eval.fitness == -float("inf"):
                    # GA/WOA: Khởi tạo quần thể hoặc bay tự do trong [lower, upper]
                    candidate_x = self.rng.uniform(lower, upper, size=dim)
                else:
                    # GA/WOA: Đột biến hoặc bao vây con mồi quanh nghiệm tốt nhất
                    scale = np.maximum(1e-4, 0.1 * (upper - lower))
                    noise = self.rng.normal(0.0, scale, size=dim)
                    # Cắt x về [lower, upper] bằng self.encoding.clip(x)
                    candidate_x = self.encoding.clip(best_x + noise)

                # =====================================================================
                # ĐÁNH GIÁ ỨNG VIÊN: GA và WOA bắt buộc chấm điểm qua evaluate_candidate
                # Hàm trả về (Evaluation, flows trên các cạnh) và tự động đếm 1 eval
                # =====================================================================
                ev, flows = self.evaluate_candidate(candidate_x)

                # Ưu tiên nghiệm hợp lệ (feasible) trước, sau đó tối đa hóa fitness
                if self.is_better(ev, best_eval):
                    best_eval = ev
                    best_flows = flows
                    best_x = candidate_x.copy()

            # Lưu lại fitness và luồng tốt nhất ở cuối vòng lặp
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
            params=self.params,
            algorithm_name="RandomSearch (Dummy)",
            best_x=best_x,
            history_flows=self.history_flows
        )
