"""
core/solver_base.py
Định nghĩa giao diện chung (abstract base class) và cấu trúc kết quả cho các Solver (GA, WOA, v.v.).
Phụ trách: Thành viên 1 (Nhóm trưởng)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import time
from typing import Any
import numpy as np

from core.graph import Graph
from core.fitness import Evaluation, evaluate, is_better
from core.paths import PathEncoding


@dataclass
class Result:
    """Kết quả trả về sau khi chạy một Solver."""
    best_flows: np.ndarray          # Nghiệm tốt nhất tìm được (vector luồng trên các cạnh)
    best_eval: Evaluation           # Đánh giá chi tiết của nghiệm tốt nhất
    history: list[float]            # Fitness tốt nhất sau mỗi vòng lặp (dùng vẽ đường hội tụ)
    evaluations: int                # Tổng số lần gọi hàm evaluate (dùng so sánh ngân sách)
    elapsed_s: float                # Thời gian chạy (giây)
    mode: str = "max_flow"          # 'max_flow' hoặc 'min_cost'
    params: dict[str, Any] = field(default_factory=dict)
    algorithm_name: str = "Solver"
    best_x: np.ndarray | None = None
    history_flows: list[np.ndarray] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Chuyển kết quả thành dictionary để tiện lưu trữ JSON hoặc xuất biểu đồ."""
        return {
            "algorithm_name": self.algorithm_name,
            "mode": self.mode,
            "evaluations": self.evaluations,
            "elapsed_s": round(self.elapsed_s, 4),
            "best_fitness": round(self.best_eval.fitness, 4),
            "best_flow": round(self.best_eval.total_flow, 4),
            "best_cost": round(self.best_eval.total_cost, 4),
            "violation": round(self.best_eval.violation, 6),
            "feasible": self.best_eval.feasible,
            "best_flows": self.best_flows.tolist(),
            "best_x": self.best_x.tolist() if self.best_x is not None else None,
            "history_length": len(self.history),
            "history_flows_length": len(self.history_flows),
            "params": self.params
        }


class SolverBase(ABC):
    """
    Lớp cha trừu tượng cho mọi thuật toán tối ưu (GA, WOA, Dummy/Random).
    Đảm bảo tính nhất quán:
      - Cùng giao diện run(max_iter) -> Result
      - Cùng cách đếm số lần gọi hàm evaluate
      - Cùng hạt giống ngẫu nhiên (seed)
    """

    def __init__(
        self,
        graph: Graph,
        mode: str = "max_flow",
        params: dict[str, Any] | None = None,
        seed: int = 42
    ):
        if mode not in ("max_flow", "min_cost"):
            raise ValueError(f"Chế độ '{mode}' không hợp lệ. Chỉ chấp nhận 'max_flow' hoặc 'min_cost'.")

        self.graph = graph
        self.mode = mode
        self.params = params.copy() if params else {}
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.eval_count = 0
        self.history: list[float] = []
        self.history_flows: list[np.ndarray] = []

        # Tích hợp mã hóa nghiệm theo đường đi (Path-based encoding)
        self.encoding = PathEncoding(graph, max_paths=self.params.get("max_paths", 200))
        self.dim = self.encoding.num_paths
        self.lower = self.encoding.lower
        self.upper = self.encoding.upper

    def generate_initial_population(self, pop_size: int) -> np.ndarray:
        """
        Khởi tạo quần thể ban đầu kích thước (pop_size, self.dim) trong miền [self.lower, self.upper]
        từ hạt giống self.seed để các thuật toán xuất phát từ cùng một tập ứng viên.
        """
        init_rng = np.random.default_rng(self.seed)
        return init_rng.uniform(self.lower, self.upper, size=(pop_size, self.dim))

    def is_better(self, candidate: Evaluation, current_best: Evaluation | None) -> bool:
        """Kiểm tra ứng viên mới có tốt hơn nghiệm tốt nhất hiện tại hay không (ưu tiên feasible)."""
        return is_better(candidate, current_best)

    def _evaluate(self, flows: np.ndarray) -> Evaluation:
        """
        MỌI solver BẮT BUỘC phải gọi hàm đánh giá qua phương thức này.
        Phương thức tự động tăng bộ đếm eval_count để đảm bảo so sánh công bằng.
        """
        self.eval_count += 1
        penalty = self.params.get("penalty", None)
        return evaluate(flows, self.graph, mode=self.mode, penalty=penalty)

    def evaluate_candidate(self, x: np.ndarray | list[float]) -> tuple[Evaluation, np.ndarray]:
        """
        Decode x -> flows (đúng chế độ self.mode), chấm điểm qua self._evaluate, trả (ev, flows).
        Chỉ được gọi self._evaluate đúng MỘT lần mỗi ứng viên.
        """
        if self.mode == "min_cost":
            flows = self.encoding.decode(
                x,
                required_flow=self.graph.required_flow,
                topup=self.params.get("topup", False)
            )
        else:
            flows = self.encoding.decode(x, required_flow=None)
        ev = self._evaluate(flows)
        return ev, flows

    def budget_exhausted(self, max_evals: int | None = None) -> bool:
        """Kiểm tra xem đã dùng hết ngân sách số lần gọi evaluate hay chưa."""
        return max_evals is not None and self.eval_count >= max_evals

    @abstractmethod
    def run(self, max_iter: int, max_evals: int | None = None) -> Result:
        """
        Chạy thuật toán với số vòng lặp tối đa max_iter hoặc ngân sách đánh giá max_evals.

        Parameters:
        -----------
        max_iter: int
            Số thế hệ / vòng lặp tối đa.
        max_evals: int | None
            Ngân sách số lần gọi evaluate tối đa. Dừng sớm nếu vượt quá.

        Returns:
        --------
        Result:
            Chứa best_flows, best_eval, history, evaluations, elapsed_s, best_x.
        """
        pass
