"""
experiments/benchmark.py
Script chạy thực nghiệm, thống kê số liệu và xuất bảng biểu so sánh GA vs WOA.
Phụ trách: Thành viên 3 (TV3)
Deadline kịch bản: 29/11 | Hoàn thiện thực nghiệm: 13/12

Quy tắc thực nghiệm:
  1. Mỗi kịch bản chạy cả hai thuật toán (GA và WOA) ở cả hai chế độ (max_flow và min_cost).
  2. Mỗi cấu hình chạy ít nhất 20 lần độc lập (20 seeds khác nhau).
  3. Cùng ngân sách đánh giá (max_evals) và cùng hạt giống để so sánh công bằng.
  4. Thu thập các chỉ số: Best, Mean, Std, Tỷ lệ Feasible, Thời gian chạy, Tốc độ hội tụ.
  5. Xuất kết quả dạng bảng CSV / Markdown và vẽ biểu đồ đường hội tụ (Convergence curves).
"""

from __future__ import annotations
from pathlib import Path
import numpy as np

from core.graph import Graph
from core.solver_base import Result


def run_benchmark(
    scenarios_dir: str = "data/scenarios",
    num_runs: int = 20,
    max_iter: int = 100,
    max_evals: int | None = 3000
) -> dict:
    """
    Chạy thực nghiệm hàng loạt trên các kịch bản trong data/scenarios.
    """
    # TODO (TV3): Hoàn thiện hàm chạy thực nghiệm tự động
    raise NotImplementedError("TODO: Phụ trách bởi Thành viên 3 (TV3) - run_benchmark")


def plot_convergence(ga_history: list[float], woa_history: list[float], title: str, save_path: str):
    """
    Vẽ biểu đồ so sánh đường cong hội tụ của GA và WOA bằng Matplotlib.
    """
    # TODO (TV3): Hoàn thiện vẽ biểu đồ hội tụ
    raise NotImplementedError("TODO: Phụ trách bởi Thành viên 3 (TV3) - plot_convergence")


if __name__ == "__main__":
    print("Script benchmark đang trong giai đoạn triển khai bởi TV3.")
