"""
core/fitness.py
Hàm đánh giá (fitness function) và kiểm tra tính hợp lệ của nghiệm cho bài toán Network Flow.
Hỗ trợ 2 chế độ:
  - 'max_flow': Tối đa hóa tổng luồng tới đích (sink).
  - 'min_cost': Tối thiểu hóa tổng chi phí gửi lượng luồng yêu cầu (required_flow).
Quy ước chung cho Solver: Solver luôn TỐI ĐA HÓA giá trị fitness.
Phụ trách: Thành viên 1 (Nhóm trưởng)
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from core.graph import Graph


@dataclass
class Evaluation:
    """Kết quả đánh giá một nghiệm (vector luồng)."""
    fitness: float          # Giá trị thích nghi (Solver luôn tối đa hóa giá trị này)
    total_flow: float       # Tổng luồng thực sự đến đích (sink)
    total_cost: float       # Tổng chi phí luồng trên các cạnh: sum(flow[e] * cost[e])
    violation: float        # Tổng vi phạm bảo toàn luồng tại nút trung gian và vượt sức chứa
    feasible: bool          # Nghiệm có thỏa mãn toàn bộ ràng buộc hay không
    shortfall: float = 0.0  # Lượng luồng còn thiếu so với required_flow (dùng cho min_cost)
    excess: float = 0.0     # Lượng luồng vượt quá required_flow (dùng cho min_cost)

    def to_dict(self) -> dict:
        return {
            "fitness": float(self.fitness),
            "total_flow": float(self.total_flow),
            "total_cost": float(self.total_cost),
            "violation": float(self.violation),
            "feasible": bool(self.feasible),
            "shortfall": float(self.shortfall),
            "excess": float(self.excess)
        }


def evaluate(
    flows: np.ndarray,
    graph: Graph,
    mode: str = "max_flow",
    penalty: float = 1000.0,
    tol: float = 1e-4
) -> Evaluation:
    """
    Đánh giá vector nghiệm flows trên đồ thị graph theo chế độ mode.

    Parameters:
    -----------
    flows: np.ndarray
        Vector 1D độ dài |E|, flows[i] là lưu lượng chạy trên cạnh i.
    graph: Graph
        Đồ thị bài toán mạng luồng.
    mode: str
        'max_flow' hoặc 'min_cost'.
    penalty: float
        Hệ số phạt cho mỗi đơn vị vi phạm ràng buộc (mặc định 1000.0).
    tol: float
        Ngưỡng dung sai xác định tính khả thi (feasible).

    Returns:
    --------
    Evaluation:
        Chứa đầy đủ thông tin fitness, total_flow, total_cost, violation, feasible.
    """
    flows = np.asarray(flows, dtype=np.float64)
    if flows.shape[0] != graph.num_edges:
        raise ValueError(
            f"Kích thước vector flows ({flows.shape[0]}) không khớp với số cạnh của đồ thị ({graph.num_edges})."
        )

    # 1. Kiểm tra vi phạm sức chứa cạnh: 0 <= flows[e] <= capacity[e]
    under_flow = np.maximum(0.0, -flows)
    over_flow = np.maximum(0.0, flows - graph.capacities)
    cap_violation = float(np.sum(under_flow) + np.sum(over_flow))

    # 2. Kiểm tra vi phạm bảo toàn luồng tại các nút trung gian V \ {s, t}
    # net_flow = B @ flows, trong đó net_flow[v] = in_flow(v) - out_flow(v)
    net_flow = graph.incidence_matrix @ flows
    s_idx = graph.source_idx
    t_idx = graph.sink_idx

    # Mặt nạ cho các nút trung gian
    intermediate_mask = np.ones(graph.num_nodes, dtype=bool)
    intermediate_mask[s_idx] = False
    intermediate_mask[t_idx] = False

    conservation_violation = float(np.sum(np.abs(net_flow[intermediate_mask])))
    total_violation = conservation_violation + cap_violation

    # 3. Tính tổng luồng thực tế đi vào nút đích (sink)
    # net_flow[t_idx] = in_flow(sink) - out_flow(sink)
    total_flow = float(max(0.0, net_flow[t_idx]))

    # 4. Tính tổng chi phí luồng trên các cạnh: sum(flow[e] * cost[e])
    # Chỉ tính chi phí trên phần luồng không âm (cắt bỏ phần âm nếu có lỗi giải thuật)
    valid_flows = np.clip(flows, 0.0, graph.capacities)
    total_cost = float(np.sum(valid_flows * graph.costs))

    shortfall = 0.0
    excess = 0.0

    # 5. Tính fitness theo từng chế độ (Quy ước: luôn TỐI ĐA HÓA fitness)
    if mode == "max_flow":
        # Tối đa hóa tổng luồng tới đích, phạt khi vi phạm bảo toàn luồng / sức chứa
        fitness = total_flow - penalty * total_violation
        feasible = (total_violation <= tol) and (total_flow > tol)

    elif mode == "min_cost":
        if graph.required_flow is None:
            raise ValueError("Chế độ 'min_cost' yêu cầu đồ thị phải có thuộc tính 'required_flow'.")

        req = graph.required_flow
        shortfall = float(max(0.0, req - total_flow))
        excess = float(max(0.0, total_flow - req))

        # Tối thiểu hóa tổng chi phí: fitness = -total_cost, trừ phạt nếu vi phạm
        # Phạt cả vi phạm bảo toàn luồng lẫn việc không đáp ứng đúng lượng luồng yêu cầu
        flow_deviation = shortfall + excess
        fitness = -(total_cost + penalty * (total_violation + flow_deviation))
        feasible = (total_violation <= tol) and (flow_deviation <= tol)

    else:
        raise ValueError(f"Chế độ '{mode}' không hợp lệ. Chỉ chấp nhận 'max_flow' hoặc 'min_cost'.")

    return Evaluation(
        fitness=float(fitness),
        total_flow=total_flow,
        total_cost=total_cost,
        violation=total_violation,
        feasible=feasible,
        shortfall=shortfall,
        excess=excess
    )


def repair_flows(flows: np.ndarray, graph: Graph) -> np.ndarray:
    """
    Hàm bổ trợ: Cắt vector luồng về phạm vi hợp lệ [0, capacity] của từng cạnh.
    Có thể được solver (GA, WOA) sử dụng sau các phép lai ghép, đột biến hoặc di chuyển vị trí.
    """
    return np.clip(flows, 0.0, graph.capacities)


def is_feasible(
    flows: np.ndarray,
    graph: Graph,
    mode: str = "max_flow",
    tol: float = 1e-4
) -> bool:
    """Kiểm tra nhanh xem vector nghiệm flows có hợp lệ hay không."""
    ev = evaluate(flows, graph, mode=mode, tol=tol)
    return ev.feasible
