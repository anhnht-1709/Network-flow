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
from collections import deque
from dataclasses import dataclass
from typing import Any
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
    penalty: float | None = None,
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
    penalty: float | None
        Hệ số phạt cho mỗi đơn vị vi phạm ràng buộc. Nếu để None, tự động tính theo tổng chi phí mạng.
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

    if penalty is None:
        max_network_cost = float(np.sum(graph.capacities * graph.costs))
        penalty = max(1000.0, max_network_cost * 10.0)

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


def is_better(candidate: Evaluation, current_best: Evaluation | None) -> bool:
    """
    So sánh hai kết quả đánh giá: ưu tiên nghiệm hợp lệ (feasible=True) trước,
    nếu cùng trạng thái hợp lệ thì chọn nghiệm có fitness lớn hơn.
    """
    if current_best is None:
        return True
    if candidate.feasible and not current_best.feasible:
        return True
    if not candidate.feasible and current_best.feasible:
        return False
    return candidate.fitness > current_best.fitness


def cut_proof(
    graph: Graph,
    required_flow: float | None = None,
    tol: float = 1e-4
) -> dict[str, Any] | None:
    """
    Tìm lát cắt s-t có sức chứa nhỏ nhất để kiểm chứng tính vô nghiệm khi
    lượng luồng yêu cầu (required_flow) vượt quá sức chứa tối đa của mạng.

    Returns:
    --------
    dict | None:
        Trả về thông tin lát cắt chứng minh vô nghiệm nếu required_flow > cut_capacity + tol,
        ngược lại trả về None.
    """
    req = graph.required_flow if required_flow is None else required_flow
    if req is None or req <= 0.0:
        return None

    n = graph.num_nodes
    s = graph.source_idx
    t = graph.sink_idx
    node_idx = {name: idx for idx, name in enumerate(graph.nodes)}

    # Xây dựng mạng dư (residual capacity matrix) để tìm lát cắt cực tiểu bằng BFS
    residual = np.zeros((n, n), dtype=np.float64)
    for edge in graph.edges:
        if edge.capacity > 0:
            u = node_idx[edge.u]
            v = node_idx[edge.v]
            residual[u, v] += edge.capacity

    while True:
        parent = [-1] * n
        parent[s] = s
        queue: deque[int] = deque([s])
        while queue and parent[t] == -1:
            u = queue.popleft()
            for v in range(n):
                if parent[v] == -1 and residual[u, v] > 1e-12:
                    parent[v] = u
                    queue.append(v)

        if parent[t] == -1:
            break

        bottleneck = float("inf")
        v = t
        while v != s:
            u = parent[v]
            bottleneck = min(bottleneck, float(residual[u, v]))
            v = u

        v = t
        while v != s:
            u = parent[v]
            residual[u, v] -= bottleneck
            residual[v, u] += bottleneck
            v = u

    # Tập đỉnh S đến được từ s trên mạng dư cuối cùng xác định lát cắt cực tiểu (S, V \ S)
    visited = {s}
    queue = deque([s])
    while queue:
        u = queue.popleft()
        for v in range(n):
            if v not in visited and residual[u, v] > 1e-12:
                visited.add(v)
                queue.append(v)

    source_side = {graph.nodes[i] for i in visited}
    cut_edges = [
        f"{e.u}->{e.v}"
        for e in graph.edges
        if e.u in source_side and e.v not in source_side
    ]
    cut_capacity = float(sum(
        e.capacity
        for e in graph.edges
        if e.u in source_side and e.v not in source_side
    ))

    if req > cut_capacity + tol:
        return {
            "required_flow": float(req),
            "cut_capacity": cut_capacity,
            "shortfall": float(req - cut_capacity),
            "source_side": [node for node in graph.nodes if node in source_side],
            "cut_edges": cut_edges
        }
    return None
