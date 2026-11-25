"""
core/paths.py
Mã hóa nghiệm theo đường đi (Path-based encoding) cho bài toán Network Flow.
Bao gồm:
  - enumerate_paths: Tìm kiếm các đường đi đơn từ source đến sink có giới hạn số lượng.
  - PathEncoding: Quản lý không gian nghiệm đường đi, cắt biên [lower, upper] và decode sang vector luồng cạnh.
"""

from __future__ import annotations
from collections import deque
import warnings
import numpy as np

from core.graph import Graph


def enumerate_paths(graph: Graph, max_paths: int = 200) -> list[list[int]]:
    """
    Tìm danh sách các đường đi đơn (không lặp nút) từ source đến sink.
    Mỗi đường đi là danh sách các chỉ số cạnh (edge indices theo graph.edges).
    Bỏ qua các cạnh có capacity == 0.

    Thuật toán:
      1. BFS ngược từ sink để tính khoảng cách (số cạnh) ngắn nhất tới sink, cắt nhánh nút cụt.
      2. Lặp theo độ dài L tăng dần (từ ngắn nhất đến num_nodes - 1), dùng DFS giới hạn độ sâu
         để thu thập các đường đi có đúng độ dài L theo thứ tự chỉ số cạnh tăng dần.
      3. Dừng ngay khi đạt max_paths.

    Parameters:
    -----------
    graph: Graph
        Đồ thị bài toán mạng luồng.
    max_paths: int
        Số lượng đường đi tối đa cần thu thập (mặc định 200).

    Returns:
    --------
    list[list[int]]:
        Danh sách các đường đi, mỗi đường là list các chỉ số cạnh.
    """
    if max_paths <= 0:
        raise ValueError(f"max_paths phải là số nguyên dương, nhận được: {max_paths}")

    source = graph.source
    sink = graph.sink

    # Xây dựng danh sách kề xuôi và danh sách kề ngược (chỉ xét cạnh có capacity > 0)
    # forward_adj[u] = [(edge_idx, v), ...]
    # reverse_adj[v] = [u, ...]
    forward_adj: dict[str, list[tuple[int, str]]] = {node: [] for node in graph.nodes}
    reverse_adj: dict[str, list[str]] = {node: [] for node in graph.nodes}

    for idx, edge in enumerate(graph.edges):
        if edge.capacity > 0:
            forward_adj[edge.u].append((idx, edge.v))
            reverse_adj[edge.v].append(edge.u)

    # Đảm bảo thứ tự cạnh duyệt luôn tăng dần theo chỉ số cạnh
    for u in forward_adj:
        forward_adj[u].sort(key=lambda item: item[0])

    # 1. BFS ngược từ sink để tính khoảng cách (số cạnh tối thiểu) từ mỗi đỉnh tới sink
    dist_to_sink: dict[str, int] = {sink: 0}
    queue: deque[str] = deque([sink])

    while queue:
        curr = queue.popleft()
        curr_d = dist_to_sink[curr]
        for pred in reverse_adj[curr]:
            if pred not in dist_to_sink:
                dist_to_sink[pred] = curr_d + 1
                queue.append(pred)

    # Nếu source không đến được sink
    if source not in dist_to_sink:
        raise ValueError("Không tìm thấy đường đi nào từ nguồn đến đích.")

    min_len = dist_to_sink[source]
    max_len = graph.num_nodes - 1

    paths: list[list[int]] = []
    search_limit = max_paths + 1

    # 2. Duyệt theo độ dài L tăng dần để ưu tiên các đường đi ngắn nhất
    for target_len in range(min_len, max_len + 1):
        if len(paths) >= search_limit:
            break

        # DFS tìm đường đi có đúng độ dài target_len
        def dfs(u: str, current_edges: list[int], visited: set[str], remaining_steps: int) -> bool:
            if remaining_steps == 0:
                if u == sink:
                    paths.append(list(current_edges))
                    if len(paths) >= search_limit:
                        return True  # Dừng tìm kiếm vì đã đủ số đường
                return False

            for edge_idx, v in forward_adj[u]:
                if v in visited:
                    continue
                # Cắt nhánh: nếu từ v không tới được sink trong số bước còn lại
                if dist_to_sink.get(v, float("inf")) > remaining_steps - 1:
                    continue

                current_edges.append(edge_idx)
                visited.add(v)

                stopped = dfs(v, current_edges, visited, remaining_steps - 1)

                visited.remove(v)
                current_edges.pop()

                if stopped:
                    return True
            return False

        dfs(source, [], {source}, target_len)

    if not paths:
        raise ValueError("Không tìm thấy đường đi nào từ nguồn đến đích.")

    if len(paths) > max_paths:
        paths = paths[:max_paths]
        warnings.warn(
            f"Số lượng đường đi vượt quá giới hạn max_paths={max_paths}. "
            f"Đã cắt bớt và giữ lại {len(paths)} đường đi ngắn nhất.",
            UserWarning,
            stacklevel=2
        )

    return paths


class PathEncoding:
    """
    Bộ mã hóa và giải mã nghiệm theo đường đi (Path-based encoding).
    Nghiệm x là một mảng 1D độ dài num_paths, trong đó x[p] là lượng luồng phân bổ cho đường p.
    """

    def __init__(self, graph: Graph, max_paths: int = 200):
        self.graph = graph
        self.paths = enumerate_paths(graph, max_paths=max_paths)
        self.num_paths = len(self.paths)
        self.lower = np.zeros(self.num_paths, dtype=np.float64)

        # upper[p] = min capacity trên các cạnh của đường p (nút cổ chai của đường đi)
        self.upper = np.array([
            min(self.graph.edges[e_idx].capacity for e_idx in p)
            for p in self.paths
        ], dtype=np.float64)

        # Tính trước chi phí đơn vị của từng đường đi: sum(cost các cạnh)
        self._path_unit_costs = np.array([
            sum(self.graph.edges[e_idx].cost for e_idx in p)
            for p in self.paths
        ], dtype=np.float64)

        # Thứ tự các đường đi sắp xếp theo chi phí tăng dần (dùng cho bước top-up min_cost)
        self._sorted_path_indices_by_cost = np.argsort(self._path_unit_costs, kind="stable")

    def clip(self, x: np.ndarray | list[float]) -> np.ndarray:
        """Cắt x về phạm vi hợp lệ [lower, upper]."""
        x_arr = np.asarray(x, dtype=np.float64)
        return np.clip(x_arr, self.lower, self.upper)

    def decode(
        self,
        x: np.ndarray | list[float],
        required_flow: float | None = None,
        topup: bool = False
    ) -> np.ndarray:
        """
        Chuyển vector lượng luồng trên đường đi x thành vector luồng trên các cạnh flows (độ dài num_edges).

        Các bước:
          1. x = clip(x). residual = capacities.copy(). flows = zeros(num_edges).
          2. Duyệt các đường theo chỉ số tăng dần 0..num_paths-1 (xác định):
             amount = min(x[p], min(residual trên các cạnh của p)).
             Cộng amount vào flows, trừ khỏi residual.
          3. Nếu required_flow is None (chế độ max_flow): trả về flows ngay.
          4. Nếu required_flow is not None (chế độ min_cost):
             - Tính total = tổng luồng tịnh tới sink.
             - Nếu total > required_flow: co đều flows bằng hệ số (required_flow / total).
             - Nếu total < required_flow và topup=True: bù luồng thiếu bằng cách duyệt các đường
               theo chi phí đơn vị tăng dần, thêm luồng còn trống từ residual.
             - Nếu total < required_flow và topup=False: giữ nguyên flows (phần thiếu sẽ bị phạt trong evaluate).

        Lưu ý:
          Top-up theo chi phí chỉ dùng làm thuật toán đối chứng, KHÔNG bật cho GA/WOA vì nó tự giải
          bài toán min_cost (thực tế x=0 trên S1 nếu bật topup sẽ tự tham lam ra chi phí tối ưu 24).
          Khi topup=False, nếu tổng luồng chưa đủ required_flow thì evaluate sẽ phạt phần thiếu.
        """
        x_clipped = self.clip(x)
        residual = self.graph.capacities.copy()
        flows = np.zeros(self.graph.num_edges, dtype=np.float64)

        # 2. Duyệt tuần tự theo chỉ số đường đi tăng dần
        for p_idx, path in enumerate(self.paths):
            if x_clipped[p_idx] <= 0.0:
                continue
            # Nút cổ chai còn lại trên các cạnh của đường p
            bottleneck = min(residual[e] for e in path)
            amount = min(x_clipped[p_idx], bottleneck)
            if amount > 0.0:
                for e in path:
                    flows[e] += amount
                    residual[e] -= amount

        # 3. Chế độ max_flow: required_flow is None
        if required_flow is None:
            return flows

        # 4. Chế độ min_cost:
        # Tính tổng luồng thực tế đi vào nút đích sink
        # net_flow = B @ flows, trong đó net_flow[sink] = in_flow(sink) - out_flow(sink)
        net_flow = self.graph.incidence_matrix @ flows
        total_flow = float(max(0.0, net_flow[self.graph.sink_idx]))

        if total_flow > required_flow:
            if total_flow > 1e-12:
                scale = required_flow / total_flow
                flows *= scale
        elif total_flow < required_flow and topup:
            deficit = required_flow - total_flow
            # Duyệt các đường theo thứ tự chi phí đơn vị tăng dần
            for p_idx in self._sorted_path_indices_by_cost:
                if deficit <= 1e-12:
                    break
                path = self.paths[p_idx]
                bottleneck = min(residual[e] for e in path)
                if bottleneck > 1e-12:
                    add_amount = min(deficit, bottleneck)
                    for e in path:
                        flows[e] += add_amount
                        residual[e] -= add_amount
                    deficit -= add_amount

        return flows
