"""
core/graph.py
Định nghĩa cấu trúc dữ liệu đồ thị cho bài toán Network Flow (Max Flow & Min Cost Flow).
Phụ trách: Thành viên 1 (Nhóm trưởng)
"""

from __future__ import annotations
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any
import numpy as np


@dataclass
class Edge:
    """Đại diện cho một cạnh có hướng trong mạng luồng."""
    u: str              # Đỉnh đầu (nguồn của cạnh)
    v: str              # Đỉnh cuối (đích của cạnh)
    capacity: float     # Sức chứa tối đa của cạnh (capacity >= 0)
    cost: float = 0.0   # Chi phí trên một đơn vị luồng qua cạnh (cost >= 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "u": self.u,
            "v": self.v,
            "capacity": self.capacity,
            "cost": self.cost
        }


@dataclass
class Graph:
    """Đại diện cho mạng luồng có hướng G = (V, E)."""
    nodes: list[str]
    edges: list[Edge]
    source: str
    sink: str
    required_flow: float | None = None
    name: str = "network"
    known_optimum: dict[str, float] | None = None

    # Các thuộc tính nội bộ hỗ trợ tính toán vector hóa nhanh
    _node_to_idx: dict[str, int] = field(init=False, repr=False)
    _capacities: np.ndarray = field(init=False, repr=False)
    _costs: np.ndarray = field(init=False, repr=False)
    _incidence_matrix: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.validate()
        self._build_index_structures()

    def validate(self) -> None:
        """Kiểm tra tính hợp lệ của dữ liệu đồ thị."""
        node_set = set(self.nodes)
        if len(node_set) != len(self.nodes):
            raise ValueError("Danh sách đỉnh (nodes) chứa đỉnh bị trùng lặp.")

        if self.source not in node_set:
            raise ValueError(f"Đỉnh nguồn '{self.source}' không nằm trong danh sách đỉnh.")
        if self.sink not in node_set:
            raise ValueError(f"Đỉnh đích '{self.sink}' không nằm trong danh sách đỉnh.")
        if self.source == self.sink:
            raise ValueError("Đỉnh nguồn và đỉnh đích không được trùng nhau.")

        for i, edge in enumerate(self.edges):
            if edge.u not in node_set:
                raise ValueError(f"Cạnh {i} ({edge.u} -> {edge.v}): đỉnh đầu '{edge.u}' không tồn tại.")
            if edge.v not in node_set:
                raise ValueError(f"Cạnh {i} ({edge.u} -> {edge.v}): đỉnh cuối '{edge.v}' không tồn tại.")
            if edge.capacity < 0:
                raise ValueError(f"Cạnh {i} ({edge.u} -> {edge.v}): capacity ({edge.capacity}) không được âm.")
            if edge.cost < 0:
                raise ValueError(f"Cạnh {i} ({edge.u} -> {edge.v}): cost ({edge.cost}) không được âm.")

        if self.required_flow is not None and self.required_flow < 0:
            raise ValueError(f"Lượng luồng yêu cầu (required_flow = {self.required_flow}) không được âm.")

    def _build_index_structures(self) -> None:
        """Xây dựng ma trận liên thuộc và mảng capacities, costs để vector hóa tính toán."""
        self._node_to_idx = {name: idx for idx, name in enumerate(self.nodes)}
        self._capacities = np.array([e.capacity for e in self.edges], dtype=np.float64)
        self._costs = np.array([e.cost for e in self.edges], dtype=np.float64)

        # Ma trận liên thuộc B kích thước (|V|, |E|):
        # B[u, e] = -1 (luồng đi ra khỏi u)
        # B[v, e] = +1 (luồng đi vào v)
        # Khi đó B @ flows = net_flow tại mỗi đỉnh (in_flow - out_flow)
        num_v = len(self.nodes)
        num_e = len(self.edges)
        b = np.zeros((num_v, num_e), dtype=np.float64)
        for e_idx, edge in enumerate(self.edges):
            u_idx = self._node_to_idx[edge.u]
            v_idx = self._node_to_idx[edge.v]
            b[u_idx, e_idx] -= 1.0
            b[v_idx, e_idx] += 1.0
        self._incidence_matrix = b

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    @property
    def source_idx(self) -> int:
        return self._node_to_idx[self.source]

    @property
    def sink_idx(self) -> int:
        return self._node_to_idx[self.sink]

    @property
    def capacities(self) -> np.ndarray:
        """Mảng 1D float chứa capacity của từng cạnh theo thứ tự edges."""
        return self._capacities

    @property
    def costs(self) -> np.ndarray:
        """Mảng 1D float chứa cost của từng cạnh theo thứ tự edges."""
        return self._costs

    @property
    def incidence_matrix(self) -> np.ndarray:
        """Ma trận liên thuộc (|V|, |E|)."""
        return self._incidence_matrix

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Graph:
        """Tạo Graph từ dictionary."""
        edges = [
            Edge(
                u=e["u"],
                v=e["v"],
                capacity=float(e["capacity"]),
                cost=float(e.get("cost", 0.0))
            )
            for e in data["edges"]
        ]
        return cls(
            name=data.get("name", "network"),
            nodes=list(data["nodes"]),
            edges=edges,
            source=data["source"],
            sink=data["sink"],
            required_flow=float(data["required_flow"]) if data.get("required_flow") is not None else None,
            known_optimum=data.get("known_optimum")
        )

    def to_dict(self) -> dict[str, Any]:
        """Xuất Graph thành dictionary."""
        res: dict[str, Any] = {
            "name": self.name,
            "source": self.source,
            "sink": self.sink,
            "nodes": list(self.nodes),
            "edges": [e.to_dict() for e in self.edges]
        }
        if self.required_flow is not None:
            res["required_flow"] = self.required_flow
        if self.known_optimum is not None:
            res["known_optimum"] = self.known_optimum
        return res

    @classmethod
    def from_json(cls, path: str | Path) -> Graph:
        """Đọc và tạo Graph từ file JSON kịch bản."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Không tìm thấy file đồ thị: {path}")
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_json(self, path: str | Path) -> None:
        """Ghi Graph ra file JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
