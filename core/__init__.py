"""
Package core: Chứa các thành phần cốt lõi của bài toán Network Flow.
"""

from core.graph import Edge, Graph
from core.fitness import Evaluation, evaluate, repair_flows, is_feasible
from core.solver_base import SolverBase, Result
from core.dummy_solver import RandomSolver
from core.paths import PathEncoding, enumerate_paths

__all__ = [
    "Edge",
    "Graph",
    "Evaluation",
    "evaluate",
    "repair_flows",
    "is_feasible",
    "SolverBase",
    "Result",
    "RandomSolver",
    "PathEncoding",
    "enumerate_paths",
]
