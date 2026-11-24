"""Bộ kiểm thử riêng cho phần cài đặt GA của TV2."""

from __future__ import annotations
import numpy as np

from core.graph import Graph
from core.ga import GASolver


def load_graph() -> Graph:
    return Graph.from_json("data/scenarios/s1_small.json")


def test_initial_population_and_operators() -> None:
    graph = load_graph()
    solver = GASolver(graph, seed=42, params={"pop_size": 12})
    population = solver.initialize_population()
    assert population.shape == (12, solver.dim)
    assert np.all(population >= solver.lower)
    assert np.all(population <= solver.upper)

    parent1 = solver.lower.copy()
    parent2 = solver.upper.copy()
    child1, child2 = solver.crossover(parent1, parent2)
    assert child1.shape == (solver.dim,)
    assert child2.shape == (solver.dim,)
    assert np.all(child1 >= solver.lower) and np.all(child1 <= solver.upper)
    assert np.all(child2 >= solver.lower) and np.all(child2 <= solver.upper)

    mutated = solver.mutate(parent1)
    assert mutated.shape == (solver.dim,)
    assert np.all(mutated >= solver.lower) and np.all(mutated <= solver.upper)


def test_ga_modes_and_history() -> None:
    graph = load_graph()
    for mode in ("max_flow", "min_cost"):
        solver = GASolver(graph, mode=mode, seed=42, params={"pop_size": 20})
        result = solver.run(max_iter=20, max_evals=250)
        assert result.algorithm_name == "GA"
        assert result.evaluations <= 250
        assert len(result.history) == len(result.history_flows)
        assert result.best_x is not None
        assert np.all(result.best_x >= solver.lower - 1e-9)
        assert np.all(result.best_x <= solver.upper + 1e-9)
        assert result.best_eval.feasible is True


def test_budget_and_reproducibility() -> None:
    graph = load_graph()
    no_budget = GASolver(graph, seed=1).run(max_iter=5, max_evals=0)
    assert no_budget.evaluations == 0
    assert len(no_budget.history) == len(no_budget.history_flows) == 1

    first = GASolver(graph, seed=123, params={"pop_size": 15}).run(max_iter=10, max_evals=120)
    second = GASolver(graph, seed=123, params={"pop_size": 15}).run(max_iter=10, max_evals=120)
    assert first.evaluations == second.evaluations
    assert np.allclose(first.best_x, second.best_x)
    assert np.allclose(first.best_flows, second.best_flows)
    assert first.history == second.history


if __name__ == "__main__":
    test_initial_population_and_operators()
    test_ga_modes_and_history()
    test_budget_and_reproducibility()
    print("TẤT CẢ KIỂM THỬ GA ĐỀU ĐẠT!")
