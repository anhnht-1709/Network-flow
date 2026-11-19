"""
test_framework.py
Script kiểm thử tích hợp khung chung và mã hóa nghiệm theo đường đi (Path-based encoding).
Phụ trách: Thành viên 1 (Nhóm trưởng)
"""

from __future__ import annotations
import sys
import warnings
from pathlib import Path
import numpy as np

# Đảm bảo import được package core
sys.path.insert(0, str(Path(__file__).parent))

from core.graph import Graph, Edge
from core.fitness import Evaluation, evaluate, is_feasible, is_better, cut_proof
from core.paths import enumerate_paths, PathEncoding
from core.dummy_solver import RandomSolver


def test_4a_s1_min_cost_flow_and_cost():
    """
    Test 4a: S1 min_cost với x = [8, 0] cho chi phí 24, với x = [8, 5] cho chi phí > 24 và total_flow = 8.
    """
    print("--- [Test 4a] S1 min_cost x=[8, 0] và x=[8, 5] ---")
    graph = Graph.from_json("data/scenarios/s1_small.json")
    encoding = PathEncoding(graph)

    # x = [8, 0]: bơm toàn bộ 8 đơn vị qua đường s-a-t (rẻ nhất, cost 2+1=3)
    flows_opt = encoding.decode([8.0, 0.0], required_flow=8.0, topup=False)
    ev_opt = evaluate(flows_opt, graph, mode="min_cost")
    assert abs(ev_opt.total_cost - 24.0) < 1e-6, f"cost kỳ vọng 24.0, nhận được {ev_opt.total_cost}"
    assert abs(ev_opt.total_flow - 8.0) < 1e-6, f"flow kỳ vọng 8.0, nhận được {ev_opt.total_flow}"
    assert ev_opt.feasible is True

    # x = [8, 5]: luồng thô = 13 > 8, co đều về 8, luồng chia cho cả 2 đường -> chi phí > 24
    flows_sub = encoding.decode([8.0, 5.0], required_flow=8.0, topup=False)
    ev_sub = evaluate(flows_sub, graph, mode="min_cost")
    assert ev_sub.total_cost > 24.0, f"cost kỳ vọng > 24.0, nhận được {ev_sub.total_cost}"
    assert abs(ev_sub.total_flow - 8.0) < 1e-6, f"flow kỳ vọng 8.0, nhận được {ev_sub.total_flow}"
    assert ev_sub.feasible is True
    print(f"  -> ĐẠT: x=[8, 0] có cost={ev_opt.total_cost:.1f}, flow={ev_opt.total_flow:.1f}; "
          f"x=[8, 5] có cost={ev_sub.total_cost:.4f} > 24, flow={ev_sub.total_flow:.1f}.")


def test_4b_decode_zeros_topup_comparison():
    """
    Test 4b: decode(zeros, required_flow=8, topup=False) cho feasible False,
    decode(zeros, required_flow=8, topup=True) cho chi phí 24.
    """
    print("--- [Test 4b] decode(zeros) với topup=False vs topup=True ---")
    graph = Graph.from_json("data/scenarios/s1_small.json")
    encoding = PathEncoding(graph)
    zeros = np.zeros(encoding.num_paths)

    # 1. topup=False: không bù luồng, total_flow = 0 != 8 -> feasible = False
    flows_notopup = encoding.decode(zeros, required_flow=8.0, topup=False)
    ev_notopup = evaluate(flows_notopup, graph, mode="min_cost")
    assert ev_notopup.feasible is False, "zeros với topup=False phải feasible=False do thiếu luồng"
    assert abs(ev_notopup.total_flow - 0.0) < 1e-6

    # 2. topup=True: bù luồng tham lam theo chi phí tăng dần -> tự đạt luồng 8 qua đường rẻ nhất -> cost = 24
    flows_topup = encoding.decode(zeros, required_flow=8.0, topup=True)
    ev_topup = evaluate(flows_topup, graph, mode="min_cost")
    assert abs(ev_topup.total_cost - 24.0) < 1e-6, f"cost kỳ vọng 24.0, nhận được {ev_topup.total_cost}"
    assert abs(ev_topup.total_flow - 8.0) < 1e-6
    assert ev_topup.feasible is True
    print(f"  -> ĐẠT: topup=False cho feasible={ev_notopup.feasible} (flow={ev_notopup.total_flow:.1f}); "
          f"topup=True bù tham lam cho cost={ev_topup.total_cost:.1f}, feasible={ev_topup.feasible}.")


def test_4c_diamond_network():
    """
    Test 4c: Mạng kim cương:
    thứ tự đường là [[0, 1], [4, 3], [0, 2, 3]] và upper = [4, 3, 4],
    max_flow decode(upper) cho total_flow = 8.
    """
    print("--- [Test 4c] Mạng kim cương: thứ tự đường, upper và max_flow ---")
    diamond_edges = [
        Edge(u="s", v="a", capacity=6.0, cost=1.0),  # 0
        Edge(u="a", v="t", capacity=4.0, cost=1.0),  # 1
        Edge(u="a", v="b", capacity=5.0, cost=1.0),  # 2
        Edge(u="b", v="t", capacity=4.0, cost=1.0),  # 3
        Edge(u="s", v="b", capacity=3.0, cost=1.0),  # 4
    ]
    graph = Graph(
        name="diamond",
        nodes=["s", "a", "b", "t"],
        edges=diamond_edges,
        source="s",
        sink="t",
        required_flow=5.0
    )

    encoding = PathEncoding(graph)

    # 1. Kiểm tra chính xác thứ tự các đường đi
    expected_paths = [[0, 1], [4, 3], [0, 2, 3]]
    assert encoding.paths == expected_paths, (
        f"Thứ tự đường kỳ vọng {expected_paths}, nhận được {encoding.paths}"
    )

    # 2. Kiểm tra upper
    expected_upper = np.array([4.0, 3.0, 4.0])
    assert np.allclose(encoding.upper, expected_upper), (
        f"upper kỳ vọng {expected_upper}, nhận được {encoding.upper}"
    )

    # 3. max_flow decode(upper) cho total_flow = 8
    flows_full = encoding.decode(encoding.upper, required_flow=None)
    ev_full = evaluate(flows_full, graph, mode="max_flow")
    assert abs(ev_full.total_flow - 8.0) < 1e-9, f"total_flow kỳ vọng 8.0, nhận được {ev_full.total_flow}"
    assert ev_full.violation < 1e-9
    assert ev_full.feasible is True

    print(f"  -> ĐẠT: paths={encoding.paths}, upper={encoding.upper.tolist()}, "
          f"max_flow decode(upper) cho total_flow={ev_full.total_flow:.1f}.")


def test_4d_no_warning_when_exact_max_paths():
    """
    Test 4d: enumerate_paths(S1, max_paths=2) không phát cảnh báo (vì S1 có đúng 2 đường).
    """
    print("--- [Test 4d] enumerate_paths(S1, max_paths=2) không cảnh báo ---")
    graph = Graph.from_json("data/scenarios/s1_small.json")
    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        paths = enumerate_paths(graph, max_paths=2)
        assert len(paths) == 2, f"Kỳ vọng 2 đường, nhận được {len(paths)}"
        assert len(captured_warnings) == 0, (
            f"Không được phát cảnh báo khi số đường đúng bằng max_paths. Bắt được: {len(captured_warnings)}"
        )
    print("  -> ĐẠT: Không có cảnh báo nào được phát ra khi số đường đúng bằng max_paths.")


def test_5_random_solver_s1_min_cost_notopup():
    """
    Test 5: RandomSolver trên S1 min_cost với topup=False: chạy 50 vòng (seed 42).
    Assert feasible == True nếu seed 42 ra được.
    """
    print("--- [Test 5] RandomSolver trên S1 min_cost với topup=False (50 vòng, seed 42) ---")
    graph = Graph.from_json("data/scenarios/s1_small.json")
    solver = RandomSolver(graph, mode="min_cost", seed=42, params={"topup": False})
    res = solver.run(max_iter=50)

    print(f"  Thuật toán: {res.algorithm_name}")
    print(f"  Best cost tìm được: {res.best_eval.total_cost:.4f}")
    print(f"  Total flow: {res.best_eval.total_flow:.4f} (Mục tiêu: {graph.required_flow})")
    print(f"  Fitness: {res.best_eval.fitness:.4f}")
    print(f"  Feasible: {res.best_eval.feasible}")
    print(f"  Best x (đường đi): {res.best_x}")
    print(f"  Số lần đánh giá: {res.evaluations}")

    # Kiểm tra tính khả thi
    assert res.best_eval.feasible is True, (
        f"RandomSolver không tìm được nghiệm feasible (feasible={res.best_eval.feasible})."
    )
    print("  -> ĐẠT: RandomSolver tìm được nghiệm feasible=True với cost=24.0 trên S1 min_cost (topup=False).")


def test_other_core_checks():
    """Kiểm tra bổ sung các tính năng cốt lõi: S1 max_flow, budget_limit, no_path_error."""
    print("--- [Test bổ trợ] S1 Max Flow, Budget Limit, No Path Error ---")
    graph = Graph.from_json("data/scenarios/s1_small.json")

    # S1 Max Flow với RandomSolver
    solver_mf = RandomSolver(graph, mode="max_flow", seed=42)
    res_mf = solver_mf.run(max_iter=50)
    assert res_mf.best_eval.feasible is True
    assert res_mf.best_eval.total_flow >= 12.5

    # Budget limit
    res_budget = solver_mf.run(max_iter=100, max_evals=500)
    assert res_budget.evaluations <= 500

    # No path error
    disconnected_edges = [
        Edge(u="s", v="a", capacity=10.0, cost=1.0),
        Edge(u="b", v="t", capacity=10.0, cost=1.0),
    ]
    graph_dis = Graph(name="dis", nodes=["s", "a", "b", "t"], edges=disconnected_edges, source="s", sink="t")
    try:
        enumerate_paths(graph_dis)
        assert False, "Phải raise ValueError"
    except ValueError as e:
        assert "đường đi" in str(e).lower()

    print("  -> ĐẠT: Các kiểm thử bổ trợ đều thành công.")


def test_shared_population_and_cut_proof():
    """Kiểm tra khởi tạo quần thể dùng chung, is_better, history_flows và cut_proof."""
    print("--- [Test 6] Quần thể dùng chung, is_better, history_flows và cut_proof ---")
    graph = Graph.from_json("data/scenarios/s1_small.json")

    s1 = RandomSolver(graph, mode="min_cost", seed=123)
    s2 = RandomSolver(graph, mode="min_cost", seed=123)
    pop1 = s1.generate_initial_population(10)
    pop2 = s2.generate_initial_population(10)
    assert pop1.shape == (10, s1.dim)
    assert np.allclose(pop1, pop2), "Hai solver cùng seed phải tạo cùng quần thể ban đầu."

    # Kiểm tra is_better ưu tiên feasible=True kể cả khi fitness thấp hơn
    infeasible_high = Evaluation(fitness=-10.0, total_flow=7.0, total_cost=10.0, violation=1.0, feasible=False)
    feasible_low = Evaluation(fitness=-500.0, total_flow=8.0, total_cost=500.0, violation=0.0, feasible=True)
    assert is_better(feasible_low, infeasible_high) is True
    assert is_better(infeasible_high, feasible_low) is False

    # Kiểm tra history_flows trong Result
    res = s1.run(max_iter=5)
    assert len(res.history_flows) == len(res.history)

    # Kiểm tra cut_proof: S1 có max_flow = 13 -> required_flow = 8 trả None, required_flow = 15 trả chứng cứ lát cắt
    assert cut_proof(graph, required_flow=8.0) is None
    proof = cut_proof(graph, required_flow=15.0)
    assert proof is not None
    assert abs(proof["cut_capacity"] - 13.0) < 1e-6
    assert abs(proof["shortfall"] - 2.0) < 1e-6
    print(f"  -> ĐẠT: Quần thể chung khớp 100%, is_better ưu tiên feasible, cut_proof={proof['cut_edges']} (cap={proof['cut_capacity']}).")


if __name__ == "__main__":
    print("=" * 65)
    print("  CHẠY BỘ KIỂM THỬ TÍCH HỢP PATH-ENCODING")
    print("=" * 65)
    test_4a_s1_min_cost_flow_and_cost()
    test_4b_decode_zeros_topup_comparison()
    test_4c_diamond_network()
    test_4d_no_warning_when_exact_max_paths()
    test_5_random_solver_s1_min_cost_notopup()
    test_other_core_checks()
    test_shared_population_and_cut_proof()
    print("\n" + "=" * 65)
    print("  TẤT CẢ CÁC BÀI TEST ĐỀU ĐẠT THÀNH CÔNG 100%!")
    print("=" * 65)
