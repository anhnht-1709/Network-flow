"""
main.py
Điểm khởi chạy chính của dự án.
Phụ trách tích hợp: Thành viên 1 (Nhóm trưởng)
"""

import sys
from pathlib import Path

# Đảm bảo đường dẫn import
sys.path.insert(0, str(Path(__file__).parent))

from core.graph import Graph
from core.dummy_solver import RandomSolver


def main():
    print("=" * 65)
    print("  ĐỒ ÁN TRÍ TUỆ NHÂN TẠO - NHÓM 8")
    print("  ĐỀ TÀI: ỨNG DỤNG GA VÀ WOA CHO BÀI TOÁN NETWORK FLOW")
    print("=" * 65)

    scenario_path = Path("data/scenarios/s1_small.json")
    if not scenario_path.exists():
        print(f"Lỗi: Không tìm thấy kịch bản mẫu tại {scenario_path}")
        return

    graph = Graph.from_json(scenario_path)
    print(f"\n[+] Đã tải kịch bản: {graph.name}")
    print(f"    Số đỉnh: {graph.num_nodes} {graph.nodes}")
    print(f"    Số cạnh: {graph.num_edges}")
    print(f"    Nút nguồn (Source): {graph.source} -> Nút đích (Sink): {graph.sink}")
    print(f"    Lượng luồng yêu cầu: {graph.required_flow}")
    if graph.known_optimum:
        print(f"    Mốc tối ưu lý thuyết: {graph.known_optimum}")

    # Chạy thử nghiệm khung chung với RandomSolver
    print("\n--- [Demo kiểm tra khung chung: Max Flow] ---")
    solver_mf = RandomSolver(graph, mode="max_flow", seed=42)
    res_mf = solver_mf.run(max_iter=50)
    print(f"Thuật toán: {res_mf.algorithm_name}")
    print(f"Thời gian: {res_mf.elapsed_s:.4f}s | Số lần đánh giá: {res_mf.evaluations}")
    print(f"Luồng cực đại tìm được: {res_mf.best_eval.total_flow:.4f}")
    print(f"Điểm Fitness: {res_mf.best_eval.fitness:.4f}")
    print(f"Khả thi (Feasible): {res_mf.best_eval.feasible}")

    print("\n--- [Demo kiểm tra khung chung: Min Cost Flow] ---")
    solver_mc = RandomSolver(graph, mode="min_cost", seed=42)
    res_mc = solver_mc.run(max_iter=50)
    print(f"Thuật toán: {res_mc.algorithm_name}")
    print(f"Thời gian: {res_mc.elapsed_s:.4f}s | Số lần đánh giá: {res_mc.evaluations}")
    print(f"Chi phí tìm được: {res_mc.best_eval.total_cost:.4f}")
    print(f"Lượng luồng đạt được: {res_mc.best_eval.total_flow:.4f} (Mục tiêu: {graph.required_flow})")
    print(f"Điểm Fitness: {res_mc.best_eval.fitness:.4f}")
    print(f"Khả thi (Feasible): {res_mc.best_eval.feasible}")

    print("\n" + "=" * 65)
    if res_mf.best_eval.feasible and res_mc.best_eval.feasible:
        print("Khung chung hoạt động hoàn hảo! Các thành viên có thể bắt đầu code.")
    else:
        print("CẢNH BÁO: Ít nhất một chế độ chưa đạt nghiệm hợp lệ (feasible=False)!")
        if not res_mf.best_eval.feasible:
            print("  - Max Flow: Nghiệm chưa thỏa mãn ràng buộc.")
        if not res_mc.best_eval.feasible:
            print("  - Min Cost Flow: Nghiệm chưa thỏa mãn ràng buộc.")
    print("Xem tài liệu giao diện chi tiết tại: docs/INTERFACE_SPEC.md")
    print("=" * 65)


if __name__ == "__main__":
    main()
