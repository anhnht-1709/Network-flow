"""
gui/app.py
Giao diện người dùng đồ họa dạng Form bằng PySide6 cho bài toán Network Flow.
Phụ trách: Thành viên 4 (TV4)
Deadline khung form: 05/12 | Giao diện hoàn chỉnh: 11/12

Yêu cầu chức năng:
  1. Cho phép nạp kịch bản đồ thị từ thư mục data/scenarios/ hoặc nhập tay.
  2. Chọn dạng bài toán: Max Flow hoặc Min Cost Flow.
  3. Chọn thuật toán: GA, WOA, hoặc RandomSearch (để test).
  4. Cấu hình các tham số: max_iter, max_evals, seed, pop_size, ...
  5. Nút bấm "Chạy tối ưu" (chạy luồng nền QThread để không đơ giao diện).
  6. Vùng hiển thị kết quả:
     - Bảng số liệu: Thời gian, số lần đánh giá, Best Fitness, Total Flow, Total Cost, Feasible.
     - Đồ thị trực quan mạng luồng (vẽ bằng NetworkX + Matplotlib canvas).
     - Đồ thị đường cong hội tụ (Fitness theo từng vòng lặp).
"""

from __future__ import annotations
import sys
from pathlib import Path


def main():
    """Điểm khởi chạy ứng dụng GUI."""
    # TODO (TV4): Khởi tạo QApplication và MainWindow với PySide6
    print("Ứng dụng GUI đang được xây dựng bằng PySide6 bởi TV4.")
    print("Có thể test trước khung form bằng RandomSolver.")


if __name__ == "__main__":
    main()
