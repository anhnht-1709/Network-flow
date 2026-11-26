# TÀI LIỆU QUY CHUẨN GIAO DIỆN CHUNG (SOLVER INTERFACE SPECIFICATION)
**Dự án:** Ứng dụng thuật toán GA và WOA cho bài toán Network Flow (Nhóm 8)  
**Phụ trách:** Thành viên 1 (Nhóm trưởng)  
**Áp dụng cho:** Tất cả các thành viên (TV2 - GA, TV3 - Dữ liệu & Thực nghiệm, TV4 - GUI, TV5 - WOA)

---

## 1. Kiến trúc tổng thể và cấu trúc Repo
```
AI_NHOM8/
├── core/
│   ├── graph.py         # Lớp Graph, Edge: đọc/ghi JSON, ma trận liên thuộc
│   ├── fitness.py       # Hàm evaluate(), Evaluation, repair_flows(), is_feasible()
│   ├── paths.py         # enumerate_paths, PathEncoding (mã hóa theo đường đi)
│   ├── solver_base.py   # Lớp cha SolverBase, Result
│   └── dummy_solver.py  # RandomSolver mẫu cho GA & WOA
├── data/scenarios/      # Chứa các file kịch bản JSON (s1_small.json, ...)
├── experiments/         # benchmark.py (TV3 phụ trách)
├── gui/                 # app.py (TV4 phụ trách)
├── report/              # Các phần báo cáo
├── docs/                # Tài liệu giao diện chung
└── main.py              # Điểm khởi chạy chương trình
```

---

## 2. Mã hóa nghiệm theo đường đi (Path-based Encoding)
> **Lý do áp dụng:** Mã hóa nghiệm trực tiếp trên từng cạnh dẫn đến việc các ràng buộc bảo toàn luồng tại nút trung gian (ràng buộc đẳng thức số thực) gần như không thể thỏa mãn ngẫu nhiên (`feasible=False`).  
> **Giải pháp:** Chuyển sang biểu diễn cá thể $x$ là **lượng luồng chạy trên từng đường đi đơn** từ `source` đến `sink`.

### 2.1. Các thông số không gian tìm kiếm trong SolverBase:
Mỗi solver kế thừa `SolverBase` tự động có sẵn các thuộc tính và phương thức:
- `self.dim`: Số chiều không gian nghiệm = số lượng đường đi đơn ($num\_paths$).
- `self.lower`: Mảng 1D float chứa cận dưới = `zeros(self.dim)`.
- `self.upper`: Mảng 1D float chứa cận trên = nút cổ chai (sức chứa nhỏ nhất trong các cạnh của từng đường đi).
- Khởi tạo quần thể chung: `self.generate_initial_population(pop_size)` sinh ma trận `(pop_size, self.dim)` từ `self.seed` để các thuật toán xuất phát từ cùng một tập ứng viên ban đầu.
- Kích thước quần thể mặc định dùng chung: hằng `DEFAULT_POP_SIZE = 30` trong `core/solver_base.py` (GA và WOA đều đọc `params["pop_size"]`, mặc định 30).
- Ghi tiến trình: `self.reset_tracking()` ở đầu `run()` và `self.record_progress(best_eval, best_flows)` cuối mỗi vòng lặp (tự lưu `history`, `history_flows`, `history_evals`).
- Cắt biên nghiệm: `self.encoding.clip(x)` đưa $x$ về đoạn $[\text{lower}, \text{upper}]$.
- So sánh nghiệm: `self.is_better(ev, best_eval)` ưu tiên nghiệm khả thi (`feasible=True`) trước, sau đó chọn nghiệm có `fitness` lớn hơn.

### 2.2. Cơ chế Decode và Đánh giá:
- Solver (GA, WOA) chỉ cần sinh và tối ưu vector $x \in [\text{lower}, \text{upper}]$.
- Khi đánh giá, gọi duy nhất: `ev, flows = self.evaluate_candidate(x)`.
  - Hàm tự động decode $x$ sang vector luồng trên các cạnh `flows`, đảm bảo tự động thỏa mãn bảo toàn luồng và sức chứa cạnh.
  - Chế độ `min_cost`: mặc định `topup=False`. KHÔNG bật bù luồng tham lam cho GA/WOA vì hai thuật toán phải tự tìm cách phối hợp luồng để đạt nghiệm tối ưu; nếu thiếu luồng, `evaluate()` sẽ phạt phần `shortfall`.
  - Hàm chấm điểm qua `self._evaluate(flows)` và tự động đếm số lần gọi `self.eval_count`.

---

## 3. Hàm đánh giá (Fitness Function) – Quy chuẩn duy nhất
> **Quy ước bất biến:** Mọi Solver luôn **TỐI ĐA HÓA (MAXIMIZE)** giá trị `fitness` và ưu tiên nghiệm hợp lệ (`feasible=True`) qua `self.is_better(ev, best_eval)`.

Hàm đánh giá: `evaluate(flows, graph, mode="max_flow"|"min_cost", penalty=None) -> Evaluation`
- Khi `penalty=None`, hệ số phạt được tính tự động lớn hơn tổng chi phí tối đa có thể của mạng ($\max(1000.0, 10 \times \sum c_e \cdot \text{cost}_e)$).
- Hàm `cut_proof(graph, required_flow)` tìm lát cắt $s$-$t$ cực tiểu để chứng minh toán học cho các kịch bản vô nghiệm ($\text{required\_flow} > \text{cut\_capacity}$).

Đối tượng trả về `Evaluation`:
- `fitness: float`: Điểm thích nghi để chọn lọc cá thể (càng cao càng tốt).
- `total_flow: float`: Tổng luồng tịnh thực tế đi vào nút đích `sink`.
- `total_cost: float`: Tổng chi phí $\sum flows[e] \times cost[e]$.
- `violation: float`: Tổng mức độ vi phạm bảo toàn luồng tại các nút trung gian và vượt sức chứa.
- `feasible: bool`: `True` nếu nghiệm hợp lệ (vi phạm $\le 10^{-4}$ và thỏa mãn đủ luồng).

### Công thức tính fitness:
1. **Chế độ `max_flow`:**  
   $$\text{fitness} = \text{total\_flow} - \text{penalty} \times \text{violation}$$
2. **Chế độ `min_cost`:** (Yêu cầu đồ thị có `required_flow` $R$)  
   $$\text{fitness} = -\big(\text{total\_cost} + \text{penalty} \times (\text{violation} + |\text{total\_flow} - R|)\big)$$

---

## 4. Giao diện Solver chuẩn (`SolverBase`) và Mã mẫu cho TV2 (GA) / TV5 (WOA)
Mọi thuật toán bắt buộc kế thừa `SolverBase` từ `core.solver_base`:

```python
from core.solver_base import SolverBase, Result
import numpy as np

class CustomSolver(SolverBase):
    def run(self, max_iter: int, max_evals: int | None = None) -> Result:
        self.reset_tracking()

        # 1. Khởi tạo quần thể ban đầu dùng chung từ self.seed
        pop = self.generate_initial_population(pop_size=30)
        best_x = np.zeros(self.dim)
        best_eval, best_flows = self.evaluate_candidate(best_x)
        self.record_progress(best_eval, best_flows)

        # 2. Vòng lặp tiến hóa / tối ưu (GA, WOA)
        for gen in range(max_iter):
            if self.budget_exhausted(max_evals):
                break
            
            # Sinh cá thể x trong khoảng [self.lower, self.upper]
            candidate_x = self.rng.uniform(self.lower, self.upper, size=self.dim)
            candidate_x = self.encoding.clip(candidate_x)

            # Đánh giá cá thể duy nhất qua evaluate_candidate
            ev, flows = self.evaluate_candidate(candidate_x)

            # Cập nhật nghiệm tốt nhất (ưu tiên feasible, sau đó tối đa hóa fitness)
            if self.is_better(ev, best_eval):
                best_eval = ev
                best_flows = flows
                best_x = candidate_x.copy()

            self.record_progress(best_eval, best_flows)

        return Result(
            best_flows=best_flows,
            best_eval=best_eval,
            history=self.history,
            evaluations=self.eval_count,
            elapsed_s=0.0, # tính thời gian thực tế
            mode=self.mode,
            params=self.params,
            algorithm_name="GA_or_WOA",
            best_x=best_x,
            history_flows=self.history_flows,
            history_evals=self.history_evals
        )
```

---

## 5. Hướng dẫn phân công chi tiết cho các thành viên

### Thành viên 2 (GA) & Thành viên 5 (WOA) – Giao việc kỹ thuật ngắn:
Mọi thuật toán (`core/ga.py` và `core/woa.py`) đều kế thừa `SolverBase` và tuân thủ các quy tắc bất biến sau:
1. **Không gian tìm kiếm & Khởi tạo:** Làm việc trên vector nghiệm đường đi $x$ kích thước `self.dim` trong `[self.lower, self.upper]`. Khởi tạo quần thể ban đầu bằng `self.generate_initial_population(pop_size)`.
2. **Kẹp biên:** Luôn dùng `candidate_x = self.encoding.clip(candidate_x)` sau khi lai ghép, đột biến hoặc cập nhật vị trí.
3. **Đánh giá cá thể:** BẮT BUỘC gọi duy nhất `ev, flows = self.evaluate_candidate(x)` trong vòng lặp (hàm tự decode với `topup=False`, chấm điểm và tự động đếm 1 lần đánh giá). Không gọi trực tiếp `_evaluate(flows)`.
4. **Mục tiêu tối ưu:** Cập nhật nghiệm tốt nhất khi `self.is_better(ev, best_eval)` trả về `True` và lưu `self.history_flows.append(best_flows.copy())` sau mỗi vòng lặp. Ghi thêm bộ đếm số lần thực hiện từng toán tử (`details`) vào `Result.params`.
5. **Dừng theo ngân sách:** Luôn kiểm tra `if self.budget_exhausted(max_evals): break` ở cả vòng lặp thế hệ lẫn vòng lặp đánh giá cá thể để dừng sớm khi hết ngân sách `max_evals`.
6. **Code mẫu chuẩn:** Tham khảo cách cài đặt chi tiết trong [`core/dummy_solver.py`](file:///Users/tramanh/Project/AI_NHOM8/core/dummy_solver.py).

### Thành viên 3 (Dữ liệu & Thực nghiệm - `experiments/benchmark.py`):
- Nạp đồ thị bằng `Graph.from_json("data/scenarios/...")`.
- Khởi tạo solver cùng `seed`, gọi `res = solver.run(max_iter, max_evals=...)`.
- Lấy `res.history` để vẽ biểu đồ đường hội tụ, **trục hoành dùng `res.history_evals`** (số lượt đánh giá), KHÔNG dùng số vòng lặp vì mỗi vòng GA và WOA tốn số lượt đánh giá khác nhau.
- Luôn truyền cùng `params={"pop_size": ...}` và cùng `max_evals` cho cả GA và WOA.
- So sánh hiệu năng dựa trên: `res.best_eval.total_flow` (với max flow), `res.best_eval.total_cost` (với min cost), `res.evaluations`, `res.elapsed_s`, và tỷ lệ nghiệm `res.best_eval.feasible`.

### Thành viên 4 (Giao diện Form GUI - `gui/app.py`):
- Form cho phép người dùng:
  1. Chọn kịch bản đồ thị (nạp từ `data/scenarios/`).
  2. Chọn chế độ bài toán (`max_flow` hoặc `min_cost`).
  3. Chọn thuật toán (`GA`, `WOA`, hoặc `RandomSearch`).
  4. Cấu hình tham số: `max_iter`, `max_evals`, `seed`, `pop_size`, ...
  5. Bấm nút **"Chạy tối ưu"** -> gọi `solver.run(max_iter)`.
  6. Hiển thị: Bảng số liệu kết quả, đồ thị mạng luồng (dùng `res.best_flows`), và đồ thị hội tụ từ `res.history`.

---
*Tài liệu đã được kiểm chứng hoạt động thông suốt với script [`test_framework.py`](file:///Users/tramanh/Project/AI_NHOM8/test_framework.py).*
