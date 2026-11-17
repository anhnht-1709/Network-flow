# Kế hoạch phân chia công việc – Dự án cuối kỳ môn Trí tuệ nhân tạo
**Nhóm 8 · 5 thành viên · Deadline nộp: 20/12/2026**  
**Đề tài:** Ứng dụng thuật toán GA và Whale Optimization Algorithm (WOA) cho bài toán Network flow

---

## 1. Tổng quan

### 1.1. Mục tiêu
Xây dựng một chương trình có giao diện dạng form, giải bài toán tối ưu luồng trong mạng (Network flow) ở cả hai dạng **max flow** và **min cost flow** bằng hai thuật toán là **GA** và **WOA**, cả hai cài đặt lại từ đầu. Sau đó so sánh hiệu năng của hai thuật toán trên nhiều tình huống, viết báo cáo theo mẫu và thuyết trình có demo.

### 1.2. Tiêu chí chấm điểm (theo đề)
| Hạng mục | Điểm | Điều nhóm phải đạt |
|---|---|---|
| **Chương trình** | 4đ | Tổ chức code rõ, tách hàm chức năng; thuật toán cài từ đầu; nhiều tình huống khác nhau; so sánh hiệu năng GA và WOA; giao diện form (viết bằng PySide6, thư viện GUI Python trong gợi ý của giảng viên) |
| **Báo cáo** | 4đ | Theo mẫu, giải thích từng phần, không dán ảnh màn hình trơn; Turnitin mặc định dưới 30%, đính kết quả vào cuối báo cáo |
| **Trình bày và hỏi đáp** | 2đ | Gọn, rõ, có demo chạy chương trình, mọi thành viên trả lời trôi chảy phần mình |

*Khai thác triệt để hai thuật toán yêu cầu, không so sánh với thuật toán ngoài đề tài.*

### 1.3. Nguyên tắc làm việc
- Nhóm trưởng chốt giao diện chung trước, sau đó mọi người code song song trên nhánh riêng, nhóm trưởng ktra rồi merge vào main.
- Mỗi người viết phần báo cáo của module mình ngay sau khi code xong, không dồn cuối.
- Báo cáo kết quả thực hiện ngay sau khi làm xong. Tổng kết quá trình vào cuối tuần.
- Hạn hoàn thành là 20/12, bản hoàn chỉnh của cả nhóm phải xong vào 17/12 để còn 3 ngày dự phòng.

---

## 2. Thiết kế kỹ thuật chung

### 2.1. Mô hình bài toán
- Mạng có hướng $G = (V, E)$. Mỗi cạnh có sức chứa ($capacity$) và chi phí đơn vị ($cost$). Có nút nguồn ($source$) và nút đích ($sink$). Nhóm giải cả hai dạng bài toán trên cùng một mạng:
  - **Max flow:** tối đa hóa tổng luồng từ nguồn đến đích.
  - **Min cost flow:** đạt lượng luồng yêu cầu ($required\_flow$) với tổng chi phí nhỏ nhất.
- Cả hai thuật toán dùng chung một hàm đánh giá (`evaluate` trong `core/fitness.py`). Fitness có hai chế độ:
  - `max_flow`: lấy tổng luồng đến nút đích làm mục tiêu, trừ phạt khi vi phạm bảo toàn luồng hoặc sức chứa.
  - `min_cost`: lấy số âm của tổng chi phí làm mục tiêu, trừ phạt khi vi phạm bảo toàn luồng, vượt sức chứa hoặc chưa đạt lượng luồng yêu cầu ($shortfall$).

### 2.2. Mã hóa nghiệm (ĐÃ CẬP NHẬT: Mã hóa theo đường đi - Path-based encoding)
- Một nghiệm là **vector lượng luồng trên các đường đi đơn** từ $source$ đến $sink$:
  $$x = [x_0, x_1, \dots, x_{P-1}]$$
  với $P = num\_paths$ (thu thập bằng thuật toán BFS ngược kết hợp DFS theo độ dài tăng dần trong `core/paths.py`).
- Mỗi thành phần $x[p]$ nằm trong $[0, upper[p]]$, với $upper[p]$ là sức chứa nhỏ nhất (nút cổ chai - bottleneck) trên các cạnh của đường đi $p$.
- Hàm `decode(x, required_flow, topup=False)` chuyển $x$ thành vector luồng trên từng cạnh (`flows` độ dài $|E|$), tự động bảo toàn luồng tại các nút trung gian và thỏa mãn sức chứa cạnh.
- **`topup=False` mặc định:** KHÔNG bật tính năng tự bù luồng tham lam theo chi phí cho GA và WOA, để hai thuật toán tự tối ưu hóa và giải bài toán min cost flow. Nếu tổng luồng chưa đạt $required\_flow$, hàm `evaluate` sẽ phạt phần thiếu ($shortfall$).
- GA và WOA **dùng chung tuyệt đối** cách mã hóa này để so sánh công bằng.

### 2.3. Giao diện solver chung (ĐÃ CẬP NHẬT: evaluate_candidate & max_evals)
Mọi solver có cùng dạng kế thừa từ `core.solver_base.SolverBase`:
- **Không gian nghiệm chuẩn:**
  - `self.dim`: số chiều nghiệm = số đường đi ($num\_paths$).
  - `self.lower`: vector cận dưới = toàn 0.
  - `self.upper`: vector cận trên = bottleneck của từng đường đi.
- **Đánh giá cá thể:**
  - BẮT BUỘC gọi qua `ev, flows = self.evaluate_candidate(x)` (decode $x \to flows$ với `topup=False`, chấm điểm qua `self._evaluate`, tự động đếm đúng 1 lần đánh giá).
- **Quy ước mục tiêu:** Solver luôn **TỐI ĐA HÓA fitness** (fitness càng lớn nghiệm càng tốt ở cả `max_flow` lẫn `min_cost`).
- **Giao diện chạy & ngân sách:**
  - `run(max_iter: int, max_evals: int | None = None) -> Result`
  - Cơ chế dừng theo ngân sách: kiểm tra `self.budget_exhausted(max_evals)` ở mỗi vòng lặp để dừng sớm khi chạm trần số lần đánh giá.
- **Theo dõi tiến trình:**
  - `self.history`: mảng lưu giá trị fitness tốt nhất sau mỗi vòng lặp (dùng vẽ đường hội tụ).
  - `self.eval_count`: đếm số lần gọi hàm đánh giá, để so sánh theo cùng ngân sách.
- **Kết quả trả về `Result`:**
  - `best_flows`: vector luồng trên các cạnh (GUI dùng để vẽ mạng luồng).
  - `best_x`: vector nghiệm tốt nhất theo đường đi.
  - `best_eval`: chi tiết đánh giá (`fitness`, `total_flow`, `total_cost`, `violation`, `feasible`).
  - `history`, `evaluations`, `elapsed_s`, `params`.

### 2.4. Cấu trúc repo
```
AI_NHOM8/
├── core/
│   ├── graph.py         # Lớp Graph, Edge, đọc và kiểm tra dữ liệu
│   ├── fitness.py       # Hàm evaluate(), Evaluation, kiểm tra tính hợp lệ
│   ├── paths.py         # enumerate_paths, PathEncoding (mã hóa theo đường đi)
│   ├── solver_base.py   # Lớp cha SolverBase, Result
│   ├── dummy_solver.py  # RandomSolver mẫu cho GA & WOA
│   ├── ga.py            # Thuật toán GA (TV2 phụ trách)
│   └── woa.py           # Thuật toán WOA (TV5 phụ trách)
├── data/scenarios/      # Các kịch bản thử nghiệm JSON
├── experiments/
│   └── benchmark.py     # Chạy thực nghiệm và xuất bảng/biểu đồ (TV3 phụ trách)
├── gui/
│   └── app.py           # Giao diện form PySide6 (TV4 phụ trách)
├── report/              # Các phần báo cáo
├── docs/
│   ├── PLAN.md          # Kế hoạch phân công công việc (file này)
│   └── INTERFACE_SPEC.md# Tài liệu quy chuẩn giao diện kỹ thuật
├── main.py              # Điểm khởi chạy chương trình
├── test_framework.py    # Kiểm thử tự động khung chung
└── .gitignore           # Bỏ qua file rác, cache
```

### 2.5. Quy ước chung
- Mỗi thành viên làm trên nhánh riêng (`feature/ga`, `feature/woa`, `feature/gui`, …), merge vào main qua nhóm trưởng.
- Mọi so sánh dùng chung hạt giống ngẫu nhiên (`seed`), cùng ngân sách đánh giá (`max_evals`), và mỗi kịch bản chạy ít nhất 20 lần.
- Docstring ngắn cho mỗi hàm; đặt tên theo tiếng Anh, giải thích bằng tiếng Việt trong báo cáo.

---

## 3. Phân công thành viên

| TV | Vai trò | Phụ trách chính | Phần báo cáo |
|---|---|---|---|
| **TV1** | Nhóm trưởng, mô hình hóa và tích hợp | `graph.py`, `fitness.py`, `paths.py`, `solver_base.py`, `dummy_solver.py`, quản lý repo, merge, ghép báo cáo | Giới thiệu và mô hình bài toán, Kết luận |
| **TV2** | GA | `ga.py`: mã hóa đường đi, khởi tạo quần thể, chọn lọc, lai ghép, đột biến | Cơ sở lý thuyết và cài đặt GA |
| **TV3** | Dữ liệu và thực nghiệm | Kịch bản thử nghiệm, `benchmark.py`, bảng và biểu đồ so sánh, chạy Turnitin | Thực nghiệm và so sánh |
| **TV4** | Giao diện | `gui/app.py`: form nhập mạng, chọn thuật toán, tham số, chạy, vẽ đồ thị luồng và đường hội tụ | Thiết kế chương trình và giao diện |
| **TV5** | WOA và thuyết trình | `woa.py`: mã hóa đường đi, cá voi bao vây, xoắn ốc bong bóng, tìm ngẫu nhiên; slide và demo | Cơ sở lý thuyết và cài đặt WOA |

---

## 4. Lịch tổng thể

| Giai đoạn | Thời gian | Mốc cần đạt |
|---|---|---|
| **Khởi động** | 17–18/11 | Chốt đề, giao việc, tạo repo |
| **Khung chung** | 19–23/11 | Giao diện solver, mã hóa đường đi và đồ thị mẫu chạy được *(Đã hoàn thành sớm ngày 17/11)* |
| **Code song song** | 24/11–05/12 | GA, WOA, kịch bản, GUI khung xong |
| **Tích hợp** | 06–08/12 | Merge toàn bộ vào main |
| **Hoàn thiện chương trình** | 09–13/12 | GUI nối thuật toán, thực nghiệm xong, Turnitin chạy thử |
| **Báo cáo và slide** | 10–16/12 | Báo cáo ghép, slide xong |
| **Chốt bản cuối** | 17/12 | Báo cáo kèm Turnitin, code main đóng băng |
| **Tập dượt** | 18–19/12 | Chạy thử demo và hỏi đáp |
| **Nộp** | 20/12 | Nộp sản phẩm |

---

## 5. Chi tiết từng bước

### Bước 0 – Họp khởi động (17–18/11)
- **Ai:** cả nhóm, TV1 chủ trì
- **Input:** file danh sách đề tài, mẫu báo cáo của giảng viên
- **Việc làm:**
  - Chốt giải cả hai dạng max flow và min cost flow trên cùng bộ kịch bản và các kịch bản dự kiến
  - Xác nhận với giảng viên việc dùng WOA làm thuật toán seminar
  - Giao việc theo bảng ở mục 3, thống nhất lịch họp
  - TV1 tạo repo và mời các thành viên
- **Output:** biên bản ngắn (dạng bài toán, phân công, lịch họp), repo có sẵn nhánh cho từng người
- **Hoàn thành khi:** mỗi người biết rõ việc, hạn và cách nộp của mình

### Bước 1 – Khung chung (19–23/11) *(TV1 đã hoàn thành xuất sắc ngày 17/11)*
- **Ai:** TV1
- **Input:** biên bản Bước 0
- **Việc làm:**
  - Viết `Graph` (nút, cạnh, capacity, cost, nguồn, đích, lượng luồng yêu cầu), kiểm tra dữ liệu đầu vào.
  - Viết `evaluate(flows)` (hỗ trợ cả max flow và min cost flow) và hàm kiểm tra tính hợp lệ.
  - Viết `PathEncoding` và `enumerate_paths`: mã hóa nghiệm theo đường đi giúp nghiệm tự động bảo toàn luồng và thỏa mãn sức chứa cạnh (`topup=False` mặc định).
  - Viết `solver_base.py` với giao diện chuẩn `evaluate_candidate`, `run(max_iter, max_evals)`, `budget_exhausted`, `history`, bộ đếm `eval_count`.
  - Cài đặt `RandomSolver` (solver giả) kiểm chứng thành công trên đồ thị mẫu S1_small.
  - Viết tài liệu quy chuẩn giao diện `docs/INTERFACE_SPEC.md` và kế hoạch `docs/PLAN.md` gửi cả nhóm.
- **Output:** `graph.py`, `fitness.py`, `paths.py`, `solver_base.py`, `dummy_solver.py`, đồ thị mẫu, kiểm thử tự động `test_framework.py` đạt 100%.

### Bước 2a – Cài GA (24/11–05/12)
- **Ai:** TV2
- **Input:** `Graph`, `SolverBase`, `PathEncoding`, tài liệu giao diện
- **Việc làm:**
  - Không gian tìm kiếm: vector $x$ kích thước `self.dim` trong khoảng `[self.lower, self.upper]`.
  - Khởi tạo quần thể trong $[lower, upper]$.
  - Chọn lọc (ví dụ tournament), lai ghép (crossover thực/blx-alpha), đột biến (Gaussian/uniform), giữ lại cá thể tốt (elitism).
  - Sau đột biến, kẹp nghiệm về biên: `candidate_x = self.encoding.clip(candidate_x)`.
  - Chấm điểm cá thể BẮT BUỘC qua `ev, flows = self.evaluate_candidate(x)`. Luôn tối đa hóa fitness (`ev.fitness > best_eval.fitness`).
  - Dừng theo ngân sách: `if self.budget_exhausted(max_evals): break`.
  - Thử trên đồ thị mẫu và các kịch bản TV3 giao.
- **Output:** `ga.py` kế thừa solver chung, trả `Result` kèm `history` và số lần đánh giá, ghi chú tham số mặc định.
- **Hoàn thành khi:** GA cho nghiệm hợp lệ trên mọi kịch bản ở cả hai dạng và tiệm cận đáp án tối ưu.
- **Hạn:** mốc giữa 29/11 (bản chạy được), bản cuối 05/12.

### Bước 2b – Cài WOA (24/11–05/12)
- **Ai:** TV5
- **Input:** `Graph`, `SolverBase`, `PathEncoding`, tài liệu giao diện
- **Việc làm:**
  - Không gian tìm kiếm: vector cá voi $x$ kích thước `self.dim` trong khoảng `[self.lower, self.upper]`.
  - Khởi tạo quần thể cá voi trong khoảng $[lower, upper]$.
  - Cài ba cơ chế cập nhật vị trí: bao vây con mồi, đường xoắn ốc bong bóng, tìm kiếm ngẫu nhiên; hệ số $a$ giảm dần từ 2 về 0.
  - Sau mỗi bước di chuyển, kẹp nghiệm về biên: `candidate_x = self.encoding.clip(candidate_x)`.
  - Chấm điểm cá thể BẮT BUỘC qua `ev, flows = self.evaluate_candidate(x)`. Luôn tối đa hóa fitness (`ev.fitness > best_eval.fitness`).
  - Dừng theo ngân sách: `if self.budget_exhausted(max_evals): break`.
  - Thử trên đồ thị mẫu và các kịch bản.
- **Output:** `woa.py` kế thừa solver chung, trả `Result` kèm `history` và số lần đánh giá, ghi chú tham số mặc định.
- **Hoàn thành khi:** WOA cho nghiệm hợp lệ trên mọi kịch bản ở cả hai dạng; giải thích được vì sao mỗi công thức trong code ứng với bước nào của thuật toán.
- **Hạn:** mốc giữa 29/11, bản cuối 05/12.

### Bước 2c – Kịch bản thử nghiệm (24–29/11)
- **Ai:** TV3
- **Input:** định dạng `Graph`, dạng bài toán đã chốt
- **Việc làm:** tạo 5–6 kịch bản khác nhau trong `data/scenarios/`, mỗi kịch bản có thêm chi phí trên cạnh và lượng luồng yêu cầu để chạy được cả max flow và min cost flow:
  - S1: mạng nhỏ (6–8 nút)
  - S2: mạng vừa (15–20 nút)
  - S3: mạng lớn (40–50 nút)
  - S4: có nút thắt cổ chai
  - S5: nhiều nguồn và nhiều đích
  - S6: kịch bản giao thông có đoạn tắc, mô phỏng bằng cách giảm sức chứa hoặc tăng chi phí của một số cạnh
  - Với mỗi kịch bản, tính đáp án tối ưu bằng thuật toán đối chứng (max flow: Edmonds-Karp; min cost flow: successive shortest path) làm mốc kiểm tra.
- **Output:** thư mục `data/scenarios/` (JSON), bảng mô tả kịch bản (số nút, cạnh, mục đích, đáp án tối ưu của từng dạng).
- **Hoàn thành khi:** TV2 và TV5 đọc được mọi kịch bản mà không sửa gì.
- **Hạn:** 29/11.

### Bước 2d – Giao diện khung (24/11–05/12)
- **Ai:** TV4
- **Input:** tài liệu giao diện solver, `RandomSolver` mẫu
- **Việc làm:**
  - Dùng PySide6 (được cộng điểm); nếu chưa kịp làm quen thì chuyển sang CustomTkinter làm phương án dự phòng (chốt trước 29/11).
  - Dựng form: nhập hoặc tải mạng, chọn dạng bài toán (max flow hoặc min cost flow), chọn thuật toán, nhập tham số (`max_iter`, `max_evals`, `seed`, ...), nút chạy, vùng hiển thị kết quả.
  - Chạy thử với `RandomSolver` của TV1.
- **Output:** `gui/app.py` chạy được với solver mẫu, bản phác giao diện để TV5 đưa vào slide.
- **Hoàn thành khi:** có thể chọn kịch bản, bấm chạy và thấy kết quả hiển thị trên form.
- **Hạn:** 05/12.

### Bước 3 – Tích hợp (06–08/12)
- **Ai:** TV1, có TV2 và TV5 hỗ trợ sửa lỗi
- **Input:** nhánh của TV2, TV3, TV4, TV5
- **Việc làm:** merge từng nhánh vào main, sửa chỗ lệch giao diện, chạy toàn bộ kịch bản với cả hai solver.
- **Output:** main chạy được cả hai thuật toán trên mọi kịch bản.
- **Hoàn thành khi:** chạy `main.py` không báo lỗi trên mọi kịch bản.
- **Hạn:** 08/12.

### Bước 4a – Nối giao diện (09–11/12)
- **Ai:** TV4
- **Input:** main đã merge
- **Việc làm:** nối form với GA và WOA ở cả hai dạng bài toán, vẽ đồ thị luồng trên mạng (dùng `best_flows`), vẽ đường hội tụ (dùng `history`), hiển thị số liệu (giá trị, thời gian, số lần đánh giá), xử lý nhập sai.
- **Output:** ứng dụng hoàn chỉnh, kèm hướng dẫn chạy ngắn.
- **Hoàn thành khi:** người chưa từng dùng chạy được một kịch bản và đọc được kết quả.
- **Hạn:** 11/12.

### Bước 4b – Thực nghiệm và so sánh (09–13/12)
- **Ai:** TV3, TV2 và TV5 hỗ trợ chỉnh tham số
- **Input:** main, các kịch bản
- **Việc làm:**
  - Mỗi kịch bản chạy cả hai thuật toán, ở cả hai dạng bài toán, ít nhất 20 lần, cùng ngân sách đánh giá (`max_evals`) và cùng seed.
  - Ghi: giá trị tốt nhất (tổng luồng hoặc tổng chi phí tùy dạng), trung bình, độ lệch chuẩn, tỷ lệ nghiệm hợp lệ, thời gian chạy, số vòng hội tụ.
  - Khảo sát ảnh hưởng của tham số chính (kích thước quần thể, tỷ lệ đột biến của GA, hệ số xoắn ốc của WOA).
  - Rút nhận xét: thuật toán nào thắng ở tình huống nào và vì sao.
- **Output:** bảng kết quả, biểu đồ hội tụ và so sánh, đoạn nhận xét ngắn cho từng kịch bản.
- **Hoàn thành khi:** mọi số liệu trong báo cáo tái lập được bằng `benchmark.py`.
- **Hạn:** 13/12.

### Bước 5 – Viết phần báo cáo (viết dần, hạn 15/12)
- **Ai:** mỗi thành viên viết phần mình ở mục 3
- **Input:** code và kết quả của phần mình, mẫu báo cáo của giảng viên
- **Việc làm:** viết lý thuyết, cách cài đặt, giải thích các đoạn code quan trọng và kết quả; không dán ảnh trơn, mỗi hình có đoạn giải thích.
- **Output:** file phần báo cáo của mỗi người trong `report/`.
- **Hạn:** lý thuyết và cài đặt xong trước 10/12, phần thực nghiệm xong trước 15/12.

### Bước 6 – Ghép báo cáo (16/12)
- **Ai:** TV1
- **Input:** các phần báo cáo trong `report/`
- **Việc làm:** ghép theo mẫu, thống nhất văn phong, đánh số hình và bảng, viết phần giới thiệu và kết luận, thêm tài liệu tham khảo.
- **Output:** báo cáo hoàn chỉnh trước kiểm tra đạo văn.
- **Hạn:** 16/12.

### Bước 7 – Kiểm tra đạo văn (chạy thử 12/12, bản cuối 17/12)
- **Ai:** TV3 chạy, TV1 sửa cùng người viết phần bị trùng
- **Input:** bản nháp báo cáo
- **Việc làm:** chạy Turnitin với cấu hình mặc định, mục tiêu dưới 30%, diễn đạt lại đoạn trùng, chạy lại lần cuối, nối kết quả vào cuối báo cáo.
- **Output:** báo cáo cuối kèm kết quả Turnitin.
- **Hoàn thành khi:** tỷ lệ trùng dưới 30%.
- **Hạn:** chạy thử 12/12 để còn kịp sửa; bản cuối 17/12.

### Bước 8 – Slide và demo (13–17/12)
- **Ai:** TV5, các thành viên góp nội dung phần mình
- **Input:** báo cáo, kết quả thực nghiệm, ứng dụng
- **Việc làm:** làm slide cô đọng; chọn kịch bản demo và viết kịch bản ngắn (bấm gì, nói gì, kết quả mong đợi); chuẩn bị bản ghi hình dự phòng khi demo trực tiếp lỗi.
- **Output:** slide, kịch bản demo, bản ghi hình dự phòng.
- **Hạn:** 17/12.

### Bước 9 – Tập dượt và nộp (18–20/12)
- **Ai:** cả nhóm
- **Input:** slide, ứng dụng, báo cáo cuối
- **Việc làm:** tập thuyết trình một lượt đầy đủ, mỗi người chuẩn bị trả lời 3–5 câu hỏi về phần mình, thống nhất cách trả lời câu hỏi chéo, nộp sản phẩm.
- **Output:** danh sách câu hỏi và trả lời dự kiến, bộ sản phẩm nộp (code, báo cáo, slide).
- **Hạn:** tập dượt 18–19/12, nộp 20/12.

---

## 6. Mốc của từng thành viên

| TV | Mốc và hạn |
|---|---|
| **TV1** | 18/11 biên bản và repo · 23/11 khung chung *(xong 17/11)* · 08/12 merge xong · 16/12 ghép báo cáo · 17/12 báo cáo cuối |
| **TV2** | 29/11 GA chạy được · 05/12 GA bản cuối · 10/12 phần lý thuyết và cài đặt GA · 13/12 hỗ trợ chỉnh tham số |
| **TV3** | 29/11 bộ kịch bản · 12/12 Turnitin chạy thử · 13/12 thực nghiệm xong · 15/12 phần thực nghiệm báo cáo · 17/12 Turnitin bản cuối |
| **TV4** | 05/12 form khung · 11/12 GUI hoàn chỉnh · 15/12 phần giao diện báo cáo · 17/12 hướng dẫn chạy |
| **TV5** | 29/11 WOA chạy được · 05/12 WOA bản cuối · 10/12 phần lý thuyết và cài đặt WOA · 17/12 slide và kịch bản demo |

---

## 7. Cấu trúc báo cáo gợi ý
*(Đối chiếu với mẫu của giảng viên; nếu mẫu khác thì theo mẫu)*
1. Giới thiệu và phát biểu bài toán Network flow (TV1)
2. Cơ sở lý thuyết và cài đặt GA (TV2)
3. Cơ sở lý thuyết và cài đặt WOA (TV5)
4. Thiết kế chương trình và giao diện, nêu rõ thư viện PySide6 và lý do chọn (TV4, TV1)
5. Thực nghiệm và so sánh hiệu năng (TV3)
6. Kết luận và hướng phát triển (TV1)
7. Tài liệu tham khảo
8. Phụ lục: kết quả Turnitin

---

## 8. Rủi ro và cách xử lý

| Rủi ro | Cách xử lý |
|---|---|
| **Khung chung trễ, các nhánh lệch giao diện** | TV1 xong Bước 1 đúng hạn 23/11 *(thực tế đã xong ngày 17/11)*; ai thấy giao diện không đủ dùng thì báo trong buổi họp, không tự sửa |
| **Nghiệm ngẫu nhiên vi phạm bảo toàn luồng** | Đã giải quyết triệt để bằng mã hóa theo đường đi `PathEncoding`: nghiệm tự động bảo toàn luồng và thỏa mãn sức chứa cạnh |
| **So sánh GA và WOA không công bằng** | Dùng chung fitness, cách mã hóa đường đi, seed và ngân sách số lần đánh giá (`max_evals`) |
| **Dồn việc báo cáo về cuối** | Mỗi người viết phần mình ngay khi code xong (hạn 10/12 cho phần lý thuyết và cài đặt) |
| **Turnitin vượt 30%** | Chạy thử sớm ngày 12/12, diễn đạt lại bằng lời của nhóm, không chép lại tài liệu |
| **Demo lỗi trong lúc trình bày** | Có bản ghi hình dự phòng và một kịch bản nhỏ chạy nhanh |
| **Thành viên bận hoặc trễ hạn** | Báo trong buổi họp gần nhất; TV1 phân lại việc hoặc nhờ người khác hỗ trợ |

---

## 9. Điều cần xác nhận
- Hạn 20/12 là hạn nộp cuối của toàn bộ sản phẩm. Nếu buổi thuyết trình diễn ra sớm hơn thì dời các mốc từ Bước 6 trở đi lên tương ứng.
- Dạng bài toán gồm cả max flow và min cost flow, làm cùng lúc; khối lượng ở Bước 2c, 4b và báo cáo tăng lên, nếu đến mốc 05/12 còn trễ thì nhóm họp lại để chốt phạm vi.
- Mẫu báo cáo của giảng viên chưa có trong tài liệu đã đọc nên cấu trúc ở mục 7 chỉ là gợi ý.
- Việc dùng WOA làm thuật toán seminar nên xác nhận với giảng viên.
