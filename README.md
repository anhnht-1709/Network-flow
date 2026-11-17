# 🌊 ỨNG DỤNG THUẬT TOÁN GA VÀ WOA CHO BÀI TOÁN NETWORK FLOW
> **Đồ án cuối kỳ môn Trí tuệ nhân tạo — Nhóm 8**  
> **Giảng viên hướng dẫn:** Bộ môn Trí tuệ nhân tạo  

---

## 📌 1. Giới thiệu đề tài

Dự án tập trung nghiên cứu, cài đặt lại từ đầu (from scratch) và so sánh thực nghiệm hiệu năng giữa hai thuật toán tiến hóa & tối ưu hóa bầy đàn:
1. **Thuật toán Di truyền (Genetic Algorithm - GA)**
2. **Thuật toán Tối ưu hóa Bầy Cá Voi (Whale Optimization Algorithm - WOA)**

cho bài toán **Tối ưu luồng trong mạng (Network Flow)** ở cả hai dạng:
- **Max Flow:** Tối đa hóa tổng lưu lượng luồng từ nguồn ($source$) đến đích ($sink$).
- **Min Cost Flow:** Đạt đúng lượng luồng yêu cầu ($required\_flow$) với tổng chi phí trên các cạnh là nhỏ nhất.

Ứng dụng được xây dựng hoàn chỉnh với bộ khung giải thuật (Core Engine), bộ kịch bản thực nghiệm đa dạng (Simulations/Benchmarks) và giao diện trực quan dạng Form viết bằng **PySide6**.

---

## 📂 2. Cấu trúc thư mục dự án

Dự án áp dụng kiến trúc phân tầng (Layered Architecture) và triết lý **Skeleton Code** (bộ khung định sẵn giao diện chuẩn, type hints và docstrings chi tiết):

```text
AI_NHOM8/
├── core/                       # [TẦNG CỐT LÕI] Thuật toán và mô hình toán học
│   ├── __init__.py             # Export các class và hàm công khai
│   ├── graph.py                # Lớp Graph, Edge, ma trận liên thuộc, nạp/xuất JSON
│   ├── fitness.py              # Hàm evaluate(), Evaluation, kiểm tra ràng buộc & tính điểm
│   ├── paths.py                # enumerate_paths, PathEncoding (Mã hóa nghiệm theo đường đi)
│   ├── solver_base.py          # Lớp cha SolverBase, Result (chuẩn hóa API cho mọi thuật toán)
│   ├── dummy_solver.py         # RandomSolver mẫu chuẩn để kiểm thử và đối chiếu
│   ├── ga.py                   # [TV2] Cài đặt thuật toán GA (Khởi tạo, Chọn lọc, Lai ghép, Đột biến)
│   └── woa.py                  # [TV5] Cài đặt thuật toán WOA (Bao vây, Xoắn ốc, Tìm kiếm ngẫu nhiên)
├── data/scenarios/             # [TẦNG DỮ LIỆU] Chứa các kịch bản mạng thử nghiệm (JSON)
│   └── s1_small.json           # Kịch bản mạng mẫu 4 nút kèm đáp án tính tay
├── experiments/                # [TẦNG THỰC NGHIỆM] Kịch bản chạy thử nghiệm và so sánh
│   └── benchmark.py            # [TV3] Chạy thực nghiệm 20 lần/kịch bản, xuất bảng và vẽ biểu đồ hội tụ
├── gui/                        # [TẦNG GIAO DIỆN] Ứng dụng Desktop trực quan
│   └── app.py                  # [TV4] Giao diện Form PySide6 (chọn mạng, cấu hình, chạy & vẽ đồ thị luồng)
├── report/                     # Báo cáo học phần và phụ lục Turnitin
├── docs/                       # Tài liệu kỹ thuật dự án
│   ├── PLAN.md                 # Kế hoạch phân công công việc chi tiết
│   └── INTERFACE_SPEC.md       # Quy chuẩn giao diện Solver & hướng dẫn kỹ thuật
├── requirements.txt            # Danh sách thư viện phụ thuộc
├── main.py                     # Điểm khởi chạy nhanh ứng dụng
└── test_framework.py           # Bộ kiểm thử tự động khung chung
```

---

## 👥 3. Phân công thành viên & Lộ trình thực hiện

| Thành viên | Vai trò | Module phụ trách chính | Mốc hạn hoàn thành |
|---|---|---|---|
| **TV1 (Nhóm trưởng)** | Kiến trúc & Tích hợp | `graph.py`, `fitness.py`, `paths.py`, `solver_base.py`, merge code, ghép báo cáo | **17/11 (Xong khung chung)** · 08/12 (Merge) · 17/12 (Báo cáo cuối) |
| **TV2** | Thuật toán GA | `core/ga.py`: Triển khai đầy đủ các toán tử GA theo đường đi | 29/11 (Bản chạy được) · **05/12 (Bản cuối)** · 10/12 (Báo cáo GA) |
| **TV3** | Dữ liệu & Thực nghiệm | `data/scenarios/`, `experiments/benchmark.py`, chạy Turnitin | 29/11 (Bộ kịch bản) · **13/12 (Thực nghiệm xong)** · 17/12 (Turnitin < 30%) |
| **TV4** | Giao diện người dùng | `gui/app.py`: Dựng form PySide6, vẽ đồ thị luồng và đường hội tụ | 05/12 (Khung form) · **11/12 (GUI hoàn chỉnh)** · 15/12 (Báo cáo GUI) |
| **TV5** | Thuật toán WOA & Demo | `core/woa.py`, Slide thuyết trình và kịch bản demo | 29/11 (Bản chạy được) · **05/12 (Bản cuối)** · 17/12 (Slide & Video demo) |

---

## ⚙️ 4. Hướng dẫn thiết lập môi trường (Setup)

> **Kho lưu trữ chính thức:** [https://github.com/anhnht-1709/Network-flow](https://github.com/anhnht-1709/Network-flow)  
> **Yêu cầu:** Máy tính đã cài đặt **Python 3.10+** (khuyến nghị Python 3.11 – 3.14).

### 4.1. Clone mã nguồn về máy
```bash
git clone https://github.com/anhnht-1709/Network-flow.git
cd Network-flow
```

### 4.2. Tạo môi trường ảo (Virtual Environment)

- **Trên Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install -r requirements.txt
  ```

- **Trên Windows (Command Prompt):**
  ```cmd
  python -m venv .venv
  .venv\Scripts\activate.bat
  pip install -r requirements.txt
  ```

- **Trên macOS / Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```

### 4.3. Chạy kiểm tra bộ khung mẫu
Sau khi kích hoạt môi trường ảo, chạy thử lệnh:
```bash
python main.py
python test_framework.py
```
Nếu màn hình xuất hiện thông báo `TẤT CẢ CÁC BÀI TEST ĐỀU ĐẠT THÀNH CÔNG 100%!`, môi trường của bạn đã sẵn sàng!

---

## 5. Quy trình làm việc với Git & GitHub dành cho thành viên

> ⚠️ **NGUYÊN TẮC**  
> 1. **TUYỆT ĐỐI KHÔNG commit trực tiếp lên nhánh `main`**.  
> 2. Mỗi thành viên chỉ làm việc trên nhánh riêng của mình (`tv2`, `tv3`, `tv4`, `tv5`).  
> 3. Chỉ sửa đúng các hàm/file được phân công, không sửa file của thành viên khác.

### 5.1. Khi bắt đầu nhận việc (Làm lần đầu)
```bash
# 1. Chuyển về main và kéo code mới nhất
git switch main
git pull origin main

# 2. Tạo nhánh riêng của bạn (thay X bằng số thứ tự của bạn, vd: tv2, tv5)
git switch -c tvX
```

### 5.2. Trong quá trình viết code (Mỗi ngày)
Mỗi khi bắt đầu ngày làm việc mới hoặc trước khi code tiếp, hãy cập nhật nhánh của bạn với code mới nhất từ `main`:
```bash
git switch main
git pull origin main
git switch tvX
git merge main
```
*Kiểm tra lại xem bạn đang ở đúng nhánh chưa bằng lệnh: `git branch` (dấu `*` phải nằm ở nhánh `tvX`).*

### 5.3. Commit và Push code lên GitHub
Khi hoàn thành một phần việc hoặc cuối buổi code:
```bash
# 1. Kiểm tra những file bạn đã sửa
git status

# 2. Chỉ add đúng file bạn phụ trách (ví dụ TV2)
git add core/ga.py
git commit -m "TV2: Hoan thien khoi tao quan the va toan tu crossover"

# 3. Đẩy nhánh lên GitHub
git push -u origin tvX
```

### 5.4. Tạo Pull Request (PR) để nộp code
1. Truy cập vào kho lưu trữ GitHub của nhóm.
2. Bạn sẽ thấy thanh màu vàng thông báo nhánh `tvX` vừa được push -> Nhấn nút **"Compare & pull request"**.
3. Đảm bảo nhánh đích: `base: main` $\leftarrow$ `compare: tvX`.
4. Đặt tiêu đề rõ ràng: `TV2: Hoàn thành thuật toán GA`.
5. Nhấn **"Create pull request"**.
6. Báo nhóm trưởng (TV1) review và merge vào `main`.

---

## 6. Quy ước lập trình cho TV2 (GA) và TV5 (WOA)

Các thuật toán `core/ga.py` và `core/woa.py` đều kế thừa `SolverBase` và tuân thủ các quy tắc cốt lõi:
1. **Không gian nghiệm:** Nghiệm $x$ là vector lượng luồng trên các đường đi đơn có độ dài `self.dim`. Miền giá trị cá thể nằm trong `[self.lower, self.upper]`.
2. **Kẹp biên nghiệm:** Sau khi lai ghép, đột biến hoặc bơi theo xoắn ốc, luôn kẹp nghiệm về biên:
   ```python
   candidate_x = self.encoding.clip(candidate_x)
   ```
3. **Đánh giá cá thể:** BẮT BUỘC chấm điểm qua phương thức chuẩn duy nhất:
   ```python
   ev, flows = self.evaluate_candidate(candidate_x)
   ```
   *(Hàm tự động decode $x \to flows$ với `topup=False`, chấm điểm và tăng bộ đếm đánh giá).*
4. **Mục tiêu tối ưu:** Thuật toán luôn **TỐI ĐA HÓA fitness** (cập nhật nghiệm tốt nhất khi `ev.fitness > best_eval.fitness`).
5. **Dừng theo ngân sách:** Luôn kiểm tra điều kiện dừng:
   ```python
   if self.budget_exhausted(max_evals):
       break
   ```
6. **Code mẫu chi tiết:** Xem trực tiếp tại [`core/dummy_solver.py`](core/dummy_solver.py).

---

## 7. Lệnh cứu nguy khi thao tác Git

- **Lỡ sửa nhầm file ngoài phân công:**
  ```bash
  git restore path/to/file_sua_nham.py
  ```
- **Hủy toàn bộ thay đổi chưa commit để quay về trạng thái sạch:**
  ```bash
  git restore .
  ```
- **Xem lịch sử các commit gần nhất:**
  ```bash
  git log --oneline -n 5
  ```

---
*Mọi thắc mắc về API hoặc tích hợp, các thành viên liên hệ ngay với TV1 (Nhóm trưởng) để được hỗ trợ.*
