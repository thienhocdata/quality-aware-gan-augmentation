# Đối chiếu phiên bản

- **3D-HyperGAMO:** code baseline nằm trong `archive/legacy/KLTN_3D_HyperGAMO_Protocol_B_v2.ipynb`; architecture C/G/D dùng tham chiếu worker M4 đã chạy, không đồng nhất trajectory với B cũ.
- **Protocol B:** giữ notebook nguyên byte. `protocol_b.ipynb` chạy kiểm tra cache CPU, không tự chia lại mẫu hoặc refit preprocessing.
- **V2:** nguồn smoke, runner, technical được trích thành `src/frozen/smoke.py`, `runner.py`, `technical.py`; launcher giữ riêng, không nạp vào kernel CPU.
- **M4:** ba bản notebook chia lô dùng cùng worker SHA; giữ **một** worker `src/frozen/m4.py` và các launcher riêng; notebook M4 là registry, không train.
- **Test:** `src/frozen/test.py` và `src/frozen/test_launcher.py` là source đã cung cấp; đã đối chiếu hash với worker trong ZIP test thực tế: SHA-256 53673fd9ce094121431164053cc6cf990c7f4f4800aa599fc8ed88b86f88e4b9.
- **Selector:** `weighting.py`, `endpoint.py`, `endgame.py` là snapshot định nghĩa (bỏ các cell auto-run). Forward chỉ thuộc workflow riêng khi phê duyệt.
- **CPU hậu nghiệm:** `M4_V2_diagnostics_3_seeds_CPU.ipynb` và `M5_V2_CPU_analysis.ipynb` giữ trong archive; không sử dụng làm kết quả mới.
- **Leakage/Pavia:** giữ trong archive để chứng minh quá trình quyết định và giới hạn, không coi là cải tiến của V2.

**Hai cặp `(1)` có SHA trùng hệt bản không hậu tố; archive chỉ giữ một bản tên không hậu tố, manifest ghi bản lặp.**

Không thực hiện xóa file gốc trên ổ người dùng. Không thể xác nhận raw train/test/selector ZIP hoặc checkpoint chỉ từ notebook source.
