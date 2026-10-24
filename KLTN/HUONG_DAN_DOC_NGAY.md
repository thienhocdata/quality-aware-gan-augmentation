# Đọc bộ đã bổ sung output

**Trạng thái mới nhất (23/10/2026):** mở `docs/SOURCE_COMPLETENESS_AUDIT_2026-10-23.md` trước khi dùng source. File `CURRENT_PROJECT_MANIFEST.json` là manifest hiện hành; `manifest.json` giữ lại inventory của bản đóng gói 17/10.

**Cập nhật 19–23/10/2026:** Phần 6 đọc tại `notebooks/selector.ipynb` và `docs/PHAN_6_KET_QUA_SELECTOR.md`. Endpoint ablation sáu C đã COMPLETE. Source endpoint cuối và source V3-A đã được gom vào `src/frozen`; output phân tích nhẹ nằm ở bundle 07–08. Notebook master chưa được viết lại để hiển thị V3-A.

## Một notebook chính

Mở `notebooks/00_MASTER_CURRENT.ipynb`. Notebook đã có output hiển thị từ lượt kiểm CPU và bao gồm endpoint ablation lẫn V3-A. `00_MASTER.ipynb` được giữ như bản master 17/10. Có thể đọc notebook hiện hành ngay mà không Run All.

Run All mặc định chỉ kiểm hash, đọc kết quả đã lưu và tính lại bảng/hình trên CPU. Không train, không khởi tạo model, không chạy lại test. Sáu notebook còn lại là các giao diện theo phần và cũng đã chạy kiểm CPU.

## Kaggle

1. Import riêng `00_MASTER_CURRENT.ipynb`.
2. Gắn `KLTN_clean_OUTPUTS_FOR_KAGGLE.zip` làm một Input. Hỗ trợ ZIP nguyên hoặc Kaggle tự giải nén.
3. Giữ GPU tắt, `RUN_ENDGAME_TRAIN_ONLY=False`; Run All để dựng báo cáo mới từ output cũ.

## Mốc 7.500 nằm ở đâu?

Ở **Phần 7 — chẩn đoán 7.500→8.000**. Sáu checkpoint đã có trong `checkpoints/03_ENDGAME_LEARNED_LOCAL_6_CKPTS.zip`.

Phần này còn PENDING vì chưa có lượt forward sáu checkpoint hoàn tất. Khi thực hiện sau này trên Kaggle GPU: gắn thêm ZIP checkpoint này, rồi bật riêng `RUN_ENDGAME_TRAIN_ONLY=True`. Chỉ forward G/heads trên train, không train hoặc test. Chưa thực thi phần này trong lần đóng gói.

## Các thư mục

- `notebooks/`: một notebook tổng + sáu notebook theo phần, có output kiểm CPU.
- `src/frozen/`: source thí nghiệm nguyên bản, khóa bằng SHA-256.
- `data/`: full cache/provenance Protocol B.
- `checkpoints/`: raw M4 đủ 12 job và gói sáu checkpoint dành cho Endgame.
- `outputs/bundles/`: tám ZIP output gộp theo phần.
- `outputs/technical/`: hai báo cáo smoke/matched runner đã PASS.
- `outputs/master_reviews/`: bảng/hình/báo cáo tổng hợp vừa dựng từ dữ liệu đã lưu.
- `docs/`: hồ sơ nguồn, kiểm CPU, migration và giới hạn.
- `archive/legacy/`: notebook lịch sử để đối chiếu, không chạy tự động.

Bản ZIP đầy đủ khá lớn vì giữ cả dữ liệu và checkpoint thật. Gói OUTPUTS_FOR_KAGGLE nhỏ hơn, dùng để đọc/tổng hợp kết quả mà không cần đưa model lên Kaggle.
