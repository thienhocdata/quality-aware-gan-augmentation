**Bản xuất bản 24/10/2026:** đọc [hướng dẫn bản đầy đủ](README_RELEASE_2026-10-24.md). Notebook hiện hành là `notebooks/00_MASTER_CURRENT.ipynb`.

# Bản đã bổ sung output và kiểm CPU

**Yêu cầu nghiên cứu hiện hành — 24/10/2026:** đọc [tiêu chuẩn giá trị nghiên cứu của người dùng](docs/YEU_CAU_GIA_TRI_NGHIEN_CUU_2026-10-24.md) trước khi đề xuất hướng mới. Kết quả cuối phải có lý do sử dụng trước các phương án phù hợp hiện có; tăng điểm so với bản cũ hoặc đủ tiêu chuẩn khóa luận không tự chứng minh giá trị.

**Cập nhật source 23/10/2026:** source trục chính đã được kiểm kê và gom an toàn. Đọc `docs/SOURCE_COMPLETENESS_AUDIT_2026-10-23.md` và `docs/PROJECT_STRUCTURE_2026-10-23.md`. Registry hiện hành nằm trong `docs/provenance/current/`; `manifest.json` là snapshot đóng gói cũ ngày 17/10.

Bắt đầu tại `HUONG_DAN_DOC_NGAY.md` và `notebooks/00_MASTER.ipynb`.

# Indian Pines — KLTN

Project được sắp lại từ 17 notebook do người dùng cung cấp. **Không train, không test inference, không khôi phục/checkpoint GPU khi mở notebook.**

## Notebook chính

| Notebook | Mục đích | Output CPU |
|---|---|---|
| `notebooks/hypergamo.ipynb` | Kiến trúc 3D-HyperGAMO và mã chuẩn của M4 | Kiểm source trên màn hình |
| `notebooks/protocol_b.ipynb` | Cache, split, tiền xử lý, giới hạn leakage | Kiểm cấu trúc cache trên màn hình |
| `notebooks/v2.ipynb` | V2 và các cổng kiểm kỹ thuật | Tình trạng source và report |
| `notebooks/m4.ipynb` | Worker và registry train M4 (4×3) | Contract/registry nếu cung cấp ZIP |
| `notebooks/test.ipynb` | Kiểm kết quả test đã lưu | `outputs/test/summary.json` nếu có report |
| `notebooks/selector.ipynb` | Phần 6: M4, hành vi weighting và endpoint ablation đã COMPLETE | Bảng gốc, mean ± SD, paired/class analysis; không train |

## Thư mục

- `src/frozen/`: source worker + launcher **giữ nguyên nội dung từ notebook**, có hash trong `manifest.json`. M4 dùng source SHA-256 `6d33ca6a0d39600f5f44c0d7e63547b1237149d625471ac73315d982ffbd8fb5`.
- `src/canonical/`: vùng code đồng bộ cho thí nghiệm tương lai; hiện chưa thay thế source frozen.
- `src/results.py`: phân tích CPU trên bảng gốc, xác minh mean ± sample SD và chênh lệch seed.
- `src/audit.py`: tiện ích đọc/đối chiếu trên CPU, là code quản lý **mới**, không phải worker đã train.
- `archive/legacy/`: notebook gốc, byte-for-byte; giữ làm bằng chứng, không chạy trực tiếp.
- `data/`, `checkpoints/`, `outputs/`: để dữ liệu thực tế; Bản WITH_OUTPUTS đã kèm dữ liệu/checkpoint và các output theo phần; bản REPORTS_ONLY giữ source + output nhẹ.
- `docs/scope.md`: trục đã chốt, gồm file canonical và quy tắc dọn.
- `docs/protocol_review.md`: giới hạn Protocol B.
- `manifest.json`: danh sách source, SHA và bản trùng.

## Khai báo đầu vào

`inputs.json` đã khai báo sẵn các gói trong project bằng đường dẫn tương đối; không cần sửa khi giải nén nguyên bộ. `m4_zips` là danh sách đường dẫn ZIP train; `protocol_b`, `test`, `weighting`, `endgame`, `technical` là đường dẫn file/thư mục hoặc `null`; `directories` là thư mục bổ sung để tìm báo cáo. Đường dẫn Windows sử dụng `\\` trong JSON. Có thể chạy notebook mà không có output: nó sẽ thông báo `SKIP` chứ không tự tạo số liệu.

## Cách chạy

1. Giải nén toàn bộ project. Mở folder `KLTN_clean` trong VS Code/Jupyter và chọn kernel Python 3.
2. Chạy từng notebook chính theo thứ tự. `Run All` chỉ dùng CPU + đọc file. Đầu vào chưa có sẽ hiển thị `SKIP`.
3. Khi cần kiểm output, cập nhật các đường dẫn một lần trong `inputs.json`. Không đổi config gốc hoặc source frozen.
4. Muốn sử dụng GPU cho **thí nghiệm mới**: tạo workflow/launcher mới có phê duyệt riêng, không thêm lệnh GPU vào `Run All`.

## Quy tắc version

- `FROZEN`: nội dung/trạng thái thực nghiệm đã xác minh, sửa phải tạo revision mới.
- `SOURCE_LOCKED`: code đã khóa nhưng ZIP output gốc chưa được tích hợp vào project.
- `PENDING`: bước kỹ thuật chưa có bằng chứng hoàn tất.
- `IN_PROGRESS`: nhánh phân tích tiếp theo, không sửa dữ liệu M4/test.

**Không đổi tên archive gốc, không xóa bản duy nhất hoặc suy luận PASS từ sự tồn tại của notebook.** Các source sử dụng global state và môi trường Kaggle nguyên bản được bảo tồn để audit; không được coi là thư viện có thể import tự do. Mọi đường dẫn file trong output mang ý nghĩa nguồn, không chỉ tên gợi nhớ.

## Tính đầy đủ

Bản tích hợp giữ các ZIP gốc/bản gộp đã kiểm trong data/, checkpoints/ và outputs/bundles/. Mốc 3 thiếu report riêng, Endgame chưa hoàn tất; không train/test lại để hợp thức hóa. Source của Protocol B được giữ nguyên trong archive vì chia ra thành module mới có thể làm thay đổi trình tự thực thi và semantics. `protocol_b.ipynb` là giao diện kiểm đọc cache, không phải bản thực thi B mới.

## Output đã bổ sung

Xem docs/integration_review.md. Dùng notebooks/00_MASTER.ipynb để đọc toàn bộ trục trong một notebook, hoặc sáu notebook theo phần. Train/test model đã hoàn tất tiếp tục được khóa; phần Endgame 7.500→8.000 mặc định tắt.

## Cập nhật Phần 6 — 19/10/2026

Xem `docs/PHAN_6_KET_QUA_SELECTOR.md` và phần mới ở cuối `notebooks/selector.ipynb`. Endpoint ablation sáu C đã hoàn tất, được kiểm từ output thật; gói phân tích nhẹ nằm ở `outputs/bundles/07_SELECTOR_ENDPOINT_ABLATION_ANALYSIS_20261019.zip`. Chẩn đoán trajectory G/head 7.500→8.000 vẫn là bước lịch sử riêng chưa có report hoàn tất. Bản local mới không tự cập nhật `00_MASTER.ipynb`, các ZIP cũ hoặc GitHub Release đã xuất.

## Cập nhật V3-A và provenance — 23/10/2026

Source thực thi V3-A và runner endpoint ablation cuối đã được bổ sung vào `src/frozen/`. Kết quả phân tích V3-A nằm ở `outputs/bundles/08_V3A_RESULTS_ANALYSIS_20261019.zip`. Không train, inference hoặc test model trong lần bổ sung này.
