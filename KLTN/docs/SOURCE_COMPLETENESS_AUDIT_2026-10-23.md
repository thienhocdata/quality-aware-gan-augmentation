# Kiểm kê source trục chính — 23/10/2026

## Kết luận

**Source cần cho trục nghiên cứu hiện tại đã đủ trên máy.** Sau khi bổ sung các bản còn nằm trong output, `KLTN_clean` hiện chứa đủ source cho:

1. tái triển khai 3D-HyperGAMO lịch sử;
2. Protocol B `50e988fae260`;
3. kiểm tra spatial overlap/leakage;
4. smoke và technical checks V2;
5. M4 bốn nhánh × ba seed;
6. test 12 classifier;
7. chẩn đoán learned weighting;
8. endpoint ablation sáu classifier;
9. V3-A distance-bias.

Kiểm tự động ghi nhận **22/22 source tồn tại, khớp SHA-256 và qua kiểm cú pháp/JSON**. Registry artifact ghi nhận **13/13 gói/notebook cốt lõi tồn tại và đúng kích thước**, các gói nhỏ được băm lại; ba archive lớn giữ hash từ lần audit byte-level trước.

Việc này xác nhận **tính đầy đủ của source**, không tự động xác nhận tính mới, chất lượng khoa học hay khả năng tái lập bit-for-bit trên môi trường GPU khác.

## Source chuẩn theo giai đoạn

| Giai đoạn | Source chuẩn trong project | Trạng thái |
|---|---|---|
| Tái triển khai lịch sử | `archive/executed/3DHyperGAMO_IndianPines_Paper_Reproduction.ipynb`, `src/frozen/paper_reproduction/` | Đủ source; protocol tách biệt khỏi B/M4 |
| Protocol B | `archive/executed/Protocol_B_executed_V8_355611502.ipynb` | Snapshot đã chạy, hash `a4b3e9ca...` |
| Tham chiếu Protocol B | `archive/legacy/KLTN_3D_HyperGAMO_Protocol_B_v2.ipynb` | Dùng đọc source; snapshot V8 ưu tiên khi mô tả run cũ |
| Leakage/overlap | `archive/legacy/KLTN_KiemChung_RoRiKhongGian_ReviewNhom.ipynb` | Source audit hỗ trợ; không phải worker train chính |
| V2 smoke | `src/frozen/smoke.py` | Source PASS đã khóa |
| Matched runner | `src/frozen/runner.py` | Source PASS đã khóa |
| Technical check | `src/frozen/technical.py` | Source còn; tên report riêng chưa truy ra |
| M4/V2 training | `src/frozen/m4.py` | Source thực thi chuẩn, SHA `6d33ca6a...` |
| Test 12 classifier | `src/frozen/test.py` | Source thực thi chuẩn, SHA `53673fd9...` |
| Weighting 8.000 | `src/frozen/weighting.py` | Phân tích ba seed hoàn tất |
| Endgame 7.500→8.000 | `src/frozen/endgame.py` | Source + sáu checkpoint có; chưa có report forward COMPLETE |
| Endpoint ablation | `src/frozen/selector_endpoint_ablation/endpoint_ablation_runner_clean_r3.py` | **Source thực thi chuẩn**, SHA `5d1e0adf...` |
| V3-A | `src/frozen/v3a/v3a_worker.py`, `v3a_orchestrator.py` | Source thực thi chuẩn, run ba seed hoàn tất |

## Cảnh báo phiên bản quan trọng

`src/frozen/endpoint.py` **không phải** runner của endpoint ablation sáu classifier. Hai file có hash khác nhau. File cũ được giữ nguyên để bảo toàn provenance; source endpoint cuối nằm trong thư mục `src/frozen/selector_endpoint_ablation/`.

Source M4 bên trong payload V3-A khớp byte với `src/frozen/m4.py`. Vì vậy V3-A đang kế thừa đúng implementation M4 đã khóa, thay vì dùng một bản M4 khác.

## Những khoảng trống còn lại

Các mục sau **không phải thiếu source cốt lõi**:

- Endgame 7.500→8.000 chưa có report forward hoàn tất. Đây là thiếu một kết quả chẩn đoán tùy chọn.
- Report Mốc 3/technical độc lập chưa xác định tên; source technical vẫn còn.
- Chưa có một environment lock duy nhất tái tạo bit-for-bit mọi phiên Kaggle. Phiên bản runtime vẫn được ghi trong contract/report từng run.
- Protocol B vẫn được bảo tồn dưới dạng notebook đã chạy; chưa tách thành module dùng chung có parity test.

Không cần train lại chỉ để lấp các khoảng trống quản lý này.

## Thay đổi quản lý đã thực hiện

- Sao chép nguyên byte source paper reproduction và snapshot Protocol B vào project canonical.
- Trích đúng runner endpoint từ backup hoàn tất; không sửa nội dung.
- Sao chép worker/orchestrator V3-A cùng provenance nhỏ vào `src/frozen/v3a/`.
- Bổ sung gói phân tích V3-A nhẹ thành bundle số 08.
- Tạo registry JSON/CSV có hash, vai trò và trạng thái bằng `tools/rebuild_registry.py`.

**Không artifact nghiên cứu nào bị xóa hoặc ghi đè.** Sau khi kiểm hash, 29 item rời/nhánh cũ được di chuyển có thể hoàn tác vào `../KLTN_archive_legacy` hoặc kho `../outputs`; manifest vị trí cũ/mới nằm trong archive. Hai cache tái sinh được xóa và ghi log riêng.

## File máy đọc

- `docs/provenance/current/source_registry.json`
- `docs/provenance/current/source_registry.csv`
- `docs/provenance/current/artifact_registry.json`
- `docs/provenance/current/artifact_registry.csv`
- `docs/provenance/current/registry_validation.json`
- `CURRENT_PROJECT_MANIFEST.json`

Chạy lại `python tools/rebuild_registry.py` chỉ đọc/hash file; không import TensorFlow, không train và không test model.
