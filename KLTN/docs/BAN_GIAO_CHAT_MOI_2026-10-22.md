# Bàn giao sang chat mới — KLTN 3D-HyperGAMO

Ngày chốt: 22/10/2026.

## 1. Trục nghiên cứu hiện tại

Bài toán chính là xử lý mất cân bằng dữ liệu trong phân loại ảnh siêu phổ Indian Pines. Baseline là bản tái triển khai 3D-HyperGAMO dưới Protocol B `50e988fae260`; không gọi đây là exact reproduction con số trong paper.

Phương pháp chính được giữ là **Learned-local V2**:

- lấy anchor đồng đều trong lớp;
- graph 3 neighbor cùng lớp;
- `rho=0.2`;
- mẫu sinh `0.8*anchor + 0.2*weighted_neighbors`;
- trọng số neighbor do G và Dense head học.

V3-A là biến thể distance-bias đã chạy đủ nhưng không được chọn thay V2.

## 2. Kết quả chính đã xác minh

Full training, ba seed 42/43/44, cùng Protocol B:

| Phương pháp | Macro-F1 trung bình ± sample SD |
|---|---:|
| Original GAN reimplementation | 81.663 ± 2.877% |
| Matched ROS | 81.485 ± 0.967% |
| Random-local | 82.226 ± 3.004% |
| Learned-local V2 | **83.009 ± 3.015%** |
| V3-A | 81.543 ± 1.701% |

V2 cao hơn Original GAN 1.345 điểm phần trăm, ROS 1.524 điểm và Random-local 0.783 điểm. Lợi ích chưa ổn định tuyệt đối qua seed và một phần lớn chênh so Random-local đến từ Oats có 10 mẫu test.

Endpoint ablation sáu classifier giữ G/head cuối:

- Learned endpoint: 82.163 ± 1.551% Macro-F1.
- Random-one-fixed: 79.784 ± 2.667%.
- Chênh trung bình +2.379 điểm, thắng 2/3 cặp.

Selector endpoint đã đo 794.496 bộ trọng số; mọi max-weight >0.999 và winner không đổi qua 64 noise cho từng anchor/seed. Đây là hành vi đã đo, chưa chứng minh là nguyên nhân làm giảm F1.

V3-A thấp hơn V2 1.465 điểm và 0/2.520 endpoint probes đổi winner khi bật distance bias trên logits V3-A. Giữ V3-A như kết quả ablation âm.

## 3. Các giới hạn cần giữ đúng khi viết

- Protocol B còn spatial/raw-source overlap.
- Validation development đã bão hòa ở một số candidate; lớp hiếm có rất ít mẫu validation.
- Test B đã được xem trong quá trình phát triển các phân tích/biến thể sau đó.
- Ba seed là ba lượt train trên cùng split/cảnh, không phải ba dataset độc lập.
- Indian Pines là dataset chính nhưng hiện chỉ có một cảnh/dataset được xác nhận đầy đủ.
- Không tuyên bố V2 vượt kết quả paper, vượt SOTA hoặc giải quyết xong leakage.
- MSE/shared Adam là lựa chọn kế thừa đã kiểm, không phải bug.

Danh sách 15 hạn chế và đối chiếu paper:

`protocol_b_review/limitations_review_2026-10-22/HAN_CHE_V2_V3A_VA_DOI_CHIEU_PAPER.md`

## 4. Quyết định tổ chức project

Chọn **`KLTN_clean/` làm project canonical**.

- `KLTN_clean` = working canonical.
- `KLTN_TRUC_CHINH` = snapshot/bàn giao cũ, chưa xóa.
- `outputs` = kho artifact/release lớn.
- `protocol_b_review` = hồ sơ audit/nghiên cứu lịch sử, không dùng làm runtime source tự động.
- `quality-aware-gan-augmentation` chưa phải canonical source hiện tại.

Mô hình quản lý đã chốt:

- một notebook master chỉ để đọc/tổng hợp;
- nhiều notebook mỏng theo từng câu hỏi;
- logic thật nằm trong source `.py` dùng chung;
- không tạo notebook riêng cho từng seed;
- source đã tạo output cũ nằm ở `src/frozen` và không sửa;
- source đồng bộ tương lai nằm ở `src/canonical`;
- output/checkpoint nhận dạng bằng manifest và hash, không chỉ bằng tên ZIP.

Không bấm Run All toàn bộ notebook để tái tạo mọi thứ. Trước train mới cần:

1. registry artifact;
2. canonical source;
3. parity checks giữa canonical và frozen;
4. khóa protocol/config;
5. smoke;
6. train theo job có resume;
7. chỉ test sau khi đủ checkpoint và quyết định đã khóa.

## 5. Vấn đề cần quyết định tiếp theo

Trước khi có lượt train cuối, cần quyết định phạm vi bằng chứng:

1. Giữ Indian Pines Protocol B như kết quả lịch sử chính.
2. Hoàn thiện fidelity audit giữa code chính thức 3D-HyperGAMO và nhánh `original_gan` của nhóm.
3. Bổ sung hoặc chuẩn hóa các baseline thể hiện trực tiếp bài toán imbalance: real-only, một cost-sensitive baseline, Matched ROS, Original GAN, Random-local ablation và V2 — tùy ngân sách.
4. Nếu chỉ còn một lượt xác nhận lớn, ưu tiên một dataset HSI thứ hai với phương pháp/config khóa trước. Indian Pines vẫn là dataset chính; dataset thứ hai chỉ xác nhận bên ngoài.
5. Báo Macro-F1 chính, kèm per-class F1/recall/support, AA, GM, OA, Kappa, confusion matrix và chi phí train.

Không mở V4 hoặc sửa đồng thời overlap, metric neighbor, k, rho, loss và anchor conditioning.

## 6. Tài liệu nguồn nên đọc đầu tiên

- `KLTN_clean/README.md`
- `KLTN_clean/docs/scope.md`
- `protocol_b_review/research_report_2026-10-19/notes/scientific_review.md`
- `protocol_b_review/limitations_review_2026-10-22/HAN_CHE_V2_V3A_VA_DOI_CHIEU_PAPER.md`
- `protocol_b_review/v3a_results_received_2026-10-19/recomputed_summary_mean_sample_SD.csv`

## 7. Yêu cầu cho chat mới

Đọc file bàn giao này và các tài liệu được chỉ ra trước khi đề xuất train hoặc sửa code. Không mặc định tạo thêm thí nghiệm. Bước kế tiếp là chốt project canonical và phạm vi lượt xác nhận cuối.
