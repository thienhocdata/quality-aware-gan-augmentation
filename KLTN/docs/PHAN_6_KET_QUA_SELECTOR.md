# Phần 6. Phân tích hành vi và giá trị của learned selector

**Trạng thái: hoàn tất phạm vi endpoint ablation đã khóa. Cập nhật ngày 19/10/2026.**

Phần này tổng hợp kết quả M4, chẩn đoán trọng số tại checkpoint cuối và một đối chứng endpoint riêng. Các kết quả được đọc từ output đã lưu; việc tổng hợp không huấn luyện hoặc chạy inference lại. Dataset chính là Indian Pines, giao thức B `50e988fae260`, 16 lớp, patch 11×11 và PCA 40 thành phần.

## 6.1. Câu hỏi và hai loại bằng chứng

Câu hỏi cơ chế được khảo sát là: **quy tắc weighting mà Learned-local V2 học được tại checkpoint cuối có đem lại lợi ích phân loại hơn quy tắc chọn một neighbor ngẫu nhiên cố định hay không?**

Hai thí nghiệm trả lời hai câu hỏi khác nhau:

| Thí nghiệm | Đối tượng so sánh | Vai trò |
|---|---|---|
| M4: bốn nhánh × ba seed | Original GAN, Matched ROS, Random-local với trọng số Dirichlet và Learned-local V2 | Đánh giá phương pháp trong quá trình huấn luyện M4 đầy đủ |
| Endpoint: hai nhánh × ba cặp seed | G/head Learned-local cuối được giữ cố định; huấn luyện C mới bằng learned weights hoặc Random-one-fixed | Đánh giá giá trị của quy tắc weighting cuối trong lịch C được ghép cặp |

Không cộng hoặc gộp chênh lệch giữa hai thí nghiệm. Random-local Dirichlet và Random-one-fixed là hai đối chứng khác nhau. Thí nghiệm endpoint không thay thế bảng M4 và không cô lập lợi ích của toàn bộ quá trình huấn luyện đối kháng.

## 6.2. Kết quả M4 làm nền cho câu hỏi cơ chế

M4 đã hoàn tất 12 classifier, tương ứng bốn nhánh và seed 42/43/44, mỗi job có 8.000 classifier updates. Kết quả dưới đây là trung bình ± độ lệch chuẩn mẫu (`ddof=1`) qua ba seed trên cùng 4.721 mẫu test.

| Phương pháp | OA (%) | AA (%) | Macro-F1 (%) | GM (%) | Kappa |
|---|---:|---:|---:|---:|---:|
| Original GAN | 87,552 ± 0,444 | 85,783 ± 1,641 | 81,663 ± 2,877 | 82,086 ± 4,027 | 0,8587 ± 0,0053 |
| Matched ROS | 87,143 ± 0,422 | 85,049 ± 0,681 | 81,485 ± 0,967 | 80,575 ± 3,269 | 0,8542 ± 0,0045 |
| Random-local, Dirichlet | 88,018 ± 1,199 | 86,016 ± 1,062 | 82,226 ± 3,004 | 81,862 ± 3,919 | 0,8644 ± 0,0134 |
| Learned-local V2 | 88,251 ± 1,229 | 86,046 ± 1,801 | 83,009 ± 3,015 | 81,793 ± 3,888 | 0,8668 ± 0,0139 |

V2 có Macro-F1 trung bình cao hơn Original GAN 1,345 điểm phần trăm, Matched ROS 1,524 điểm và Random-local 0,783 điểm. Tuy nhiên, V2 chỉ thắng Random-local ở hai trong ba seed; tại seed 43, V2 thấp hơn cả ba đối chứng về Macro-F1. V2 cũng không đứng đầu GM. Vì vậy, bảng M4 hỗ trợ nhận định cải thiện trung bình trong cấu hình đã kiểm tra, chưa hỗ trợ nhận định vượt trội ổn định ở mọi seed hoặc mọi độ đo.

Phân tích lớp trước đây còn cho thấy Oats, với 10 mẫu test, đóng góp khoảng 0,685 trong tổng 0,783 điểm tăng Macro-F1 so với Random-local. Kết quả này được giữ nguyên trong diễn giải; không loại lớp hoặc seed bất lợi để làm đẹp bảng. Môi trường train M4 seed 42 khác phiên bản tf-keras/NumPy của seed 43/44, nên SD không được diễn giải như chỉ phản ánh một nguồn ngẫu nhiên duy nhất.

## 6.3. Hành vi trọng số tại checkpoint 8.000

Với anchor \(x_a\), ba neighbor cùng lớp và \(\rho=0,2\), V2 sinh mẫu theo:

\[
\tilde x(a,z,c)=0,8x_a+0,2\sum_{j\in N_c(a)}w_j(z,c,a)x_j.
\]

G tạo features từ noise và điều kiện lớp. Dense head tạo logits trên các supports của lớp, sau đó graph của anchor giới hạn ba logits tham gia softmax. Anchor tác động qua graph và phép trộn; G/head không nhận trực tiếp feature của anchor làm điều kiện học.

Chẩn đoán cuối đã đo đủ ba seed, 15 learned heads mỗi seed, tổng 4.138 anchor và 64 noise mỗi anchor. Lớp đa số không có learned head. Trong toàn bộ 794.496 bộ trọng số đã khảo sát, trọng số lớn nhất đều vượt 0,999; không anchor nào đổi neighbor trội qua 64 noise. Logits vẫn thay đổi theo noise nhưng softmax rất tập trung, khiến biến thiên patch ở endpoint rất nhỏ.

Đây là mô tả của các checkpoint cuối và tập noise đã khảo sát. Không suy thành mọi noise, không xác định thời điểm hiện tượng xuất hiện và không gọi đó là bằng chứng nguyên nhân seed 43 kém. Đầu ra vẫn gồm 80% anchor và 20% tổ hợp neighbor; kết quả không có nghĩa sao chép nguyên một patch hoặc mọi mẫu sinh đều giống nhau.

Quan sát này dẫn tới đối chứng endpoint: nếu cơ chế gần chọn một neighbor, cần so giá trị của lựa chọn học được với một lựa chọn ngẫu nhiên cố định trước khi sửa noise, nhiệt độ softmax hoặc kiến trúc.

## 6.4. Thiết kế endpoint ablation

Dùng G và 15 Dense heads từ native final checkpoint M4 Learned-local tại 8.000 C updates của từng seed 42/43/44. Các mô hình sinh được khôi phục đúng source/tensor hashes, giữ cố định và gọi với `training=False`.

Với mỗi G seed, huấn luyện hai classifier mới, C seed tương ứng, trên final train 5.528 mẫu:

- **Learned weights:** dùng trực tiếp trọng số softmax thật của checkpoint trong công thức V2.
- **Random-one-fixed:** với mỗi anchor, chọn đều một trong ba neighbor bằng RNG riêng trước training, rồi giữ lựa chọn đó cố định trong cả job. Patch bổ sung là \(0,8x_a+0,2x_{r(a)}\).

Tổng cộng sáu C-only job. Hai nhánh trong mỗi cặp có cùng khởi tạo C và Adam slots, dữ liệu, graph, real IDs, fake-class schedule, anchors, noise và dropout. C dùng MSE, legacy Adam với learning rate \(2\times10^{-4}\), \(\beta_1=0,5\), 8.000 updates; global Adam clock cuối là 136.000 theo lịch M4. Những lượt D/G bị bỏ chỉ tăng iteration counter, không dùng zero-gradient updates. G/head không được huấn luyện lại.

Đây là đối chứng toàn bộ quy tắc weighting endpoint: Learned dùng soft weights thật, Random-one-fixed dùng một neighbor. Thiết kế chưa tách riêng hiệu ứng danh tính neighbor khỏi mức độ mềm của trọng số. Seed G/C và random map thay cùng block; ba cặp không tách được phương sai do từng nguồn.

## 6.5. Kết quả endpoint ablation

| Cặp seed G/C | Learned Macro-F1 (%) | Random-one-fixed Macro-F1 (%) | Learned − Random (điểm phần trăm) |
|---|---:|---:|---:|
| 42 | 82,512 | 82,857 | −0,345 |
| 43 | 83,510 | 78,067 | +5,442 |
| 44 | 80,466 | 78,429 | +2,038 |
| Trung bình ± SD mẫu | **82,163 ± 1,551** | **79,784 ± 2,667** | **+2,379** |

SD mẫu của ba paired differences là 2,909 điểm phần trăm. Learned thắng hai trong ba cặp; seed 42 thua. Ba cặp là đơn vị lặp của thí nghiệm; không coi 4.721 pixel test như 4.721 lần huấn luyện độc lập.

| Độ đo | Learned: TB ± SD mẫu | Random-one-fixed: TB ± SD mẫu |
|---|---:|---:|
| OA (%) | 87,192 ± 0,903 | 86,705 ± 1,661 |
| AA (%) | 85,263 ± 1,221 | 83,439 ± 1,854 |
| Macro-F1 (%) | 82,163 ± 1,551 | 79,784 ± 2,667 |
| GM (%) | 79,907 ± 4,580 | 76,269 ± 1,454 |
| Kappa | 0,854850 ± 0,010282 | 0,849095 ± 0,018714 |
| NLL ↓ | 1,034228 ± 0,185781 | 0,993130 ± 0,128543 |
| Brier ↓ | 0,225038 ± 0,017608 | 0,228192 ± 0,028805 |

GM được bổ sung từ recall 16 lớp trong dự đoán đã lưu. NLL dùng đúng clipping \(10^{-12}\) của runner. Learned có NLL trung bình kém hơn; kết quả không hỗ trợ nhận định cải thiện mọi mặt hoặc hiệu chuẩn xác suất tốt hơn.

Ở seed 43, F1 Learned cao hơn ở 14/16 lớp, hòa một và thấp hơn một. Lớp Buildings–Grass–Trees–Drives có F1 56,180% so với 15,267%, recall 50/99 so với 10/99. Riêng lớp này đóng góp 2,558 trong tổng 5,442 điểm Macro-F1 tăng, khoảng 47%; phần còn lại vẫn dương.

Ở seed 42, Random-one-fixed đúng riêng 158 mẫu và Learned đúng riêng 54 mẫu; OA Learned giảm 2,203 điểm phần trăm, trong khi Macro-F1 giảm 0,345 điểm. Ở seed 44, Learned đúng riêng 212 mẫu so với 176 của Random; Macro-F1 tăng 2,038 điểm. Các số đếm đúng riêng mô tả vị trí khác biệt, không xác định nguyên nhân mô hình học được hoặc mắc lỗi.

Endpoint seed 43 có lợi không giải thích hoặc đảo ngược kết quả seed 43 trong M4 cũ: classifier và quá trình huấn luyện khác nhau. Không gán chênh lệch M4 cũ cho một selector hỏng từ các quan sát này.

## 6.6. Kết luận và giới hạn

> Trong M4 trên Indian Pines theo Protocol B, Learned-local V2 đạt Macro-F1 trung bình 83,009 ± 3,015%, cao hơn Random-local Dirichlet 0,783 điểm phần trăm và thắng ở hai trong ba seed. Chẩn đoán cuối cho thấy trọng số tập trung gần một neighbor và ít thay đổi theo noise. Trong endpoint ablation riêng, giữ cố định G/head và huấn luyện sáu classifier mới theo lịch ghép cặp, learned weighting đạt 82,163 ± 1,551% Macro-F1, so với 79,784 ± 2,667% của Random-one-fixed; chênh lệch trung bình +2,379 điểm phần trăm, với hai trong ba cặp có lợi. Kết quả cung cấp bằng chứng thực nghiệm cho giá trị của quy tắc weighting cuối so với đối chứng ngẫu nhiên cố định đã xét, dù chưa chứng minh lợi thế ổn định hoặc lợi ích riêng của quá trình huấn luyện đối kháng.

Test Indian Pines đã được xem trước khi thiết kế endpoint ablation, nên phần này là nghiên cứu hậu nghiệm. Protocol B có chồng lấn không gian; kết quả không chứng minh đã xử lý leakage hoặc tổng quát hóa sang vùng độc lập. Chỉ ba cặp và biến thiên giữa seed còn đáng kể; không tuyên bố ý nghĩa thống kê hoặc tính mới thuật toán chỉ từ bảng này.

Chẩn đoán 7.500→8.000 của **G/head M4** vẫn chưa có report forward hoàn tất trong hồ sơ đã xác minh. Đó là trạng thái khác với endpoint ablation sáu C mới đã hoàn tất. Các checkpoint C tại 7.500 có trong backup hiện tại không thay thế checkpoint hoặc chẩn đoán trajectory G. Phần chưa chạy này không được dùng để suy luận bão hòa xuất hiện khi nào và không được biến thành điều kiện buộc train lại.

Phạm vi endpoint ablation được đóng sau kết quả này. Các giả thuyết về k, rho, metric neighbor, temperature hoặc noise vẫn là hướng tương lai, chưa phải cải tiến được chứng minh và không tự mở thành lượt train mới.

## 6.7. Nguồn và cách đọc hồ sơ

- M4: `outputs/bundles/03_M4_TEST_RESULTS.zip`, đủ 12 classifier; notebook dùng bảng gốc để tính mean/SD và paired differences.
- Chẩn đoán weights: `outputs/bundles/05_V2_WEIGHTING_8000.zip`, ba seed, 15 heads/seed, train-only forward đã hoàn tất trước đây.
- Endpoint analysis: `outputs/bundles/07_SELECTOR_ENDPOINT_ABLATION_ANALYSIS_20261019.zip`, gồm sáu predictions gốc, metrics, per-class, confusion matrices, báo cáo và audit.
- Backup endpoint đầy đủ: `D:/HUIT/Chương trình học/HK1 ( 2026- 2027)/KLTN - GAN/outputs/6. Selector Analysis/selector_endpoint_ablation_clean_backup.zip`; 130.717.579 bytes, SHA-256 `7dcdcbdd1a62948a704162334c992b35674cdbcffd38843e5360426b319fc062`.
- Audit endpoint đã kiểm 311 điều kiện CPU về file hashes, counters, provenance, pairing và phép tính từ predictions. Lượt tổng hợp không chạy TensorFlow, không tái kiểm numerical restore và không thay checkpoint. Gói phân tích nhẹ không chứa lại weights; giữ backup gốc để bảo toàn sáu classifier và native checkpoints.

`notebooks/selector.ipynb` hiển thị bảng và kết luận bằng cách đọc các gói đã lưu; Run All vẫn là CPU và không chạy lại training/test inference. Các ZIP đầy đủ hoặc GitHub Release đã xuất trước ngày này không tự cập nhật theo bản local mới.
