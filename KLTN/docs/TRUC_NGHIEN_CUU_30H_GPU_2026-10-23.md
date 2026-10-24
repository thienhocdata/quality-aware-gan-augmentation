# Đề xuất trục nghiên cứu 30 giờ GPU — CHƯA ĐỦ ĐIỀU KIỆN CHẠY FULL

**Ngày chốt:** 23/10/2026  
**Dataset giữ lại:** Indian Pines  
**Phương pháp gốc giữ lại:** 3D-HyperGAMO  
**Trạng thái:** bản đề xuất đang phản biện; chưa khóa trục, chưa khóa năm nhánh và chưa khuyến nghị chạy full.

**Đính chính sau rà soát ngày 23/10/2026:** Bản đầu chốt quá sớm. GAN gần ROS không chỉ ra D là nút thắt; 17/34 optimizer slots không tương đương 50% runtime; thay GD bằng GC là thay phân bổ mục tiêu học, không cô lập thuần tín hiệu D; ba seed và ngưỡng trung bình ±1 điểm không xác nhận non-inferiority. Các mục thí nghiệm bên dưới là ứng viên để thẩm định, không phải đặc tả chạy đã duyệt. Xem [bản phản biện và quyết định chưa train](RUI_RO_KE_HOACH_30H_GPU_2026-10-23.md).

## 1. Hướng ứng viên, chưa phải quyết định

Không tiếp tục V4/TR-HyperGAMO, không thêm attention/Transformer, loss mới, adaptive neighborhood hoặc một Generator phức tạp khác trong lượt 30 giờ GPU sắp tới.

Trục đề xuất là:

> **Đánh giá và tối ưu vai trò của tín hiệu Discriminator trong 3D-HyperGAMO cho phân loại HSI mất cân bằng trên Indian Pines, với mục tiêu giữ chất lượng lớp thiểu số trong khi giảm chi phí huấn luyện.**

Tên làm việc: **Compute-Aware 3D-HyperGAMO**. Đây chưa phải tên phương pháp cuối. Nếu cấu hình tốt nhất bỏ hoàn toàn Discriminator thì không gọi nó là GAN; khi đó tên đúng phải là một biến thể *classifier-guided generative oversampling* kế thừa HyperGAMO.

V2, V3-A, selector diagnostics và leakage experiments vẫn giữ nguyên vai trò trong hồ sơ nghiên cứu. Chưa có quyết định chuyển V2 sang phụ lục hoặc chuyển backbone sang Original GAN; lựa chọn đó còn cần lý do và một mục tiêu đã thống nhất.

## 2. Vì sao đổi trục là hợp lý

### 2.1. Bằng chứng local tạo ra câu hỏi nghiên cứu rõ

Kết quả M4 trên cùng Protocol B, ba seed 42/43/44:

| Hệ thống | Macro-F1 trung bình ± SD (%) |
|---|---:|
| Original GAN | 81,663 ± 2,877 |
| Matched ROS | 81,485 ± 0,967 |
| Random-local | 82,226 ± 3,004 |
| Learned-local V2 | 83,009 ± 3,015 |

Original GAN hơn Matched ROS **0,178 điểm phần trăm Macro-F1** trong trung bình ba seed. Tuy nhiên các chênh theo seed là **+3,480; +0,351; −3,297 điểm**, SD của chênh là **3,392 điểm**. Trung bình gần 0 do các chênh trái chiều triệt tiêu; đây không phải bằng chứng tương đương, cũng không chỉ ra D là nguyên nhân. Nó chỉ tạo động cơ khảo sát hệ thống sinh mẫu, chưa đủ để ưu tiên bỏ D.

### 2.2. Đây là nút thắt chi phí thật trong source

Mỗi loop M4 có 34 vị trí optimizer logic:

- 1 `C_real`;
- 1 `D_real`;
- 1 `C_fake`;
- 1 `D_fake`;
- 15 `GC`, cập nhật G/head qua mục tiêu classifier;
- 15 `GD`, cập nhật G/head qua mục tiêu discriminator.

Như vậy, `D_real + D_fake + 15 GD = 17/34` vị trí, đúng một nửa số slot. Chưa đo tỷ trọng wall time/FLOPs của chúng. Source có `skip_iterations`, chỉ tăng logical Adam clock và không thực hiện zero-gradient update. Nó giữ clock theo hợp đồng M4, nhưng không giữ Adam moments hay BatchNorm giống nhánh có update: các khác biệt này phải được định nghĩa là một phần của intervention hoặc kiểm soát riêng.

### 2.3. Câu hỏi có cơ sở từ paper gốc

GAMO gốc đã có ablation `GAMO\D`. Kết quả của nhánh bỏ D thay đổi theo dataset: full GAMO thường tốt hơn, nhưng mức lợi ích của D không cố định. Vì vậy, kiểm tra D trên HSI không phải suy đoán vô căn cứ. Nó cũng không phải một thuật toán hoàn toàn mới; đóng góp của khóa luận nằm ở **đánh giá có kiểm soát và tối ưu chất lượng–chi phí trong 3D-HyperGAMO trên Indian Pines**.

### 2.4. Vì sao không chọn thêm kiến trúc mới

Không gian “real-patch guidance + conditional discriminator + class consistency + CNN–Transformer” đã có công trình 3D-GIGAMO. Các hướng local convex weighting, boundary guidance và filtering cũng đã có prior art gần. Một V4 nhiều mô-đun sẽ tốn GPU nhưng rất khó quy nguyên nhân nếu kết quả tăng hoặc giảm. Ngược lại, thay đổi tần suất/việc sử dụng D chỉ tác động một cơ chế đã định nghĩa rõ.

## 3. Câu hỏi nghiên cứu

### RQ1 — Giá trị của D

Khi giữ nguyên dữ liệu, classifier exposure, fake-class schedule, Generator, tổng số cập nhật G/head và logical optimizer clock, recipe `GC+GD` khác recipe `GC+GC` như thế nào? Đây là so sánh thay thế mục tiêu dưới cùng số update, chưa phải tác dụng riêng của D. GC dùng target `1-condition`, tức tạo mẫu khó cho C, không phải tối ưu nhãn đúng cho G.

### RQ2 — Điểm vận hành chất lượng–chi phí

Có thể giảm tần suất D mà vẫn giữ chất lượng lớp thiểu số, đồng thời giảm đáng kể thời gian GPU hay không?

### RQ3 — Giá trị của sinh mẫu

Các cấu hình sinh mẫu có đem lại lợi ích đủ rõ so với matched real oversampling dưới cùng classifier exposure và fake-class allocation hay không?

## 4. Năm nhánh ứng viên — chưa quyết định chạy

Tất cả dùng **3D-HyperGAMO original/global-support**, không dùng V2 local selector làm backbone của thí nghiệm mới.

| Nhánh | Lịch trong một loop | Mục đích |
|---|---|---|
| A — Full | 2 C + 2 D + 15 GC + 15 GD | Mốc 3D-HyperGAMO đầy đủ |
| B — D25 matched | 30 cập nhật G/head mỗi loop; 25% loop dùng 15 GC + 15 GD, 75% loop dùng 30 GC; D real/fake chỉ ở loop D | Điểm vận hành giảm 75% tần suất D nhưng giữ ngân sách G/head |
| C — No-D matched | 2 C + 30 GC; bỏ D/GD, giữ logical clock | Cô lập tín hiệu D khi ngân sách G/head bằng A |
| D — No-D fast | 2 C + 15 GC; bỏ D/GD, giữ logical clock | Biến thể thực dụng giảm mạnh số route |
| E — Matched ROS | 2 C; synthetic slot được thay bằng real resampling theo cùng fake-class schedule | Kiểm tra sinh mẫu có hơn lặp mẫu thật hay không |

`D25` là một điểm thăm dò coarse được đăng ký trước để giảm 75% tần suất D. Nó không được gọi là tần suất tối ưu. Không quét nhiều tỷ lệ rồi chọn theo test.

### 4.1. So sánh nào trả lời điều gì

- **A so với C:** hiệu ứng của recipe thay GD bằng thêm GC, khi tổng số cập nhật G/head được giữ bằng nhau; không gọi là hiệu ứng riêng của D.
- **A, B, C:** đường đáp ứng thô khi tỷ lệ GD là 100%, 25%, 0%, với cùng ngân sách G/head.
- **C so với D:** tác dụng của 15 lượt GC bổ sung; tách “bỏ D” khỏi “giảm tổng compute”.
- **D so với E:** classifier-guided generation có hơn real resampling trong cấu hình nhanh hay không.
- **A so với D:** đánh đổi chất lượng–thời gian của hệ thống đầy đủ và hệ thống rút gọn.

Không dùng riêng A so với D để kết luận D có ích, vì hai nhánh khác cả tín hiệu lẫn tổng số cập nhật G.

## 5. Hợp đồng thực nghiệm bắt buộc

Giữa các nhánh phải giữ:

- Protocol B `50e988fae260`, train final 5.528 và test 4.721;
- patch 11×11, PCA 40 và scaler fit train-only;
- cùng kiến trúc C/G, initializer và seed;
- cùng real batches, fake classes, noise, class condition, target và dropout fixture khi route tương ứng tồn tại;
- 8.000 C updates, tương ứng 4.000 loop;
- cùng checkpoint cadence, hash, resume contract và môi trường TensorFlow/Keras;
- test tensor không được load trong worker train;
- route bị bỏ dùng assignment cho logical Adam iteration; không dùng zero-gradient optimizer update.

Phải lưu trace hash đủ để chứng minh các nhánh đã dùng cùng dữ liệu/lịch đã đăng ký. Mỗi nhánh chạy seed 42/43/44. Không thêm seed chỉ cho nhánh đang có kết quả đẹp.

## 6. Cổng trước khi chạy full

### G0 — Fidelity và pre-registration, CPU

1. Ghi rõ đây là controlled reimplementation theo cơ chế 3D-HyperGAMO, không gọi Protocol B là reproduction nguyên trạng paper.
2. Khóa source SHA, config, năm nhánh, seed, metric, quyết định và giới hạn thời gian.
3. Kiểm việc restore và output registry mà không load test trong train.

### G1 — Gradient/profiler audit, tối đa 1 GPU giờ

Trên train-only, seed 42:

- đo norm gradient GC và GD trên G/head;
- đo cosine GC–GD;
- đo wall time từng route C, D, GC, GD sau warm-up;
- kiểm D/G/C freeze đúng;
- kiểm finite loss/gradient;
- kiểm logical clock, checkpoint và resume.

Audit này không chọn kiến trúc theo accuracy. Nó xác nhận `GD` có thực sự truyền tín hiệu và cho dự toán compute.

### G2 — Smoke 200 C updates

Chạy cả năm nhánh ở seed 42. Chỉ kiểm wiring, fairness contract, hash, NaN, checkpoint/resume. Không đọc test và không diễn giải accuracy.

### G3 — Benchmark thời gian

Chạy 500–1.000 loop hoặc dùng profiler ổn định để dự báo tổng thời gian. Chỉ chạy full nếu tổng dự báo không vượt **24 GPU giờ**, giữ ít nhất **6 giờ dự phòng** cho retry, test một lần và đóng gói.

Nếu dự báo vượt 24 giờ, ưu tiên bốn nhánh A/C/D/E; B là nhánh đầu tiên được bỏ theo quy tắc đã ghi trước. Không giảm seed sau khi đã nhìn kết quả.

## 7. Metric và tiêu chí quyết định

### 7.1. Độ đo

- **Primary:** Macro-F1, trung bình và SD qua seed; paired difference theo seed.
- **Bắt buộc:** AA, GM, OA, Kappa, NLL, Brier.
- **Theo lớp:** precision, recall, F1, support và confusion matrix.
- **Mất cân bằng:** trung bình F1/recall của nhóm 25% lớp có ít mẫu train nhất, tập lớp được xác định trước khi train.
- **Chi phí:** wall time, GPU giờ, số actual optimizer calls, số logical calls, peak GPU memory và kích thước checkpoint.
- **Đồ thị chính:** Macro-F1–GPU time Pareto; per-class F1 deltas; paired seed deltas.

### 7.2. Ngưỡng thực dụng được khóa trước

Một nhánh rút gọn đạt **ngưỡng đánh đổi kỹ thuật dự kiến** nếu đồng thời (các ngưỡng dưới đây chưa được biện minh/chốt):

1. Macro-F1 trung bình không thấp hơn Full quá 1,0 điểm phần trăm;
2. thời gian train giảm ít nhất 25%;
3. GM không thấp hơn Full quá 1,0 điểm;
4. không có lớp thuộc nhóm tail giảm recall quá 10 điểm ở ít nhất 2/3 seed.

Một nhánh được xem là **cải thiện accuracy rõ** nếu Macro-F1 tăng ít nhất 1,0 điểm và không đánh đổi GM/tail recall theo ngưỡng trên.

Các ngưỡng là ứng viên tiêu chí kỹ thuật, không phải kiểm định thống kê. Cho phép mất một phần Macro-F1 để giảm thời gian không phải Pareto dominance. Trung bình ba seed nằm trong biên 1 điểm không đủ để xác nhận statistical non-inferiority.

## 8. Cách kết luận cho mọi khả năng

| Kết quả | Kết luận hợp lệ |
|---|---|
| A hơn C rõ ở đa số seed | D có giá trị thực nghiệm trong recipe này; không nên bỏ hoàn toàn |
| B gần A nhưng nhanh hơn | Có thể giảm tần suất D; B là ứng viên tối ưu chất lượng–chi phí |
| C gần hoặc hơn A | Recipe thay GD bằng GC có điểm trung bình gần hoặc hơn A; chưa chứng minh D vô ích |
| D nằm trong biên đánh đổi đã khóa và nhanh hơn ≥25% | Báo một ứng viên chất lượng–thời gian; chưa gọi là statistical non-inferiority chỉ bằng ba trung bình |
| E nằm trong ±1 điểm so với mọi nhánh sinh và nhanh hơn nhiều | Không có căn cứ thực dụng để trả chi phí GAN trên Protocol B |
| Không nhánh nào hơn A hoặc E theo tiêu chí | Giữ 3D-HyperGAMO gốc hoặc ROS; báo kết quả âm, không phát minh V4 để cứu bảng |

Lượt train không bảo đảm cải thiện và cũng không bảo đảm phân biệt được các recipe nếu độ bất định lớn. Smoke/gradient/profile PASS chỉ xác nhận vận hành kỹ thuật và dự toán chi phí, không dự báo lợi ích Macro-F1. Chưa đầu tư full suite trước khi giải quyết các khoảng trống trong bản phản biện.

## 9. Phạm vi phát biểu

Protocol B là benchmark cố định trong cùng cảnh Indian Pines. Nó thay split, preprocessing scope và model-selection rule so với code công khai, nên không so trực tiếp số điểm với paper như cùng một thí nghiệm.

Patch train/test còn chồng lấn nguồn pixel; test B đã được xem trong các vòng nghiên cứu trước. Vì vậy, lượt mới là **pre-registered controlled follow-up trên Protocol B**, không phải blind external confirmation và không chứng minh spatial generalization. Ba seed thay đổi stochastic training, không tạo ba cảnh độc lập.

Các giới hạn này không làm thí nghiệm vô nghĩa. Chúng giới hạn đóng góp ở mức: **phân tích cơ chế và tối ưu hiệu quả tính toán của 3D-HyperGAMO dưới một protocol cố định cho Indian Pines**.

## 10. Phân vai trong luận văn — đề xuất chưa chốt

### Giữ trong trục chính

- source/paper reproduction của 3D-HyperGAMO;
- Protocol B, manifest, preprocessing và checkpoint contracts;
- baseline Original GAN, Matched ROS, CNN/ROS/cost-sensitive nếu cùng protocol;
- M4 route accounting và kết quả đã có;
- overlap audit như giới hạn protocol;
- thí nghiệm compute-aware D mới.

### Chuyển sang nghiên cứu phụ/thất bại có thông tin

- V2 learned-local và endpoint selector analysis;
- V3-A distance correction;
- TR-HyperGAMO/V4 chưa train;
- Pavia feasibility;
- source-collision, endgame và các chẩn đoán phục vụ lịch sử.

Không xóa artifact gốc. Chỉ thay vai trò của chúng trong luận văn.

## 11. Tài liệu nền chính

- Roy et al., **3D-HyperGAMO**, DOI `10.1109/TGRS.2021.3052048`, official code: https://github.com/mhaut/3D-HyperGAMO
- Mullick et al., **GAMO**, ICCV 2019: https://openaccess.thecvf.com/content_ICCV_2019/html/Mullick_Generative_Adversarial_Minority_Oversampling_ICCV_2019_paper.html
- Heusel et al., **TTUR**, NeurIPS 2017: https://papers.nips.cc/paper_files/paper/2017/hash/8a1d694707eb0fefe65871369074926d-Abstract.html
- Ouyang et al., **Accelerated WGAN Update Strategy**, WACV 2021: https://openaccess.thecvf.com/content/WACV2021/html/Ouyang_Accelerated_WGAN_Update_Strategy_With_Loss_Change_Rate_Balancing_WACV_2021_paper.html
- 3D-GIGAMO, Remote Sensing 2026: https://www.mdpi.com/2072-4292/18/19/3343

TTUR và Accelerated WGAN chỉ cung cấp động cơ chung rằng lịch G/D ảnh hưởng động lực học và chi phí. Chúng không chứng minh tỷ lệ 25% đúng cho loss MSE của 3D-HyperGAMO; tỷ lệ đó phải được đánh giá như một điểm vận hành đăng ký trước.

