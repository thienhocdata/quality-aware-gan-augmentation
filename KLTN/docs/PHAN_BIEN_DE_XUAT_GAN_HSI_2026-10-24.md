# Phản biện PDF kiến trúc GAN HSI bản 0.9

Ngày: 24/10/2026. Đã đọc 20 trang và kiểm trực quan các công thức, bảng, sơ đồ liên quan. Không sửa PDF gốc, không chạy TensorFlow/GPU, không train hoặc test inference.

Nguồn: `C:/Users/ADMIN/OneDrive/Desktop/DE_XUAT_KIEN_TRUC_GAN_HSI_2026-10-24.pdf`.

SHA-256: `88510770b1fd882174b54d360897611ed1262fef13054e13e73a2fd32605bd09`.

## 1. Quyết định

**Giữ hướng real-conditioned residual GAN và phép so A0/A1. Sửa bản đặc tả trước runner dài; chưa chạy nguyên bản 0.9 trong 30 giờ.**

Không tìm thấy lỗi dấu loss/kích thước tensor khiến thiết kế bất khả thi. Đây là một proposal thăm dò hợp lệ, phù hợp phạm vi người dùng đã cho: Indian Pines + GAN là bắt buộc, phần còn lại có thể thiết kế lại. Nó chưa phải kiến trúc đã chứng minh có lợi thế cạnh tranh.

Đề nghị giữ câu hỏi chính: **một GAN sinh biến thiên từ mẫu thật có hơn ROS/augmentation đơn giản hay không; và factorized residual có đem lại lợi ích so với free residual trong cùng scaffold hay không?**

## 2. Phần đúng nên giữ

- Trang 3, 8: nhận ra `G(a,z,c)=a` có thể khớp đúng phân phối train theo lớp; adversarial loss thuần không tự buộc G sinh biến thiên hữu ích.
- Trang 7–8: `U[121,4] V[4,40]/sqrt(4)` cho residual rank tối đa 4; scalar envelope giữ rank. Công thức đúng.
- Trang 8: `Delta=R*T/sqrt(R²+||T||²+epsilon)` cho `||Delta||<R` với R>0 và epsilon không âm; tránh triệt amplitude bằng hard clipping như bản trust-region cũ.
- Trang 10: conditional hinge D, `L_G=-E[D(fake,c)]`, bỏ sigmoid, stop-gradient fake ở D step và gradient D→G ở G step có dấu/luồng hợp lý.
- Trang 5, 11: C downstream không cấp gradient về G là redesign có chủ ý, không phải lỗi triển khai. Không cần ép quay lại target `1-condition` của M4.
- Trang 7, 9: không áp wavelength smoothness máy móc lên thứ tự PCA components là đúng.
- Trang 12–18: development panel, profiler, full-state checkpoint, tách thăm dò khỏi xác nhận và không ép 8.000 C updates là các quyết định đúng.

## 3. Các điểm cần sửa hoặc khóa

### A. H2 rank 4 là giả thuyết có thể thử, chưa phải điểm mạnh được chứng minh

**Trang 7–8.** Low rank là một prior có căn cứ trong HSI, nhưng rank của ảnh/khối phổ không đồng nghĩa rank của biến thiên cùng lớp hữu ích cho augmentation. `U` có thể là checkerboard; rank 1 vẫn có nhiễu không gian rất mạnh. `V` tự do không bảo đảm spectral smoothness, đúng nhãn tâm hoặc physical validity.

Cách làm rẻ trước lượt dài: đo singular spectrum và lỗi xấp xỉ rank 4 của **hiệu hai patch train cùng lớp**, có thống kê center/context và lớp hiếm. Không dùng test/development donors cho phép đo đó. Kết quả chỉ giúp thẩm định prior; không thay downstream evaluation.

Không bắt buộc chứng minh trước rằng rank 4 tối ưu. Có thể giữ nó là một lựa chọn thăm dò đã khóa nếu thống kê không bác tính khả thi; không quét rank theo test.

### B. A0/A1 thay cả cấu trúc, capacity và động lực tối ưu

**Trang 7–8, 15.** Đếm theo Dense input 256, có bias:

| Output head | Số tham số |
|---|---:|
| A0: 256→4.840 | 1.243.880 |
| A1: 256→484 và 256→160 | 165.508 |

Head A0 lớn khoảng **7,52 lần**. PDF đã thừa nhận khác biệt này; không phải lỗi bị bỏ quên. A0/A1 là so sánh hai gói parameterization hợp lệ. Nếu A1 thắng, claim đầu tiên là “structured compact generator hữu ích trong điều kiện đã thử”, chưa phải tác dụng thuần của rank độc lập với số tham số.

Không bắt buộc thêm arm thứ ba ngay trong 30 giờ. Capacity-matched free output có thể là đối chứng xác nhận sau nếu muốn phát biểu mạnh về rank.

### C. Hai loss non-identity/noise cần một đối chứng đúng cơ chế

**Trang 8–9.** PDF đã cảnh báo đúng rằng distance floor không bảo đảm mẫu có ích. Cần chuyển cảnh báo này thành phép so cụ thể. Một residual rank 1 thay đổi theo z ở phần context cũng có thể làm giảm hai penalty, trong khi không bổ sung thông tin lớp trung tâm.

Khuyến nghị định nghĩa arm SIMPLE thành **random residual có cùng envelope**, và khi kiểm H2 có thể dùng rank-4 random residual làm null control gần cơ chế. Khóa donor/scale/noise và exposure. Đo center/context displacement, nearest-real/duplicate rate và downstream per-class F1. Không thêm hai baseline mới nếu có thể dùng chính slot SIMPLE đã dự trù.

Tại exact identity hoặc output hoàn toàn bỏ noise, Euclidean norm không khả vi tại 0. Một số implementation cho gradient 0, một số có thể NaN nếu tính sqrt-sum không an toàn. Penalty không bảo đảm tự thoát trạng thái này. Cần test trường hợp 0 rõ ràng và initializer tạo đường học nonzero.

### D. Initializer, epsilon và quy tắc chọn lambda còn thiếu

**Trang 7–9, 18.** A1 bilinear: gradient theo U tỷ lệ với V và ngược lại. Cả hai factor rất nhỏ có thể làm đường học yếu hơn A0 nhiều lần. Khóa initializer bằng phân bố residual ban đầu, không chỉ câu “gần 0”. Kiểm riêng gradient U/V/trunk/Ea; không gộp thành G changed.

Scaling `U→alpha U`, `V→V/alpha` không đổi T nhưng đổi gradient theo tham số. Vì vậy không hiệu chỉnh lambda bằng cách so norm toàn bộ parameter gradients của A0 với A1. Nên quy định đo gradient tại **effective residual/output** trên một bộ anchor/noise train cố định; sau đó giữ cùng lambda cho hai arm.

Tách epsilon ở denominator norm² và epsilon chia scale. Mệnh đề `Delta≈T` gần gốc cần epsilon nhỏ so với R². Với support scale gần 0, phải quy định rõ sample bị loại khỏi diagnostic, khỏi regularizer hay khỏi GAN batch; không âm thầm nâng radius. Mỗi lớp cần đủ donor để tính scale; nếu chỉ một support phải có policy rõ.

### E. Full-patch locality chưa bảo vệ nhãn pixel trung tâm

**Trang 5–6, 8–10.** Nhãn là center label, nhưng `s_a` và hai penalty đều đo toàn patch. Context gồm 120 pixel có thể chi phối scale. D dùng GAP nên cần kiểm nó có thật sự nhạy với thông tin lớp tâm, không chỉ texture/context.

Cần khóa trước một lựa chọn: giữ D hiện tại nhưng đo center/context đầy đủ, hoặc bổ sung center branch vào D chung cho A0/A1. Đây là quyết định thiết kế; chưa có dữ liệu để khẳng định center branch chắc chắn tăng F1. Projection conditioning không phải giấy chứng nhận đúng nhãn.

### F. Freeze D bao gồm spectral-normalization state

**Trang 10, 12–13.** Không BatchNorm không có nghĩa không có nontrainable state. Spectral normalization có power-iteration vectors; wrapper có thể cập nhật state/kernel khi forward tùy implementation.

Phải khóa framework/version, lớp nào được SN, D mode ở D/G routes và test snapshot toàn bộ D trainable/nontrainable state khi cập nhật G. Gradient qua D đến fake vẫn phải tồn tại. Không âm thầm tắt SN nếu runtime không hỗ trợ.

### G. Real-only/ROS đang có khả năng trùng nhau; SIMPLE chưa thành thuật toán

**Trang 11–13.** Nếu Real-only dùng balanced resampling và mọi arm có cùng class schedule, matched ROS có thể chính là cùng thuật toán. Nếu chỉ Real-only dùng weighted loss, lại thêm intervention khác.

Chọn một cách rõ: Real-only theo empirical prior và ghi khác class exposure; ROS làm primary matched-exposure control. Hoặc gộp hai arm nếu thực chất giống nhau. Khóa SIMPLE thành một công thức, không để runner tự chọn giữa mixup/local/random residual.

### H. Development panel và job matrix phải cụ thể trước full

**Trang 12, 15–18.** PDF thừa nhận chưa có panel khả dụng, đây là khoảng trống chưa giải quyết chứ không phải một panel sai đã được chạy.

Cần bổ sung: rule split, tọa độ/seed/count mỗi lớp, fit scope, mức footprint reuse, primary metric, số lần query development, endpoint G/C được phép đánh giá. Không yêu cầu lại strict all-16 source-disjoint patch 11×11 vì geometry đã chỉ ra không khả thi cho một số lớp.

Job matrix tối thiểu phải ghi `panel × GAN arm × G seed × G endpoint × C seed × augmentation manifest`. Nhiều C seed từ một G không thành nhiều independent G seeds.

“Mỗi arm 9 giờ” là cap chi phí, không phải same-training-budget. Primary A0/A1 nên cùng G/D updates và downstream C exposure; runtime là outcome. So equal walltime là câu hỏi phụ có số update khác.

## 4. Đối chiếu tài liệu: phần đúng và phần nên bổ sung

- **ADC-GAN:** đúng khi cảnh báo class CE có thể làm giảm diversity trong AC-GAN generator-agnostic. Paper không chứng minh mọi C-guidance đều xấu; nó đề xuất generator-aware discriminative classifier. Vì vậy bỏ C→G ở scaffold là lựa chọn giảm rủi ro, không phải kết luận bắt buộc từ paper. [ICML source](https://proceedings.mlr.press/v162/hou22a.html).
- **Noise dependence:** cần dẫn trực tiếp [Diversity-Sensitive cGAN, ICLR 2019](https://research.google/pubs/diversity-sensitive-conditional-generative-adversarial-networks/). Nó hỗ trợ động cơ regularize noise response, không xác nhận công thức floor/tau trong PDF là tối ưu cho HSI.
- **D:** thêm citation [projection cGAN](https://arxiv.org/abs/1802.05637) và [spectral normalization](https://arxiv.org/abs/1802.05957); nguồn triển khai cụ thể phải được kiểm riêng.
- **Low rank:** [WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/He_Self-Supervised_Learning_with_Spectral_Low-Rank_Prior_for_Hyperspectral_Image_Reconstruction_WACV_2025_paper.html) hỗ trợ low-rank spectral prior trong reconstruction, không chứng minh rank-4 minority augmentation. Nên ghi đúng khoảng cách từ prior đến H2.
- **Modern HSI GAN:** SMPCA-GAN/S2A-Swin/3D-GIGAMO là các nguồn liên quan thực. Chưa xác minh code/runtimes của mọi paper, nên không ghi “đã reproduct” hoặc lấy điểm khác protocol xếp hạng.

## 5. Sửa bản 1.0 theo phạm vi hữu hạn

1. Giữ family G shared + anchor/noise/class, conditional D và downstream C; giữ A0/A1 làm câu hỏi thăm dò.
2. Khóa initializer/output scale, epsilon/zero-scale, lambda-calibration location và SN-state contract.
3. Chốt center/context policy; định nghĩa SIMPLE là đối chứng sát giả thuyết; gộp hoặc phân vai R/ROS.
4. Ghi một development protocol và job matrix có thể chạy, cùng budget/update/exposure và endpoint query limits.
5. Đo train-only scale/rank và thực hiện một smoke/profiler hữu hạn. Nếu cùng recipe thì tiếp tục từ checkpoint, không khởi chạy lại từ đầu chỉ để đổi tên pilot thành full.

Chưa cần thêm Transformer, teacher, hardness loss, trust-region hay ADA đồng loạt. Sau scaffold, chỉ mở một thay đổi dựa trên failure mode hoặc lợi ích downstream đã đo.

## 6. Giá trị có thể có và giới hạn đóng góp

30 giờ có thể trả lời feasibility và tạo tín hiệu lựa chọn giữa A0/A1; chưa đủ mặc định để bảo vệ superiority. Các đối chứng Real/ROS/SIMPLE đủ cho tín hiệu phát triển ban đầu, chưa đủ chứng minh cạnh tranh hiện đại.

Trước kết luận cuối cần ít nhất một đối chứng GAN gần cơ chế có thể chạy, một C đủ mạnh và đánh giá bổ sung theo cùng contract. Nếu chỉ C yếu hưởng lợi thì utility không tự chuyển sang hệ mạnh. Nếu A1 thắng mà A0 cũng tương đương, đóng góp nằm ở shared conditional GAN hoặc compactness, chưa chắc ở rank prior.

**Điểm đáng giữ của PDF:** nó đã tạo một can thiệp cụ thể có thể bị bác bỏ, nhận ra identity equilibrium và tách đúng giai đoạn phát triển. Điểm cần hoàn thiện là bằng chứng phù hợp của prior và contract thực thi/đánh giá, không phải nghĩ thêm tên kiến trúc.

## 7. Những kiểm tra đã làm trong lần review này

- Trích/đọc 20 trang, render toàn bộ bằng PDFium; xem contact sheets và công thức ở độ phân giải đầy đủ.
- Đối chiếu source M4 và hồ sơ protocol/kết quả đã có; không dùng output cũ để chọn hyperparameter mới.
- Tính đại số bằng NumPy trên tensor giả: rank trước/sau envelope ≤4; norm bound; rank-1 checkerboard counterexample; factor-scaling symmetry; ảnh hưởng epsilon gần gốc; số tham số đầu ra.
- Kết quả CPU lưu ở `protocol_b_review/architecture_proposal_pdf_2026-10-24/algebra_checks.json`. Đây không phải TensorFlow forward/backward smoke và không phải thống kê residual trên Indian Pines thật.

**Kết luận vận hành:** GO cho hoàn thiện đặc tả và prototype có giới hạn. Chưa GO chạy dài nguyên bản 0.9. Không có lý do từ phép rà soát này để bỏ toàn bộ hướng residual GAN, và chưa có bằng chứng nó sẽ thắng V2 hoặc paper hiện đại.

## 8. Cập nhật sau phụ lục quyết định thiết kế v1.0

Đã đọc toàn bộ phụ lục người dùng gửi tại `C:/Users/ADMIN/.codex/attachments/ef110ad0-0dac-4882-9a41-fcf8d30b1a08/Pasted text.txt`. Phụ lục được dùng làm quyết định ưu tiên khi khác bản 0.9.

**Những vấn đề đã được xử lý thực chất:** H2 giới hạn đúng ở gói parameterization + capacity; rank audit dùng same-class patch differences; initializer và calibration tại effective residual; epsilon/nearzero policy; SIMPLE random rank4; R/ROS phân vai; D GAP được giữ có chủ ý và có center/context diagnostics; SN toàn-state freeze; same-update comparison; đầy đủ resume; tách G seed khỏi C seed. Không tiếp tục coi các mục này là lỗi chưa được nhận biết.

**Quyết định:** dùng v1.0 làm nền cho CPU audit và prototype/profiler hữu hạn. Không thêm kiến trúc/arm trong lần review này. Còn bốn quyết định thực thi cần điền:

1. ROS fallback nằm ở downstream-only hay cả GAN loop? Định nghĩa eligible anchor/support pools, class prior của D real/fake và đường gradient tương ứng. Nếu target D-real rộng hơn pool G, phải ghi rõ lý do/khả năng bao phủ; không mặc nhiên bắt hai pool giống nhau hoặc tự coi khác nhau là bug.
2. Khóa phân phối random U/V và radial scale của SIMPLE. Cùng R_a là cùng upper bound, chưa phải cùng realized perturbation magnitude. Không bắt match hậu nghiệm theo test; nếu chỉ match cap thì diễn giải là cap-matched control.
3. Phân biệt hard technical STOP (NaN, sai freeze/state/label/data contract) với cơ chế cần REVISE. G bỏ noise không tự bác H1 utility: một biến đổi phụ thuộc anchor nhưng xác định vẫn có thể hữu ích. Chỉ claim noise diversity phải bị giới hạn tương ứng.
4. Điền development manifest/count/hash, job matrix, endpoints/query limits và numeric decision rules trước lượt dài. Một G seed có thể tạo tín hiệu thăm dò để GO xác nhận; không đủ superiority ổn định nhưng không làm toàn bộ 30 giờ thăm dò vô giá trị.

Lưu ý tên metric: cosine trên PCA coefficients không tự có ý nghĩa spectral angle trên phổ vật lý. Nếu báo SAM, ghi rõ original/reconstructed spectrum và preprocessing.

Đây là bổ sung contract, không phải yêu cầu tạo thêm một kiến trúc hoặc chạy lại các thí nghiệm V2 lịch sử.
