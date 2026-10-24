# Thiết kế lại GAN — phạm vi và lựa chọn sơ bộ cho giai đoạn thăm dò

Ngày: 24/10/2026. Không train, không chạy test mới. Đây là nghiên cứu lựa chọn cơ chế; chưa phải đặc tả kiến trúc đã khóa. Các câu hỏi GPT web người dùng nhắc trong tin nhắn mới chưa được đính kèm, nên chưa được đối chiếu ở bản này.

## 1. Phạm vi đã được người dùng cho phép

Indian Pines là dataset gốc/chính. Giữ GAN kế thừa bài toán/cơ chế sinh mẫu hỗ trợ phân loại của 3D-HyperGAMO. Có thể thiết kế lại G, D, C, loss, biểu diễn, preprocessing, lịch học và protocol. Không bắt buộc giữ V2 hoặc recipe paper. Có thể bổ sung dataset ở giai đoạn sau.

30 giờ GPU là ngân sách phát triển để tạo bàn đạp cho cải tiến cuối. Nó không mặc định phải đủ toàn bộ seed, ablation và dataset xác nhận cuối cùng. Mục tiêu là thu được một nền tảng có thể tiếp tục, cùng bằng chứng đủ để chọn hoặc loại cơ chế; không chỉ thêm một bảng test B.

## 2. Câu hỏi nên phục vụ

> Có thể học các biến thiên phổ–không gian hữu ích của lớp thiểu số từ ít mẫu thật, thay cho việc chỉ chọn/trộn support hữu hạn, và chứng minh lợi ích trước một classifier mạnh cùng các đối chứng augmentation phù hợp hay không?

Đây là câu hỏi ứng viên. Việc vượt ra ngoài convex hull chỉ mở rộng miền biểu diễn; nó không tự bảo đảm giữ nhãn hoặc tăng F1.

## 3. Những cơ chế có căn cứ để chuyển giao

| Nguồn sơ cấp | Cơ chế có ích cho thiết kế | Điều không được suy quá |
|---|---|---|
| [3D-HyperGAMO, code tác giả](https://github.com/mhaut/3D-HyperGAMO) | Sinh mẫu phục vụ phân loại mất cân bằng, G/D/C và convex support synthesis | Không bắt buộc giữ Dense theo số support hoặc shared Adam khi đã tuyên bố redesign |
| [SMPCA-GAN, JSTARS 2023](https://research.manchester.ac.uk/en/publications/permuted-kpca-and-smote-to-guide-gan-based-oversampling-for-imbal/) | Hướng dẫn G bằng pattern có cấu trúc thay pure noise; kiểm soát lẫn lớp | Lẫn phân phối lớp khác spatial train–test source reuse |
| [S2A-Swin, 2026](https://www.mdpi.com/2072-4292/18/6/935) | Conditional WGAN-GP, regularization phổ/không gian, classifier spectral–spatial | Không áp smoothness theo bước sóng trực tiếp trên thứ tự PCA components |
| [3D-GIGAMO, 2026](https://www.mdpi.com/2072-4292/18/19/3343) | Real-patch guidance, G implicit, conditional D và HCT classifier | OA IP 99,37% là điểm toàn hệ thống trên random 10% train; không phải Macro-F1, lợi ích riêng của G hoặc bằng chứng spatial-independent |
| [ReACGAN, NeurIPS 2021](https://proceedings.neurips.cc/paper_files/paper/2021/hash/c5ab6cebaca97f7171139e4d414ff5a6-Abstract.html) | Chuẩn hóa feature và data-to-data/class relationships để ổn định conditional generation | Kết quả RGB không chứng minh đủ dữ liệu cho HSI có vài support hiếm |
| [ADC-GAN, ICML 2022](https://proceedings.mlr.press/v162/hou22a.html) | Auxiliary discriminative classifier xét class và real/fake; chỉ ra rủi ro giảm diversity của class-consistency thông thường | Không coi CE đúng nhãn là bảo đảm đồng thời về semantics và diversity |
| [BAGAN-GP, tác giả](https://arxiv.org/abs/2011.00189) | Supervised autoencoder initialization và học cấu trúc chung giữa lớp trước khi sinh minority | Pretraining vẫn có thể bị majority chi phối; cần thiết kế sampling phù hợp |
| [GAN limited data, NeurIPS 2020](https://proceedings.neurips.cc/paper_files/paper/2020/hash/8d30aa96e72440759f74bd2306c1fa3d-Abstract.html) | Discriminator overfit là một failure mode; augmentation có thể giúp | Không áp color/cutout RGB máy móc lên phổ hoặc tâm nhãn HSI |

Các nguồn cho phép tổng hợp một thiết kế có cơ sở. Chúng chưa chứng minh tổ hợp mới sẽ thắng hoặc có tính mới thuật toán.

## 4. Hai họ kiến trúc đáng cân nhắc

### A. Shared real-conditioned residual patch GAN — ưu tiên về mục tiêu sinh patch

Luồng:

```text
anchor HSI + class + noise
          ↓
shared spectral–spatial encoder / conditional residual G
          ↓
synthetic HSI patch
       ↙        ↘
conditional D   classifier C
```

G nhận anchor trực tiếp và dùng chung tham số giữa các lớp. Noise phải đi vào phép biến đổi; đầu ra có thể là residual quanh patch thật. Không cần ma trận Dense có số cột bằng số support mỗi lớp, và không giới hạn toàn bộ output vào convex combination của support đã thấy.

D xét real/fake có điều kiện lớp. C là một classifier phổ–không gian đủ mạnh, nhưng giữ cùng C trong so sánh lợi ích riêng của bộ sinh. Khởi tạo qua representation/reconstruction học từ train là một khả năng, không phải module mặc định bắt buộc.

Động cơ: chia sẻ kiến thức biến thiên giữa lớp; dùng patch thật làm cấu trúc nền; học biến đổi thay cho lookup support. Đây là suy luận thiết kế, không phải kết quả đã có.

Rủi ro cần phân biệt: G trả lại anchor; noise không ảnh hưởng; D ghi nhớ; biến đổi từ majority làm sai rare class; thay context nhưng làm ngữ nghĩa tâm nhãn sai. Residual nhỏ không tự chứng minh đúng lớp.

### B. Conditional feature-space GAN — phương án phát triển nhẹ hơn

Encoder E học representation phổ–không gian; G sinh feature có điều kiện theo class/anchor; D xét feature–class; C học từ real và generated features. Có thể cache train embeddings để làm nhiều đối chứng rẻ.

[f-CLSWGAN, CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Xian_Feature_Generating_Networks_CVPR_2018_paper.html) cung cấp cơ sở feature generation; [FSHyperRGAN, TGRS 2024](https://ieeexplore.ieee.org/document/10741331/) là cơ sở gần HSI cho adversarial generation, embedding/reconstruction và relational matching. Mục tiêu của các paper không trùng hoàn toàn bài toán hiện tại.

Rủi ro: E có thể làm mất thông tin minority trước khi G hoạt động; G không khôi phục được những semantics đó. Feature win không chứng minh sinh patch vật lý tốt. Đổi E và GAN cùng lúc chỉ cho phép kết luận toàn hệ thống nếu không có đối chứng cùng E.

**Quyết định sơ bộ:** ưu tiên đặc tả A nếu mục tiêu cuối là sinh patch HSI có ích. B là phương án so sánh ở mức thiết kế/đánh giá hoặc dự phòng nếu patch synthesis không phù hợp ngân sách. Chưa full-train cả hai họ.

## 5. Không cần bỏ toàn bộ ý tưởng trust-region

Điểm nên giữ là kiểm soát class intrusion và độ lớn biến đổi. Hai lỗi của bản cũ có thể sửa:

1. `q=[center,mean]` không quan sát mọi chiều patch. Cần giới hạn full-patch norm hoặc một subspace được kiểm hạng; không gọi q-distance là giấy chứng nhận nhãn.
2. Không học rho rồi clip theo 1/rho khiến amplitude bị triệt. Có thể dùng parameterization trực tiếp `delta = R * v / sqrt(1+||v||^2)`. Gradient không bị triệt đại số ở mọi điểm ngoài biên như hard clipping, nhưng vẫn có thể nhỏ khi v lớn.

Một khả năng khác là giới hạn theo hướng, thay vì một quả cầu bị bó bởi mẫu khác lớp gần nhất. Với `x=a+t*u`, `||u||=1`, điều kiện gần a hơn support khác lớp b tương đương:

`2*t*dot(u,b-a) <= ||b-a||^2`.

Hướng đi xa b không bị b giới hạn. Chỉ các đối thủ có dot dương đặt upper bound cho t. Có thể parameterize t bên trong miền hợp lệ bằng sigmoid, cùng một giới hạn locality hữu hạn. Đây vẫn chỉ là ràng buộc theo support quan sát được, không phải bảo toàn nhãn thật. Hình học Voronoi/safe oversampling có prior art; chưa xác minh tính mới của kết hợp cụ thể này.

Các sửa chữa trên tạo ứng viên có thể kiểm nghiệm. Chưa đủ lý do để đồng thời thêm directional geometry, hardness loss, teacher mới và Transformer vào một lượt.

## 6. Giao thức phục vụ phát triển

- Protocol B và output cũ giữ nguyên làm lịch sử, không dùng test B để tune mô hình mới.
- Một all-16 limited-label development panel có thể đo bài toán imbalance, nhưng phải báo overlap thật.
- Một spatial stress panel trên các lớp/khối còn đủ geometry có thể xem lợi ích có giữ được khi giảm source reuse. Không gộp điểm của panel thiếu lớp vào Macro-F1 all-16.
- Nếu chọn patch nhỏ hơn để có all-16 source-disjoint development, trước hết kiểm feasibility và counts bằng CPU; không chọn patch vì điểm accuracy đẹp.
- Một split mới cùng Indian Pines không biến thành dataset/cảnh xác nhận độc lập. Dataset bổ sung ở giai đoạn sau là lựa chọn được người dùng cho phép.

## 7. Thiết kế đối chứng phải thể hiện lợi ích thực

Cùng một C mạnh, dữ liệu và exposure phù hợp:

1. Real-only với một chiến lược imbalance ở loss/sampling được định trước.
2. Matched ROS và một augmentation/interpolation đơn giản phù hợp.
3. Conditional GAN scaffold đối chứng.
4. Scaffold cùng đúng một cơ chế cải tiến đang nghiên cứu.

Original HyperGAMO/V2 cũ chỉ là điểm tham chiếu lịch sử nếu đã đổi split, C hoặc preprocessing. Muốn so trực tiếp, phải đưa các cách sinh mẫu vào cùng thiết lập mới.

Không tự gọi điểm toàn hệ thống tăng là lợi ích G. Có thể báo hai câu hỏi riêng: hệ mới tốt hơn đến đâu, và trong cùng hệ classifier/encoder đó GAN góp được bao nhiêu.

## 8. 30 giờ là trần đầu tư phát triển, không phải dự báo thời gian

Phân bổ tham khảo cần hiệu chỉnh theo profiler: tối đa 3 giờ plumbing/smoke/profile; 7 giờ representation và baselines; 12 giờ một candidate và lặp kiểm tra; 5 giờ một ablation quan trọng; 3 giờ dự phòng/inference/lưu kết quả.

Pilot đạt kỹ thuật thì tiếp tục từ checkpoint của chính pilot nếu recipe giữ nguyên, không train lại từ đầu. Cache preprocessing/embedding chỉ dùng lại trong cùng split/fit scope. Lưu G/D/C, optimizer, RNG, counters và config/hash để resume.

Không ép 8.000 C updates của M4 lên một mô hình mới. Khóa budget phù hợp trước khi đọc held-out scores; ghi cả training exposure và compute thực.

Mục tiêu cuối giai đoạn: một scaffold GAN đủ tốt để tiếp tục, kết quả đối chứng development, những failure mode đã định lượng và quyết định cơ chế nào đáng phát triển cuối. Chưa phải bằng chứng một model hoàn hảo, SOTA hoặc đã giải quyết leakage.

## 9. Việc chưa được chốt

- Cấu hình số của encoder/G/D/C, loss cuối và optimizer.
- Hướng residual trong raw bands hay PCA space; spectral priors phải tương thích representation.
- Protocol development, class counts và panel geometry thực tế.
- Giữ controlled hardness, geometry control hay chỉ một conditional residual baseline trước.
- Dự toán runtime và scope phù hợp GPU thực.
- Các câu hỏi GPT web người dùng sẽ gửi tiếp.

Đây là các quyết định thiết kế cần hoàn thiện trước runner, không phải yêu cầu chạy lại toàn bộ chẩn đoán V2 lịch sử.
