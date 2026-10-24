# Đánh giá và chỉnh sửa notebook Protocol B — 05/10/2026

**Kết luận:** thiết kế B phù hợp với mục tiêu hiện tại, nhưng v1 có lỗi triển khai cần sửa trước GPU. Đã tạo `KLTN_3D_HyperGAMO_Protocol_B_v2_reviewed.ipynb` cạnh v1; v1 được giữ nguyên. Đây là đánh giá có kiểm soát dựa trên HyperGAMO, không phải tái lập nguyên trạng bảng điểm paper.

## Những lỗi đã xác nhận và sửa

| Vị trí v1 (cell index tính từ 0) | Vấn đề và bằng chứng | Chỉnh sửa v2 |
|---|---|---|
| 21 — Development split | MAT array lưu theo Fortran order. `dev_val_mask.ravel()[chosen]=True` ghi vào bản sao: trên dữ liệu thật v1 cho train4422/val0 rồi fail assertion. | Ghi trực tiếp tọa độ hàng/cột; xác nhận train4422/val1106 và đủ16lớp. |
| 37 — ROS | Bootstrapping mọi lớp làm mất tâm thật. Seed42 final chỉ giữ4746/5528 tâm; lớp lớn nhất giữ870/1390. | Giữ tất cả tâm gốc, chỉ thêm bản lặp phần thiếu. Final đủ5528 tâm thật,22240 presentations,1390/lớp. |
| 19 và các cache | Run ID bỏ qua nhiều cấu hình train; cache/model được dùng lại chỉ vì tồn tại file. | Run ID gồm full configuration và source digest; cache/preprocessing có contract/hashes; marker complete kiểm hash weights. |
| 6 — Nguồn GT | Tên GT không nhận alias chính thức `indian_pines_corrected_gt.mat`; local mặc định không trỏ đến bộ đã có. | Nhận alias/explicit path và chọn theo hash nguồn đã audit; nhiều bản cùng nội dung không gây chọn nhầm. |
| 48/53 — Final CNN | Giữ epoch của dev trên pool train lớn hơn làm thêm khoảng25% optimizer updates. | Khóa LR và đúng số optimizer updates được chọn; final có thể dừng giữa epoch, không dùng validation. |
| 61 — C-real-only | Cùng8000C updates nhưng Adam clock khác GAN; bỏ batch cuối28 và đổi cách đi qua mẫu. | Hai real calls mỗi vòng, cùng partition/tail; tiến clock qua các lượt D/G bị bỏ. Vẫn ghi giới hạn empirical-vs-deficit class prior. |
| 33 và hàm train | In GPU inventory nhưng có thể train nặng trên CPU; tự cài tf-keras2.20 có thể thay TF giữa kernel. | Huấn luyện cần GPU + smoke đã qua; kiểm TF/tf-keras cùng minor, không tự nâng runtime giữa phiên. |
| 37/53/61/67 — Job dài | Chỉ lưu cuối lượt; mất tiến độ khi phiên dừng; thiếu chặn NaN. | Recovery weights+optimizer+counters+NumPy/batch state; tuning/finalCNN lưu mỗi epoch, GAN/control mỗi500Cupdates; không ghi complete với NaN/Inf. |
| 65 — Smoke | Chỉ inference nên không biết C/D bị update sai hay G không học. | Smoke C-real,D-real,G-vs-C,G-vs-D; kiểm mạng thay đổi, mạng frozen/inactive, optimizer clock và checkpoint roundtrip. |
| 84/86 — Cổng test | `{}` cũng được coi là cải tiến đã khóa; không cần model cải tiến hoàn tất. | Quyết định có schema/lý do; registry kiểm mọi model/seed/hash/spec; snapshot cuối bất biến trước test. Có lựa chọn baseline-only được khai báo rõ nếu không có cải tiến có căn cứ. |
| 76 — Diagnostics | Fake-fake512 so với real-real10/16/21 có confound số mẫu; khoảng cách float32 có cancellation. | Float64 distance; lấy real/fake cùng số lượng, lặp32lần; random convex/copy/class mean controls và weight variance theo noise. |
| 12 — Chữ ký phổ | Trung bình phổ dùng cả nhãn test trước gate. | Hình chữ ký phổ chỉ dùng B train; bản đồ toàn cảnh chỉ sau test. |

## Những điều được giữ vì có cơ sở

- B có5528train/4721test, giữ16lớp, patch11×11, PCA40, imbalance139:1. Việc đổi B không chữa overlap.
- PCA/MinMax fit tâm dev train ở development, fit lại toàn bộ5528tâm ở final. Final khởi tạo weights từ đầu.
- C/G/D, MSE, generator softmax convex support, target `1-cond`, smoothing của D và shared Adam giữ đường tính của tác giả.
- 8000lần cập nhật C tương ứng4000vòng real/fake. Không tự đổi thành8000GAN cycles.
- Balanced validation NLL vẫn là tiêu chí chính được định trước của notebook v1; Macro-F1 và epoch làm tie-break. Đây là lựa chọn riêng, không được gọi là biện pháp sửa spatial leakage.
- Ba CNN thử4LR; GAN baseline giữ LRpaper. Được so sánh như các pipeline đã khai báo; không tuyên bố mọi phương pháp có cùng ngân sách tuning.

## Phạm vi kết luận

Train/validation/test không trùng tâm nhưng footprint vẫn giao pixel. Sau sửa mask, validation1106/1106 và test4721/4721 patch có giao với train. Khóa test chỉ loại việc chọn model bằng điểm test, không làm test độc lập không gian.

Oats có2mẫu validation và10mẫu test; lớp7 có3mẫu validation và12mẫu test. Một lỗi test đổi recall tương ứng10 và8,33điểm phần trăm. Báo per-class support; không suy rộng3seed cố định thành bất định do split hoặc cảnh mới.

Convex averaging tự có thể co phương sai. High entropy từng vector weights chưa chứng minh đa dạng qua noise; low variance/near-copy riêng lẻ chưa đủ kết luận generator hỏng. C-only control kiểm soát nhiều yếu tố hơn v1, nhưng vẫn cần đối chứng cùng lịch lấy real theo deficit nếu muốn cô lập riêng *nội dung tổng hợp* khỏi tăng lượt học lớp hiếm. Chưa tự thêm thuật toán GAN mới.

## Kiểm tra đã thực hiện

- Notebook schema và mọi code cell parse thành công.
- Run All mặc định `preflight` được kiểm trên bộ MAT nguồn thật, không huấn luyện hay tính điểm test.
- Split4422/1106, master5528/4721; regression trên chính MAT Fortran-layout qua.
- PCA/scaler train-only, cache reload và từ chối sai contract qua.
- ROS giữ mọi tâm gốc và count1390/lớp; khoảng cách rất nhỏ được tính ổn định; diagnostic cardinality matching qua.
- Cổng test từ chối dict rỗng và danh sách cải tiến rỗng.

**Chưa xác nhận runtime TensorFlow/GPU hoặc chạy đủ8000updates.** Máy local chưa có TF; bộ cài thử trong môi trường riêng không tải được. Strong smoke và checkpoint roundtrip đã có trong v2 nhưng phải chạy trên Kaggle GPU trước development/baseline. Resume không hứa bitwise-identical cho TF Dropout sau restart.

## Thứ tự chạy

1. Import v2 vào Kaggle; `CHE_DO_CHAY='preflight'`, Run All trên CPU. Cuối notebook in `CPU PREFLIGHT PASSED` và ZIP báo cáo nhỏ.
2. Bật GPU, restart kernel; `CHE_DO_CHAY='smoke'`. Nếu thiếu tf-keras, làm theo lệnh phiên bản khớp rồi restart. Chỉ đi tiếp khi smoke update/freeze/checkpoint qua.
3. `development`: tuning ba CNN; khóaLR/sốupdates.
4. `baseline`: final baCNN, C-only control, GAN ba seed; giữ config và source. Checkpoint tiếp tục khi run folder còn hoặc được phục hồi từ Kaggle Input.
5. `diagnostics`: đọc model đã hoàn thành; khảo sát và khóa quyết định cải tiến có căn cứ.
6. `test` + gateTrue: chỉ sau registry đủ artifacts. Mọi phương pháp khóa trước khi đọc điểm test.

Lưu Version kèm Output sau mỗi phiên. Gắn Output trước đó vào Input để v2 khôi phục thư mục run cùng cấu hình; không tiếp tục từ checkpoint A/v1.

Đối chiếu paper chi tiết được nối ở phần dưới.

---


# Đối chiếu Protocol B với paper và mã nguồn

Ngày đối chiếu: 05/10/2026. Đối tượng: `KLTN_3D_HyperGAMO_Protocol_B_v1.ipynb`.

Đây là đánh giá thiết kế thí nghiệm và đối chiếu tài liệu; không xác nhận notebook đã train thành công hay đã có kết quả test. Các nhận xét về triển khai v1 cần được đọc cùng báo cáo kiểm tra code và bản chỉnh sửa v2.

## Kết luận chính

Protocol B phù hợp để **đánh giá có kiểm soát các cách xử lý mất cân bằng trên một benchmark Indian Pines cố định trong cùng ảnh**. B thay đổi định nghĩa split, phạm vi fit tiền xử lý và quy tắc chọn mô hình so với mã công khai của tác giả. Vì vậy, không gọi B là tái lập nguyên trạng paper, và không dùng chênh lệch điểm với paper làm bằng chứng độc lập rằng một phương pháp tốt hơn.

## Bảng đối chiếu

| Thành phần | Paper hoặc mã công khai | Protocol B và cách diễn giải |
|---|---|---|
| Bài toán và cơ chế sinh | [Roy et al., 3D-HyperGAMO](https://profile.iiita.ac.in/srdubey/publications/2022_3D-HyperGAMO.pdf) dùng oversampling có điều kiện cho lớp thiểu số. [GAMO gốc](https://openaccess.thecvf.com/content_ICCV_2019/html/Mullick_Generative_Adversarial_Minority_Oversampling_ICCV_2019_paper.html) dùng generator tổ hợp lồi và trò chơi ba mạng. | B giữ cơ chế tổ hợp lồi của patch cùng lớp. Mục tiêu cần kiểm chứng là hiệu quả phân lớp, đặc biệt theo lớp hiếm; độ đa dạng mẫu sinh là chẩn đoán cơ chế. |
| Kiến trúc | [dense_net3D_SA.py, commit ghim](https://github.com/mhaut/3D-HyperGAMO/blob/3f92889aec4a7e2c8f7a2db4c8a9bf167f296f5e/dense_net3D_SA.py) định nghĩa C, D, G và head softmax trên support. | Kiến trúc v1 và phép nhân ma trận `weights @ support` tương đương cấu trúc nguồn về mặt toán học. Sự tương đương này chưa xác nhận cách đóng băng mạng hoặc cập nhật tham số khi chạy. |
| Thông số GAN | Paper Roy nêu LR 0,0002, latent 100, patch 11×11 và 40 thành phần sau giảm chiều. | Các giá trị B có cơ sở từ paper; không cần đổi chúng chỉ vì các paper khác dùng giá trị khác. |
| Ngân sách 8.000 | [main.py, commit ghim](https://github.com/mhaut/3D-HyperGAMO/blob/3f92889aec4a7e2c8f7a2db4c8a9bf167f296f5e/main.py) train C một lần trên real và một lần trên fake, rồi tăng `step` thêm 2. | 8.000 **lần cập nhật C** tương ứng 4.000 vòng real/fake. Phải phân biệt số vòng GAN, số lần cập nhật C/D/G/head và epoch. |
| Định nghĩa DIP | [auxil.py, commit ghim](https://github.com/mhaut/3D-HyperGAMO/blob/3f92889aec4a7e2c8f7a2db4c8a9bf167f296f5e/auxil.py) remap nhãn DIP rồi loại các tâm train khỏi GT test, không lọc bất đồng nhãn. Audit local xác nhận cách này cho 5.689/4.711. | B giữ train khi nhãn DIP khớp Standard GT, cho 5.528/4.721. Các count từng lớp khớp [Table 3 của TransHSI](https://www.mdpi.com/2072-4292/15/22/5331). Cách ghi đúng là “split nhãn nhất quán, khớp số mẫu DASE đã công bố”; chưa đủ bằng chứng khẳng định bảng điểm Roy dùng chính phép lọc B. |
| Tiền xử lý | `auxil.py` fit PCA và MinMax trên toàn bộ cube trước chia mẫu. | B fit trên tâm train của giai đoạn tương ứng. Đây là khác biệt protocol cần công bố. Train-only fit không loại pixel ngoài train khỏi vùng lân cận của patch. |
| Model selection | `main.py` xem test mỗi 500 step và lưu kết quả có test OA tốt nhất. | B khóa test và chọn đầu cuối GAN theo ngân sách cố định. Đây là cách chọn mô hình khác nguồn, nên điểm cuối B không so trực tiếp như cùng thí nghiệm với best-test score nguồn. |
| Validation | [Audebert et al.](https://arxiv.org/pdf/1904.10674) mô tả 5% train làm validation; [DeepHyperX công khai hiện tại](https://github.com/nshaud/DeepHyperX/blob/master/main.py) gọi `sample_gt(train_gt, 0.95, mode="random")`. | Dev validation 20% của B là thiết kế riêng. Không viện dẫn Audebert như bằng chứng rằng validation này độc lập về footprint patch. Mã hiện tại cũng không chứng minh chính xác trạng thái mã lúc xuất bản paper. |
| Balanced validation NLL | Không phải quy tắc model selection được lấy nguyên từ các nguồn trên. | Là tiêu chí liên tục, cho mỗi lớp đóng góp ngang nhau, giúp xử lý trial hòa Macro-F1. Nó không sửa chồng lấn và không biến validation thành phép đo tổng quát không gian. |
| CNN baseline | CNN-3D trong bảng Roy dẫn các phương pháp CNN3D trước đó; chưa có căn cứ coi đó là C real-only trong B. | Ba CNN dùng cùng C trong B giúp kiểm soát kiến trúc. Không lấy CNN3D 75,09% trong paper làm điểm dự kiến hay điểm bắt buộc của CNN thường B. |
| Chia dữ liệu và chồng lấn | [Liang et al.](https://arxiv.org/abs/1605.05829) và [nghiên cứu về spatial non-disjoint evaluation](https://ieeexplore.ieee.org/document/8046029/) giải thích sai lệch do phụ thuộc không gian và spatial features. [TAP-Net](https://www.mdpi.com/2072-4292/13/2/324) thay cách partition để kiểm soát leakage. | B giữ benchmark trong cùng ảnh và ghi rõ overlap. Khóa test bảo vệ model selection; không tạo độc lập không gian. TAP-Net là đối chiếu về protocol, không phải điểm số cùng điều kiện để xếp hạng trực tiếp. |

## Phạm vi công bằng và ablation

Ba CNN có cùng lưới 4 LR, cùng development split và cùng quy tắc chọn cấu hình. GAN giữ một LR và ngân sách từ nguồn. Điều này hợp lệ nếu câu hỏi là **so sánh các hệ thống theo quy trình được khai báo trước**, nhưng chưa phải tối ưu mọi phương pháp với ngân sách tìm kiếm bằng nhau, cũng không chứng minh GAN đã đạt cấu hình tối ưu.

CNN thường, ROS và class weight đều dùng C, nhưng ROS có nhiều lượt trình bày mẫu và bước cập nhật hơn khi đếm theo epoch. Phải lưu số bước, số mẫu trình bày, batch size và thời gian. Khi chuyển từ development train sang toàn bộ 5.528 mẫu, phải công bố quy tắc chuyển ngân sách; giữ cùng epoch sẽ làm số bước tăng.

Ưu tiên **ablation cùng C** trước khi đổi dataset: giữ kiến trúc, loss, optimizer và ngân sách C rõ ràng giúp xác định ảnh hưởng của mẫu sinh. Trong v1, `C-real-only matched` khớp kiến trúc, MSE, LR, β1 và số lần cập nhật C, nhưng chưa khớp hoàn toàn trạng thái optimizer: HyperGAMO dùng chung Adam giữa nhiều mạng, còn real-only dùng Adam riêng. Vì vậy, chưa diễn giải chênh lệch như đã loại hết confound tối ưu.

## Giới hạn của diagnostics mẫu sinh

- Generator tổ hợp lồi có thể làm phương sai giảm; `variance_ratio < 1` đơn lẻ không chứng minh mode collapse.
- Entropy/effective support đo mức phối hợp support **trong một mẫu**. Mọi noise cho cùng vector weights có entropy cao vẫn có thể tạo một mẫu trung bình giống nhau.
- Nearest-neighbor distance phụ thuộc số mẫu đối chiếu. So fake–fake trên 512 mẫu với real–real trên 10/16/21 mẫu có confound cardinality; cần nhóm cùng kích thước và phép tổng hợp đã khai báo trước.
- GAMO nhằm sinh mẫu khó ở gần biên lớp; “khó phân loại”, “gần support” và “generator thất bại” là các kết luận khác nhau. Chỉ kết luận cơ chế khi nhiều tín hiệu và ảnh hưởng phân lớp đồng thuận.
- Hình PCA 2D/3D là minh họa. Sự tách hoặc nhập cụm trên hình không tự chứng minh chất lượng mẫu sinh trong toàn bộ không gian patch.

## Kết luận được phép báo cáo

Kết quả cuối mô tả hiệu quả trên **fixed within-scene Indian Pines benchmark**, với patch vẫn có phụ thuộc không gian. Ba seed trên cùng split phản ánh biến thiên do huấn luyện ngẫu nhiên, không phản ánh biến thiên do chọn split hay khả năng sang scene mới.

Các lớp test rất nhỏ cần kèm support: Oats có 10 mẫu, một lỗi đổi recall 10 điểm phần trăm; Grass-pasture-mowed có 12 mẫu, một lỗi đổi khoảng 8,33 điểm. Báo OA cùng AA, Macro-F1, GM, per-class metrics và dự đoán từng seed. Không suy luận độc lập thống kê từ từng pixel khi các patch phụ thuộc.

Không cần mở lại khảo sát A/B để thực hiện đối chiếu này. Công việc trước train là sửa và xác nhận triển khai, khóa ngân sách/selection, rồi giữ nguyên protocol khi đánh giá các baseline và ablation. Báo cáo này không đề xuất một thuật toán GAN mới và không xác nhận thành công runtime.
