# Nền tảng và thiết kế nghiên cứu

## 1. Tóm tắt đề tài bằng ngôn ngữ đơn giản

Trong một tập dữ liệu ảnh phân loại, có thể tồn tại lớp có rất nhiều ảnh và lớp
chỉ có rất ít ảnh. Mô hình phân loại thường học tốt lớp đa số nhưng bỏ sót lớp
thiểu số. Đề tài nghiên cứu việc dùng mạng sinh đối nghịch (GAN) để học phân
phối ảnh của lớp thiểu số và tạo thêm ảnh mới. Tuy nhiên, ảnh GAN sinh ra không
được mặc nhiên coi là dữ liệu tốt. Chúng phải được đánh giá về chất lượng, tính
đúng lớp, độ đa dạng và giá trị đối với bài toán phân loại trước khi được đưa vào
tập huấn luyện.

Ý tưởng trung tâm của đề tài là:

> Tối ưu GAN trước, sau đó chọn lọc ảnh sinh phù hợp, rồi kiểm chứng xem tập ảnh
> đã chọn có giúp mô hình phân loại nhận diện lớp thiểu số tốt hơn các phương
> pháp cân bằng dữ liệu truyền thống hay không.

Đề tài không nhằm chứng minh trước rằng GAN luôn tốt hơn. Kết luận phải được rút
ra từ thực nghiệm có kiểm soát.

## 2. Bài toán thực tế

Giả sử tập ảnh huấn luyện có hai lớp:

- Lớp đa số: 5.000 ảnh.
- Lớp thiểu số: 500 ảnh.

Nếu mô hình luôn dự đoán mọi ảnh là lớp đa số thì độ chính xác đã đạt khoảng
90,9%, nhưng Recall của lớp thiểu số bằng 0. Ví dụ này cho thấy Accuracy có thể
cao trong khi mô hình hoàn toàn thất bại ở lớp quan trọng.

Mất cân bằng làm mô hình có xu hướng:

1. tối ưu quyết định theo lớp xuất hiện nhiều;
2. học biểu diễn kém đầy đủ cho lớp thiểu số;
3. đặt biên quyết định bất lợi cho lớp thiểu số;
4. đạt Accuracy cao nhưng Macro F1, Balanced Accuracy hoặc Recall lớp thiểu số
   thấp.

Trong ảnh y tế, lớp thiểu số thường là ca bệnh hiếm hoặc tổn thương nguy hiểm.
Thu thập thêm ảnh thật có thể khó vì chi phí, quyền riêng tư và yêu cầu chuyên
gia gán nhãn. Tổng quan hệ thống về tăng cường dữ liệu ảnh y tế cũng chỉ ra rằng
các kỹ thuật tăng cường trải dài từ biến đổi hình học đơn giản tới mô hình sinh,
và hiệu quả phụ thuộc vào cơ quan, phương thức ảnh và tác vụ cụ thể
([Garcea et al., 2023](https://pubmed.ncbi.nlm.nih.gov/36549032/)).

## 3. Đầu vào và đầu ra

### 3.1. Đầu vào

- Tập ảnh có nhãn và mất cân bằng theo lớp.
- Tập train, validation và test được chia trước mọi thao tác cân bằng.
- Vector nhiễu `z` và nhãn lớp `y` nếu dùng GAN có điều kiện.
- Cấu hình huấn luyện: kích thước ảnh, batch size, learning rate, số epoch,
  kiến trúc và seed.

### 3.2. Đầu ra trung gian

- Generator và discriminator/critic đã huấn luyện.
- Ảnh nhân tạo cho từng lớp thiểu số.
- Điểm chất lượng, tính đúng lớp, độ đa dạng và độ gần trùng của ảnh sinh.
- Tập ảnh sinh đã được chọn lọc.

### 3.3. Đầu ra cuối cùng

- Mô hình phân loại ảnh sau khi huấn luyện trên từng phương án cân bằng.
- Bảng so sánh Recall lớp thiểu số, Macro F1, Balanced Accuracy và PR-AUC.
- Đánh giá chất lượng và độ bao phủ của GAN.
- Kết luận về điều kiện mà dữ liệu GAN có ích, không có ích hoặc gây hại.

Luồng tổng quát:

```text
Ảnh train lớp thiểu số ---> GAN ---> Nhiều ảnh ứng viên
                                      |
                                      v
                         Đánh giá và chọn lọc ảnh
                                      |
Ảnh train thật -----------------------+---> Huấn luyện classifier
                                                   |
Ảnh test thật, chưa từng sử dụng ------------------+---> Đánh giá
```

## 4. GAN hoạt động như thế nào?

GAN gốc gồm hai mạng nơ-ron được huấn luyện đối nghịch
([Goodfellow et al., 2014](https://arxiv.org/abs/1406.2661)):

- **Generator `G`** biến vector nhiễu `z` thành ảnh giả `G(z)`.
- **Discriminator `D`** cố phân biệt ảnh thật với ảnh do `G` sinh.

Trò chơi minimax ban đầu được viết:

```text
min_G max_D V(D,G)
= E_x~pdata[log D(x)] + E_z~pz[log(1 - D(G(z)))]
```

Trực giác:

1. `D` học phát hiện ảnh giả.
2. `G` nhận gradient gián tiếp từ `D` và học cách tạo ảnh khó bị phát hiện hơn.
3. Hai mạng cùng thay đổi nên mục tiêu của mỗi mạng không đứng yên.

DCGAN thay các mạng kết nối đầy đủ bằng kiến trúc tích chập và đưa ra một số
ràng buộc kiến trúc giúp GAN phù hợp hơn với ảnh
([Radford et al., 2015](https://arxiv.org/abs/1511.06434)). Đây là baseline dễ
hiểu, nhưng chưa chắc là lựa chọn cuối cùng khi dữ liệu ít hoặc huấn luyện bất
ổn định.

Với dữ liệu nhiều lớp, Conditional GAN đưa nhãn `y` vào cả generator và
discriminator, giúp yêu cầu mô hình sinh đúng lớp
([Mirza & Osindero, 2014](https://arxiv.org/abs/1411.1784)):

```text
x_fake = G(z, y)
```

Điều này phù hợp hơn việc huấn luyện một GAN hoàn toàn độc lập cho mỗi lớp nhỏ,
nhưng hiệu quả thực tế vẫn phải kiểm chứng.

## 5. Vì sao GAN có thể sinh ảnh kém?

Không nên gom mọi nguyên nhân vào câu “GAN chưa tối ưu”. Có ít nhất bốn nhóm
nguyên nhân khác nhau.

### 5.1. Cấu hình và quá trình huấn luyện chưa tốt

- Learning rate quá lớn làm loss dao động; quá nhỏ làm học rất chậm.
- Discriminator quá mạnh khiến generator nhận tín hiệu gradient kém.
- Generator quá mạnh tạm thời có thể khai thác điểm yếu của discriminator.
- Batch size, normalization, optimizer hoặc kiến trúc không phù hợp.
- Chọn checkpoint chỉ dựa trên epoch cuối thay vì validation metric.

Đây là phần có thể cải thiện bằng tuning, theo dõi metric và lựa chọn checkpoint.

### 5.2. Dữ liệu thật có giới hạn

GAN chỉ học từ dữ liệu được cung cấp. Nếu lớp thiểu số quá ít, gán nhãn sai,
chứa ảnh mờ hoặc thiếu các kiểu hình quan trọng, GAN không thể tự biết đầy đủ
phân phối ngoài dữ liệu đó. Generator có thể ghi nhớ, pha trộn sai đặc trưng hoặc
chỉ tái tạo một số kiểu phổ biến.

Vì vậy, GAN không tạo ra “tri thức y khoa mới” một cách đáng tin cậy. Nó xấp xỉ
phân phối quan sát được từ mẫu huấn luyện.

### 5.3. Bản chất tối ưu đối nghịch không ổn định

Hai mạng cùng học tạo ra một trò chơi động thay vì một mục tiêu tối ưu cố định.
Một lỗi điển hình là **mode collapse**: nhiều vector `z` khác nhau tạo ra ảnh
gần giống nhau. Ảnh có thể trông đẹp nhưng chỉ bao phủ một phần nhỏ phân phối
thật.

WGAN-GP thay cách huấn luyện GAN bằng critic và gradient penalty để thực thi
ràng buộc Lipschitz ổn định hơn so với weight clipping
([Gulrajani et al., 2017](https://arxiv.org/abs/1704.00028)). Đây là một ứng viên
để so sánh với DCGAN, không phải sự bảo đảm tuyệt đối rằng ảnh luôn tốt.

### 5.4. “Ảnh đẹp” khác “ảnh hữu ích”

Một ảnh có thể nhìn hợp lý với mắt người nhưng:

- không chứa dấu hiệu phân biệt của đúng lớp;
- quá giống ảnh đã có nên không thêm thông tin;
- chứa artifact mà classifier lợi dụng như một đường tắt;
- gần biên hoặc sai nhãn;
- làm thay đổi phân phối train theo hướng xa dữ liệu test thật.

Do đó, mục tiêu cuối không chỉ là giảm FID hoặc tạo ảnh đẹp. Phải đo tác động
trên classifier bằng tập test thật độc lập.

## 6. Thế nào là “GAN đã được tối ưu”?

Trong nghiên cứu thực nghiệm, không thể chứng minh một mạng sâu đã đạt tối ưu
toàn cục. Cụm từ phù hợp hơn là **GAN đã được chọn cấu hình và checkpoint theo
một quy trình hợp lý, công khai và tái lập được**.

Quy trình dự kiến:

1. Cố định cách chia dữ liệu và seed.
2. Chọn một không gian siêu tham số nhỏ có lý do.
3. Huấn luyện nhiều cấu hình trên train.
4. Theo dõi loss nhưng không dùng loss làm tiêu chí duy nhất.
5. Đánh giá ảnh tại các checkpoint bằng validation data và metric sinh ảnh.
6. Kiểm tra lưới ảnh với cùng một tập vector `z` cố định qua các epoch.
7. Kiểm tra độ đa dạng và dấu hiệu mode collapse.
8. Chọn checkpoint theo quy tắc đã khai báo trước.
9. Lặp lại cấu hình tốt nhất trên nhiều seed.

Có thể thử TTUR, tức learning rate khác nhau cho generator và discriminator.
Công trình đề xuất TTUR cũng giới thiệu FID để đo khoảng cách giữa biểu diễn của
phân phối ảnh thật và ảnh sinh
([Heusel et al., 2017](https://arxiv.org/abs/1706.08500)).

## 7. Đánh giá GAN theo nhiều chiều

Không có một metric đơn lẻ trả lời đầy đủ “GAN có tốt không”. Đề tài nên dùng
nhiều tầng đánh giá.

### 7.1. Fidelity: ảnh có giống miền ảnh thật không?

- Quan sát lưới ảnh theo checkpoint.
- FID giữa ảnh thật và ảnh sinh; thấp hơn thường biểu thị hai phân phối gần hơn
  trong không gian đặc trưng.
- Nếu là ảnh chuyên ngành, cân nhắc trích đặc trưng từ backbone phù hợp miền dữ
  liệu thay vì chỉ phụ thuộc Inception huấn luyện trên ImageNet.

FID phụ thuộc kích thước mẫu và backbone, vì vậy chỉ so sánh khi dùng cùng quy
trình, cùng số ảnh và cùng implementation.

### 7.2. Diversity/Coverage: GAN có bao phủ nhiều kiểu ảnh không?

- Khoảng cách cặp trong không gian embedding.
- Tìm nearest neighbor giữa ảnh sinh và train để phát hiện sao chép.
- Precision/Recall cho mô hình sinh: precision gần với fidelity, recall gần với
  độ bao phủ phân phối. Cách tách hai chiều này được đề xuất vì một điểm tổng hợp
  như FID khó phân biệt các kiểu thất bại
  ([Sajjadi et al., 2018](https://arxiv.org/abs/1806.00035)).

### 7.3. Class consistency: ảnh có đúng lớp không?

Dùng một classifier tham chiếu được huấn luyện chỉ từ ảnh thật để ước lượng xác
suất ảnh sinh thuộc lớp yêu cầu. Tuy nhiên, classifier có thể sai hoặc thiên lệch,
nên điểm này chỉ là tín hiệu lọc, không phải “chân lý”. Với ảnh y tế, một tập mẫu
nhỏ nên được chuyên gia hoặc người hướng dẫn kiểm tra nếu có điều kiện.

### 7.4. Utility: ảnh có giúp nhiệm vụ cuối không?

Đây là tầng quyết định. Giữ nguyên kiến trúc classifier và quy trình huấn luyện,
chỉ thay chiến lược dữ liệu. Sau đó đo trên test thật:

- Recall theo từng lớp.
- Macro F1.
- Balanced Accuracy.
- PR-AUC, đặc biệt khi lớp dương hiếm.
- Confusion matrix.

## 8. Phương pháp chọn lọc ảnh đề xuất

Phương pháp ban đầu không cần tuyên bố là một thuật toán hoàn toàn mới. Nó là
một pipeline có giả thuyết rõ và có thể thực nghiệm loại bỏ từng thành phần.

Mỗi ảnh sinh `x_g` có thể nhận ba nhóm điểm:

```text
S_class(x_g):     mức nhất quán với nhãn cần sinh
S_realism(x_g):   mức gần miền ảnh thật
S_diversity(x_g): đóng góp về độ đa dạng, không gần trùng
```

Một quy tắc tổng hợp đơn giản:

```text
S(x_g) = w1*S_class + w2*S_realism + w3*S_diversity
```

Quy trình chọn:

1. Sinh nhiều ảnh ứng viên hơn số lượng cần bổ sung.
2. Loại ảnh có class confidence dưới ngưỡng.
3. Loại ảnh quá xa cụm ảnh thật trong embedding space.
4. Loại ảnh quá gần ảnh train để giảm nguy cơ ghi nhớ.
5. Gom cụm hoặc chọn theo khoảng cách để giữ độ đa dạng.
6. Lấy top-k hoặc lấy mẫu có trọng số từ tập còn lại.

Các trọng số và ngưỡng chỉ được chọn trên train/validation. Không nhìn test để
chọn ngưỡng.

Rủi ro quan trọng: nếu chỉ lấy ảnh classifier tự tin nhất, hệ thống có thể giữ
toàn “mẫu dễ” và loại các ca khó hữu ích. Vì vậy cần so sánh ít nhất ba chính
sách:

- Không lọc.
- Chỉ lọc chất lượng/đúng lớp.
- Lọc chất lượng kết hợp đa dạng.

## 9. Câu hỏi và giả thuyết nghiên cứu

### RQ1 — GAN nào phù hợp hơn trong điều kiện dữ liệu lớp thiểu số hạn chế?

So sánh DCGAN baseline với cGAN hoặc WGAN-GP bằng fidelity, coverage, độ ổn
định giữa các seed và chi phí huấn luyện.

### RQ2 — Tối ưu GAN có đủ để mọi ảnh sinh đều hữu ích không?

So sánh classifier dùng toàn bộ ảnh từ checkpoint tốt với classifier dùng ảnh
được chọn lọc từ cùng checkpoint.

### RQ3 — Chọn lọc ảnh có cải thiện phân loại lớp thiểu số không?

Giả thuyết H1: GAN có lọc đạt Macro F1 và Recall lớp thiểu số cao hơn GAN không
lọc trong cùng ngân sách số ảnh bổ sung.

### RQ4 — Càng thêm nhiều ảnh GAN có càng tốt không?

Thử các tỷ lệ bổ sung 25%, 50%, 100% khoảng thiếu hụt và cân bằng hoàn toàn.
Giả thuyết H2: hiệu quả tăng tới một mức rồi bão hòa hoặc giảm khi ảnh sinh bắt
đầu lấn át ảnh thật.

### RQ5 — GAN có hơn phương pháp truyền thống không?

So sánh với random oversampling, augmentation hình học/màu phù hợp, class
weight và focal loss. Không coi SMOTE trực tiếp trên pixel là baseline mặc định;
nếu dùng SMOTE, nên áp dụng trên không gian đặc trưng và giải thích giới hạn.

## 10. Thiết kế thực nghiệm

### 10.1. Nguyên tắc chống rò rỉ dữ liệu

1. Chia dữ liệu theo đối tượng/bệnh nhân nếu dữ liệu có nhiều ảnh của cùng một
   người; không chia ngẫu nhiên từng ảnh làm cùng bệnh nhân xuất hiện ở train và
   test.
2. Chỉ train GAN bằng train split.
3. Chỉ fit bộ lọc, scaler, embedding rule và threshold trên train/validation.
4. Test split giữ nguyên, không cân bằng và chỉ dùng ở đánh giá cuối.

### 10.2. Ma trận thí nghiệm tối thiểu

| Mã | Dữ liệu huấn luyện classifier | Mục đích |
|---|---|---|
| E0 | Ảnh thật mất cân bằng | Baseline |
| E1 | Random oversampling | So sánh sao chép mẫu |
| E2 | Augmentation truyền thống | So sánh biến đổi ảnh |
| E3 | Class weight hoặc focal loss | So sánh can thiệp ở loss |
| E4 | Ảnh thật + GAN không lọc | Đo utility của GAN thô |
| E5 | Ảnh thật + GAN lọc đúng lớp/chất lượng | Đo tác dụng lọc |
| E6 | Ảnh thật + GAN lọc chất lượng và đa dạng | Phương pháp đầy đủ |

### 10.3. Kiểm soát để so sánh công bằng

- Cùng train/validation/test split.
- Cùng classifier, augmentation nền, số epoch và early stopping.
- Cùng số ảnh bổ sung giữa E1, E2, E4, E5 và E6 khi so sánh trực tiếp.
- Tối thiểu ba seed nếu tài nguyên cho phép.
- Báo trung bình và độ lệch chuẩn, không chỉ báo lần chạy tốt nhất.
- Tách tuning GAN khỏi tuning classifier.

### 10.4. Ablation study

Ablation trả lời thành phần nào thực sự tạo ra cải thiện:

```text
Full filter
- bỏ class consistency
- bỏ realism constraint
- bỏ diversity constraint
- thay top-k bằng random selection
```

Nếu bỏ một thành phần mà kết quả không thay đổi, chưa có bằng chứng rằng thành
phần đó cần thiết.

## 11. Cách diễn giải kết quả

### Trường hợp A: FID tốt hơn và classifier tốt hơn

Có bằng chứng nhất quán rằng ảnh sinh vừa gần phân phối thật vừa hữu ích cho tác
vụ. Vẫn cần xem Recall từng lớp và nearest-neighbor để loại khả năng sao chép.

### Trường hợp B: FID tốt hơn nhưng classifier không tốt hơn

Chất lượng thị giác/phân phối tổng quát chưa chuyển thành thông tin phân biệt
lớp. Bộ lọc có thể đang chọn mẫu dễ, hoặc FID không nhạy với dấu hiệu chuyên
ngành.

### Trường hợp C: FID không tốt nhưng classifier tốt hơn

Ảnh sinh có thể hoạt động như regularization hoặc bổ sung biến thiên quanh biên
quyết định. Không nên kết luận ảnh có tính chân thực chuyên môn chỉ từ utility.

### Trường hợp D: GAN kém hơn augmentation truyền thống

Đây vẫn là kết quả nghiên cứu hợp lệ. Có thể kết luận rằng ở quy mô dữ liệu,
miền ảnh và ngân sách tính toán đã xét, GAN không tạo lợi ích tương xứng; phân
tích nguyên nhân và điều kiện thất bại chính là đóng góp.

## 12. Phạm vi phù hợp cho khóa luận

Để tránh đề tài quá rộng, phiên bản khả thi là:

- Một bộ dữ liệu chính có mất cân bằng tự nhiên.
- Một classifier cố định, ví dụ ResNet18.
- DCGAN làm baseline sinh ảnh.
- Một biến thể ổn định/có điều kiện, ví dụ cGAN hoặc WGAN-GP.
- Một pipeline lọc ba tiêu chí.
- Sáu đến bảy nhóm thí nghiệm và ablation.
- Ảnh 64x64 hoặc 128x128 ở giai đoạn đầu, sau đó chỉ tăng độ phân giải nếu GPU
  và thời gian cho phép.

Nghiên cứu trên dữ liệu y tế chỉ nên được mô tả là hỗ trợ bài toán phân loại
trong phạm vi bộ dữ liệu công khai, không tuyên bố khả năng chẩn đoán lâm sàng.
Một tổng quan gần đây về bài toán ít dữ liệu y tế ghi nhận hạn chế phổ biến về
external validation và khả năng công khai dữ liệu/mã nguồn; điều này củng cố
nhu cầu báo cáo minh bạch và tránh kết luận vượt quá dữ liệu
([Piffer et al., 2024](https://pubmed.ncbi.nlm.nih.gov/39655846/)).

## 13. Đóng góp kỳ vọng

Nếu thực nghiệm được thực hiện đầy đủ, khóa luận có thể trình bày bốn đóng góp:

1. Một quy trình tái lập để huấn luyện và đánh giá GAN trong điều kiện lớp thiểu
   số hạn chế.
2. Một cơ chế chọn lọc ảnh sinh dựa trên tính đúng lớp, độ gần dữ liệu thật và
   độ đa dạng.
3. Một đánh giá có kiểm soát giữa GAN không lọc, GAN có lọc và các baseline
   truyền thống.
4. Phân tích mối quan hệ giữa chất lượng mô hình sinh và utility downstream,
   bao gồm cả trường hợp GAN không giúp classifier.

Điểm mới không nằm ở câu “dùng GAN sinh ảnh”, mà ở câu:

> Không phải mọi ảnh GAN sinh ra đều có giá trị như nhau; khóa luận xây dựng và
> kiểm chứng một quy trình tối ưu–đánh giá–chọn lọc nhằm giữ những ảnh có khả
> năng đóng góp tốt hơn cho nhận diện lớp thiểu số.

## 14. Những việc cần quyết định tiếp theo

1. Chọn miền dữ liệu và bộ dữ liệu cụ thể.
2. Kiểm tra giấy phép, số lớp, số ảnh, độ phân giải và đơn vị chia dữ liệu.
3. Đo tỷ lệ mất cân bằng thực tế.
4. Xác định GPU, bộ nhớ và thời gian huấn luyện khả dụng.
5. Chốt bài toán nhị phân hay đa lớp.
6. Chốt DCGAN + cGAN hay DCGAN + WGAN-GP.
7. Viết data card và protocol chia dữ liệu trước khi train.

## 15. Tài liệu nền tảng nên đọc

1. Goodfellow et al. (2014), [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661).
2. Mirza & Osindero (2014), [Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784).
3. Radford et al. (2015), [Unsupervised Representation Learning with DCGANs](https://arxiv.org/abs/1511.06434).
4. Gulrajani et al. (2017), [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028).
5. Heusel et al. (2017), [TTUR and Fréchet Inception Distance](https://arxiv.org/abs/1706.08500).
6. Sajjadi et al. (2018), [Assessing Generative Models via Precision and Recall](https://arxiv.org/abs/1806.00035).
7. Chen et al. (2022), [GANs in Medical Image Augmentation: A Review](https://pubmed.ncbi.nlm.nih.gov/35276550/).
8. Garcea et al. (2023), [Data Augmentation for Medical Imaging: A Systematic Review](https://pubmed.ncbi.nlm.nih.gov/36549032/).

