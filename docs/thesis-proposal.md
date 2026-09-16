# ĐỀ CƯƠNG KHÓA LUẬN TỐT NGHIỆP

## 1. Thông tin chung

**Tên đề tài dự kiến:**

> Nghiên cứu mô hình mạng sinh đối nghịch và ứng dụng giải quyết mất cân bằng
> dữ liệu ảnh lỗi công nghiệp

**Tên tiếng Anh dự kiến:**

> Research on Generative Adversarial Networks for Imbalanced Industrial Defect
> Image Data

**Tên nhấn mạnh phương pháp đề xuất (nếu được phép điều chỉnh):**

> Nghiên cứu phương pháp sinh ảnh lỗi công nghiệp có kiểm soát bằng mạng sinh
> đối nghịch nhằm giải quyết mất cân bằng dữ liệu

**Lĩnh vực:** Khoa học dữ liệu, học sâu, thị giác máy tính, mô hình sinh.

**Ghi chú về trạng thái:** Đây là đề cương làm việc. Tên bộ dữ liệu, kiến trúc
cuối cùng và mức điều khiển độ nghiêm trọng của lỗi chỉ được chốt sau bước khảo
sát tài liệu và dữ liệu.

## 2. Đặt vấn đề và lý do chọn đề tài

Trong kiểm tra chất lượng sản phẩm bằng thị giác máy tính, dữ liệu thu được
thường mất cân bằng: ảnh sản phẩm bình thường xuất hiện với số lượng lớn, trong
khi mỗi loại lỗi như nứt, xước, lõm, rỗ hoặc thiếu chi tiết chỉ xuất hiện với số
lượng nhỏ. Đây là hệ quả tự nhiên của dây chuyền sản xuất, vì lỗi là sự kiện ít
xảy ra nhưng lại có ý nghĩa quan trọng đối với kiểm soát chất lượng.

Khi được huấn luyện trên dữ liệu như vậy, mô hình phân loại hoặc phát hiện lỗi có
thể thiên lệch về lớp đa số. Accuracy tổng thể vẫn có thể cao, nhưng Recall của
các lớp lỗi hiếm thấp, dẫn đến bỏ sót sản phẩm lỗi. Các phương pháp truyền thống
như random oversampling, biến đổi hình học và gán trọng số cho hàm mất mát có
thể giảm ảnh hưởng của mất cân bằng, nhưng chúng không nhất thiết tạo ra biến
thiên lỗi mới có ý nghĩa.

GAN tạo ảnh thông qua quá trình học đối nghịch giữa generator và discriminator
([Goodfellow et al., 2014](https://arxiv.org/abs/1406.2661)). Do đó, GAN có tiềm
năng học phân phối của lớp lỗi và sinh thêm dữ liệu. Tuy nhiên, sử dụng GAN như
một bộ sinh ảnh ngẫu nhiên còn ba vấn đề:

1. ảnh sinh có thể kém chất lượng hoặc không đúng loại lỗi;
2. ảnh có thể thiếu đa dạng hoặc gần sao chép dữ liệu huấn luyện;
3. ảnh nhìn chân thực chưa chắc giúp mô hình nhận diện lỗi tốt hơn.

Đề tài vì vậy không chỉ nghiên cứu việc dùng GAN để tăng số lượng ảnh. Hướng đề
xuất là xây dựng quy trình sinh lỗi **có điều kiện và có kiểm soát**, trong đó
generator nhận ảnh sản phẩm bình thường, thông tin loại lỗi và vùng cần tạo lỗi.
Ảnh sinh sau đó được đánh giá và chọn lọc trước khi bổ sung vào tập huấn luyện.
Hiệu quả phải được kiểm chứng bằng khả năng nhận diện lớp lỗi trên tập ảnh thật
độc lập.

## 3. Phát biểu bài toán

Cho tập dữ liệu ảnh có nhãn:

```text
D = {(x_i, y_i)} với i = 1,...,N
```

trong đó `x_i` là ảnh sản phẩm và `y_i` là nhãn bình thường hoặc loại lỗi. Phân
phối số lượng mẫu theo lớp không đồng đều:

```text
n_majority >> n_minority
```

Mục tiêu là học mô hình sinh có điều kiện:

```text
x_fake = G(x_normal, z, c, m[, s])
```

với:

- `x_normal`: ảnh sản phẩm bình thường;
- `z`: vector nhiễu tạo biến thiên;
- `c`: nhãn loại lỗi cần sinh;
- `m`: mask hoặc thông tin vị trí lỗi;
- `s`: mức độ lỗi, là thành phần mở rộng nếu dữ liệu và thời gian cho phép;
- `x_fake`: ảnh sản phẩm có lỗi nhân tạo.

Ảnh `x_fake` cần thỏa mãn bốn yêu cầu:

1. có đặc trưng gần với ảnh lỗi thật;
2. đúng loại và vùng lỗi được yêu cầu;
3. bảo toàn vùng không lỗi;
4. cung cấp thêm thông tin hữu ích cho mô hình nhận diện lỗi.

Bài toán cuối cùng là xác định liệu tập huấn luyện được bổ sung bằng ảnh GAN có
kiểm soát có cải thiện khả năng nhận diện lớp thiểu số so với dữ liệu gốc, GAN
không kiểm soát và các phương pháp cân bằng truyền thống hay không.

## 4. Mục tiêu nghiên cứu

### 4.1. Mục tiêu tổng quát

Nghiên cứu và xây dựng phương pháp tăng cường dữ liệu ảnh lỗi công nghiệp lớp
thiểu số bằng GAN có điều kiện, kết hợp đánh giá và chọn lọc ảnh sinh, nhằm cải
thiện hiệu quả nhận diện lỗi trên dữ liệu mất cân bằng.

### 4.2. Mục tiêu cụ thể

1. Hệ thống hóa cơ sở lý thuyết về GAN, các biến thể phù hợp với ảnh và những
   vấn đề thường gặp khi huấn luyện GAN.
2. Khảo sát các nhóm phương pháp xử lý mất cân bằng dữ liệu ở mức dữ liệu, hàm
   mất mát và mô hình.
3. Khảo sát các phương pháp sinh khuyết tật/lỗi công nghiệp và xác định khoảng
   trống nghiên cứu phù hợp với phạm vi khóa luận.
4. Lựa chọn bộ dữ liệu ảnh lỗi công nghiệp công khai theo tiêu chí xác định
   trước, phân tích phân bố lớp và xây dựng protocol chia dữ liệu chống rò rỉ.
5. Xây dựng mô hình phân loại baseline và các phương pháp cân bằng truyền thống.
6. Xây dựng GAN baseline để có cơ sở đánh giá độ ổn định và chất lượng ảnh sinh.
7. Đề xuất mô hình GAN có điều kiện để sinh loại lỗi tại vùng được chỉ định và
   bảo toàn vùng không lỗi.
8. Xây dựng cơ chế đánh giá, phát hiện gần trùng và chọn lọc ảnh sinh.
9. Thực nghiệm so sánh công bằng và ablation study để xác định đóng góp của từng
   thành phần.
10. Xây dựng ứng dụng minh họa quy trình sinh lỗi và tác động của cân bằng dữ
    liệu lên mô hình nhận diện.

## 5. Câu hỏi nghiên cứu

**RQ1.** Mô hình GAN nào phù hợp với dữ liệu ảnh lỗi công nghiệp mất cân bằng và
giới hạn về số lượng mẫu?

**RQ2.** Việc đưa điều kiện loại lỗi và vị trí lỗi vào generator có giúp ảnh sinh
đúng yêu cầu hơn GAN sinh không kiểm soát hay không?

**RQ3.** Ràng buộc bảo toàn vùng không lỗi có giảm các thay đổi không mong muốn
trên sản phẩm hay không?

**RQ4.** Chọn lọc ảnh dựa trên tính đúng lớp, độ chân thực và độ đa dạng có giúp
mô hình nhận diện lớp lỗi tốt hơn việc sử dụng toàn bộ ảnh GAN sinh ra không?

**RQ5.** Tăng cường dữ liệu bằng GAN có kiểm soát có cải thiện Macro F1,
Balanced Accuracy và Recall lớp thiểu số so với oversampling, augmentation
truyền thống và re-weighting hay không?

**RQ6.** Số lượng ảnh GAN bổ sung ảnh hưởng như thế nào đến hiệu quả nhận diện;
có tồn tại điểm bão hòa hoặc suy giảm khi ảnh sinh quá nhiều hay không?

## 6. Giả thuyết nghiên cứu

- **H1:** GAN có điều kiện đạt tỷ lệ ảnh đúng loại và đúng vùng lỗi cao hơn GAN
  sinh không kiểm soát.
- **H2:** Ràng buộc bảo toàn làm giảm sai khác ngoài mask mà không làm suy giảm
  đáng kể chất lượng vùng lỗi.
- **H3:** Với cùng ngân sách ảnh bổ sung, ảnh GAN đã chọn lọc mang lại Macro F1
  và Recall lớp thiểu số cao hơn ảnh GAN không lọc.
- **H4:** GAN có kiểm soát không nhất thiết luôn vượt mọi baseline, nhưng có lợi
  thế rõ hơn khi lớp lỗi có ít mẫu và biến thiên hình thái không thể tạo chỉ bằng
  xoay, lật hoặc thay đổi màu.
- **H5:** Utility không tăng tuyến tính theo số lượng ảnh sinh; sử dụng quá nhiều
  ảnh nhân tạo có thể làm lệch phân phối train và giảm hiệu quả trên ảnh thật.

Giả thuyết sẽ được chấp nhận hoặc bác bỏ dựa trên kết quả thực nghiệm; không xem
đây là kết luận có sẵn.

## 7. Đối tượng và phạm vi nghiên cứu

### 7.1. Đối tượng nghiên cứu

- Mô hình GAN cho dữ liệu ảnh.
- Các phương pháp xử lý mất cân bằng dữ liệu ảnh.
- Ảnh bề mặt hoặc sản phẩm công nghiệp có lớp lỗi hiếm.
- Mô hình phân loại ảnh dùng để đo utility của dữ liệu sinh.

### 7.2. Phạm vi bắt buộc

- Một bộ dữ liệu công khai chính.
- Bài toán phân loại lớp bình thường và các loại lỗi, hoặc phân loại đa lớp lỗi.
- Điều khiển tối thiểu theo loại lỗi và vùng lỗi.
- Bảo toàn vùng ngoài mask.
- So sánh với GAN baseline và ít nhất ba phương pháp truyền thống.
- Đánh giá trên tập test chỉ gồm ảnh thật.
- Độ phân giải khởi đầu 64x64 hoặc 128x128 để phù hợp tài nguyên.

### 7.3. Phạm vi nâng cao

- Điều khiển mức độ nghiêm trọng của lỗi.
- Sinh ảnh ở độ phân giải cao hơn.
- Phát hiện hoặc phân đoạn lỗi thay vì chỉ phân loại.
- Đánh giá bởi chuyên gia sản xuất.

Các nội dung nâng cao chỉ triển khai sau khi phần bắt buộc đã hoạt động và có
kết quả tái lập.

### 7.4. Ngoài phạm vi

- Triển khai trực tiếp lên dây chuyền sản xuất thật.
- Chứng nhận hệ thống kiểm tra chất lượng ở cấp công nghiệp.
- Tuyên bố GAN có thể thay thế hoàn toàn việc thu thập và gán nhãn ảnh lỗi thật.

## 8. Tiêu chí chọn bộ dữ liệu

Không chốt dữ liệu chỉ vì phổ biến. Bộ dữ liệu cần đáp ứng:

1. có giấy phép sử dụng nghiên cứu rõ ràng;
2. có ảnh bình thường và ít nhất hai loại lỗi;
3. tồn tại mất cân bằng tự nhiên hoặc kịch bản long-tail có cơ sở;
4. có nhãn lớp đáng tin cậy;
5. ưu tiên có mask hoặc bounding box vùng lỗi;
6. đủ ảnh để chia train/validation/test;
7. không có nhiều ảnh gần trùng giữa các tập;
8. kích thước phù hợp GPU và thời gian khóa luận.

Các ứng viên sẽ được lập bảng so sánh trước khi quyết định. MVTec AD có thể được
khảo sát nhưng không mặc nhiên chọn, vì thiết kế anomaly detection của nó có thể
khác bài toán phân loại mất cân bằng cần nghiên cứu.

## 9. Cơ sở lý thuyết

### 9.1. GAN cơ bản

Generator `G` sinh ảnh từ nhiễu, discriminator `D` phân biệt thật/giả. Hàm mục
tiêu minimax:

```text
min_G max_D E_x[log D(x)] + E_z[log(1 - D(G(z)))]
```

Quá trình đối nghịch giúp generator học phân phối dữ liệu, nhưng có thể gặp mất
ổn định gradient và mode collapse.

### 9.2. DCGAN

DCGAN sử dụng mạng tích chập cho generator và discriminator, là baseline phù
hợp để kiểm chứng pipeline ảnh
([Radford et al., 2015](https://arxiv.org/abs/1511.06434)).

### 9.3. Conditional GAN

Conditional GAN đưa nhãn điều kiện vào cả generator và discriminator
([Mirza & Osindero, 2014](https://arxiv.org/abs/1411.1784)). Đề tài mở rộng điều
kiện từ nhãn lớp sang ảnh bình thường và mask vị trí lỗi.

### 9.4. WGAN-GP

WGAN-GP dùng critic và gradient penalty nhằm làm huấn luyện ổn định hơn so với
weight clipping
([Gulrajani et al., 2017](https://arxiv.org/abs/1704.00028)). Biến thể này là
ứng viên khi GAN baseline dao động hoặc mode collapse rõ rệt.

### 9.5. Mất cân bằng dữ liệu

Ba nhóm xử lý chính:

- **Data-level:** undersampling, oversampling, augmentation, sinh dữ liệu.
- **Algorithm-level:** class weighting, focal loss, class-balanced loss.
- **Hybrid:** kết hợp điều chỉnh dữ liệu và hàm mất mát.

Đề tài phải so sánh với baseline đủ mạnh để tránh kết luận GAN tốt chỉ vì đối
chứng quá yếu.

## 10. Phương pháp nghiên cứu đề xuất

### 10.1. Kiến trúc tổng quát

```text
Ảnh bình thường x -----------------------------+
                                                   |
Nhãn loại lỗi c --> Encoder/Embedding -------------+--> Generator --> ảnh lỗi x_fake
                                                   |
Mask vị trí m ------------------------------------+
                                                   |
Nhiễu z ------------------------------------------+

Ảnh thật/ảnh sinh + điều kiện -----------------------> Discriminator/Critic
```

Generator chỉ nên thay đổi vùng liên quan đến mask; vùng còn lại cần gần với
ảnh đầu vào.

### 10.2. Hàm mất mát dự kiến

```text
L_G = L_adv
    + lambda_cls * L_cls
    + lambda_mask * L_mask
    + lambda_pres * L_preserve
    + lambda_div * L_diversity
```

- `L_adv`: làm ảnh sinh gần phân phối ảnh thật.
- `L_cls`: bảo đảm đúng loại lỗi.
- `L_mask`: khuyến khích lỗi xuất hiện trong vùng yêu cầu.
- `L_preserve`: hạn chế thay đổi ngoài vùng lỗi.
- `L_diversity`: giảm nguy cơ nhiều đầu vào nhiễu sinh cùng một kiểu lỗi.

Không nhất thiết đưa toàn bộ loss vào phiên bản đầu. Mỗi thành phần chỉ được giữ
nếu có định nghĩa đo lường, cách cài đặt và ablation chứng minh giá trị.

Một dạng loss bảo toàn đơn giản:

```text
L_preserve = ||(1-m) * (x_fake - x_normal)||_1
```

### 10.3. Đánh giá và chọn lọc ảnh sinh

Mỗi ảnh ứng viên được đánh giá theo:

- **Class consistency:** có đúng loại lỗi yêu cầu không?
- **Realism:** có nằm gần phân phối ảnh lỗi thật không?
- **Localization:** lỗi có nằm trong vùng mask không?
- **Preservation:** vùng ngoài mask có bị thay đổi không?
- **Diversity:** ảnh có bổ sung biến thiên thay vì gần trùng không?
- **Memorization check:** ảnh có gần sao chép ảnh train không?

So sánh ba chiến lược:

1. dùng toàn bộ ảnh GAN;
2. lọc theo chất lượng và đúng lớp;
3. lọc theo chất lượng, đúng lớp và đa dạng.

## 11. Thiết kế thực nghiệm

### 11.1. Chia dữ liệu

- Chia train/validation/test trước khi cân bằng.
- Nếu nhiều ảnh thuộc cùng một sản phẩm hoặc chuỗi chụp, chia theo sản phẩm hoặc
  nhóm để tránh rò rỉ.
- GAN chỉ học từ train.
- Threshold lọc và siêu tham số chọn trên validation.
- Test giữ phân phối thật và không chứa ảnh nhân tạo.

### 11.2. Các nhóm thí nghiệm

| Mã | Phương pháp | Vai trò |
|---|---|---|
| E0 | Dữ liệu gốc mất cân bằng | Baseline chính |
| E1 | Random oversampling | Baseline data-level |
| E2 | Augmentation truyền thống | Baseline biến đổi ảnh |
| E3 | Class weight/Focal Loss | Baseline algorithm-level |
| E4 | GAN sinh không kiểm soát | Baseline mô hình sinh |
| E5 | GAN có điều kiện loại và vị trí lỗi | Kiểm tra điều khiển |
| E6 | E5 + ràng buộc bảo toàn | Kiểm tra bảo toàn |
| E7 | E6 + chọn lọc ảnh | Phương pháp đầy đủ |

### 11.3. Thí nghiệm theo tỷ lệ ảnh sinh

Ảnh GAN bổ sung theo các mức:

- 25% khoảng thiếu hụt;
- 50% khoảng thiếu hụt;
- 100% khoảng thiếu hụt;
- cân bằng hoàn toàn với lớp đa số, nếu khác mức 100% theo định nghĩa sử dụng.

### 11.4. Ablation study

- Bỏ điều kiện loại lỗi.
- Bỏ mask vị trí.
- Bỏ loss bảo toàn.
- Bỏ loss đa dạng.
- Bỏ bước lọc.
- Thay chọn lọc bằng chọn ngẫu nhiên cùng số lượng.

### 11.5. Tính tái lập

- Cố định và công bố seed.
- Lưu toàn bộ cấu hình YAML.
- Lưu phiên bản dữ liệu và quy tắc chia tập.
- Chạy tối thiểu ba seed cho thí nghiệm chính nếu tài nguyên cho phép.
- Báo trung bình và độ lệch chuẩn.
- Không chỉ lựa chọn và báo lần chạy tốt nhất.

## 12. Chỉ số đánh giá

### 12.1. Chất lượng mô hình sinh

- FID, được giới thiệu trong công trình TTUR
  ([Heusel et al., 2017](https://arxiv.org/abs/1706.08500)).
- Precision và Recall cho phân phối sinh để tách fidelity khỏi coverage
  ([Sajjadi et al., 2018](https://arxiv.org/abs/1806.00035)).
- Tỷ lệ ảnh đúng lớp.
- IoU hoặc độ phủ giữa vùng lỗi sinh và mask yêu cầu, nếu trích được vùng lỗi.
- Sai khác trung bình ngoài mask.
- Khoảng cách nearest-neighbor với ảnh train.
- Độ đa dạng trong không gian đặc trưng.

### 12.2. Hiệu quả nhận diện lỗi

- Recall theo lớp lỗi.
- Precision theo lớp.
- Macro F1.
- Balanced Accuracy.
- PR-AUC.
- Confusion matrix.

Accuracy chỉ là chỉ số phụ vì có thể gây hiểu nhầm trên dữ liệu mất cân bằng.

### 12.3. Hiệu quả tính toán

- Thời gian huấn luyện.
- Bộ nhớ GPU.
- Số tham số.
- Thời gian sinh và lọc một ảnh.

## 13. Tiêu chí thành công

Phương pháp được xem là có bằng chứng hỗ trợ khi:

1. sinh được ảnh đúng điều kiện tốt hơn GAN baseline;
2. giảm thay đổi ngoài vùng lỗi so với phiên bản không có loss bảo toàn;
3. không có dấu hiệu sao chép nghiêm trọng hoặc mode collapse rõ ràng;
4. cải thiện ổn định Macro F1 hoặc Recall lớp thiểu số trên test thật so với E0;
5. phương pháp chọn lọc vượt GAN không lọc trong cùng ngân sách ảnh bổ sung;
6. kết quả được lặp lại trên nhiều seed hoặc được trình bày kèm độ bất định.

Không đặt trước một phần trăm cải thiện tùy ý trước khi có baseline. Nếu GAN
không vượt augmentation truyền thống, khóa luận vẫn có giá trị khi quy trình
đúng và nguyên nhân thất bại được phân tích thuyết phục.

## 14. Đóng góp dự kiến

1. Tổng hợp có hệ thống mối liên hệ giữa GAN, sinh lỗi công nghiệp và xử lý dữ
   liệu ảnh mất cân bằng.
2. Xây dựng pipeline sinh lỗi có điều kiện theo loại và vị trí, kết hợp ràng buộc
   bảo toàn vùng không lỗi.
3. Xây dựng quy trình đánh giá–chọn lọc ảnh sinh theo nhiều chiều thay vì chỉ
   nhìn chất lượng trực quan.
4. Đánh giá có kiểm soát utility của ảnh GAN so với các phương pháp cân bằng
   truyền thống.
5. Cung cấp mã nguồn, cấu hình và protocol thực nghiệm có khả năng tái lập.

Các mục trên là **đóng góp dự kiến**, chưa phải tuyên bố tính mới. Research gap
và tính mới cuối cùng phải được điều chỉnh sau tổng quan tài liệu.

## 15. Ứng dụng minh họa

Ứng dụng dự kiến có các chức năng:

1. tải ảnh sản phẩm bình thường;
2. chọn loại lỗi;
3. vẽ hoặc tải mask vị trí lỗi;
4. sinh một hoặc nhiều ảnh lỗi;
5. hiển thị điểm chất lượng và ảnh bị loại;
6. so sánh phân bố lớp trước/sau cân bằng;
7. hiển thị kết quả classifier trước/sau tăng cường dữ liệu.

Ứng dụng chỉ minh họa kết quả nghiên cứu, không tuyên bố sẵn sàng triển khai
trong dây chuyền thật.

## 16. Nội dung dự kiến của khóa luận

### Chương 1. Tổng quan đề tài

- Bối cảnh kiểm tra lỗi công nghiệp.
- Vấn đề mất cân bằng dữ liệu.
- Lý do sử dụng GAN.
- Mục tiêu, câu hỏi, phạm vi và đóng góp.

### Chương 2. Cơ sở lý thuyết và nghiên cứu liên quan

- Học sâu cho phân loại ảnh.
- Mất cân bằng dữ liệu và các phương pháp xử lý.
- GAN, DCGAN, cGAN và WGAN-GP.
- Đánh giá mô hình sinh.
- Sinh lỗi công nghiệp và research gap.

### Chương 3. Phương pháp đề xuất

- Phát biểu bài toán và ký hiệu.
- Kiến trúc hệ thống.
- Generator và discriminator/critic.
- Biểu diễn điều kiện loại/vị trí lỗi.
- Các hàm loss.
- Thuật toán chọn lọc ảnh.
- Quy trình huấn luyện và suy luận.

### Chương 4. Thực nghiệm và kết quả

- Bộ dữ liệu và protocol chia tập.
- Môi trường, siêu tham số và baseline.
- Kết quả chất lượng ảnh sinh.
- Kết quả downstream classifier.
- Ablation study.
- Phân tích lỗi và giới hạn.

### Chương 5. Ứng dụng minh họa

- Yêu cầu và kiến trúc ứng dụng.
- Chức năng.
- Giao diện và kịch bản sử dụng.

### Chương 6. Kết luận và hướng phát triển

- Trả lời từng câu hỏi nghiên cứu.
- Đóng góp đạt được.
- Hạn chế.
- Hướng mở rộng.

## 17. Kế hoạch thực hiện dự kiến

| Giai đoạn | Nội dung | Sản phẩm kiểm tra |
|---|---|---|
| 1 | Chốt bài toán và đề cương | Đề cương được duyệt |
| 2 | Tổng quan tài liệu | Bảng literature review và research gap |
| 3 | Khảo sát, chọn dữ liệu | Data card và protocol chia tập |
| 4 | Xây dựng classifier baseline | Bảng E0–E3 |
| 5 | Huấn luyện GAN baseline | Checkpoint, ảnh mẫu và metric |
| 6 | Xây dựng GAN có kiểm soát | Kết quả E4–E6 |
| 7 | Lọc ảnh và ablation | Kết quả E7 và ablation |
| 8 | Phân tích, viết khóa luận | Chương 1–4 hoàn chỉnh |
| 9 | Xây dựng ứng dụng | Demo và hướng dẫn chạy |
| 10 | Rà soát và chuẩn bị bảo vệ | Bản nộp, slide, kịch bản |

Mỗi giai đoạn chỉ chuyển tiếp khi sản phẩm kiểm tra đã được đọc và chấp nhận.

## 18. Rủi ro và phương án dự phòng

| Rủi ro | Ảnh hưởng | Phương án |
|---|---|---|
| Dữ liệu lỗi quá ít | GAN ghi nhớ hoặc mode collapse | Augmentation khi train GAN, giảm phạm vi lớp, dùng transfer learning |
| Không có mask lỗi | Không đánh giá được localization | Tạo mask thô/bounding box hoặc chuyển vị trí lỗi thành phần nâng cao |
| GAN huấn luyện không ổn định | Ảnh chất lượng thấp | DCGAN baseline, WGAN-GP, chọn checkpoint và giới hạn độ phân giải |
| GPU hạn chế | Thời gian huấn luyện dài | 64x64/128x128, mixed precision, Colab/Kaggle |
| Ảnh đẹp nhưng classifier không tăng | Không đạt giả thuyết utility | Phân tích failure case, tỷ lệ ảnh sinh và bộ lọc; báo kết quả trung thực |
| Research gap quá gần công trình cũ | Đóng góp yếu | Thu hẹp vào preservation, utility-aware filtering hoặc extreme imbalance |

## 19. Sản phẩm dự kiến

- Báo cáo khóa luận.
- Mã nguồn PyTorch.
- Cấu hình và hướng dẫn tái lập thí nghiệm.
- Checkpoint chọn lọc, nếu giấy phép và dung lượng cho phép.
- Bảng kết quả baseline, GAN và ablation.
- Ảnh minh họa và phân tích lỗi.
- Ứng dụng demo.
- Repository GitHub quản lý phiên bản.

## 20. Tài liệu nền tảng ban đầu

1. Goodfellow, I. et al. (2014). [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661).
2. Mirza, M. & Osindero, S. (2014). [Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784).
3. Radford, A. et al. (2015). [Unsupervised Representation Learning with DCGANs](https://arxiv.org/abs/1511.06434).
4. Gulrajani, I. et al. (2017). [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028).
5. Heusel, M. et al. (2017). [GANs Trained by a Two Time-Scale Update Rule](https://arxiv.org/abs/1706.08500).
6. Sajjadi, M. et al. (2018). [Assessing Generative Models via Precision and Recall](https://arxiv.org/abs/1806.00035).

Danh mục này mới là nền tảng phương pháp. Giai đoạn tổng quan tiếp theo phải bổ
sung công trình về dữ liệu lỗi công nghiệp, controllable defect synthesis,
anomaly generation, class imbalance và đánh giá utility downstream.

## 21. Tóm tắt đề cương trong một đoạn

Khóa luận nghiên cứu GAN trong bài toán mất cân bằng dữ liệu ảnh lỗi công
nghiệp. Thay vì chỉ sinh ảnh lớp thiểu số từ nhiễu, phương pháp dự kiến sử dụng
ảnh sản phẩm bình thường, nhãn loại lỗi và mask vị trí để sinh lỗi có kiểm soát,
đồng thời bảo toàn vùng không lỗi. Ảnh sinh được đánh giá về độ chân thực, đúng
lớp, đúng vị trí, đa dạng và nguy cơ sao chép trước khi bổ sung vào tập huấn
luyện. Hiệu quả được kiểm chứng trên classifier cố định và tập test ảnh thật,
so sánh với dữ liệu gốc, oversampling, augmentation, re-weighting và GAN không
kiểm soát. Nghiên cứu tập trung trả lời khi nào và trong điều kiện nào dữ liệu
GAN thực sự cải thiện khả năng nhận diện lỗi hiếm.

