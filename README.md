# Quality-Aware GAN Augmentation

Nghiên cứu tăng cường dữ liệu lớp thiểu số bằng GAN có kiểm soát chất lượng cho
bài toán phân loại ảnh mất cân bằng.

## Câu hỏi nghiên cứu

Ảnh lớp thiểu số do GAN sinh ra và được chọn lọc theo chất lượng, tính đúng lớp
và độ đa dạng có cải thiện Macro F1 và Recall của lớp thiểu số tốt hơn các
phương pháp cân bằng dữ liệu truyền thống hay không?

## Phạm vi dự kiến

- Huấn luyện và tối ưu GAN trên lớp thiểu số.
- Đánh giá chất lượng và độ đa dạng của ảnh sinh.
- Lọc ảnh kém chất lượng, sai lớp hoặc gần trùng lặp.
- So sánh với baseline, oversampling, augmentation và class weighting.
- Đánh giá tác động trên cùng một mô hình phân loại.

## Quy trình

```text
Dữ liệu mất cân bằng -> GAN -> Ảnh sinh -> Chọn lọc -> Tập huấn luyện cân bằng
                                                         |
                                                         v
                                          Mô hình phân loại và đánh giá
```

## Cấu trúc dự án

```text
configs/        Cấu hình thí nghiệm
data/           Dữ liệu cục bộ (không commit ảnh)
notebooks/      Khám phá dữ liệu
src/            Mã nguồn chính
  data/         Tiền xử lý và dataloader
  gan/          Generator, discriminator và huấn luyện
  filtering/    Chấm điểm và chọn lọc ảnh sinh
  classifier/   Mô hình phân loại
  evaluation/   Chỉ số và báo cáo kết quả
scripts/        Điểm chạy các quy trình
tests/          Kiểm thử
outputs/        Kết quả, ảnh mẫu và checkpoint cục bộ
reports/        Hình, bảng và nội dung phục vụ khóa luận
app/            Ứng dụng minh họa
```

## Bắt đầu

Yêu cầu Python 3.11 trở lên.

```bash
python -m venv .venv
pip install -r requirements.txt
python scripts/check_environment.py
```

## Nguyên tắc thực nghiệm

- Chia train/validation/test trước mọi thao tác sinh hoặc cân bằng dữ liệu.
- Chỉ dùng tập train để huấn luyện GAN và tạo ảnh bổ sung.
- Giữ nguyên tập test theo phân phối thực tế.
- Lưu seed, cấu hình, checkpoint và chỉ số cho từng lần chạy.
- Dùng Macro F1, Balanced Accuracy và Recall lớp thiểu số làm chỉ số chính.

## Trạng thái

Giai đoạn 1: xây dựng đề cương, chọn bộ dữ liệu và triển khai baseline.

## Tài liệu nghiên cứu

- [Nền tảng và thiết kế nghiên cứu](docs/research-foundation.md): giải thích chi
  tiết bài toán, GAN, ảnh sinh kém, phương pháp đề xuất và kế hoạch thực nghiệm.
