# KLTN — 3D-HyperGAMO trên Indian Pines

Bộ nghiên cứu đã tổng hợp, cập nhật ngày **24/10/2026**:

- [Toàn bộ source, notebook, tài liệu và output nhẹ](KLTN/).
- [Notebook tổng hợp hiện hành, có output đã lưu](KLTN/notebooks/00_MASTER_CURRENT.ipynb).
- [Tải bản đầy đủ gồm dữ liệu và checkpoint](https://github.com/thienhocdata/quality-aware-gan-augmentation/releases/tag/kltn-complete-2026-10-24).
- [Danh mục từng file và SHA-256](KLTN/PUBLICATION_MANIFEST_2026-10-24.json).
- [Đề xuất kiến trúc mới và phụ lục v1.0](KLTN/docs/proposals/) — đang ở trạng thái đề xuất, chưa train.

Source thực thi Protocol B, M4/V2, selector endpoint và V3-A được bảo tồn cùng
kết quả/provenance. Git chứa source và báo cáo nhẹ; Release chứa bản đầy đủ có
cache Protocol B, 12 model M4, 6 checkpoint Learned-local 7.500/8.000 và 12 classifier.
Tải hai phần và ghép theo hướng dẫn trong Release để có ZIP đầy đủ.

[Hướng dẫn phạm vi và trạng thái bằng chứng](KLTN/README_RELEASE_2026-10-24.md):
Endgame còn thiếu báo cáo forward hoàn tất, và báo cáo M3 riêng chưa xác định.
Các gói endpoint/V3-A là output phân tích đã tổng hợp, không phải mọi backup
thô lịch sử. Lần xuất bản không train hoặc chạy lại test.

Các thư mục ngoài `KLTN/` bên dưới được giữ lại từ giai đoạn đề cương trước;
đọc `KLTN/` để theo dõi trục nghiên cứu hiện hành.

---

## Đề cương giai đoạn trước — Quality-Aware GAN Augmentation

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
- [Đề cương nghiên cứu](docs/thesis-proposal.md): đề cương khóa luận theo hướng
  sinh ảnh lỗi công nghiệp có kiểm soát để xử lý mất cân bằng dữ liệu.
