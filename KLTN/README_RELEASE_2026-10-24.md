# Bản tổng hợp đầy đủ — 24/10/2026

Mở `notebooks/00_MASTER_CURRENT.ipynb` để đọc kết quả đã lưu. Thư mục hiện hành tên `KLTN`; tên `KLTN_clean` trong hồ sơ cũ là tên của snapshot trước.

Bản này gồm toàn bộ nội dung bộ tổng hợp hiện hành: source thực thi đã khóa, notebook và output đã lưu, dữ liệu Protocol B, 12 model M4, 6 checkpoint 7.500/8.000, 12 classifier gọn, báo cáo selector/endpoint/V3-A và đề xuất kiến trúc 24/10. Gói classifier trước đây ở ngoài thư mục đã được sao chép vào `checkpoints/`; registry hiện hành dùng đường dẫn nội bộ này. Source frozen và checkpoint không đổi.

## Git và bản tải đầy đủ

Git chứa source, notebook, tài liệu và output nhẹ. Ba ZIP lớn ở `data/` và `checkpoints/` được phát hành trong bản tải đầy đủ trên Releases. [Tải bản đầy đủ](https://github.com/thienhocdata/quality-aware-gan-augmentation/releases/tag/kltn-complete-2026-10-24), ghép hai phần bằng `GHEP_FILE.ps1`, rồi giải nén ZIP.

`PUBLICATION_MANIFEST_2026-10-24.json` ghi kích thước và SHA-256 của từng file trước khi đóng gói; manifest không tự liệt kê chính nó.

## Phạm vi bằng chứng

Các gói endpoint và V3-A là output phân tích đã tổng hợp, không phải mọi backup thô của tất cả phiên lịch sử. Bộ checkpoint M4 và Endgame là gói đầy đủ được liệt kê. Endgame có checkpoint nhưng chưa có báo cáo forward 7.500→8.000 hoàn tất; report kỹ thuật M3 riêng chưa xác định. Đề xuất kiến trúc 24/10 chưa có kết quả train. Các trạng thái này được giữ đúng như hồ sơ, không tạo số liệu mới.

Lần xuất bản này chỉ kiểm file/hash/ZIP, không train và không inference trên test.
