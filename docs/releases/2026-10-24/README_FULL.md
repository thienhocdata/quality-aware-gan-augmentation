# KLTN — bản tổng hợp đầy đủ ngày 24/10/2026

Gồm kiến trúc/source thí nghiệm, notebook có output, tài liệu, dữ liệu Protocol B,
12 model M4, checkpoint Learned-local 7.500/8.000 của ba seed và gói 12 classifier.
Kèm output selector/endpoint/V3-A đã tổng hợp và đề xuất kiến trúc ngày 24/10.

## Tải và mở

1. Tải **cả hai** file `KLTN_COMPLETE_2026-10-24.zip.part01` và `.part02`.
2. Tải `FULL_ARCHIVE_MANIFEST.json` và `GHEP_FILE.ps1` vào cùng thư mục.
3. Trong PowerShell tại thư mục đó, chạy `powershell -ExecutionPolicy Bypass -File .\GHEP_FILE.ps1`.
   Script kiểm SHA-256 từng phần, ghép ZIP rồi kiểm SHA-256 toàn bộ.
4. Giải nén ZIP bằng 7-Zip và mở `KLTN/notebooks/00_MASTER_CURRENT.ipynb`.
   Notebook hiện hành đọc kết quả có sẵn; không cần train lại.

Nếu có GitHub CLI, có thể tải cả gói bằng:

```powershell
gh release download kltn-complete-2026-10-24 --repo thienhocdata/quality-aware-gan-augmentation --dir KLTN_download
```

`PROJECT_FILE_MANIFEST.json` liệt kê kích thước và SHA-256 từng file trong bộ
tổng hợp. Bản source trên nhánh Git không chứa ba ZIP lớn; bản Release này có đủ.

## Trạng thái kết quả

Đây là toàn bộ **bộ đã tổng hợp**, không phải mọi backup thô của tất cả phiên cũ.
Output endpoint/V3-A là các gói phân tích gọn đã kiểm; không kèm tất cả checkpoint
thô V3-A. Checkpoint M4 và Endgame được giữ nguyên. Endgame chưa có báo cáo
forward 7.500→8.000 hoàn tất. Báo cáo M3 riêng chưa xác định. Đề xuất kiến trúc
mới nằm trong `docs/proposals/`, còn ở trạng thái đề xuất, chưa train.

Lần xuất bản chỉ kiểm byte/hash/CRC và đóng gói; không huấn luyện hay chạy test.
Release cũ ngày 17/10 được giữ riêng, không ghi đè.
