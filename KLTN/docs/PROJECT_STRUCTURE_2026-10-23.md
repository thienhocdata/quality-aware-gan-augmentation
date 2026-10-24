# Cấu trúc quản lý project

`KLTN_clean` là project canonical. Các thư mục còn lại ở cấp cha là kho artifact, audit, archive hoặc hỗ trợ workspace; không phải project runtime chính.

Sau lần sắp xếp 23/10/2026, các nhánh cũ và file rời nằm trong `../KLTN_archive_legacy`; classifier bundle dùng lại nằm trong `../outputs/classifier_bundles`. Manifest di chuyển được lưu trong kho archive.

```text
KLTN_clean/
├── notebooks/                 # notebook đọc/tổng hợp; mặc định không train
├── src/
│   ├── frozen/               # source đã tạo output lịch sử; không sửa
│   └── canonical/            # nơi viết source đồng bộ cho lượt xác nhận sau
├── archive/
│   ├── executed/             # snapshot notebook thực tế đã chạy
│   └── legacy/               # notebook lịch sử/tham chiếu
├── data/                      # Protocol B cache/source archive
├── checkpoints/               # checkpoint/model archive lớn
├── outputs/
│   ├── bundles/              # output nhẹ theo thứ tự nghiên cứu
│   ├── technical/            # smoke/technical reports
│   └── master_reviews/       # báo cáo CPU tính lại từ output đã lưu
├── docs/
│   └── provenance/current/  # registry JSON/CSV và kết quả kiểm hash
└── tools/                     # công cụ quản lý CPU-only
```

## Quy tắc

1. Không sửa file trong `src/frozen`; mọi thay đổi thuật toán tạo file/version mới trong `src/canonical`.
2. Nhận dạng run bằng protocol ID, source SHA, config/contract và checkpoint hash; không chỉ dựa vào tên ZIP.
3. Notebook chỉ điều phối/hiển thị. Logic dùng chung của thí nghiệm mới phải nằm trong `src/canonical`.
4. Output cũ chỉ đọc. Không ghi đè báo cáo hoặc checkpoint.
5. Không nhân đôi archive hàng GB chỉ để đổi thư mục; dùng registry để quản lý vị trí và hash.
6. Chỉ xóa bản cũ sau khi có danh sách duplicate theo hash, bản sao lưu và sự đồng ý riêng.

## Thứ tự bundle hiện tại

1. Protocol B results
2. M4 training results
3. M4 saved test results
4. V2 classification diagnostics
5. V2 learned-weighting at 8,000
6. Endgame 7,500→8,000 status (chưa COMPLETE)
7. Selector endpoint ablation
8. V3-A negative ablation

`manifest.json` là inventory của gói ngày 17/10/2026. `CURRENT_PROJECT_MANIFEST.json` và `docs/provenance/current/` phản ánh trạng thái mới nhất.
