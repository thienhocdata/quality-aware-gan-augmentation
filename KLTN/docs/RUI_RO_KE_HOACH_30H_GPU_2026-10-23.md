# Phản biện kế hoạch 30 giờ GPU — quyết định hiện tại: chưa chạy full

Ngày: 23/10/2026. Chỉ đọc source và output; không train hoặc test inference mới.

## Kết luận trực tiếp

Đề xuất tối ưu D là một câu hỏi thực nghiệm có cơ sở, nhưng chưa đủ căn cứ để chọn nó làm trục mới hoặc đầu tư ngay 15 job. Bản trả lời trước đã chốt quá sớm và dùng một số diễn giải mạnh hơn dữ liệu cho phép.

## Các lỗi/rủi ro đã xác minh

| Điểm | Bằng chứng thực tế | Hệ quả |
|---|---|---|
| GAN gần ROS chưa cô lập D | GAN−ROS Macro-F1 theo seed: +3,480; +0,351; −3,297 điểm; mean +0,178; paired SD 3,392 | Trung bình nhỏ do triệt tiêu giữa seed. Chưa chứng minh tương đương, chưa chỉ ra thành phần nào dư thừa. |
| Thay GD bằng GC không chỉ bỏ D | `m4.py` frame đặt `classifier_targets=1-condition`; GC ép G tạo mẫu khó cho C | No-D matched tăng gấp đôi số route gây khó cho C. A−C là hiệu ứng thay recipe, không phải tác dụng thuần của D. |
| D có vai trò bảo vệ phân phối | GAMO gốc đưa D vào để chống mẫu nằm ngoài phân phối lớp dù vẫn ở convex hull | Bỏ D có thể giảm chất lượng/ngữ nghĩa lớp; convex combination không bảo đảm nhãn đúng. |
| Số optimizer slots không đo runtime | 17/34 slot liên quan D/GD; chưa profile thời gian từng route | Không suy tiết kiệm 50%. Nhánh matched vẫn chạy gần đủ G backward và thêm GC. |
| Clock giống không có nghĩa động lực học giống | Shared legacy Adam; `skip_iterations` chỉ tăng iteration; G có BatchNorm; C/D có recorded Dropout | Phải khóa vị trí slot, số G forward ở training mode, BN policy và RNG route. Không dùng cùng seed để khẳng định trajectories đã match. |
| D25 chưa có lý do dữ liệu | 25% được đề xuất như một tỷ lệ coarse | Đây là hyperparameter mới. Đăng ký trước làm giảm lựa chọn hậu nghiệm nhưng không chứng minh tỷ lệ hợp lý hoặc tối ưu. |
| Smoke/gradient không có cổng hiệu quả | 200 updates và norm/cosine chỉ đo trạng thái cục bộ | Gradient nhỏ ở endpoint không cho phép bỏ D trong toàn training; smoke PASS không dự báo F1. |
| Ngưỡng 1 điểm/25% chưa được biện minh | Các ngưỡng do bản đề xuất đặt, không xuất phát từ yêu cầu chất lượng hoặc power analysis | Không gọi Pareto dominance khi cho phép giảm F1. Không xác nhận non-inferiority từ point estimate ba seed. |
| Protocol đánh giá vẫn yếu ở đúng chỗ người dùng cảnh báo | Validation development bão hòa; test B đã xem; overlap còn; một cảnh | Không có validation đủ sức chọn candidate hiện tại. Giữ Protocol B chỉ trả lời so sánh có điều kiện trong cùng cảnh; không sửa được bằng nhiều GPU hơn. |
| Tính mới và giữ trục | GAMO gốc đã có GAMO\\D; chuyển Original thay V2 chưa được biện minh | Có thể là nghiên cứu tối ưu hệ thống cho KLTN, chưa phải đóng góp thuật toán mới; phải thống nhất mục tiêu trước. |
| Dự toán 30 giờ chưa có dữ liệu cùng môi trường | Timing cũ không cùng toàn bộ recipe/runtime; benchmarks 500–1000 loop cũng tốn compute | Phải tính cả setup, compile, audit, benchmark, checkpoint I/O, retry, test và số GPU đồng thời. 2 GPU × 1 giờ = 2 device-hours; quota dịch vụ có thể tính theo quy tắc riêng. |

## Rủi ro người dùng đã nêu trong tài liệu bàn giao

Tài liệu pasted `5c7c8855-2dc0-4d26-b5ab-116d00b4d40f` đã nêu:

1. Trust-region chỉ bảo đảm hình học theo biểu diễn q, chưa bảo đảm ngữ nghĩa lớp.
2. Clipping có thể kéo mẫu về anchor và triệt gradient của rho.
3. Classifier-derived hardness có thể không đáng tin khi validation bão hòa.
4. Test B đã được xem, hiệu ứng nhỏ/dao động và spatial dependence làm suy luận khó.
5. Các kiến trúc guidance/consistency đã có prior art.

Hướng tối ưu D tránh rủi ro 1–2 của trust-region do không dùng công thức đó. Nó **chưa giải quyết 3–4**. Prior art về GAMO\\D khiến rủi ro tính mới vẫn còn. Vì vậy không thể lấy lý do “đổi hướng” để coi các cảnh báo đã được xử lý.

## Cái gì đủ chắc và cái gì chưa

- Đủ chắc: source có 34 optimizer slots/loop; GC target là `1-condition`; data/output cũ tồn tại; GAN−ROS có biến thiên lớn; Protocol B còn các giới hạn đã audit.
- Chưa đủ: D là nút thắt chính; bỏ/giảm D giữ F1; giảm ≥25% runtime; 30 giờ đủ cho suite mới; 3 seed phân biệt được biên 1 điểm; cấu hình mới đáng thay V2 làm trục.
- Không có thí nghiệm nào bảo đảm kết quả dương hoặc không có rủi ro khoa học. Có thể kiểm tra implementation và hạn chế phạm vi đầu tư trước khi chạy.

## Quyết định hiện tại và điều kiện đi tiếp

**Không chạy full suite theo bản năm nhánh hiện tại. Không tự động nối audit PASS sang full train.**

Chỉ có thể duyệt một lượt train khi có ba thứ:

1. **Một claim hữu ích đã thống nhất:** ví dụ giữ chất lượng trong biên chấp nhận và giảm thời gian, hoặc cải thiện lớp thiểu số. Không dùng lẫn các mục tiêu khi kết quả bất lợi.
2. **Một thiết kế đo đúng claim:** định nghĩa thay đổi GD/GC/clock/BN, protocol đánh giá, cách thể hiện bất định. Nếu mục tiêu chỉ là so recipe trong Protocol B thì ghi rõ phạm vi đó; nếu mục tiêu là cải thiện generalization phải có đánh giá phù hợp hơn, trước khi train.
3. **Một dự toán khả thi:** profile thực tế trong cùng runtime, tính toàn bộ công việc; giới hạn chi phí cho bước chuẩn bị; full train là quyết định tách riêng sau khi đọc report.

Một audit train-only ngắn có thể đo cost và kiểm implementation, nhưng không làm thay nhiệm vụ thiết kế giao thức đánh giá. Nếu không có tiêu chí hiệu quả đủ phân biệt trong ngân sách, giữ 30 giờ cho việc khác thay vì cố chạy để có bảng mới.

## Nguồn đối chiếu

- `KLTN/src/frozen/m4.py`: C/D/G architectures, `frame`, `build_models`, `controlled_update`, `skip_iterations`, `train_once`.
- `protocol_b_review/m4_test_received_2026-10-16/received/test_by_seed.csv`: recompute GAN−ROS theo seed.
- `KLTN/docs/PHAN_6_KET_QUA_SELECTOR.md` và hồ sơ hạn chế 22/10/2026.
- GAMO, ICCV 2019: https://openaccess.thecvf.com/content_ICCV_2019/html/Mullick_Generative_Adversarial_Minority_Oversampling_ICCV_2019_paper.html
- GAMO supplemental, có ablation GAMO\\D: https://openaccess.thecvf.com/content_ICCV_2019/supplemental/Mullick_Generative_Adversarial_Minority_ICCV_2019_supplemental.pdf
