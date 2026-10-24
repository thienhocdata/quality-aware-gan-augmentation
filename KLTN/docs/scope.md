# Trục chính, kiểm tra đầu vào và quy tắc dọn bộ nghiên cứu

**Yêu cầu bổ sung 24/10/2026:** [Giá trị nghiên cứu và lý do lựa chọn sử dụng](YEU_CAU_GIA_TRI_NGHIEN_CUU_2026-10-24.md) là tiêu chuẩn cho các đề xuất tiếp theo. Các trạng thái trục bên dưới mô tả lịch sử đã làm; không phải phê duyệt một lượt train mới hoặc bằng chứng phương pháp đã có giá trị cạnh tranh.

Ngày chốt: 17/10/2026. Theo yêu cầu người dùng: kiểm lại những gì cần cho trục chính, chọn phần cần giữ, loại phần không cần khỏi bộ chính, sau đó tạo một notebook hoàn chỉnh. Tài liệu này ghi phạm vi và tiêu chí thực hiện; chưa phải notebook hợp nhất hoặc bằng chứng thực thi mô hình mới.

## 1. Trục chính được giữ

**3D-HyperGAMO gốc → tái hiện/Protocol B trên Indian Pines → Learned-local V2 → đối chứng M4 bốn nhánh × ba seed → test gốc 12 classifier → phân tích selector.**

- Dataset chính: **Indian Pines**, giữ 16 lớp, patch 11 × 11, PCA 40 và protocol B `50e988fae260` trong các run M4 đã có.
- V2 đang được khảo sát: anchor-uniform, graph ba neighbor cùng lớp, `rho=0.2`, trọng số neighbor từ Dense heads. Giữ đúng source đã chạy; không thêm temperature, loss, anchor conditioning hoặc đổi dataset trong lúc hợp nhất.
- Bốn nhánh M4: `original_gan`, `matched_ros`, `random_local`, `learned_local`; seed 42, 43, 44; checkpoint cuối tại 8.000 C updates.
- Câu hỏi trung tâm của chẩn đoán hiện tại: **quy tắc chọn neighbor được học có hữu ích hơn quy tắc đơn giản hay không?** Chẩn đoán tập trung vào 7.500 → 8.000 và đường noise → G feature → logits → weights → patch. Đối chứng nearest-one/random-one-fixed chưa được tự thêm thành một thí nghiệm đã hoàn tất.

Các nhánh spatial leakage, source collision và khảo sát Pavia được giữ như lịch sử/giới hạn dẫn tới quyết định. Chúng không được nhập thành các cải tiến thuộc V2 hoặc thành kết quả thắng của V2.

## 2. Những việc cần kiểm trước khi ráp notebook

| Kiểm tra | Tiêu chí và nguồn bằng chứng | Trạng thái hiện tại |
| --- | --- | --- |
| Source thực sự đã chạy | Đối chiếu worker trong raw M4, contract và notebook; lấy source từ output đã chạy làm chuẩn | Đã đối chiếu worker giữa hai raw M4 ZIP và ba launcher. Không lấy tên file/default của launcher để suy ra cấu hình run |
| Dữ liệu và tiền xử lý | Đúng cache B, final train 5.528, class order, PCA/scaler và hashes; phân biệt development/final | Các gói gốc còn đầy đủ; giữ B cache làm phụ thuộc bắt buộc |
| Registry M4 | Đủ 4 nhánh × 3 seed, đúng final/8.000 updates, checkpoint C/G/heads và metadata tương ứng | Đủ 12 job tại các gói canonical dưới đây; từng đối chứng có cấu trúc mạng khác nhau, không bắt ROS phải có G |
| Đối chứng thực tế | Giữ schedule/exposure, initial C, anchor traces, optimizer clock và runtime metadata theo source/contract | Dùng chứng nhận và traces của M4 đã chạy; không diễn giải SD như chỉ chịu ảnh hưởng seed khi môi trường train khác giữa seed 42 và 43/44 |
| Test và cách tổng hợp | Dùng predictions gốc 12/12, cùng 4.721 tâm; đối chiếu y_true/y_pred/CM/metrics; mean ± sample SD qua ba seed | Đã có output test và audit số liệu. Không chọn seed tốt nhất hoặc test lại khi mở notebook |
| Weighting tại 8.000 | Ba seed, 15 learned heads/seed; logits Dense thật, train-only, BN inference, không đổi state | Báo cáo `COMPLETE_3_SEEDS` và các audit còn đủ. Có logits/weights/noise đã lưu; chưa lưu đầy đủ intermediate G features |
| Endgame 7.500 → 8.000 | Đủ sáu checkpoint đúng nguồn; nếu chạy forward thì phải đủ ba cặp và không cập nhật state | Checkpoint đã đủ. Chưa có report Endgame hoàn tất; hai ZIP cũ là failed-input reports. Không ghi PASS thay cho phép forward chưa chạy |
| Notebook hợp nhất | Tách definitions khỏi actions; kiểm cú pháp, identity/schema và đường dẫn; dùng output đã lưu làm mặc định | Chưa tạo notebook hợp nhất. Không ghép nối nguyên các cell tự chạy của notebook cũ |

Không chạy lại smoke, train hoặc test để làm cho thư mục trông đầy đủ hơn. Mốc 3 riêng ghi `chưa tìm thấy report`; không dành thêm lượt chạy chỉ để bổ sung biên bản này.

Worker M4 được lấy làm chuẩn cho schedule/Dropout của đối chứng mới. Protocol B lịch sử cung cấp cache và xuất xứ kiến trúc; nó không phải cùng một trajectory train với M4. Các nguồn `test_worker.py` dạng template có thể khác worker test đã thực thi; phải lấy đúng source từ output test hoặc literal đã khớp hash trong notebook gốc.

Giữ **full B bundle**: worker M4 nguyên trạng gọi phần kiểm development artifacts ngay cả khi chạy `stage=final`. Gói chỉ có final train đủ cho helper chẩn đoán train-only, nhưng chưa đủ để thay cache B của runner huấn luyện gốc.

## 3. Bộ gốc cần giữ làm chuẩn

| Nhóm | Bản chuẩn | Vai trò |
| --- | --- | --- |
| Protocol B | [KLTN_HyperGAMO_B_50e988fae260.zip](<C:/Users/ADMIN/OneDrive/Desktop/KLTN_HyperGAMO_B_50e988fae260.zip>) | Dữ liệu, preprocessing, split/config và nguồn B |
| M4 seed 42: GAN gốc | [M4_IP_final_original_gan_seed42_REPORT.zip](<C:/Users/ADMIN/OneDrive/Desktop/M4_IP_final_original_gan_seed42_REPORT.zip>) | Classifier, checkpoint 8.000 và metadata run |
| M4 seed 42: ba nhánh còn lại | [results (1).zip](<C:/Users/ADMIN/OneDrive/Desktop/results (1).zip>) | Raw jobs, checkpoints và worker |
| M4 seed 43/44: tám job | [Downloads/results.zip](<C:/Users/ADMIN/Downloads/results.zip>) | Raw jobs của cả bốn nhánh, checkpoints 7.500/8.000 và worker |
| Test M4 gốc | [M4_IP_TEST_20261016T022123Z_f9775859_REPORT.zip](<C:/Users/ADMIN/OneDrive/Desktop/M4_IP_TEST_20261016T022123Z_f9775859_REPORT.zip>) | Predictions, metrics và CM của 12 classifier |
| Weighting endpoint | [V2_LEARNED_WEIGHTING_3SEEDS_20261016T114801Z_418a270e.zip](<C:/Users/ADMIN/OneDrive/Desktop/V2_LEARNED_WEIGHTING_3SEEDS_20261016T114801Z_418a270e.zip>) | Logits/weights và chẩn đoán train-only ba seed |
| Source notebook và modules | [Source map](<C:/Users/ADMIN/OneDrive/Tài liệu/ChatGPT/KLTN/protocol_b_review/main_axis_consolidation_2026-10-17/source_map.md>) | Chỉ rõ phiên bản thực thi và các entrypoint cần chỉnh khi hợp nhất |

Các gói classifier-only, seed42 export, recovered test, diagnostics và checkpoint rút gọn là các sản phẩm dẫn xuất tiện sử dụng. Không dùng chúng thay bộ gốc nếu chúng không giữ đủ state/source/dữ liệu. Chúng có thể được giữ trong nhánh analysis/export thay vì tính là thí nghiệm mới.

## 4. Quy tắc giữ, gộp và loại

| Nhãn | Xử lý |
| --- | --- |
| `KEEP_MAIN` | Dữ liệu/source/checkpoint/output đang hỗ trợ trực tiếp trục chính, hoặc phụ thuộc để tái hiện kết quả |
| `KEEP_HISTORY` | Bằng chứng khảo sát và nhánh đã dừng; đưa khỏi bộ chính nhưng vẫn giữ một bản truy xuất được |
| `EXACT_DUPLICATE` | Chỉ xem xét bỏ bản thừa khi bytes/hash đã khớp, đã chọn bản chuẩn, kiểm phụ thuộc đường dẫn và có nhật ký thao tác |
| `REBUILDABLE` | Cache sinh lại, fixture/mock, bytecode hoặc file trung gian; chỉ dọn khi xác nhận không còn là phụ thuộc hay bằng chứng duy nhất |
| `REVIEW_NEEDED` | Chưa xác định vai trò, nguồn hoặc mức trùng; giữ nguyên cho đến khi làm rõ |

- Hai file cùng tên không mặc nhiên trùng. `Downloads/results.zip` là M4 seed 43/44; gói 4,79 GiB ở ổ D là thí nghiệm bốn phương pháp cũ.
- Gói 4,79 GiB chứa 23 file bổ sung ngoài 297 file khớp với bản comparison-ver2; chưa được gán toàn gói là bản trùng để xóa.
- Bản lỗi Endgame không nhập vào kết quả khoa học thành công; giữ một bản nhỏ trong lịch sử để biết vì sao từng dừng.
- Không xóa notebook cũ đang là bản nguồn duy nhất hoặc thay nội dung gốc bằng một notebook mới chưa được đối chiếu.
- Thao tác dọn chỉ áp dụng các file nghiên cứu đã đưa vào danh mục. Không mở rộng thành dọn thư mục cá nhân hoặc runtime dùng chung.

## 5. Thứ tự thực hiện và cấu trúc bộ mới

1. Khóa danh mục source/dữ liệu/checkpoint/output và liên kết giữa chúng.
2. Chốt một bản chuẩn cho từng artifact; liệt kê bản trùng, lịch sử và phần tái tạo được. Chưa đụng tới file chưa rõ vai trò.
3. Tạo bộ mới theo thứ tự `01_Source_Protocol`, `02_Data_Checkpoints`, `03_M4_Test_Results`, `04_Selector_Diagnostics`, `90_History`, `99_Inventory`. Trong lúc kiểm kê dùng manifest trỏ tới bản gốc, tránh sao chép nhiều GB chỉ để sắp folder.
4. Ráp và xác minh notebook hợp nhất với danh mục này. Cell có chú thích mục tiêu, input, output và phạm vi suy luận; source/contract/hashes được giữ. `Run All` mặc định đọc kết quả đã lưu và tổng hợp báo cáo.
5. Khi bộ mới truy xuất đủ, mới thực hiện dọn bản thừa theo danh sách và nhật ký. Không xóa bản gốc trước khi xác minh sự thay thế. File lịch sử hoặc chưa rõ vai trò được giữ ngoài bộ chính thay vì xóa vĩnh viễn.

Đây là thứ tự phụ thuộc để hoàn thành việc dọn: việc quyết định loại có thể làm ngay sau audit; thao tác xóa bản duy nhất chỉ được làm sau khi bản giữ lại đã được kiểm. Không yêu cầu người dùng chạy lại thí nghiệm để phục vụ dọn file.

## 6. Nội dung notebook hợp nhất

1. Mục tiêu, protocol và kiến trúc; nguồn nào được tái hiện chính xác, phần nào là V2.
2. Cấu hình và tự phát hiện input, registry và identity checks.
3. Dữ liệu/split/preprocessing/neighbor graph; kiểm source reuse và giới hạn protocol dựa trên bằng chứng đã lưu.
4. Định nghĩa đúng C/D/G/heads, schedule và cách lưu/restore theo worker M4 đã thực thi. Các action tốn tài nguyên được tách khỏi đường `Run All` mặc định.
5. Lịch sử train 12 job và checkpoint cuối; đọc kết quả đã có.
6. Test predictions gốc, metrics từng seed/lớp, mean ± SD và so sánh ghép cặp.
7. Learned weighting endpoint ba seed: đọc raw probe đã lưu, tái tính bảng/hình.
8. Chẩn đoán 7.500 → 8.000: phân biệt inputs đã đủ với forward chưa hoàn tất; chỉ chạy train-only inference khi chọn action tương ứng, không chọn tham số từ validation/test.
9. Kết luận, giới hạn và các câu hỏi còn mở. Không gán lợi ích nhân quả cho selector từ chẩn đoán hậu nghiệm.
10. Xuất bảng, hình, manifest nguồn/hashes và một ZIP báo cáo.

Test Indian Pines đã được xem; các phân tích/thiết kế sau đó phải được ghi là hậu nghiệm. M4 có cải thiện trung bình nhưng chưa nhất quán qua seed; việc hợp nhất không đổi cách diễn giải này và không biến run cũ thành một test mù mới.

## 7. Trạng thái thao tác

Lượt này chỉ đọc source, metadata và kết quả đã lưu; tạo bảng kiểm kê/phân loại. **Chưa train, chưa inference, chưa test lại; chưa xóa hoặc di chuyển file nguồn; chưa tạo notebook hợp nhất.**

Tài liệu đối chiếu: [Artifact map](<C:/Users/ADMIN/OneDrive/Tài liệu/ChatGPT/KLTN/protocol_b_review/main_axis_consolidation_2026-10-17/artifact_map.md>), [Cleanup map](<C:/Users/ADMIN/OneDrive/Tài liệu/ChatGPT/KLTN/protocol_b_review/main_axis_consolidation_2026-10-17/cleanup_map.md>).
