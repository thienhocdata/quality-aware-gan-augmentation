# V3-A — một lần Save & Run All trên Kaggle

Mở/import **selector.ipynb** trong gói này. Đây là phiên bản thực thi V3-A; notebook phân tích V2 đã chốt trên local vẫn được giữ ở bộ KLTN_clean. V3-A là ứng viên chưa có kết quả, không thay kết luận V2 bằng một kết quả giả.

## Trước khi chạy

1. Gắn **full Output/cache Protocol B** `KLTN_HyperGAMO_B_50e988fae260` làm Kaggle Input. Có thể dùng Input đã gắn ở các lượt trước; hỗ trợ thư mục Kaggle tự giải nén hoặc ZIP. Không cần upload lại các checkpoint G V2 để train V3 fresh.
2. Bật GPU. Runner tự dùng tối đa hai GPU, mỗi process chỉ thấy một GPU. Một GPU sẽ chạy nối tiếp.
3. Chọn **Save Version → Save & Run All**, giữ việc lưu Output. Không cần chạy từng cell hoặc đổi ACTION giữa chừng.

Mã M4 khóa, worker V3, baseline M4 đã lưu và metadata gốc được nhúng trong notebook. Runner không cài TensorFlow, không đổi worker M4 và không dùng test để chọn beta/hyperparameter.

## Luồng tự động

1. Kiểm source, hashes, train 5.528, class order và graph; kiểm test cache có sẵn bằng metadata, chưa đọc test tensors.
2. Train mới seed 42 đến 200 C updates, cùng C/D/G/Dense recipe M4 và beta=0. Kiểm đường học, ownership/freeze, geometry và trạng thái hữu hạn trong chính các updates đó.
3. Lưu native checkpoint 200; process mới restore và đối chiếu đủ weights/BN/Adam/beta/dropout/audit. Gate lỗi thì dừng, lưu ZIP và không chạy dài.
4. Nếu gate đạt, seed 42 tiếp tục từ 200 đến **tổng 8.000**; chạy seed 43/44 đến 8.000 theo cùng source. Tối đa hai job đồng thời; không chạy lại 200 đầu.
5. Đủ ba job hợp lệ mới mở locked test B 4.721 mẫu. Lưu kết quả ba V3 classifier và đối chiếu với **full M4 V2 lịch sử**, không phải C-only endpoint.
6. Lưu `V3A_IP_ONERUN_FULL_BACKUP.zip` và báo cáo cuối. Không cần bấm FINALIZE riêng.

## Cấu hình và ý nghĩa

- V3-A thêm beta scalar tự do âm/dương cho mỗi trong 15 head; khởi tạo 0. G/D/C/Dense đều train fresh. Giữ k=3, rho=.2, MSE, legacy Adam và schedule M4.
- Khoảng cách Euclidean trên patch PCA/scaler train 4.840 giá trị; q=(d−mean)/max(population std,1e−8) theo ba neighbor của từng anchor; exact ties trả 0.
- 200 C updates tương ứng 100 loops và 3.400 optimizer calls. Toàn bộ ba seed gồm 24.000 C updates; pilot nằm trong ngân sách đó.
- Mặc định giới hạn toàn pipeline 7,5 giờ, pilot khoảng tối đa 30 phút cộng phần dự phòng lưu, mỗi process full tối đa ba giờ rồi tự resume khi còn thời gian phiên. Đây không phải cam kết thời gian hoàn thành: phụ thuộc GPU, runtime và hạn mức Kaggle.
- Runtime được kiểm thật trên Kaggle: TensorFlow/tf-keras 2.20.x tương thích và NumPy 2.x. Phiên bản thực tế của lượt V3 mới được khóa chung qua ba seed. Baseline M4 lịch sử có khác biệt runtime; báo cáo không tự gọi paired-identical.

## Output được giữ

- Source/config/identity/provenance và các log.
- Pilot gate, fresh-restore verification, checkpoint pilot 200 và native checkpoint cuối còn giữ.
- Classifier weights và native C/D/G/head/beta/Adam state từng seed; beta trajectory và train-only selector probes.
- `y_true`, `y_pred`, probabilities, coords, confusion matrix cho mỗi classifier.
- OA/AA/Macro-F1/GM/Kappa/NLL/Brier, per-class, mean ± sample SD, chênh lệch từng seed với full V2.
- Manifest có trạng thái **COMPLETE** chỉ khi đủ cả ba classifier và test. FAILED/PAUSED được hiển thị rõ, không giả lập số liệu.

ZIP backup được thay atomically theo các snapshot checkpoint đã commit. `/kaggle/working` và ZIP trong đó chưa phải bản sao ngoài phiên: cần Output của phiên Save & Run All được lưu thành công hoặc tải ZIP về. Nếu Kaggle ngắt/hết GPU trước khi xong, chỉ có thể tiếp tục từ checkpoint đã được giữ lại; không bảo đảm một phiên luôn đủ thời gian.

## Resume nếu bị Kaggle ngắt

Gắn `V3A_IP_ONERUN_FULL_BACKUP.zip` đã lưu thành một Input rồi Run All lại cùng notebook. Bootstrap tìm backup tương thích theo source/config identity. Job hoàn tất được kiểm và giữ lại; job dở tiếp tục từ native checkpoint. Không mở test nếu còn thiếu job.

## Trạng thái kiểm tra trước khi giao

Đã kiểm local bằng CPU: cú pháp, geometry/standardization, pins lịch sử, discovery dữ liệu thật từ ZIP và thư mục, bootstrap payload và các gate file/hash. **Máy tạo notebook không có TensorFlow; chưa forward/backprop/GPU thật.** Các kiểm tra runtime đó là phần đầu của lượt Kaggle tự động, chưa được gán PASS ở local.

Ngay cả khi V3 thắng full V2, kết quả vẫn là hậu nghiệm trên test B đã xem và có chồng lấn không gian; không tự chứng minh quan hệ nhân quả, tính mới hoặc tổng quát hóa sang vùng độc lập. Báo đủ cả kết quả không thuận lợi.
