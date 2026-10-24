# Yêu cầu của người dùng về giá trị nghiên cứu

Ngày ghi nhận: 24/10/2026. Đây là yêu cầu trực tiếp của người dùng, dùng làm tiêu chuẩn cho các đề xuất nghiên cứu tiếp theo.

## Phạm vi thiết kế được người dùng làm rõ ngày 24/10/2026

- Bắt buộc giữ Indian Pines làm dataset gốc/chính; có thể thêm dataset ở các bước sau.
- Bắt buộc giữ GAN, kế thừa nền tảng/bài toán của 3D-HyperGAMO. Bỏ D không phải trục chính hoặc phương án cuối mặc định.
- Có thể thiết kế lại toàn bộ G/D/C, biểu diễn, loss, optimizer, lịch cập nhật, preprocessing và protocol. Không bắt buộc giữ V2, 3-NN, rho=0,2, patch 11×11 hoặc recipe paper gốc cho các run mới.
- Những thay đổi đó tạo thí nghiệm mới; source/checkpoint/output lịch sử vẫn giữ nguyên và chỉ dùng so sánh khi điều kiện tương thích.
- 30 giờ GPU sắp tới là ngân sách thăm dò để tạo nền tảng cho cải tiến cuối, không mặc định là ngân sách xác nhận hoàn chỉnh của mọi nhánh × mọi seed/dataset.
- Mục tiêu là tìm cơ chế GAN có lợi ích sử dụng thực tế, không tự hạ mục tiêu thành chẩn đoán V2 hoặc chỉ cắt chi phí D.
- Không hứa một model hoàn hảo hoặc thắng trước khi có dữ liệu. Phải chủ động thiết kế và phản biện giả thuyết, thay vì lấy thiếu bằng chứng endpoint làm điều kiện cấm mọi thiết kế mới.

## Yêu cầu cốt lõi

Không lấy việc đủ điều kiện làm khóa luận, chạy được GAN, ghép các kỹ thuật hoặc tăng điểm so với bản cũ của nhóm làm bằng chứng tự động rằng phương pháp có giá trị.

Kết quả cuối phải trả lời được:

> Với một bài toán/điều kiện sử dụng cụ thể, vì sao một người nghiên cứu hoặc người dùng nên lựa chọn phương pháp của nhóm thay vì các phương án hiện có phù hợp?

Không bắt buộc phát minh thuật toán hoàn toàn mới. Nhưng phải có một tối ưu hoặc lợi ích thực nghiệm có ý nghĩa đối với một vấn đề được xác định và có bằng chứng.

## Điều cần chứng minh trước khi gọi là cải tiến có giá trị

1. Xác định người dùng/điều kiện áp dụng, vấn đề còn tồn tại và hạn chế của các phương án hiện có trong điều kiện ấy.
2. Giải thích cơ chế đề xuất giải quyết đúng vấn đề nào; phân biệt bằng chứng với giả thuyết.
3. Chọn đối chứng đủ mạnh, phù hợp và còn liên quan, gồm phương pháp đơn giản cũng như phương pháp hiện đại có thể thực hiện trong ngân sách. Không chỉ so với bản cũ của nhóm.
4. Chứng minh lợi ích cụ thể: nhận diện lớp thiểu số, độ ổn định, hiệu quả dữ liệu, chi phí, bộ nhớ hoặc một tiêu chí ứng dụng đã xác định. Không tự đổi tiêu chí sau khi xem test.
5. Báo đánh đổi và điều kiện phương pháp không còn đáng lựa chọn. Không hứa một phương án phải thắng mọi metric hoặc mọi dataset.

## Cách diễn giải cần tránh

- Không tự động gọi mức tăng khoảng 1 điểm phần trăm là thành công. Phải xét độ bất định, lớp hưởng lợi, protocol, chi phí và đối chứng.
- Không xem kỹ thuật cũ là vô giá trị chỉ vì tuổi đời. Giá trị được quyết định bởi lợi ích có bằng chứng trong điều kiện sử dụng.
- Không dùng kiến trúc lớn/phức tạp hoặc tên gọi mới thay cho bằng chứng.
- Không dùng điểm gần 100% của paper khác để so trực tiếp khi chưa đối chiếu split, preprocessing, train size, leakage, classifier và model selection. Đồng thời, khác protocol không phải lý do để bỏ qua các phương pháp hiện đại hoặc né đối chứng mạnh.
- Không coi một ablation có câu hỏi rõ là đã đủ chọn làm contribution chính. Nó có thể chỉ là bước xác định cơ chế.

## Trạng thái hiện tại

- V2 đã có kết quả cải thiện trong các điều kiện đã chạy; chưa chứng minh một lợi thế lựa chọn rõ trước các phương án hiện đại phù hợp.
- Tối ưu/bỏ D đang là ứng viên nghiên cứu, chưa phải hướng đã chốt hoặc một phương pháp đã chứng minh giá trị.
- Chưa duyệt full suite 30 giờ chỉ từ các đề xuất hiện tại.
- Indian Pines và nền 3D-HyperGAMO được giữ theo yêu cầu hiện hành; mọi thay đổi phạm vi cần được trình bày rõ.

## Mẫu bắt buộc cho đề xuất tiếp theo

Trước kiến trúc hoặc runner GPU, mỗi đề xuất phải nêu ngắn gọn:

1. Vấn đề và bằng chứng nó đáng giải quyết.
2. Đối tượng/điều kiện mà kết quả sẽ hữu ích.
3. Phương án hiện có tốt nhất/phù hợp nhất cần so.
4. Lợi ích kỳ vọng, cơ sở cơ chế và rủi ro.
5. Thí nghiệm có thể phân biệt lợi ích ấy trong ngân sách.
6. Điều kiện chấp nhận, bác bỏ hoặc chưa kết luận.

Nếu chưa điền được các mục này bằng lập luận và dữ liệu cụ thể, chưa chuyển sang full training.
