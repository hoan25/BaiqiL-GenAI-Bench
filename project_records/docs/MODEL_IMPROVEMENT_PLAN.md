# Vòng cải thiện sau demo — đề xuất để review

Ngày 2026-10-29. Đã chạy các đối chứng cosine/interaction/hybrid; chưa thay model phục vụ. Baseline đã khóa là MLP seed42; giữ nguyên báo cáo validation, holdout và external20.

## Vòng 2 được người dùng cho phép: tự tạo cặp bổ sung

- Người dùng cho phép tự tạo, đánh giá và test để tìm kết quả tạm chấp nhận. Thử weak supervision từ embeddings train có sẵn; không cần tạo ảnh mới hay tải dataset.
- Ghép ảnh train với prompt thuộc nhóm train khác, lọc text similarity <0,5 để giảm prompt gần nghĩa. Nhãn1 là **giả định tổng hợp có thể sai**, không phải nhãn con người; không thay ratings.csv hoặc điền thay người chấm development40.
- Khóa trước ba tỷ lệ mẫu bổ sung 10%,25%,50% và ba seed42/43/44; hybrid features, cấu hình head như vòng trước. Chỉ train/validation; không mở lại holdout. Lưu nguồn chỉ số từng cặp và checksum input.
- Ngưỡng thử nghiệm tạm thời cho từng seed: validation MAE≤0,803777; MAE trên external20 đã thấy≤1,165216 (giảm20% so baseline); cả ba kiểm thứ tự demo phải đúng. Đây là ngưỡng kỹ thuật phát triển do agent đề xuất, không phải người dùng ký nghiệm thu hoặc bằng chứng độc lập.
- Nếu không đạt, ghi FAIL chất lượng và giữ baseline; không hạ ngưỡng sau khi xem kết quả. Nếu đạt, vẫn cần đánh giá độc lập mới trước tuyên bố cải thiện tổng quát và review trước thay checkpoint phục vụ.

## Vấn đề cần giải quyết

Chấm cao cho ảnh không khớp prompt và ranking sai rõ ràng. Chẩn đoán ở reports/results/G7_prompt_sensitivity_20261029.md tái hiện lỗi trực tiếp trong predictor, không phụ thuộc UI. Vì vậy sửa giao diện hoặc lấy cosine nhân5 không giải quyết hợp lệ bài toán hồi quy.

## Thứ tự thí nghiệm nhỏ

1. Đối chứng hồi quy chỉ từ cosine, fit trên nhãn train. Giữ raw cosine là similarity, không tự quy đổi tuyến tính tùy ý thành thang1–5. Báo cùng metrics trên validation.
2. Thêm đặc trưng tương tác: cosine, tích từng phần tử text×image, trị tuyệt đối text−image. Fit scaler chỉ trên train; thử head nhỏ và seed42 trước, không mở rộng hàng loạt hyperparameter.
3. So sánh các ứng viên với baseline cũ trên cùng validation. Kiểm các ca sai đã biết như một bộ kiểm tra chẩn đoán đã thấy, không gọi chúng là đánh giá độc lập. Báo cả tradeoff chất lượng tổng thể và nhạy prompt, không chỉ chọn ví dụ đẹp.
4. Chỉ cân nhắc ranking loss/hard negatives nếu các đối chứng đơn giản chưa đủ. Prompt tráo ngẫu nhiên không tự có nhãn1/5; nếu thêm supervision mới, cần giao thức nhãn/review rõ ràng.

## Chống sai lệch đánh giá

- Test GenAI-Bench đã mở ngày27/10: không dùng để tuning, không chạy đi chạy lại rồi gọi là test độc lập. Mọi tuyên bố về phiên bản mới phải nêu rõ điều này.
- External20 và hai ảnh demo đã được dùng chẩn đoán; không dùng làm bằng chứng cải thiện tổng quát độc lập.
- Trước xác nhận model mới, chuẩn bị một bộ ảnh/prompt mới chưa dùng điều chỉnh, có cả nhiều ảnh cùng prompt để kiểm ranking và cùng ảnh với prompt đúng/sai; chấm độc lập trước dự đoán. Quy mô/giao thức phải chốt trước, không dừng thu thập theo metric đẹp.
- Không thay checkpoint web trước khi có kiểm artifact, kết quả đối chứng và review. Không sửa nhãn để model thắng.

## Nghiệm thu và bước cần người dùng

Có thể hoàn thiện báo cáo baseline hiện tại như một demo nghiên cứu có hạn chế, hoặc thực hiện vòng cải thiện trên. Không đánh dấu chất lượng sản phẩm đạt chỉ vì test phần mềm xanh. Review của nhóm vẫn cần cho G7.
