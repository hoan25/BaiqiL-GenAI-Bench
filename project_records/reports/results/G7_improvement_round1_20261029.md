# Vòng cải thiện 1 — chưa có ứng viên thay baseline

Người dùng cho phép tiếp tục đến khi kết quả ổn định. Vòng này thử 3 feature map × 3 seed42/43/44, mỗi run fit mean/Ridge/MLP. Không đọc/tính metric holdout; không chỉnh nhãn. Checkpoint mới nằm trong .cache/interactions-20261029 và .cache/hybrid-20261029, không ghi đè baseline.

| MLP feature map | Validation MAE mean ± population std (3 seed) |
|---|---:|
| Baseline nối text/image (bằng chứng cũ) | 0.771642 ± 0.002564 |
| Chỉ cosine | 0.923569 ± 0.001431 |
| Cosine + product + abs difference | 0.836740 ± 0.001509 |
| Nối text/image + các tương tác trên | 0.780171 ± 0.005307 |

Giữ kiến trúc64-ReLU, epochs100/patience15/lr0,001, alpha Ridge1, scaler fit train. Primary42 cố định; không chọn seed44 của hybrid chỉ vì MAE đẹp nhất. Các head reload delta0. Hai test feature maps PASS, gồm thứ tự batch, trường hợp aligned/misaligned và dữ liệu không hợp lệ.

## Chẩn đoán trên dữ liệu đã thấy

- Cosine Ridge/MLP qua cả3 kiểm thứ tự: táo thắng ô tô khi prompt táo; ảnh ô tô hợp prompt ô tô hơn prompt xe đạp/phi hành gia. Nhưng sai số validation cao hơn rõ rệt; external20 MAE vẫn khoảng1,39–1,43, chưa phải giải pháp đạt.
- Interaction MLP và hybrid MLP vẫn thất bại ca táo so ô tô ở cả3 seed; chỉ sửa feature chưa đủ. Hybrid42 validation MAE0,784479, external20 đã thấy MAE1,4441. Không tự triển khai lên web.
- Ridge không ngẫu nhiên trong thiết kế này, kết quả3 seed giống nhau không phải bằng chứng mạnh về độ bền dữ liệu.
- External20 và các ca prompt được dùng như development diagnostics, không là tập kiểm chứng độc lập mới. Không chọn model theo test holdout đã mở. Không có ngưỡng chất lượng đã được người dùng nghiệm thu, nên không gọi kết quả ổn định về seed là chất lượng đạt.

## Phần cần thêm dữ liệu thật

Đã chuẩn bị data/development40/pairs.csv và rate.html: 40 cặp từ ảnh đã thấy ghép prompt khác, không gán nhãn tự động. Người dùng chấm theo ảnh thực; không mặc định mọi cặp tráo là1/5. Sau khi có nhãn có thể làm vòng phát triển bổ sung, cần kiểm độc lập trên ảnh/prompt mới trước xác nhận cải thiện tổng quát.

Bằng chứng: G7_interaction_training_20261029.json, G7_hybrid_training_20261029.json, G7_candidate_diagnostics_20261029.json. Các metric không bị ghi đè khi vòng đầu kém. Baseline/web giữ nguyên.
