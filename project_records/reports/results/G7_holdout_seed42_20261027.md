# Test holdout — ứng viên MLP seed42 đã khóa

- Ngày 2026-10-27; 1.440 cặp, 240 nhóm prompt test từ manifest đã khóa.
- Giao thức và hash đầu vào được lưu trước khi tính metric: configs/holdout_seed42_20261027.json.
- Giữ MLP seed42 chọn từ validation trước đây; đã xem external20 nhưng không sửa model/scaler/threshold hoặc chọn lại seed. Không fit bất kỳ thứ gì trên test.
- scripts/evaluate_holdout.py: 2 gate tests PASS; chạy chính thức exit0. Hash input kiểm trước và sau; không ghi đè output có sẵn. Dự đoán và protocol hash nằm trong JSON cùng tên.

| Model | MAE | RMSE | Spearman | Pairwise accuracy |
|---|---:|---:|---:|---:|
| Mean train | 0.925712 | 1.056265 | N/A | 0.000000 |
| CLIP cosine | N/A | N/A | 0.178064 | 0.562694 |
| Ridge seed42 | 0.912457 | 1.130086 | 0.295013 | 0.646632 |
| **MLP seed42 (đã chọn)** | **0.791899** | **0.966854** | **0.437097** | **0.670466** |

Pairwise tính trong cùng prompt: 1.930 cặp đủ chênh nhãn >=0,5; loại 1.670 cặp tie/gần tie. Prediction tie nhận 0 điểm theo giao thức cũ nên mean=0 không phải mức ngẫu nhiên 50%. Cosine không quy đổi thành thang 1–5 nên không tính MAE/RMSE.

MLP giữ lợi thế so với các đối chứng trên holdout; đây là quan sát sau đánh giá, không phải lý do chọn lại model. Không suy ra 67,05% là xác suất chấm đúng cho mọi ảnh. External20 vẫn có MAE1,457/Spearman âm, nên khả năng tổng quát hóa sang bộ ảnh khác còn hạn chế. Không chạy lại tuning rồi dùng cùng test để báo đánh giá độc lập.

E2E-03 có bằng chứng thực thi; release/UAT/review chéo chưa hoàn thành. Web tiếp tục hiển thị metrics validation từ artifact gốc, không thay nhãn của bảng thành test hoặc sửa artifact đã review.
