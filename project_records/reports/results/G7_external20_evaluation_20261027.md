# G7 — Đánh giá thăm dò 20 ảnh Muse AI

- Ngày: 2026-10-27. Model cố định: MLP seed42, `mlp-b8dba09efa0e`.
- 20 ảnh, mỗi ảnh một prompt đánh giá, 60 điểm do người dùng cung cấp dưới tên hoan/huy/an. Tính độc lập của người chấm chưa được xác minh riêng.
- Sửa thiếu header trong ratings.csv, đồng bộ ratings_corrected.csv; không thay điểm/nhận xét/ngày. Bản gốc được giữ trong data/external20/results/evaluation_20261027/ratings_original.csv.
- PASS kiểm 60 khóa ảnh/người duy nhất, 3 người/ảnh, điểm nguyên 1–5; checksum 20 ảnh khớp intake. PASS inference CPU 20/20, không tải hoặc train lại.

| Chỉ số so với trung bình 3 người | Kết quả |
|---|---:|
| MAE | 1.456520 |
| RMSE | 1.623083 |
| Spearman | -0.307534 |

## Các lỗi đáng chú ý

| Ảnh | Người chấm trung bình | Model | Nhận xét |
|---|---:|---:|---|
| 20 | 1.000 | 4.663 | Đã xem ảnh trực tiếp: ô tô đỏ trước nhà trắng, prompt yêu cầu xe đạp đỏ. Model chấm cao dù sai đối tượng chính. |
| 09 | 5.000 | 2.767 | Người chấm đánh giá đúng quan hệ trái/phải, model chấm thấp; cần review trực quan thêm trước kết luận nguyên nhân. |
| 14 | 1.333 | 3.530 | Theo nhận xét người chấm, ảnh có thìa trái với prompt phủ định. |
| 10 | 2.000 | 4.099 | Theo nhận xét người chấm, mèo dưới ghế thay vì trên ghế. |

Model không phân biệt tốt mức phù hợp trên bộ thăm dò này; tương quan thứ hạng âm. Chạy không lỗi không đồng nghĩa chất lượng đạt. Không đủ căn cứ dùng điểm như phán quyết tự động đáng tin cậy.

14/20 ảnh được cả ba người chấm 5; phân bố nhãn tập trung và bộ ảnh được thiết kế có chủ đích. Đây không phải mẫu đại diện, không thay test holdout, không so trực tiếp mức tăng MAE với validation để suy ra mức giảm tổng quát. Không tính pairwise accuracy giữa các prompt khác nhau. Chưa chứng minh nguyên nhân lỗi thuộc encoder hay regression head.

## Bằng chứng và việc tiếp theo

- G7_external20_evaluation_20261027.json: dự đoán từng mẫu, checksum CSV và metric.
- data/external20/results/evaluation_20261027/predictions.csv: bảng phân tích.
- Lệnh đã chạy: `.venv/Scripts/python.exe -X utf8 .cache/evaluate_external20.py` với HF_HUB_OFFLINE=1, OMP_NUM_THREADS=4, MKL_NUM_THREADS=4; exit 0. Script được lưu kèm thư mục kết quả để truy vết, không chạy lại vào cùng output.
- Đánh giá hiện chỉ là baseline đã khóa. Nếu cải thiện model theo các lỗi này, cần bộ kiểm tra mới; không tuning rồi tái tuyên bố độc lập trên external20.
- G7 chưa DONE: còn review các ví dụ, holdout sau khóa release, peak RAM, máy sạch, model card và UAT/demo. Giữ nguyên checkpoint hiện tại và báo cáo trung thực hạn chế.
