# Kết quả training GenAI-Bench và kiểm chứng local

Đối chiếu ngày 2026-10-26, gồm bằng chứng đã tạo ở phiên trước nhưng chưa ghi nhật ký. Source training: `2b48cca`; ZIP `training-20261020T174855207187Z.zip`. Không chạy lại training để cập nhật báo cáo.

- Kaggle Tesla T4, torch 2.8.0+cu126; bootstrap/pip check/import sạch; 30 pipeline tests PASS (3,13 giây).
- 9.598 cặp × 512 chiều; extraction 256,17 giây; fingerprint CLIP không đổi.
- Train/validation/test: 6.718/1.440/1.440. Test **chưa đánh giá**.
- CPU replay độc lập: hash lock/manifest/embeddings/9 checkpoint khớp; embeddings hữu hạn và chuẩn hóa; scaler khớp thống kê train; validation metrics khớp tolerance 1e-5, best epoch và chọn head đúng.
- Bằng chứng: `G3_full_training_review.json`; smoke API checkpoint thật `G4_full_model_http.json`. Script tái kiểm tra: `scripts/review_training.py`; không ghi đè run cũ.

| Model, validation seed 42 | MAE | RMSE | Spearman | Pairwise accuracy |
|---|---:|---:|---:|---:|
| M0 mean | 0,937855 | 1,073653 | N/A | 0,000000 |
| M1 CLIP cosine | N/A | N/A | 0,184345 | 0,562371 |
| Ridge | 0,867710 | 1,083012 | 0,404081 | 0,622165 |
| MLP | 0,773777 | 0,950154 | 0,480619 | 0,661340 |

Pairwise chỉ so ảnh cùng prompt: 1.940 cặp đủ khoảng cách nhãn ≥0,5; 1.660 cặp nhãn gần nhau bị loại. Quy tắc hiện tại cho prediction tie 0 điểm, nên mean predictor có accuracy 0; không diễn giải đây là random baseline 50%. Cosine M1 chưa ánh xạ sang thang 1–5 nên MAE/RMSE không áp dụng.

MLP MAE ba seed: 42 = 0,773777; 43 = 0,773113; 44 = 0,768037. Mean ± population std = **0,771642 ± 0,002564**. Giữ **seed 42** là run chính đã định trước; không đổi sang seed 44 vì điểm tốt hơn. Seed 42 best epoch 20, dừng ở epoch 35.

Checkpoint chọn: `mlp.pt`, SHA256 `b8dba09efa0ef038ff69393ffe5bcdc7e96f742f13f7131ead3f14cb228d18dc`. Review chỉ cho phép demo local với metrics validation, chưa phải nghiệm thu release hoặc khẳng định tổng quát hóa trên test/ảnh ngoài dataset.

Quyết định dữ liệu và đoạn giải trình dùng cho báo cáo: [DATA_DECISIONS.md](../DATA_DECISIONS.md). Raw Parquet không có trong ZIP training; kiểm local dựa trên manifest đã khóa và embeddings/checkpoint, không giả nhận đã đọc lại ảnh nguồn.
