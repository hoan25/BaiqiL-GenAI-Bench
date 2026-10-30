# Vòng 2 — ghép sai prompt với weak supervision

Ngày 2026-10-29. **PASS_WITH_LIMITATIONS ở phạm vi ứng viên phát triển; chưa nghiệm thu release, chưa thay model web.**

## Giao thức và dữ liệu

Người dùng cho phép agent tự tạo, đánh giá và thử nghiệm. Tái sử dụng embeddings đã khóa; không tải dataset hoặc sinh thêm ảnh. Chỉ ghép prompt và ảnh trong train, khác prompt_group và text similarity <0,5. Nhãn1 là giả định weak supervision, có thể có false negatives; không phải đánh giá của người và không sửa nhãn gốc. Các tỷ lệ10%/25%/50% tương ứng671/1679/3359 cặp bổ sung. Seed tạo cặp1701; seed huấn luyện42/43/44; hybrid features2049, hidden64, tối đa100epoch, early stopping15 theo validation MAE.

Mọi ngưỡng được ghi trước khi chạy vòng này: validation MAE≤0,803777; MAE external20 đã thấy≤1,165216; đúng cả ba kiểm thứ tự demo. Cả ba seed phải đạt để chấp nhận một cấu hình đơn. **Không cấu hình đơn nào đạt điều kiện tất cả seed**: tỷ lệ25% có seed44 MAE1,186986 vượt ngưỡng external20. Giữ nguyên kết luận FAIL này.

Sau khi thấy dao động giữa seed, đăng ký riêng phương án trung bình đều cả ba seed, rồi kiểm cả ba tỷ lệ. Không chọn/bỏ seed theo kết quả. Đây là một ứng viên mới; không biến FAIL của các thành viên thành PASS. Ngưỡng sai số và thứ tự được giữ nguyên cho đầu ra ensemble. Việc chọn cấu hình tiếp tục dùng external20 đã thấy, vì vậy đây chỉ là development.

## Kết quả ensemble

| Tỷ lệ cặp bổ sung | MAE validation | MAE external20 đã thấy | Ba kiểm thứ tự | Gate phát triển |
|---|---:|---:|---|---|
|10%|0,760844|1,175399|3/3|FAIL|
|25%|0,758046|1,145833|3/3|PASS_WITH_LIMITATIONS|
|50%|0,755106|1,231266|3/3|FAIL|

Ứng viên25% so với baseline đang phục vụ: validation MAE0,773777→0,758046; external20 MAE1,456520→1,145833 (giảm khoảng21,3%). External20 Spearman −0,307534→0,320541. Validation pairwise0,661340→0,663402, mức tăng nhỏ. Không kết luận có ý nghĩa thống kê hay tổng quát hóa tốt hơn từ20 ảnh.

| Kiểm tra đã biết | Baseline | Ensemble25% |
|---|---:|---:|
|Prompt táo, ảnh táo|4,073|4,344|
|Prompt táo, ảnh ô tô|4,634|1,802|
|Prompt ô tô, ảnh ô tô|4,700|3,076|
|Prompt xe đạp, ảnh ô tô|4,663|1,322|
|Prompt phi hành gia, ảnh ô tô|4,714|1,044|

## Kiểm chứng thực thi

- `pytest tests/unit/test_synthetic_negatives.py -q`:2PASS/0,44giây; khác nhóm, tái lập, bỏ qua prompt quá giống và từ chối tập chỉ một nhóm.
- `scripts/experiment_synthetic_negatives.py --root .cache/reviewed-training-20261020T174855207187Z --output .cache/synthetic-negatives-20261029`:exit0,9lượt huấn luyện;27head save/load delta0; input checksum và pair order khớp.
- `scripts/check_interaction_candidates.py --experiments .cache/synthetic-negatives-20261029 --output reports/results/G7_synthetic_diagnostics_20261029.json`:exit0;18Ridge/MLP được kiểm bằng CLIP thật trên20ảnh+8cặp chẩn đoán. Ridge lặp seed không phải ba phép thử độc lập.
- `scripts/check_synthetic_ensembles.py`:exit0;9MLP đúng checksum; batch/single16mẫu mỗi head lệch<1e-5. Các metric và checkpoint hash trong `G7_synthetic_ensembles_20261029.json`.
- Holdout: **NOT RUN lại**. Không thay production predictor, không chạy lại UAT/API cho ứng viên chưa tích hợp.

## Giới hạn và quyết định

Giữ ứng viên ensemble25% để review/tích hợp thử nghiệm sau; giữ nguyên baseline và bằng chứng lịch sử. Chưa sửa hết các ca khó: ảnh08 vẫn4,127; ảnh10 vẫn4,003 dù là kế hoạch mismatch; không suy ra nhãn đúng chỉ từ planned_relation. Sai số bình quân1,146/5 còn đáng kể. Các mẫu này đã dùng chọn mô hình nên không thể làm test độc lập nữa. Ba head có thể làm tăng thời gian head inference; chưa đo latency toàn API của ensemble.

Nhiệm vụ tiếp theo trước release: kiểm bộ ảnh/prompt mới với nhãn độc lập, artifact loader/version/metrics dành riêng cho ensemble, E2E và benchmark toàn API, nhóm review/UAT. Không cần người dùng tạo lại20ảnh hiện có hoặc chạy Kaggle để tái hiện vòng thử CPU này. Không tuyên bố đã hoàn thành G7.
