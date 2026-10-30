# Chẩn đoán lỗi chấm điểm sau demo

Ngày 2026-10-29. Model giữ nguyên `mlp-b8dba09efa0e`, seed42. Chỉ dùng hai ảnh external20 đã xem và bốn prompt; không đọc/tính lại test holdout, không fit.

| Prompt đánh giá | Điểm ảnh táo01 | Điểm ảnh ô tô20 | Cosine ảnh ô tô20 |
|---|---:|---:|---:|
| Quả táo đỏ trên đĩa trắng | 4.073 | 4.634 | 0.167 |
| Ô tô đỏ trước nhà trắng | 3.858 | 4.700 | 0.237 |
| Xe đạp đỏ trước nhà trắng | 3.737 | 4.663 | 0.176 |
| Phi hành gia đi trên Mặt Trăng | 3.912 | 4.714 | 0.055 |

## Đã xác minh

- Predictor trực tiếp và encoder + head trực tiếp cho điểm giống nhau, max delta0. Kết quả táo/ô tô tái hiện số liệu demo; lỗi không cần UI hay session để xuất hiện.
- Ở ảnh ô tô, đổi prompt giữa bốn câu chỉ làm score dao động0,079919; prompt phi hành gia nhận điểm cao nhất. CLIP cosine phản ứng rõ hơn và cao nhất với prompt ô tô trong bốn câu này.
- Pipeline features hiện nối text/image embedding thành1024 chiều; không đưa cosine, tích từng phần tử hoặc khoảng cách trực tiếp vào head. MLP64-ReLU có thể học tương tác, nhưng không được đảm bảo sẽ học tốt. Ridge trên vector nối là tổng hai đóng góp tuyến tính trước clamp, không có tương tác ảnh–text trực tiếp.
- Trạng thái script PASS_EXECUTION_NOT_MODEL_QUALITY: kiểm chạy và provenance thành công, không phải chất lượng đạt.

## Giả thuyết và điều chưa biết

Hành vi phù hợp với việc head dựa nhiều vào tín hiệu riêng của ảnh hơn mức phù hợp với prompt trong các ca này. Chưa chứng minh cơ chế hoặc mức độ trên toàn dữ liệu. Chưa đủ để kết luận CLIP luôn đúng, tất cả prompt bị bỏ qua, hoặc chỉ thêm cosine chắc chắn khắc phục lỗi. Không suy ra accuracy từ tám tổ hợp không có bộ nhãn độc lập mới.

Lệnh: `.venv/Scripts/python.exe -X utf8 scripts/diagnose_prompt_sensitivity.py --model .cache/models/clip-vit-base-patch32/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268 --training .cache/reviewed-training-20261020T174855207187Z/training-seed42 --images data/external20/images/images --output reports/results/G7_prompt_sensitivity_20261029.json`.

Exit0; HF offline, OMP/MKL4. JSON lưu hash ảnh/training report và từng dự đoán. Checkpoint và các metric đã công bố giữ nguyên.
