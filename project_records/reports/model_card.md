# PromptAlign AI — Model card

Ngày cập nhật: 2026-10-27. Trạng thái: demo nghiên cứu, chưa nghiệm thu release.

## Mô hình và mục đích

CLIP ViT-B/32 frozen (revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`) mã hóa ảnh và prompt. Regression head MLP seed42 dự đoán độ phù hợp 1–5; phiên bản phục vụ `mlp-b8dba09efa0e`. Ridge và trung bình train là đối chứng. Nhóm chưa fine-tune CLIP hoặc huấn luyện ranking loss; không tuyên bố cải tiến contrastive learning.

Dùng để nghiên cứu và demo chấm ảnh/so sánh 2–6 ảnh cùng prompt. Điểm không phải xác suất đúng, không phải đánh giá vẻ đẹp, không phải lời giải thích chi tiết ảnh. Chat điều hướng chức năng bằng intent, không phải chatbot VLM tổng quát.

## Dữ liệu và giao thức

Nguồn: BaiqiL/GenAI-Bench, revision `94e543b0801feade1419873335191de90b68d164`; 12 shard, 1.600 prompt, 9.598 cặp hợp lệ. Train/validation/test: 6.718/1.440/1.440 cặp, chia theo nhóm prompt. Nhãn là trung bình 3 đánh giá con người. Hai nhãn lỗi bị loại, không sửa/điền; lý do giữ prompt gần trùng trong train xem [DATA_DECISIONS.md](DATA_DECISIONS.md).

Chọn model theo validation, primary seed42 định trước, seed43/44 dùng báo ổn định. Đã đánh giá holdout một lần ngày 2026-10-27 theo configs/holdout_seed42_20261027.json, không chỉnh model sau external20 hoặc test. Không được gọi validation là test; web vẫn hiển thị validation từ artifact gốc.

## Kết quả và giới hạn

| Tập đánh giá | MAE | RMSE | Spearman |
|---|---:|---:|---:|
| GenAI-Bench validation, MLP seed42 | 0.773777 | 0.950154 | 0.480619 |
| GenAI-Bench test holdout, MLP seed42 | 0.791899 | 0.966854 | 0.437097 |
| External20 Muse AI, trung bình 3 người chấm | 1.456520 | 1.623083 | -0.307534 |

Validation pairwise accuracy 0.661340 trên các cặp cùng prompt đủ chênh nhãn. External20 mỗi ảnh khác prompt nên không tính pairwise. External20 chỉ 20 mẫu được chọn có chủ đích, 14 mẫu được cả 3 người chấm 5; không phải mẫu đại diện. Tính độc lập của ba người chấm do người dùng cung cấp chưa được xác minh riêng.

Đã quan sát lỗi nghiêm trọng: ảnh ô tô đỏ nhưng prompt yêu cầu xe đạp đỏ, model chấm 4.663 trong khi nhãn người là 1.0. Không dùng điểm làm quyết định tự động đáng tin cậy; cần người xem ảnh kiểm lại. Chưa xác định nguyên nhân nằm ở encoder hay head. Nếu cải thiện theo external20, phải dùng tập mới để kiểm chứng độc lập.

Tiếng Việt chưa được đánh giá riêng; giao thức hiện dùng prompt tiếng Anh. Tokenizer cắt ở 77 token, có thể bỏ chi tiết cuối prompt dài. Chất lượng trên ảnh thật, ảnh khác miền hoặc nguồn sinh khác chưa được chứng minh từ validation GenAI-Bench. Không suy ra hiệu năng trên nhiều ảnh cùng prompt từ external20.

## Tái lập, nguồn và quyền sử dụng

Xem [hướng dẫn web](../docs/WEB_LOCAL.md), [review training](results/G3_full_training_review.md), [external20](results/G7_external20_evaluation_20261027.md). Model chạy CPU local từ artifact và CLIP cache, không cần phiên Kaggle online. Không phân phối trọng số/dataset trong Git.

Dataset là BaiqiL/GenAI-Bench; encoder là openai/clip-vit-base-patch32. Phiên bản và checksum được lưu trong artifact. Chưa hoàn tất kiểm tra giấy phép cho gói bàn giao; tài liệu này không cấp quyền tái phân phối dataset, encoder hoặc ảnh Muse AI. Metadata quyền sử dụng ảnh external20 còn trống; nguồn Muse AI được người dùng xác nhận, prompt tạo ảnh giữ nguyên.

## Nghiệm thu còn thiếu

Holdout của ứng viên đã khóa đã hoàn tất; dependency sạch trên cùng máy đã kiểm. Còn review chéo, UAT/demo cuối và giới hạn bàn giao trên máy khác. Kiểm thử kỹ thuật thành công không thay cho chất lượng mô hình hoặc người review xác nhận. G6 là tùy chọn. Nguồn/giấy phép và phần chưa xác minh: [SOURCES_AND_LICENSES.md](SOURCES_AND_LICENSES.md).
