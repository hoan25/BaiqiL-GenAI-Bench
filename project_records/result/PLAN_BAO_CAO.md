# Kế hoạch viết báo cáo

## 1. Mục tiêu và phạm vi

Đề tài: **Hệ thống hội thoại đánh giá mức độ phù hợp giữa mô tả (prompt) và ảnh bằng CLIP, Regression và Ranking Learning**.

Báo cáo trình bày phần đã triển khai: CLIP frozen, mean/cosine baseline, Ridge/MLP và ứng dụng FastAPI–React. Xếp hạng hiện tại dựa trên điểm hồi quy; chưa huấn luyện pairwise ranking loss, chưa fine-tune CLIP. Nếu giữ “Ranking Learning” trong tên đề tài, phải nêu rõ giới hạn này trong phần phạm vi và kết luận.

## 2. Dàn ý đề xuất

Độ dài tham khảo 25–35 trang nội dung, điều chỉnh theo mẫu môn học; chưa bao gồm phụ lục.

| Phần | Nội dung cần viết | Tư liệu hiện có | Dung lượng gợi ý |
|---|---|---|---|
| Tóm tắt | Bài toán, dữ liệu, phương pháp, kết quả validation và giới hạn | TONG_HOP_KET_QUA.md | 1 trang |
| Chương 1: Giới thiệu | Động cơ, mục tiêu, đầu vào/đầu ra, phạm vi MVP | Kế hoạch gốc, README dự án | 3–4 trang |
| Chương 2: Cơ sở lý thuyết | CLIP/contrastive pretraining, cosine, Ridge, MLP, MAE/RMSE/Spearman/pairwise | Source ML; bổ sung và xác minh tài liệu học thuật gốc | 4–5 trang |
| Chương 3: Dữ liệu và phương pháp | 12 shard, nhãn trung bình 3 người, QA, loại 2 cặp, group split, embeddings, scaler và cấu hình train | evidence/G1_kaggle_full_qa_review.json, DATA_DECISIONS.md, training_seed42.json | 5–6 trang |
| Chương 4: Thiết kế và triển khai | Luồng VS Code → GitHub → Kaggle → local, kiến trúc API/UI, session, upload, checkpoint | Source backend/frontend; hình giao diện thật | 4–5 trang |
| Chương 5: Thực nghiệm | Protocol cố định, bảng M0/M1/M2, 3 seed, early stopping, kiểm thử và CPU latency | tables/, evidence/ | 6–8 trang |
| Chương 6: Kết luận | Đóng góp thực tế, hạn chế, việc chưa hoàn thành, hướng phát triển | Bản tổng hợp và checklist bên dưới | 2–3 trang |
| Phụ lục | Cấu hình, checksum, lệnh tái lập, log, phân công, kịch bản demo | SOURCES.json, docs hướng dẫn của repository | Theo nhu cầu |

## 3. Bảng và hình cần đưa vào

1. Bảng split theo prompt/cặp ảnh (`tables/data_split.csv`).
2. Sơ đồ pipeline: prompt + ảnh → CLIP frozen → nối 2 embedding → scaler train-only → MLP → điểm 1–5 → xếp hạng.
3. Bảng so sánh validation (`tables/validation_metrics.csv`), ghi N/A đúng nghĩa và quy tắc ties.
4. Bảng độ ổn định 3 seed (`tables/mlp_seeds.csv`); mean±population std, không gọi là khoảng tin cậy.
5. Đường validation MAE theo epoch có thể vẽ từ `tables/training_history_seed42.csv`; đánh dấu best epoch 20, dừng ở 35. Không đặt train MSE và val MAE trên cùng trục như cùng một đại lượng.
6. Bảng latency (`tables/cpu_latency.csv`) và ảnh giao diện thật (`figures/G5-real-model-20261026.png`).
7. Phân tích external20 và holdout: chỉ thêm khi đã có bằng chứng; hiện ghi chưa đánh giá.

## 4. Phân công đề xuất cho nhóm 3 người

| Vai trò | Phần phụ trách | Sản phẩm cần bàn giao |
|---|---|---|
| Thành viên dữ liệu | Chương 1, dữ liệu chương 3, nguồn ảnh external20 | Giải trình QA, bảng split, nguồn/nhãn hợp lệ |
| Thành viên ML | Chương 2, phương pháp chương 3, thực nghiệm chương 5 | Công thức, cấu hình, bảng metric và diễn giải |
| Thành viên ứng dụng | Chương 4, kiểm thử/hiệu năng, đóng gói tài liệu | Kiến trúc, ảnh UI, hướng dẫn và kịch bản demo |
| Cả nhóm | Tóm tắt, chương 6, rà soát chéo | Thống nhất thuật ngữ/số liệu, UAT và báo cáo cuối |

Đây là phân công đề xuất, chưa phải xác nhận công việc của từng thành viên.

## 5. Trình tự hoàn thiện và điều kiện nghiệm thu

| Thứ tự | Việc tiếp theo | Điều kiện hoàn thành |
|---|---|---|
| 1 | Viết bản nháp từ kết quả đã có; review các mốc G0–G5 còn thiếu xác nhận | Mỗi kết luận có nguồn; phân biệt kiểm kỹ thuật với nghiệm thu toàn mốc |
| 2 | Hoàn thiện external20 | Tên ở cột reviewer; score là số 1–5; chấm độc lập trước xem dự đoán; ghi nguồn/quyền sử dụng còn thiếu |
| 3 | Chạy model seed 42 cố định trên external20 sau khi chấm | Có bảng dự đoán/nhãn/sai số và ví dụ; không tuning bằng bộ này |
| 4 | UAT, máy sạch, đo RAM peak, hoàn thiện model card | Có log/cấu hình, người review và kết quả thật; không suy ra từ build PASS |
| 5 | Chốt cấu hình/model/release rồi đánh giá holdout một lần | 1.440 cặp test; ghi metric thật, giữ nguyên quyết định model sau khi xem test |
| 6 | Bổ sung tài liệu tham khảo, rà soát, xuất PDF/Word và demo | Đúng mẫu môn học, mọi bảng/hình có chú thích, không còn số liệu nháp không nhãn |

Không cần chạy lại QA/training Kaggle đã đạt chỉ để viết báo cáo. G6 (ranking loss, caption, VLM) là mở rộng, chỉ xem xét sau MVP. Hiện gói này chưa có tài liệu tham khảo học thuật đã xác minh; người viết cần kiểm nguồn gốc trước khi trích dẫn, không dùng đường dẫn repository như thay thế cho bài báo nền tảng.
