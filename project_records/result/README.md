# Bộ tài liệu phục vụ báo cáo PromptAlign AI

Ngày tổng hợp: **27/10/2026**. Đây là bản tổng hợp bằng chứng hiện có, chưa phải biên bản nghiệm thu release.

- [PLAN_BAO_CAO.md](PLAN_BAO_CAO.md): dàn ý, phân công đề xuất và trình tự hoàn thiện báo cáo.
- [TONG_HOP_KET_QUA.md](TONG_HOP_KET_QUA.md): nội dung tổng hợp, phương pháp, kết quả và giới hạn.
- `tables/`: bảng CSV số liệu validation, ba seed, split và latency để nhập Word/Excel.
- `evidence/`: bản chụp các biên bản/JSON/log nhỏ và cấu hình training; giữ nội dung nguồn.
- [Giao diện dùng model thật](figures/G5-real-model-20261026.png): ảnh E2E, dùng ảnh tổng hợp để kiểm luồng.
- [SOURCES.json](SOURCES.json): đường dẫn gốc và SHA256 của bằng chứng đã đóng gói.
- [CHECKS.json](CHECKS.json): kết quả kiểm tra gói báo cáo.

Các bảng chính lấy từ CPU replay trong `G3_full_training_review.json`; cấu hình/lịch sử epoch lấy từ artifact training seed 42. Sai khác chữ số cuối giữa training và replay là sai số số học, không trộn hai nguồn vào cùng bảng.

Kế hoạch triển khai và kiểm thử gốc vẫn là [PROJECT_PLAN.md](../PROJECT_PLAN.md), [TEST_PLAN.md](../TEST_PLAN.md); tiến độ tại [docs/PROGRESS.md](../docs/PROGRESS.md). Không sao chép hai kế hoạch gốc vào đây. Các đường dẫn ra ngoài `result` cần repository đi kèm; bằng chứng chính và bảng kết quả đã được đóng gói trong thư mục này. Không kèm dataset, embeddings hoặc trọng số lớn.

Khi bổ sung kết quả mới, cập nhật bản tổng hợp và bảng tương ứng từ artifact thực, ghi nguồn/hash mới; không tự đổi trạng thái holdout/UAT thành PASS. Các báo cáo lịch sử có thể ghi trạng thái chờ đã được giải quyết về sau; đọc cùng bản tổng hợp ngày 27/10/2026.
