# Quy trình tiếp tục PromptAlign AI

- Trước mỗi phiên làm việc, đặc biệt sau khi hết quota hoặc mất ngữ cảnh, đọc đầy đủ `PROJECT_PLAN.md`, `TEST_PLAN.md` và `docs/PROGRESS.md`.
- Kiểm tra trạng thái thực tế: danh sách file, Git status/diff nếu có Git, mã nguồn, cấu hình, artifact và log test liên quan. Nhật ký là chỉ dẫn, không thay thế bằng chứng.
- Tiếp tục từ nhiệm vụ chưa nghiệm thu đầu tiên theo G0 → G7 và thứ tự phụ thuộc. Không làm lại phần đã hoàn thành và có kiểm thử đạt nếu không có thay đổi hoặc bằng chứng hồi quy.
- Chỉ chạy lại kiểm thử liên quan khi code, dependency, môi trường thay đổi hoặc bằng chứng cũ thiếu/không còn phù hợp.
- Theo điều chỉnh người dùng ngày 2026-10-20: phát triển code local theo thứ tự phụ thuộc, tách cổng kiểm thử local khỏi nghiệm thu thực nghiệm. ENV-06 Kaggle chờ tích hợp GPU, không chặn G1-local. Không chạy thực nghiệm chính thức trước full-data QA và môi trường sạch; không đánh dấu toàn mốc DONE chỉ từ fixture hoặc khi thiếu review. Xem lộ trình đầu PROJECT_PLAN.md.
- Sau mỗi nhóm công việc, cập nhật `docs/PROGRESS.md`: file thay đổi, lệnh chạy, kết quả thực tế, đường dẫn bằng chứng, vấn đề tồn tại và nhiệm vụ tiếp theo cụ thể.
- Ghi PASS chỉ khi thực sự kiểm tra thành công; dùng FAIL, BLOCKED hoặc NOT RUN với lý do rõ ràng. Không tạo metric giả hoặc coi fixture là thực nghiệm thật.
- Nếu sắp dừng phiên, ghi checkpoint sớm; không dựa vào khả năng ghi thêm sau khi quota đã hết. Khi tiếp tục, đối chiếu cả thay đổi chưa ghi nhật ký.
- Không tự commit/push, tải toàn bộ dataset hoặc checkpoint lớn. Không ghi đè dữ liệu/checkpoint có sẵn.
- Ngoại lệ đã được người dùng cho phép ngày 2026-10-19: tải CLIP để tiếp tục G0. Lưu checkpoint trong `.cache/` được Git ignore và ghi revision/checksum; không hỏi lại quyền tải CLIP đã cấp.
- Người dùng đã yêu cầu cập nhật code lên GitHub theo luồng VS Code → GitHub → Kaggle → VS Code. Commit/push source phục vụ yêu cầu này đã được cho phép; phải dùng đúng repository người dùng chỉ định, không push checkpoint, dataset, môi trường hoặc token.
- Hai file kế hoạch ở thư mục gốc là nguồn tham chiếu; không tạo bản sao trong docs dễ lệch nội dung.
