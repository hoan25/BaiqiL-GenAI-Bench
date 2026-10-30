# Nguồn và giấy phép — kiểm tra 2026-10-27

| Thành phần | Nguồn chính thức đã đọc | Kết quả / phạm vi |
|---|---|---|
| GenAI-Bench | https://huggingface.co/datasets/BaiqiL/GenAI-Bench | Metadata và mục Licensing Information ghi Apache-2.0. Đọc trang hiện tại; URL README theo revision dataset bị công cụ web từ chối truy cập, chưa xác minh nội dung giấy phép ở revision đã dùng. |
| CLIP | https://github.com/openai/CLIP/blob/main/LICENSE | Repository ghi MIT, copyright OpenAI 2021. Không thay thế việc kiểm gói trọng số cụ thể khi phân phối. |
| Model card CLIP đúng revision | https://huggingface.co/openai/clip-vit-base-patch32/raw/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268/README.md | Đọc được. Mô hình phục vụ nghiên cứu; ngôn ngữ ngoài tiếng Anh chưa được đánh giá có chủ đích; hạn chế đếm và phân biệt chi tiết. Đây là phạm vi sử dụng/giới hạn, không phải tên giấy phép. |
| External20 | Người dùng xác nhận https://muse.ai/, giữ prompt tạo ảnh | https://muse.ai/terms yêu cầu đăng nhập, công cụ web chỉ đọc được “Not Logged In”. Chưa xác minh điều khoản output/version hoặc license_or_permission; không dùng điều khoản Connector thay cho quyền ảnh của người dùng. |

Hiện chỉ bàn giao source, tài liệu và metric; dataset/CLIP/ảnh external20 không đưa lên Git. Giữ attribution và nguồn khi viết báo cáo. Nhãn license của dataset không phải kết luận bao quát mọi quyền liên quan đến nội dung ảnh. Chưa cấp một giấy phép mới cho toàn dự án thay nhóm tác giả.

Trích dẫn công trình: Baiqi Li và cộng sự (2024), *GenAI-Bench: Evaluating and Improving Compositional Text-to-Visual Generation*, arXiv:2406.13743. CLIP: Radford và cộng sự (2021), *Learning Transferable Visual Models From Natural Language Supervision* (xem liên kết trong model card nguồn).
