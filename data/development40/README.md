# 40 cặp bổ sung để phát triển

Mở rate.html, điền tên người chấm, xem ảnh và prompt, chọn điểm rồi tải CSV. Lưu CSV tải về vào thư mục này. Có thể xuất bản đang làm dở; điểm trống không được coi là nhãn. Trang không lưu tự động khi đóng: hãy tải CSV trước khi rời trang.

Ảnh lấy từ external20 đã thấy, prompt lấy từ mẫu khác theo offset cố định1 và7; đây là cặp ứng viên sai, không phải nhãn1 mặc định. Chấm theo nội dung thực tế. Mỗi người chấm độc lập, không xem điểm người khác/model. Có thể chấm pilot với một người trước, nhưng phải nêu hạn chế nếu chưa có nhiều người.

Bộ này dùng phát triển/huấn luyện bổ sung khi có nhãn thật, không dùng làm test độc lập. Khi dùng ảnh external20 để phát triển, bộ đó là tập đã thấy trong vòng cải thiện; báo cáo baseline cũ giữ nguyên lịch sử. Trước nghiệm thu model mới cần tập đánh giá mới chưa dùng điều chỉnh.
