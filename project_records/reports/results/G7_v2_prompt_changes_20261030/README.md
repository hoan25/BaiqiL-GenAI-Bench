# Kiểm độ nhạy khi đổi prompt — 2026-10-30

User cho phép chạy thử với20ảnh v2 hiện có. Giữ nguyên model/ảnh/prompt gốc; tạo mỗi ảnh một prompt đổi đúng một chi tiết và một prompt không liên quan lấy từ ảnh khác (offset7). Tổng60cặp inference. Không dùng CSV điểm mô phỏng hoặc tự gán điểm1–5.

Protocol ghi trước inference trong protocol.json: danh sách prompt/ảnh/checksum, ensemble25% trung bình3seed đã chốt29/10, baselineMLP42, tolerance1e-5. Kỳ vọng prompt gốc có điểm cao hơn prompt sửa dựa trên người dùng cho biết ảnh khớp prompt; chưa phải nhãn ưu tiên độc lập được review đầy đủ. Đây là kiểm độ nhạy đổi prompt trên cùng ảnh, không phải metric ranking nhiều ảnh cùng prompt của GenAI-Bench.

| Phép kiểm | Baseline | Ensemble25% | CLIP cosine |
|---|---:|---:|---:|
|Điểm gốc cao hơn khi đổi1chi tiết|12/20|17/20|17/20|
|Điểm gốc cao hơn prompt không liên quan|10/20|20/20|20/20|
|Giảm ít nhất0,25điểm khi đổi1chi tiết|0/20|9/20|Không áp dụng|
|Mức giảm trung bình khi đổi1chi tiết|0,026588|0,410626|0,020389 raw cosine|
|Mức giảm trung bình khi đổi nội dung|0,102211|2,427631|0,187150 raw cosine|

Không diễn giải17/20 thành85% chính xác tổng quát. Tín hiệu cosine cũng đúng17/20, nên không tuyên bố ensemble cải thiện khả năng phân biệt chi tiết hơn encoder từ phép thử này. Chỉ9/20 ca sửa chi tiết giảm ít nhất0,25điểm; hạ điểm rất nhỏ chưa đồng nghĩa phát hiện lỗi tốt.

## Ba lỗi còn lại của ensemble

Đã xem trực tiếp ảnh05,08,15 sau inference để kiểm nội dung. Ảnh05 có2quả cam,08có5vịt,15ô đang đóng; prompt sửa tương ứng3quả cam/3vịt/ô mở. Mô hình vẫn chấm prompt sửa cao hơn:

|Ảnh|Gốc|Sửa sai|
|---|---:|---:|
|05|3,629121|3,819320|
|08|3,300347|3,608043|
|15|4,268610|4,310899|

Không chỉnh ảnh, nhãn hoặc prompt để che ba lỗi này. Không fit lại head, thay threshold theo kết quả hoặc đánh giá lại holdout. Điểm của từng ca trong scores.csv và result.json.

## Thực thi và trạng thái

`scripts/check_v2_prompt_changes.py` bằng .venv Python, HF offline, OMP/MKL4: exit0. Kiểm cấu trúc20phép thay duy nhất PASS; checksum3checkpoint trước load và20ảnh trước/sau inference khớp. CLIP/head thật; không mock.

Execution PASS; chất lượng cải thiện so baseline ở phép kiểm này nhưng còn lỗi đếm và trạng thái. Chưa thay model web hoặc ký nghiệm thu G7. Chưa có bộ nhãn người1–5 xác minh để đo MAE độc lập; không lấy điểm AI làm nhãn thật. Bộ v2 nay đã có dự đoán quan sát, phải công khai nếu dùng tiếp để điều chỉnh mô hình.
