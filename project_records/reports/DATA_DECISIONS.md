# Quyết định xử lý dữ liệu dùng trong báo cáo

Ngày xác nhận: **2026-10-21**. Người dùng đồng ý giữ các quyết định dưới đây sau khi yêu cầu giải thích để viết báo cáo. Lock trong run training ghi reviewer `hoan`. Đây là xác nhận quyết định dữ liệu, không thay thế review chéo toàn bộ code hoặc nghiệm thu sản phẩm.

Nguồn: GenAI-Bench revision `94e543b0801feade1419873335191de90b68d164`; 12 shard, 1.600 prompt, 9.600 cặp dự kiến. Manifest giữ nguyên SHA256 `4ed1e3c7909c81c239beaa2890f4adbcaef7545217e1d309a8b023e084182ca2`.

## 1. Loại hai cặp có nhãn không hợp lệ

| Nguồn (row đánh số từ 0) | Quyết định |
|---|---|
| train-00001-of-00012.parquet, row 97, DALLE_3 | Loại cặp ảnh–prompt này |
| train-00001-of-00012.parquet, row 118, DeepFloyd_I_XL_v1 | Loại cặp ảnh–prompt này |

Validator yêu cầu ba rating nguyên trong [1,5]; hai cặp không đáp ứng điều kiện. ZIP QA chỉ lưu lý do lỗi, không lưu giá trị rating gốc bị loại, nên không kết luận giá trị sai cụ thể. Không sửa, clamp hoặc điền nhãn vì chưa có căn cứ để khôi phục đánh giá của con người; làm vậy có thể đưa mục tiêu do nhóm tự tạo vào thực nghiệm.

Chỉ loại hai cặp lỗi, giữ các ảnh/nhãn hợp lệ khác của hai prompt. Giảm 2/9.600 cặp (~0,0208%), còn **9.598 cặp**; các prompt vẫn được chia theo nhóm nên không tạo rò rỉ split. Hai model tương ứng còn 1.599 cặp, bốn model còn lại mỗi model 1.600. Hai cặp bị loại thuộc train; train/val/test = 6.718/1.440/1.440.

## 2. Giữ hai prompt “puppy's head” và “puppy's paw”

Hai câu giống gần hết về từ vựng nhưng yêu cầu người trong ảnh đặt tay lên **đầu** hoặc **chân** của chó con. Đó là khác biệt nội dung có ý nghĩa cho bài toán text–image alignment. Độ giống chuỗi 0,96 chỉ dùng để phát hiện ứng viên review, không đủ để kết luận hai yêu cầu tương đương.

Giữ cả hai nhằm bảo toàn sự khác biệt quan hệ/vị trí cần học. Cả hai thuộc **train**, không có ứng viên này xuyên train/validation/test. Không tuyên bố CLIP hoặc model hiện tại đã phân biệt tốt chi tiết này chỉ từ quyết định giữ dữ liệu.

## 3. Giữ hai prompt bàn làm việc chỉ khác dấu phẩy trong train

Hai câu tương đương nội dung, khác dấu phẩy trước “while”; similarity từ vựng ~0,9956. Hai nhóm đã cùng **train** và manifest không có ảnh trùng checksum. Giữ bản ghi nguồn và split đã chốt để bảo toàn ảnh cùng đánh giá của con người, thay vì tùy ý bỏ một bộ ảnh hoặc chia lại sau khi xem kết quả.

**Giới hạn phải báo cáo:** hai nhóm câu không phải hai ý tưởng độc lập; giữ chúng có thể tăng trọng số tương đối của nội dung này trong train. Không gọi toàn dataset là “không có trùng ngữ nghĩa”. Không có rò rỉ xuyên split từ cặp này theo manifest đã kiểm. Nếu có phiên bản dữ liệu/split mới trong nghiên cứu khác, cần gộp hai câu vào cùng nhóm trước khi chia; không áp dụng chia lại trong thực nghiệm hiện tại.

## Đoạn có thể dùng trong báo cáo

“Sau kiểm tra 12 shard của GenAI-Bench, nhóm giữ 9.598 trong 9.600 cặp ảnh–mô tả và loại hai cặp không đáp ứng điều kiện nhãn gồm ba điểm nguyên trong khoảng 1–5. Nhóm không nội suy hoặc sửa nhãn khi thiếu căn cứ từ đánh giá gốc. Kiểm tra gần trùng phát hiện hai cặp mô tả cùng nằm trong tập huấn luyện: cặp thay đổi vị trí ‘đầu/chân chó con’ được giữ vì khác nghĩa; cặp chỉ khác dấu phẩy được giữ cùng các bản ghi ảnh và nhãn nguồn, với giới hạn rằng nội dung này có thể được tăng trọng số trong train. Split được giữ cố định; không phát hiện prompt chuẩn hóa hoặc ảnh trùng checksum xuyên split. Kiểm tra gần trùng dựa trên từ vựng không chứng minh đã loại mọi tương đồng ngữ nghĩa.”

Các quyết định đã được đề xuất từ QA trước training; người dùng xác nhận lại trong phiên nhận ZIP training. Không dựa trên test holdout để chọn cách xử lý.
