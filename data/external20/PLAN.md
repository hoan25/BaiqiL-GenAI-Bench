# Kế hoạch thu thập 20 ảnh mới và prompt để phân tích lỗi

## Mục tiêu và phạm vi

Tạo **20 ảnh khác nhau**, mỗi ảnh có một prompt đánh giá, nằm ngoài GenAI-Bench. Đây là bộ kiểm tra thăm dò ngoài dataset để tìm kiểu lỗi, không thay cho test holdout và không đủ để chứng minh chất lượng tổng quát.

Không dùng bộ này để huấn luyện/chỉnh hyperparameter rồi tiếp tục gọi nó là bộ kiểm tra độc lập. Nếu sửa model dựa trên các lỗi tìm được, cần một bộ ảnh mới để kiểm chứng phiên bản sau.

## Cấu trúc

```text
data/external20/
  PLAN.md            # hướng dẫn
  samples.csv        # 20 kế hoạch ảnh và prompt, nguồn còn trống
  ratings.csv        # 3 lượt chấm độc lập/mẫu, điểm còn trống
  images/            # đặt 01.png ... 20.png tại đây
  results/           # agent ghi dự đoán, metric và phân tích sau
```

JPG/PNG/WebP đều được; nếu đổi đuôi ảnh, sửa `image_path` trong samples.csv. Không chỉ đổi tên .jpg thành .png. Dùng ảnh vừa phải (ví dụ 512–1024 px), tối đa 10 MB/ảnh. Không dùng ảnh chứa thông tin cá nhân nhạy cảm.

## Cách lấy ảnh — không bắt buộc mua công cụ

**Cách thuận tiện nếu bạn có công cụ tạo ảnh:** dùng cột `creation_prompt_en` để tạo từng ảnh. Lưu nguyên ảnh trả về, không chọn lại chỉ vì model dự đoán điểm thấp. Nếu một lần tạo bị lỗi kỹ thuật, ghi số lần tạo lại trong `notes`. Công cụ tạo có thể hiểu sai yêu cầu; phải chấm theo ảnh thực tế, không theo câu lệnh đã gửi.

**Nếu không có công cụ tạo ảnh:** tự chụp các cảnh đơn giản trong kế hoạch hoặc chọn ảnh bạn đã tự chụp, chưa dùng trong dự án. Có thể đổi cảnh khó tạo sang cảnh bạn thực hiện được, nhưng sửa cả hai prompt và ghi lý do trước khi xem điểm model. Ghi source_type=self_photo; các ảnh thật này là kiểm tra dịch chuyển miền so với ảnh AI trong dữ liệu huấn luyện.

**Nếu lấy từ Internet:** chỉ dùng nguồn có quyền sử dụng rõ ràng; lưu URL trang gốc, tác giả và giấy phép của từng ảnh. Không mặc định ảnh từ Google/Pinterest được phép dùng. Nếu không biết prompt sinh gốc, để creation_prompt_en trống và tự viết evaluation_prompt_en mô tả ảnh; không ghi prompt tự viết là prompt gốc. Không lấy lại GenAI-Bench hoặc ảnh minh họa đã dùng để kiểm model trước đó.

Không cần có prompt gốc mới đánh giá được độ phù hợp: input là ảnh và **prompt đánh giá** do nhóm đặt. Phải phân biệt prompt đánh giá với prompt đã dùng tạo ảnh.

## 20 ca được chia thế nào?

| Nhóm | ID | Cần quan sát |
|---|---|---|
| Cảnh đơn giản | 01–04 | Vật thể, màu, nền dễ nhận biết |
| Số lượng và thuộc tính | 05–08 | Đếm, màu đối tượng; có ca đổi số/màu trong prompt |
| Quan hệ và hành động | 09–12 | Trái/phải, trên/dưới, tương tác; có ca đảo quan hệ |
| Chi tiết và phủ định | 13–16 | Có/không có vật thể, chữ trong ảnh |
| Tình huống khó | 17–20 | Phong cách, vật nhỏ, nhiều thuộc tính, vật thể không đúng |

Cột `planned_relation` chỉ là **ý định thiết kế** (match/mismatch), không phải nhãn thật và không phải yêu cầu model phải cho một mức điểm cụ thể. Trong 20 kế hoạch có 6 ca mismatch có chủ đích; việc ảnh thực tế có đúng thiết kế hay không phải được xác nhận khi xem ảnh. Các cảnh khó không được tự coi là ảnh lỗi.

## Quy trình thực hiện

1. Mở samples.csv bằng Excel/VS Code. Đọc cả hai prompt trước khi tạo/chọn ảnh. `creation_prompt_en` dùng tạo ảnh; **evaluation_prompt_en mới là prompt gửi vào web chấm điểm**.
2. Lưu 20 ảnh vào images/, đúng tên file của mỗi dòng. Điền nguồn: source_type, source_name, source_url_or_reference, license_or_permission, collected_date; thêm ghi chú nếu thay đổi kế hoạch. Không đổi tên/ghi đè ảnh sau khi đã chấm.
3. Trước khi xem điểm model, 3 thành viên xem độc lập từng ảnh cùng evaluation_prompt_en, điền ratings.csv. Không xem prompt tạo ảnh hoặc điểm của nhau để suy ra nhãn. Reviewer A/B/C cần được thay bằng tên/mã người chấm thật. Không điền điểm giả nếu thiếu người; nếu chỉ có một người, ghi giới hạn đó trong báo cáo.
4. Khi đủ ảnh/prompt và đánh giá, báo agent thư mục đã hoàn tất. Agent kiểm file, checksum, ảnh trùng, trường trống và provenance; giữ cố định model MLP seed 42 hiện tại rồi chạy một lần trên bộ này. Agent không sửa nhãn để khớp dự đoán.
5. Phân tích theo mẫu và theo nhóm: điểm người chấm trung bình/độ lệch, điểm model, sai số tuyệt đối; xem các ca sai nhiều và nhận xét cụ thể. Không gọi score 4/5 là 80% chính xác.

Nếu muốn xem web trước khi chấm, chỉ dùng ảnh thử khác ngoài 20 ảnh này để tránh bị ảnh hưởng bởi dự đoán.

## Thang đánh giá con người 1–5

- **1:** Nội dung chính không phù hợp hoặc mâu thuẫn rõ với prompt.
- **2:** Chỉ khớp một phần nhỏ; sai vật thể/quan hệ quan trọng.
- **3:** Khớp phần chính nhưng thiếu/sai chi tiết đáng kể.
- **4:** Phù hợp hầu hết yêu cầu; còn sai khác nhỏ.
- **5:** Phù hợp đầy đủ các yêu cầu quan sát được của prompt.

Chấm **độ phù hợp ảnh–prompt**, không chấm vẻ đẹp riêng. Nếu không nhìn rõ số lượng/chữ/vật nhỏ, ghi sự không chắc chắn trong comment. Không ép các mức điểm xuất hiện với tần suất bằng nhau.

## Báo cáo đầu ra dự kiến

Sau khi có dữ liệu thật: bảng 20 mẫu, MAE/RMSE và Spearman nếu xác định được; phân tích lỗi định tính theo nhóm. N=20 nhỏ nên không kết luận mạnh về khác biệt model. Mỗi ảnh có prompt khác nhau nên không tính pairwise ranking giữa các ảnh khác prompt. Nếu muốn kiểm ranking, tạo thêm một bộ nhiều ảnh cho cùng prompt, tách khỏi bộ này.

Các file CSV hiện là **kế hoạch chưa thu thập/chưa chấm**, không phải kết quả thực nghiệm. Ảnh và results được Git ignore; source và template có thể version hóa, nhưng kiểm nội dung CSV trước khi push nếu có thông tin cá nhân.
