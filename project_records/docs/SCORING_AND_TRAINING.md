# Các yếu tố chấm điểm và quá trình huấn luyện PromptAlign AI

Cập nhật: 2026-10-30. Web đã quay về MLP seed42, phiên bản `mlp-b8dba09efa0e`, theo yêu cầu người dùng. Ensemble cải tiến được lưu như thí nghiệm, hiện không phục vụ web.

## 1. Dự án đang chấm điều gì?

Chấm **mức độ phù hợp của ảnh với prompt**, trả một điểm tổng thể từ1đến5. Đây không phải điểm đẹp của ảnh, xác suất đúng, hoặc phần trăm chính xác. Ví dụ4/5 không có nghĩa80% nội dung đúng.

Hai phần cần phân biệt:

- **Rubric cho người chấm:** xem từng yêu cầu trong prompt, đối chiếu ảnh rồi cho một điểm tổng thể.
- **Mô hình trên web:** học dự đoán điểm tổng thể từ đặc trưng ảnh và văn bản. Hiện chưa có bộ kiểm đếm vật thể, OCR, nhận diện màu hay quan hệ không gian riêng để chấm từng tiêu chí.

Do đó, mô hình không thực hiện công thức “đúng màu+1, đúng số lượng+1”, không có trọng số công bố riêng cho từng tiêu chí, cũng không tự giải thích chắc chắn chi tiết nào làm mất điểm.

## 2. Các yếu tố dành cho người chấm

Chỉ xét yêu cầu thực sự có trong prompt. Không thêm tiêu chí sau khi xem ảnh để cố trừ điểm.

|Yếu tố|Nội dung cần đối chiếu|Ví dụ sai|
|---|---|---|
|Đối tượng chính|Đúng vật/người/con vật, có đủ thành phần quan trọng?|Yêu cầu xe đạp nhưng ảnh là ô tô|
|Số lượng|Đúng số lượng được chỉ định, nhất là “exactly”?|Yêu cầu2quả cam, ảnh có3|
|Thuộc tính|Màu sắc, chất liệu, trang phục, đặc điểm có đúng đối tượng?|Ấm trà vàng thành đỏ; màu vòng cổ bị gán nhầm con vật|
|Vị trí và quan hệ|Trái/phải, trên/dưới, trong/ngoài, ở giữa có đúng?|Cốc ở trái thay vì phải; chó nằm trên thay vì dưới ghế|
|Hành động và trạng thái|Đang làm gì, ngồi/nằm, đóng/mở…?|Yêu cầu ô đóng nhưng ảnh ô mở|
|Bối cảnh và cách thể hiện|Nền, địa điểm, góc chụp, phong cách nếu được yêu cầu?|Yêu cầu bãi biển nhưng ảnh trong nhà|
|Chi tiết đặc biệt và loại trừ|Chữ, giờ đồng hồ, yêu cầu “không có…”?|Biển không ghi BAKERY; đồng hồ sai3:00; có người dù prompt yêu cầu không có|

Nếu prompt quy định trái/phải theo người xem thì dùng góc nhìn người xem. Chi tiết bị che hoặc không đọc được phải ghi “không xác định được”; không tự giả định đúng. Hình ảnh do AI sinh vẫn có thể sai prompt.

## 3. Thang điểm tổng thể 1–5

Đây là hướng dẫn nhóm dùng chấm ảnh bổ sung; không khẳng định nó là nguyên văn rubric của nguồn GenAI-Bench.

|Điểm|Diễn giải|
|---|---|
|5 — Khớp đầy đủ|Đối tượng chính và tất cả yêu cầu rõ ràng đều được thể hiện, không thấy sai lệch đáng kể|
|4 — Khớp tốt|Nội dung chính đúng; một sai lệch nhỏ ở chi tiết phụ, không đổi ý nghĩa cảnh|
|3 — Khớp một phần|Đúng đối tượng/cảnh chính nhưng sai hoặc thiếu yêu cầu quan trọng, như số lượng chính xác, màu được chỉ định hoặc vị trí|
|2 — Khớp ít|Chỉ đúng một phần nhỏ; sai nhiều yêu cầu quan trọng hoặc thiếu thành phần chính|
|1 — Không khớp|Sai nội dung/đối tượng chính hoặc ảnh gần như không liên quan|

Không cộng/trừ cố định cho mỗi lỗi hoặc lấy trung bình cơ học các yếu tố. Xét tầm quan trọng của yêu cầu trong toàn prompt; một lỗi đối tượng chính nặng hơn một sai lệch nền nhỏ. Ghi lý do cụ thể cho điểm dưới5. Ví dụ: “Có3quả cam thay vì2”, không chỉ ghi “ảnh chưa tốt”.

Nếu tất cả ảnh thực sự đúng thì toàn điểm5 vẫn hợp lệ. Khi đó bộ ảnh không đủ kiểm khả năng phát hiện mismatch; có thể giữ ảnh và tạo prompt đối chứng sai một chi tiết. Phép kiểm đó đo thứ tự/độ nhạy điểm, không tự tạo nhãn người1–5.

## 4. Dự án đã train mô hình chưa?

**Có. Dự án đã huấn luyện bộ hồi quy chấm điểm Ridge và MLP trên GenAI-Bench.** CLIP được dùng dưới dạng pretrained và đóng băng; nhóm chưa huấn luyện CLIP từ đầu, chưa fine-tune CLIP, chưa huấn luyện chatbot hoặc mô hình sinh ảnh.

|Thành phần|Công cụ/cách làm|Trạng thái|
|---|---|---|
|CLIP ViT-B/32|Hugging Face Transformers + PyTorch, pretrained OpenAI|Chỉ trích đặc trưng, không cập nhật trọng số|
|Mean baseline|Trung bình nhãn train|Đối chứng đơn giản|
|Ridge regression|NumPy giải bài toán hồi quy tuyến tính L2; head lưu bằng PyTorch|Đã fit, dùng đối chứng|
|MLP regression|PyTorch, mạng1024→64→1, ReLU, đầu ra1+4×sigmoid|Đã train; seed42 đang dùng trên web|
|Ensemble thử nghiệm|Ba MLP hybrid, trung bình dự đoán|Đã train local và thử web; đã rollback theo yêu cầu|

### Dữ liệu huấn luyện

- Nguồn: `BaiqiL/GenAI-Bench`,12shard,1.600prompt,9.598cặp ảnh–prompt hợp lệ sau loại2cặp nhãn lỗi.
- Nhãn học là **trung bình3điểm người đánh giá từ dataset**, không phải điểm cosine hoặc điểm AI tự sinh.
- Train6.718cặp; validation1.440; test1.440. Chia theo nhóm prompt để các ảnh cùng prompt không nằm xuyên split.
- Scaler chỉ fit train; validation dùng chọn checkpoint/early stopping. Holdout gốc đã đánh giá một lần ngày27/10, không dùng lại để tối ưu.
- Các bộ20ảnh người dùng cung cấp được dùng đánh giá/chẩn đoán như ghi trong báo cáo; không phải nguồn huấn luyện chính ban đầu.

### Train bằng gì, ở đâu?

Luồng đã thực hiện: **viết code trong VS Code → đồng bộ GitHub → chạy script trên Kaggle → tải artifact về máy → suy luận bằng FastAPI/React local**.

1. Trên Kaggle, GPU Tesla T4 chạy CLIP để trích embeddings của9.598cặp. Môi trường được ghi nhận: Python3.13.15, PyTorch2.8.0+cu126.
2. Ridge/MLP được fit trên embeddings bằng **CPU trong phiên Kaggle**. Không nên ghi toàn bộ quá trình train head là GPU: GPU dùng ở bước trích đặc trưng.
3. Chạy3seed42/43/44. MLP dùng Adam, learning rate0,001, MSE loss; tối đa100epoch, dừng sau15epoch không cải thiện validation MAE, giữ checkpoint tốt nhất. Tối đa100không có nghĩa mọi run đều chạy đủ100epoch.
4. Tải training ZIP về máy và kiểm lại9head trên CPU. Head MLP seed42 được dùng cho web; web không phụ thuộc phiên Kaggle đang mở.
5. Vòng cải thiện ngày29/10 chạy **CPU local**, dùng embeddings có sẵn; thêm các cặp ghép sai trong train với nhãn1 giả định, ghi rõ weak/synthetic supervision. Ensemble25% có1.679cặp tổng hợp thêm vào6.718cặp train và3seed. Nhãn giả định không được xem là nhãn con người. Vòng này không fine-tune CLIP.

Notebook Kaggle chỉ gọi script, phần huấn luyện nằm trong source Python. VS Code là công cụ viết/chạy code, không phải thuật toán huấn luyện.

## 5. Điểm trên web được tính như thế nào?

```text
Ảnh → CLIP image encoder → vector512chiều ┐
                                         ├→ nối1024chiều → chuẩn hóa → MLP → điểm1–5
Prompt → CLIP text encoder → vector512chiều┘
```

Ngoài điểm MLP, web hiển thị CLIP cosine là độ tương đồng giữa hai vector. Cosine không phải điểm1–5. Khi so sánh nhiều ảnh, web sắp thứ hạng theo điểm MLP; đây không phải bằng chứng đã train ranking loss.

Mô hình học từ điểm tổng thể, không có nhãn riêng từng tiêu chí để bảo đảm hiểu được số lượng, vị trí hay đóng/mở. Do đó ảnh sai chi tiết vẫn có thể được điểm cao; đổi rubric trong tài liệu không tự làm mô hình thay đổi.

## 6. Bằng chứng và giới hạn hiện tại

- Baseline MLP42: validation MAE≈0,774; holdout MAE≈0,792. Đây là sai số trung bình trên thang1–5, không phải tỷ lệ chính xác.
- Ensemble thử nghiệm có cải thiện ở một số kiểm tra, nhưng còn lỗi đếm và trạng thái. Web hiện đã quay về baseline theo yêu cầu; không nhầm hai phiên bản khi viết báo cáo.
- Bảng `ratings_simulated_3_students.csv` của bộv2 được người dùng xác nhận là AI/mô phỏng. Không được báo cáo là ba người chấm dù có tên Hoan/Huy/An. Nhận xét sau đó “toàn bộ ảnh ổn” không tự biến bảng mô phỏng thành60đánh giá độc lập.
- Chưa nghiệm thu release; kiểm thử API/web đạt không đồng nghĩa mọi dự đoán đều đúng.

### File để đối chiếu

- [Huấn luyện Kaggle](../../code/scripts/kaggle_train.py), [notebook điều phối](../../code/notebooks/02_train.ipynb).
- [Code fit Ridge/MLP](../../code/src/training/train_regression.py), [kiến trúc head](../../code/src/models/regression_head.py).
- [Biên bản kiểm training](../reports/results/G3_full_training_review.md), [số liệu replay](../reports/results/G3_full_training_review.json).
- [Kết quả holdout](../reports/results/G7_holdout_seed42_20261027.md).
- [Vòng cải thiện](../reports/results/G7_improvement_round2_20261029.md).
- [Kiểm đổi prompt v2](../reports/results/G7_v2_prompt_changes_20261030/README.md), [bằng chứng rollback](../reports/results/G7_baseline_rollback_20261030.json).
- Checkpoint đang dùng: `.cache/reviewed-training-20261020T174855207187Z/training-seed42/mlp.pt`.

Tài liệu này giải thích rubric và thực trạng triển khai, không thay thế PROJECT_PLAN.md hoặc TEST_PLAN.md.
