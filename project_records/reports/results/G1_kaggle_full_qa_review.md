# Review kết quả QA Kaggle — 2026-10-21

- Source commit: `7b621d131c90e0d0055398096955f188ff8bbd95`; working tree trên Kaggle sạch.
- ZIP người dùng lưu trong `G3_fixture_training/data-qa-20261020T165712283760Z.zip`. Đây là **QA dữ liệu thật**, không phải training G3 hoặc checkpoint mới. Giữ nguyên các file fixture G3 cũ.
- CRC ZIP PASS. Báo cáo chi tiết, hash, log và kiểm tra manifest: [G1_kaggle_full_qa_review.json](G1_kaggle_full_qa_review.json).
- Dataset revision: `94e543b0801feade1419873335191de90b68d164`.
- Manifest SHA256: `4ed1e3c7909c81c239beaa2890f4adbcaef7545217e1d309a8b023e084182ca2`.

## Kết quả thực tế

| Kiểm tra | Kết quả |
|---|---|
| Bootstrap, dependency imports, pip check | PASS trên Kaggle, không còn lỗi sitecustomize |
| Test dữ liệu trên Kaggle | 18 PASS, 0.66 giây |
| Đọc dữ liệu | 12 shard, 1.600 prompt, 9.600 cặp dự kiến |
| Giữ / loại | 9.598 / 2 |
| Train / validation / test | 6.718 / 1.440 / 1.440 cặp |
| Nhóm prompt train / validation / test | 1.120 / 240 / 240 |
| Checksum manifest, ID, mapping, ratings/mean/std | PASS khi kiểm độc lập local |
| Split tái lập khi đảo thứ tự manifest | PASS, seed 42 |
| Prompt chuẩn hóa xuyên split | 0 |
| Nhóm ảnh trùng checksum trong manifest | 0 |
| Prompt gần trùng ở ngưỡng từ vựng 0.95 | 2 cặp, cả hai cùng train |

## Đề xuất xử lý để review và khóa split

1. Giữ loại `train-00001-of-00012.parquet`, row 97, `DALLE_3` và row 118, `DeepFloyd_I_XL_v1`. Validator báo rating không phải số nguyên trong [1,5]. ZIP không chứa giá trị gốc của hai mẫu này nên không suy đoán giá trị, sửa nhãn hay điền thiếu. Tổng thiếu 2 cặp được đối chiếu đúng với manifest; mỗi model tương ứng còn 1.599 cặp.
2. Giữ hai prompt “hand on a puppy's head” và “hand on a puppy's paw”: khác vị trí cơ thể cần đánh giá, cùng train.
3. Hai prompt về bàn làm việc chỉ khác dấu phẩy: tương đương về nội dung, cùng train. Giữ các bản ghi nguồn trong split hiện tại; nếu đổi split sau này phải nhóm chúng cùng nhau. Không có bằng chứng rò rỉ qua validation/test.
4. Giữ nguyên manifest checksum ở trên làm ứng viên khóa split. Không chia lại để thay đổi kết quả và không đánh giá holdout.

**Trạng thái:** kiểm chứng kỹ thuật artifact PASS; review chéo và phê duyệt split chính thức còn chờ. Đây không phải nghiệm thu toàn bộ G0/G1 hay kết quả chất lượng model. Raw Parquet/ảnh không có trong ZIP nên local không kiểm lại byte ảnh/checksum shard; các kiểm tra này dựa trên log QA Kaggle.

## Lỗi tải 404

Notebook có HTML `href='/kaggle/working/data-qa-....zip'`: đường dẫn filesystem bị dùng làm URL của trình duyệt. File ZIP thực vẫn hợp lệ. Đã bỏ FileLink tuyệt đối trong notebook source, in tên/kích thước ZIP và hướng dẫn tải qua Output / Files. Cú pháp hai cell compile PASS; UI Kaggle bản sửa chưa chạy lại. Không cần chạy lại QA chỉ để kiểm thay đổi hiển thị này.

## Bước kế tiếp

Review các quyết định dữ liệu ở trên, khóa manifest theo checksum, rồi nối runner GPU extraction/training với manifest đã khóa. Giữ dataset trên Kaggle nếu phiên còn mở; không cần chạy lại bootstrap G0, QA hoặc tải lại dữ liệu chỉ vì liên kết 404.
