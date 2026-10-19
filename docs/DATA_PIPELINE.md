# G1 — pipeline dữ liệu local

## Trạng thái và nguồn

Adapter dùng các cột `Prompt`, `HumanRatings` và sáu cột ảnh theo [dataset card GenAI-Bench](https://huggingface.co/datasets/BaiqiL/GenAI-Bench/blob/main/README.md), đối chiếu ngày 2026-10-20. Ảnh Parquet được đọc từ struct `bytes/path`, chỉ sử dụng bytes nhúng; path ngoài không tự mở. Cột thiếu làm run FAIL; ảnh/nhãn lỗi được loại có lý do. Chưa đọc Parquet thật của toàn bộ dataset, nên tương thích dữ liệu thực vẫn chờ xác nhận.

## Chạy trong VS Code hoặc Kaggle

```powershell
.\.venv\Scripts\python.exe -X utf8 -m pip install -r requirements-data.in
.\.venv\Scripts\python.exe -X utf8 scripts/prepare_data.py --input data/raw --output data/manifests/run-001
```

`--input` chứa các shard local, có thể trong thư mục con. Lệnh không tải dữ liệu. Kaggle dùng cùng script bằng Python của môi trường đã chuẩn bị. Output phải là thư mục mới; không ghi đè run/split cũ.

- `manifest.jsonl`: source shard/row/model, prompt, ratings, mean/std, kích thước/checksum ảnh và split đề xuất. Không chứa ảnh nhị phân.
- `qa.json`: checksum/size nguồn, phiên bản Python/PyArrow/Pillow, Git commit và dirty flag, thống kê, mẫu loại, trùng prompt và ứng viên gần trùng.
- Exit 0 = CHECKS_PASS trong phạm vi file đã đọc; exit 2 = REVIEW_REQUIRED; exit 1 = FAIL (báo cáo lỗi được lưu nếu tạo được thư mục output).

Mặc định kỳ vọng 12 shard/1.600 hàng. Chỉ đổi `--expected-shards` và `--expected-rows` khi kiểm thử subset/fixture, không dùng để che dữ liệu thiếu. Batch đọc 8 hàng, không nạp toàn bộ bytes ảnh vào RAM. QA mở tất cả ảnh của các cặp có nhãn hợp lệ; lỗi nhãn được loại trước giải mã ảnh.

Split đề xuất 70/15/15 theo nhóm prompt chuẩn hóa; số lượng thực do làm tròn được báo cáo. Chuẩn hóa NFKC/casefold/khoảng trắng không giải quyết mọi tương đồng ngữ nghĩa. Near-duplicate dùng SequenceMatcher ngưỡng 0.95, là ứng viên cần người review; không tự gộp những prompt có thể khác nghĩa. Run có ứng viên gần trùng hoặc mẫu loại chưa đủ điều kiện khóa split. DATA full và review vẫn bắt buộc trước thực nghiệm chính thức.

## Kiểm thử không cần dataset

```powershell
.\.venv\Scripts\python.exe -X utf8 -m pytest tests/unit/test_data.py tests/integration/test_parquet_pipeline.py -q
```

Integration test tạo Parquet nhỏ trong thư mục tạm: hai shard, nhiều row group, prompt trùng xuyên shard, bytes hỏng, schema thiếu, đường dẫn ảnh ngoài và chạy CLI subprocess. Các fixture không phải kết quả GenAI-Bench.
