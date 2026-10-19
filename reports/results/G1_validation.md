# G1-local — bước đầu pipeline dữ liệu

- Ngày: 2026-10-20; người thực hiện: Codex; review: chưa có.
- Môi trường: CPython 3.13.5, `.venv` đã có Pillow/pytest; không đổi dependency.
- Contract chuẩn hóa, validation nhãn/ảnh, manifest và group split: **13 test PASS** (`G1_local_tests.txt`).
- CLI: **PASS** trên 10 hàng/60 cặp tổng hợp (`G1_fixture_smoke.json`); seed 42, split 42/6/12 cặp. Không train model.
- Mapping sáu model, nhãn không hợp lệ, ảnh hỏng, nguồn trùng, Unicode prompt, split không rò rỉ và không phụ thuộc thứ tự hàng đã có test.
- Cập nhật: adapter Parquet/QA/provenance và CLI đã triển khai; integration test bằng Parquet tổng hợp PASS. Toàn suite **34 PASS, 0 FAIL, 0 SKIP** (`G1_parquet_tests.txt`); `pip check` PASS sau cài PyArrow 21.0.0.
- G1-local: **LOCAL CHECKS PASS**, review chưa có. Near-duplicate là ứng viên cần review trước khi khóa split; chưa nghiệm thu toàn mốc.
- G1-full: **NOT RUN**. Chưa đọc toàn bộ 12 shard; chưa có manifest/split chính thức. Không dùng các con số fixture làm metric thực nghiệm.
