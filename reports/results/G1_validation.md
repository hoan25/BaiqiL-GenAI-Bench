# G1-local — bước đầu pipeline dữ liệu

- Ngày: 2026-10-20; người thực hiện: Codex; review: chưa có.
- Môi trường: CPython 3.13.5, `.venv` đã có Pillow/pytest; không đổi dependency.
- Contract chuẩn hóa, validation nhãn/ảnh, manifest và group split: **13 test PASS** (`G1_local_tests.txt`).
- CLI: **PASS** trên 10 hàng/60 cặp tổng hợp (`G1_fixture_smoke.json`); seed 42, split 42/6/12 cặp. Không train model.
- Mapping sáu model, nhãn không hợp lệ, ảnh hỏng, nguồn trùng, Unicode prompt, split không rò rỉ và không phụ thuộc thứ tự hàng đã có test.
- G1-local: **IN PROGRESS**. Còn adapter Parquet, integration test file, QA report/near-duplicate, provenance và review.
- G1-full: **NOT RUN**. Chưa đọc toàn bộ 12 shard; chưa có manifest/split chính thức. Không dùng các con số fixture làm metric thực nghiệm.
