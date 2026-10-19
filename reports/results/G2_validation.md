# G2-local — kiểm chứng code

- Ngày: 2026-10-20; thực hiện: Codex; review: chưa có.
- Môi trường: `.venv`, torch 2.8.0+cpu, transformers 4.57.6; không thêm dependency.
- `python -m pytest tests/unit/test_metrics.py tests/integration/test_embeddings.py -q`: **13 PASS**, exit 0; log `G2_local_tests.txt`.
- `scripts/check_encoder.py` với CLIP cache G0: **PASS**, output `G2_encoder_checks.json`; shape [2,512], max batch/single delta 1.7881393432617188e-07, frozen state không thay đổi, long prompt truncation đạt.
- `scripts/extract_embeddings.py` với manifest fixture `.cache/g2-pipeline-fixture/qa/manifest.jsonl`, batch 4, CPU, `--development`: **PASS**, 18 vector pairs chiều 512; `G2_fixture_extraction/result.json`. Đây là thực thi model thật trên dữ liệu tổng hợp, không phải benchmark dataset thật.
- G2-local checks PASS; full-data M0/M1 và GPU extraction mới **NOT RUN**. Chưa nghiệm thu toàn mốc; thiếu review, QA/split chính thức và provenance Git/config/tokenizer đầy đủ cho thực nghiệm.
