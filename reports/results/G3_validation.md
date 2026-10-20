# G3-local — huấn luyện và checkpoint

- Ngày: 2026-10-20; thực hiện: Codex; review: chưa có.
- Môi trường: Python 3.13.5, torch 2.8.0+cpu, NumPy 2.5.4; không đổi dependencies.
- Unit tests ban đầu: **9 PASS**, 9.63 giây. Kiểm Ridge tham chiếu, scaler train-only, MLP overfit 32 mẫu, early stopping/best epoch, seed, reload/range, input lỗi và test holdout không được đọc.
- Bộ unit + CLI integration: **10 PASS, 0 FAIL, 0 SKIP**, 22.41 giây, exit 0; `G3_local_tests.txt`.
- CLI thực tế: `scripts/train_regression.py --manifest .cache/g2-pipeline-fixture/qa/manifest.jsonl --extraction reports/results/G2_fixture_extraction --output reports/results/G3_fixture_training --epochs 30 --patience 5 --development`: **PASS**, exit 0.
- Mean/Ridge/MLP đều lưu và đọc lại thành công, max delta = 0. MLP best epoch 3, chạy 8 epoch. Mean được chọn vì MAE validation thấp nhất (tie với Ridge); nhãn fixture đều bằng 2 nên không phải kết quả khoa học. Bằng chứng: `G3_fixture_training/result.json`; checkpoint được ignore khỏi Git.
- G3-local đã có kiểm chứng code. Toàn mốc **NOT ACCEPTED**: chưa train/full-data validation thật, chưa review chéo. G0 ENV-06, G1-full và provenance thực nghiệm vẫn chờ tích hợp GPU.
