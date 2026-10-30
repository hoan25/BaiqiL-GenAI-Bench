# G2-local — embeddings và baseline

Code tái sử dụng nằm trong `src/models/clip_encoder.py`, `src/training/extract_embeddings.py` và `src/evaluation/`. Không cần notebook để phát triển hoặc kiểm thử.

## Chạy extraction phát triển

Sau khi tạo candidate manifest theo `docs/DATA_PIPELINE.md`, dùng Python của môi trường G0 + `requirements-data.in`:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/extract_embeddings.py --manifest data/manifests/run-001/manifest.jsonl --data-root data/raw --model .cache/models/clip-vit-base-patch32/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268 --output artifacts/embeddings-dev-001 --batch-size 16 --development
```

Output phải là thư mục mới. `--development` bắt buộc: CLI này chưa phải entrypoint thực nghiệm chính thức và không xác nhận full-data QA/review. Kaggle có thể dùng cùng script với `--device cuda` sau khi môi trường và dữ liệu được nghiệm thu. Yêu cầu CUDA nhưng không có GPU sẽ FAIL, không âm thầm dùng CPU.

- `embeddings.npz`: `pair_ids`, `text`, `image`, `cosine`, theo đúng thứ tự manifest; đọc bằng `numpy.load(..., allow_pickle=False)`. File NPZ được Git ignore.
- `result.json`: phạm vi, trạng thái, manifest/embedding checksum, fingerprint trọng số trước/sau, phiên bản torch/transformers, thời gian và validation M0/M1. Khi lỗi sau tạo thư mục output, ghi FAIL và exit 1.
- Nguồn ảnh được đối chiếu checksum và prompt với manifest. Thiếu nguồn hoặc trùng ID làm FAIL. Bộ nhớ giữ các vector và một batch ảnh, không giữ toàn bộ ảnh dataset.
- Encoder chỉ đọc model local, eval/frozen/inference mode; ảnh chuyển RGB, text truncation theo giới hạn model. Không tải checkpoint tự động.

M0 chỉ fit trung bình nhãn train. M1 báo Spearman và pairwise trên cosine thô, không báo MAE/RMSE như điểm người 1–5. Validation không đọc nhãn/prediction test. Pairwise chỉ so cùng prompt; chênh nhãn dưới 0.5 hoặc bằng nhau bị loại, prediction tie nhận 0 điểm và được đếm riêng. Spearman không xác định trả JSON `null`.

## Kiểm chứng hiện tại

- 13 test metrics/baseline/extraction PASS: `reports/results/G2_local_tests.txt`.
- Encoder CLIP thật trên CPU với ảnh tổng hợp: `G2_encoder_checks.json`; batch/single delta dưới 1e-4, vector unit norm, truncation và frozen state đạt.
- CLI Parquet → embeddings → validation trên 18 cặp tổng hợp: `G2_fixture_extraction/result.json`. Nhãn fixture đều bằng 2 nên MAE M0 bằng 0 và Spearman/ranking N/A là kết quả kiểm thử, **không phải chất lượng mô hình trên GenAI-Bench**.
- Dữ liệu fixture local nằm trong `.cache/g2-pipeline-fixture`; không đưa lên GitHub. Model dùng revision đã tải ở G0, checksum file trong `reports/results/clip_download.json`.

Còn chờ full-data QA, khóa split/review, tích hợp provenance đầy đủ (Git/config/tokenizer) và kiểm chứng extraction GPU trước thực nghiệm chính thức. G3-local tiếp tục với Ridge/MLP trên fixture, không sử dụng holdout để tuning.
