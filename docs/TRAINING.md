# G3-local — huấn luyện scoring head

`src/models/regression_head.py` cung cấp mean/Ridge/MLP; `src/training/train_regression.py` fit và chọn model theo validation. Chỉ dùng NumPy/PyTorch đã có, không thêm dependency.

Features là nối text embedding và image embedding theo thứ tự `[text, image]`. Không dùng nhãn, tên generator hoặc metadata nguồn làm feature. Scaler lấy mean/std từ train; chiều có std dưới 1e-8 dùng scale 1. Ridge giải hệ có L2 penalty (intercept không penalty); MLP có một hidden layer ReLU và đầu ra `1 + 4*sigmoid`. Mean/Ridge clamp về [1,5].

MLP chạy CPU full-batch, Adam, seed cố định; chọn epoch có MAE validation thấp nhất, dừng sau `patience` epoch không cải thiện. So sánh mean/Ridge/MLP cùng validation; khi MAE bằng nhau ưu tiên thứ tự mean, Ridge, MLP. Không bắt buộc MLP thắng baseline. Code chỉ đọc nhãn và features train/val; test giữ lại cho release.

## Chạy với output G2

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/train_regression.py --manifest data/manifests/run-001/manifest.jsonl --extraction artifacts/embeddings-dev-001 --output artifacts/training-dev-001 --epochs 100 --patience 15 --development
```

Thư mục extraction phải có `embeddings.npz` và `result.json` PASS từ G2; checksum manifest/embeddings phải khớp. `--development` bắt buộc vì full-data QA và nghiệm thu môi trường Kaggle chưa xong. Không dùng checkpoint fixture này làm model sản phẩm.

Output gồm `mean.pt`, `ridge.pt`, `mlp.pt` và `result.json` lưu config, seed, versions, Git commit/working-tree status, provenance extraction, lịch sử train/val, model được chọn và checksum checkpoint. Output phải là thư mục mới; checkpoint không ghi đè và được Git ignore. `load_head` đọc CPU với `weights_only=True`, gồm kiến trúc, scaler và weights; CLI kiểm tra reload sai lệch dưới 1e-5. Khi thất bại sau khi tạo output, ghi FAIL và exit 1.

## Kiểm chứng và giới hạn

```powershell
.\.venv\Scripts\python.exe -X utf8 -m pytest tests/unit/test_regression.py tests/integration/test_training_cli.py -q
```

Test gồm Ridge đối chiếu lời giải một chiều, scaler train-only, MLP overfit 32 mẫu tổng hợp, early stopping/best epoch, seed, checkpoint reload/range, không đọc test targets/features, input lỗi, provenance sai và CLI không ghi đè.

Đã chạy CLI trên embeddings CLIP của 18 cặp tổng hợp G2; nhãn toàn bằng 2 nên mean thắng/tie Ridge, không có ý nghĩa benchmark. Xem `reports/results/G3_fixture_training/result.json`. Chưa có huấn luyện/metric GenAI-Bench thật, chưa multi-seed trên dataset hoặc review chéo. Provenance encoder/tokenizer và gate thực nghiệm chính thức vẫn cần hoàn thiện khi tích hợp Kaggle.
