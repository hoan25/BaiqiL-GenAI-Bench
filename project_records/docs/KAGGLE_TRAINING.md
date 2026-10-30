# Chạy extraction và training sau QA

Source xử lý nằm trong VS Code/GitHub; notebook chỉ lấy source và gọi `scripts/kaggle_train.py`. Không chạy lại notebook QA đã đạt.

## Review dữ liệu trước khi chạy

Đọc [biên bản QA](../reports/results/G1_kaggle_full_qa_review.md). Ba quyết định cần người review xác nhận:

1. Giữ loại hai cặp nhãn lỗi: shard 1, row 97/DALLE_3 và row 118/DeepFloyd_I_XL_v1; không điền hoặc sửa nhãn.
2. Giữ cặp prompt “puppy's head” và “puppy's paw” vì khác nghĩa, cùng train.
3. Giữ cặp prompt bàn làm việc chỉ khác dấu phẩy trong cùng train; giữ nguyên split.

Runner chốt đúng 9.598 cặp, manifest SHA256 `4ed1e3c7909c81c239beaa2890f4adbcaef7545217e1d309a8b023e084182ca2`, dataset revision `94e543b0801feade1419873335191de90b68d164`. Người review điền tên và xác nhận trong notebook; code không tự tạo xác nhận thay người dùng. Đây là review quyết định dữ liệu theo PROJECT_PLAN.md, không phải tuyên bố toàn bộ mốc/release DONE.

## Các bước trên Kaggle

1. Import [notebooks/02_train.ipynb](https://github.com/hoan25/BaiqiL-GenAI-Bench/blob/main/notebooks/02_train.ipynb).
2. Bật Internet và GPU. Notebook dùng một GPU; head Ridge/MLP chạy CPU.
3. Cung cấp ZIP QA đã tải: `data-qa-20261020T165712283760Z.zip`. Nếu nó còn ở `/kaggle/working`, notebook tìm được. Nếu mở phiên mới, thêm ZIP vào một Kaggle Input riêng tư. Runner cũng nhận **thư mục đã giải nén** có `result.json`, `dataset.json`, `qa/qa.json`, `qa/manifest.jsonl`; từng file đều phải khớp checksum. Nếu tìm thấy nhiều kết quả, điền `QA_ARCHIVE` bằng đường dẫn ZIP hoặc thư mục đúng.
4. Sau review, điền `REVIEWER = "Tên người review"` và `ACCEPT_DATA_REVIEW = True` ở cell cuối. Nếu không đồng ý quyết định dữ liệu, dừng trước bước này và báo vấn đề.
5. Nếu đã mount đủ 12 shard, điền `DATA_ROOT` đúng thư mục và đặt `DOWNLOAD_DATASET=False`. Mặc định để trống DATA_ROOT, runner tái dùng cache revision trong `/kaggle/working`; nếu cache thiếu và `DOWNLOAD_DATASET=True`, việc chạy cell cho phép tải đúng revision (~4,76 GB). Không cần tự tải dataset về máy local.
6. Run All. Không cài pip/torch thủ công. Source phải sạch; notebook lấy bản mới từ main.
7. Tải **`training-*.zip` mới nhất** qua **Output / Files**. Lưu vào `reports/results/kaggle/` trên máy và báo đường dẫn. Nếu lỗi, vẫn tải ZIP để có stage và log đầy đủ.

Không cần upload lại cả repository. Không dùng link `/kaggle/working/...` làm URL trình duyệt.

## Runner thực hiện

- Venv mới, bootstrap bằng launcher đã kiểm chứng sạch trên Kaggle; torch 2.8.0 cu126 và dependencies trong requirements-train.txt.
- Pip check/import/version/GPU và các test liên quan trước extraction. Cài GPU mới vẫn cần kiểm chứng trên phiên thật.
- Đọc archive QA đã review, tạo lock có tên reviewer; xác minh checksum 12 shard trước extraction. Không chạy lại QA hoặc chia lại split.
- Tải/tái dùng CLIP revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`; lưu checksums model/tokenizer; frozen embeddings, kiểm model không đổi.
- Huấn luyện mean/Ridge/MLP ở seed 42/43/44. Seed 42 được chốt là run chính; head được chọn bằng validation MAE. Seed 43/44 để báo độ ổn định, không chọn seed tốt nhất. MLP tối đa 100 epoch, patience 15.
- Baseline và head chỉ báo validation; không dùng test để chọn model. Test embeddings có thể được cache nhưng test labels/predictions không được dùng cho fit hay metric trong run này.
- Xuất logs, lock, manifest, embeddings, checkpoint head, checksums, reload deltas, metrics và provenance. ZIP không chứa Parquet/ảnh gốc/CLIP weights/venv. Các file đó nằm ngoài thư mục run.

## Trạng thái và bước sau

Code đã có unit/integration và kiểm fixture CLIP thật local; chưa chạy runner mới trên Kaggle GPU hoặc huấn luyện full-data. PASS của runner chỉ chứng minh execution/validation, không phải release hoặc điểm chất lượng bảo đảm.

Sau khi nhận ZIP: kiểm checksum, số vector 9.598, CPU reload, scaler train-only, so M0/M1/M2 và độ ổn định các seed; sau đó tích hợp checkpoint thật vào web. Predictor hiện vẫn chặn chế độ production cho tới bước nghiệm thu artifact. Không đánh giá test holdout trước khi chốt model.
