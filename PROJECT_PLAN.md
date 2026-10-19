# PROMPTALIGN AI — KẾ HOẠCH TRIỂN KHAI BÀI TẬP LỚN

**Đề tài:** Hệ thống hội thoại đánh giá mức độ phù hợp giữa mô tả (prompt) và ảnh AI bằng CLIP, Regression và Ranking Learning.  
**Nhóm:** 3 người · **Dự kiến:** 5 tuần · **Phát triển:** VS Code · **GPU:** Kaggle khi cần.  
**Dataset chính:** `BaiqiL/GenAI-Bench` (12 Parquet shards; mục tiêu theo mô tả là 1.600 prompt × 6 ảnh = 9.600 cặp, cần xác nhận sau khi nạp đầy đủ).

## A. Phạm vi và định nghĩa thành công

### MVP — PHẢI HOÀN THÀNH

1. Chat UI tương tác: gợi ý câu hỏi, đính kèm ảnh, hiển thị kết quả ngay trong hội thoại và giữ ngữ cảnh phiên.
2. Chấm một ảnh so với prompt trên thang 1–5 bằng mô hình huấn luyện (không ngụy trang điểm cosine thành điểm người đánh giá).
3. Xếp hạng 2–6 ảnh theo cùng prompt, hiển thị từng điểm và thứ tự.
4. Dashboard có **số liệu đo thực tế** cho M0 (mean), M1 (CLIP zero-shot), M2 (CLIP + Ridge/MLP), gồm MAE, RMSE, Spearman và pairwise ranking accuracy khi áp dụng được.
5. Backend FastAPI + React chạy được trên CPU máy demo mà không phụ thuộc Kaggle đang mở.
6. Source code, test tự động, checkpoint, hướng dẫn chạy, báo cáo thực nghiệm có thể tái lập.

### Mở rộng — CHỈ LÀM SAU MVP

7. M3: Pairwise Ranking Loss và/hoặc fine-tuning CLIP (có thí nghiệm đối chứng).
8. Image captioning bằng BLIP/SmolVLM pretrained; gợi ý prompt theo mẫu từ caption. Không tự nhận CLIP có khả năng tạo văn bản.
9. Trợ lý trả lời câu hỏi mở về ảnh bằng VLM nhỏ; không huấn luyện chatbot lớn từ đầu.

**Không làm:** sinh ảnh text-to-image, huấn luyện LLM/VLM từ đầu, khẳng định độ chính xác của một thành phần ảnh chỉ bằng cosine, dùng API trả kết quả giả làm kết quả mô hình.

## B. Kiến trúc và nguyên tắc tích hợp

```text
React (chat, upload, comparison, dashboard)
                  |
                  v
FastAPI (validation, session/chat state, task router)
        |                    |                 |
        v                    v                 v
CLIP + scoring head      BLIP (optional)    metrics.json
        |
        v
CPU inference trong máy demo, checkpoint xuất từ Kaggle

VS Code + Git  ->  Kaggle train/extract GPU  ->  artifact có version  ->  backend VS Code
```

- Chat controller dùng intent có cấu trúc và các nút chức năng; không cần LLM để điều hướng MVP.
- Frontend được phát triển sớm với mock API **được gắn nhãn mock**, trước khi tích hợp checkpoint thật.
- Chỉ duy nhất backend được truy cập predictor; frontend không biết cấu trúc PyTorch.
- `src/` chứa code ML tái sử dụng; notebook Kaggle chỉ điều phối, không sao chép logic khó kiểm soát.
- Mọi lần huấn luyện lưu `seed`, bản dữ liệu/split, commit hash, cấu hình, thư viện, metric, checkpoint.

## C. Cấu trúc thư mục đề xuất

```text
promptalign-ai/
  README.md
  .gitignore
  .env.example
  requirements-train.txt
  requirements-api.txt
  configs/{data.yaml,train.yaml}
  data/{manifests/,samples/}                 # Không commit toàn bộ ảnh/Parquet
  notebooks/{01_eda.ipynb,02_clip.ipynb,03_regression.ipynb,04_ranking.ipynb}
  src/
    data/{ingest.py,validate.py,split.py,manifest.py}
    models/{clip_encoder.py,regression_head.py,ranking_head.py}
    training/{extract_embeddings.py,train_regression.py,train_ranking.py}
    evaluation/{metrics.py,evaluate.py}
    inference/{predictor.py,captioner.py}
  artifacts/{metrics.json,model_config.json} # Trọng số lớn lưu ngoài Git
  backend/app/{main.py,schemas.py,routes/,services/}
  frontend/src/{pages/,components/,services/,hooks/}
  tests/{unit/,integration/,e2e/,fixtures/}
  reports/{figures/,results/,model_card.md}
  docs/{PROJECT_PLAN.md,TEST_PLAN.md}
```

## D. Chia việc

| Nhóm | Vai trò | Đầu ra chính | Kiểm thử bắt buộc |
|---|---|---|---|
| Thành viên 1 | Data & Evaluation | Nạp 12 shard, manifest, split, metric, phân tích lỗi | Data integrity, leakage, metric reference |
| Thành viên 2 | ML & Kaggle | CLIP, embeddings, Ridge, MLP, ranking, artifact | Shapes, reproducibility, holdout, CPU parity |
| Thành viên 3 | Backend & Frontend | FastAPI, chat, upload, ranking UI, dashboard | API, React, E2E, upload security |
| Cả 3 | Tích hợp, báo cáo, demo | Release và video/demo | UAT, báo cáo thực nghiệm, chạy máy sạch |

**Review chéo:** M1/ML review code dữ liệu của M2; M2/ML review điểm số của M3/backend; M3 review API spec và end-to-end; mỗi PR có ít nhất một người còn lại review.

## E. Mốc tiến độ có quality gate (5 tuần)

| Mốc | Tuần | Công việc | Minh chứng bàn giao | Điều kiện chuyển mốc |
|---|---|---|---|---|
| G0: Môi trường | 1 | Git, cấu trúc dự án, Python/Node, requirements khóa phiên bản, test runner | Repo, README setup, log `pip check`, smoke test | Máy VS Code và Kaggle import được thư viện, test chạy được |
| G1: Dataset | 1 | Nạp 12 Parquet, kiểm ảnh/nhãn, manifest, chia theo prompt | manifest phiên bản hóa, báo cáo chất lượng, split | Không trùng prompt giữa split, tất cả cặp ảnh/nhãn truy xuất đúng |
| G2: M0/M1 | 2 | Mean baseline, CLIP frozen embeddings, cosine, metrics | embeddings, M0/M1 bảng metric, logs | Số embedding = số cặp hợp lệ, không NaN, batch/CPU chạy ổn |
| G3: M2 | 3 | Ridge + MLP, validation, early stopping, checkpoint | trọng số, scaler, configs, val metrics | Model load lại đúng output; so sánh công bằng trên cùng split |
| G4: Backend | 3–4 | API predict/compare/metrics/chat; trạng thái phiên | OpenAPI, integration test, service | Trường hợp đúng/sai nhận HTTP hợp lý; không trả kết quả mock |
| G5: Frontend | 3–4 | Chat UI, upload, kết quả, dashboard | giao diện React, screenshot E2E | Luồng người dùng từ upload đến kết quả hoạt động |
| G6: ML nâng cao (optional) | 4 | Pairwise ranking; caption optional | thí nghiệm ablation; caption demo | Không gây hồi quy MVP; chứng minh hoặc báo cáo không cải thiện |
| G7: Release | 5 | Test holdout một lần, thử ảnh mới, đo CPU, báo cáo | zip/repo, model, video, báo cáo, hướng dẫn | Test suite xanh; UAT đạt; demo offline với Kaggle |

**Quy tắc kiểm thử sau mỗi phần:** Viết unit test -> chạy test -> tích hợp -> integration test -> xác nhận evidence -> review -> mới đánh dấu DONE. Không chuyển mốc chỉ dựa trên việc notebook chạy không báo lỗi.

## F. Điều kiện khoa học cần giữ

1. Gộp 12 shard trước khi chia dữ liệu chính thức. `train-...parquet` là tên shard, **không phải train split của dự án**.
2. Tạo định danh cặp ảnh như `(shard_file,row_in_shard,image_model)`; không chỉ `row_id` trong shard.
3. Group split theo `prompt` chuẩn hóa hoặc định danh prompt toàn cục đã kiểm trùng; sáu ảnh cùng prompt nằm trong cùng split. Hạn chế prompt gần trùng nếu phát hiện.
4. Nhãn đích = trung bình của ba `HumanRatings[model]`; lưu cả ba điểm gốc và `std`; kiểm `1 <= rating <= 5`.
5. CLIP cosine là similarity thô, **không phải** điểm 1–5. Chỉ hồi quy được fit trên train để ánh xạ score; validation chọn hyperparameter; test dùng khi chốt release.
6. Spearman của predictor hằng số là **không xác định**; dashboard hiển thị N/A thay vì 0.
7. Chấm điểm xếp hạng trên các cặp ảnh **cùng prompt**; loại cặp có nhãn gần bằng nhau hoặc báo riêng tie.
8. Tách `prompt` dùng cho dự đoán khỏi nhãn sinh bởi BLIP; caption là gợi ý và có thể sai.
9. Báo cáo nêu rõ CLIP pretrained đã học contrastive trước đó; nếu tuyên bố "cải tiến Contrastive Learning" phải có thí nghiệm fine-tuning/Ranking mới do nhóm thực hiện, không chỉ gọi pretrained model.

## G. API contract MVP

| Method/route | Input | Kết quả | Lưu ý |
|---|---|---|---|
| `GET /api/health` | — | trạng thái load model, version | không để lộ secret |
| `POST /api/evaluate` | `prompt`, 1 image | score 1–5, cosine, model_version | validate kiểu, file, kích thước |
| `POST /api/compare` | `prompt`, 2–6 images | danh sách score và rank | stable tie handling |
| `GET /api/metrics` | — | metric thực đã tính, split, model_version | không hardcode kết quả giả |
| `POST /api/uploads` | 1–6 image files (multipart), session_id | image IDs tạm thời thuộc phiên | xác minh nội dung và giới hạn file |
| `POST /api/chat` | session_id, text, image IDs (nếu có) | type + payload (text/score/ranking/chart) | follow-up tái sử dụng prompt/ảnh |
| `POST /api/caption` (optional) | image | caption, model_version | mô hình caption riêng |

**Định dạng response chung:** `request_id`, `type`, `data`, `model_version`, `elapsed_ms`, `warnings` (nếu có); lỗi có `code`, `message`, `details` khi phù hợp.

### Giới hạn đề nghị cho MVP

- Ảnh: JPG/PNG/WebP; tối đa 10 MB/ảnh; tối đa 6 ảnh mỗi lần; giới hạn độ phân giải khi giải mã.
- Prompt: bắt buộc với chấm điểm/so sánh; độ dài hợp lý (ví dụ 1–1.000 ký tự) và trả lỗi rõ ràng nếu vượt giới hạn.
- Session: giữ prompt gần nhất/ảnh upload trong ngữ cảnh session riêng; không chia sẻ giữa người dùng. Image IDs phải thuộc đúng session, có TTL và cơ chế dọn tệp tạm.
- Chấm bằng CPU, cache encoder/model sau khởi động. Đo thời gian p50/p95 trên máy thật rồi mới chốt SLA; không đưa ra số tốc độ chưa benchmark.

## H. Ngân sách GPU và checkpoint

- G0/G1: CPU; G2: Kaggle GPU để trích embeddings; G3: Ridge/MLP thường CPU được; G6: GPU cho fine-tuning nếu chọn.
- Kế hoạch ước tính toàn dự án 7–19 giờ GPU cho các thí nghiệm gọn, **không phải cam kết**; giữ dự phòng theo hạn mức thực tế của Kaggle.
- Kaggle ghi checkpoint, metric và config vào `/kaggle/working/`, `Save Version`, sau đó tải artifact về VS Code.
- Không commit 4,76 GB dữ liệu hoặc trọng số vài trăm MB lên Git thường.
- Pin dependencies; không chạy `pip install -U` tràn lan. Log lỗi trước đây: `huggingface-hub==2.2.0` không khớp bản transformers hiện dùng; phải xác minh import sau khi restart runtime.

## I. Checklist trạng thái tại thời điểm lập kế hoạch

| Hạng mục | Trạng thái | Căn cứ |
|---|---|---|
| Tải thử shard đầu tiên | ĐÃ THỰC HIỆN (theo ảnh notebook) | Đọc 134 prompt |
| Tạo metadata shard đầu tiên | ĐÃ THỰC HIỆN (theo ảnh notebook) | 804 cặp, đủ 6 cặp/prompt |
| Vẽ phân bố nhãn | ĐÃ THỰC HIỆN (theo ảnh notebook) | mean khoảng 3,588 |
| Chia shard mẫu theo prompt | ĐÃ THỰC HIỆN (theo ảnh notebook) | train 93, val 20, test 21; assert không chồng ID |
| Data QA đầy đủ và 12 shard | CHƯA KIỂM CHỨNG | Cần G1 |
| Import CLIP + trích embedding | ĐANG VƯỚNG / CHƯA XÁC NHẬN | Từng lỗi xung đột thư viện |
| M0/M1/M2/M3 metric | CHƯA KIỂM CHỨNG | Không có metric thực nghiệm |
| FastAPI/React + E2E | CHƯA THỰC HIỆN | Cần G4/G5 |

Không gọi những mục "ĐÃ THỰC HIỆN" ở trên là đã qua kiểm thử tự động toàn diện. Các kết quả chỉ được quan sát từ ảnh Kaggle đã gửi.

## J. Định nghĩa DONE áp dụng cho MỌI hạng mục

- [ ] Code chạy thành công với cấu hình và hướng dẫn được ghi lại.
- [ ] Unit test phù hợp được viết và pass.
- [ ] Integration test hoặc notebook smoke test liên quan pass.
- [ ] Các ca biên/ca lỗi được kiểm thử, không crash không rõ nguyên nhân.
- [ ] Có bằng chứng: log test, metric, screenshot hoặc model artifact.
- [ ] Không lộ secrets, không commit dữ liệu/trọng số lớn.
- [ ] Có một người khác review và đồng ý.
- [ ] Cập nhật README, trạng thái mốc, phiên bản artifacts.

**File kèm theo:** `TEST_PLAN.md` có danh mục test case chi tiết, lệnh chạy và tiêu chí nhận kết quả.
