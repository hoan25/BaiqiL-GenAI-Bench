# Web local — FastAPI + React

## Khởi động không có model

Tại thư mục gốc, cài `requirements-api.txt` vào `.venv` rồi chạy:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m uvicorn backend.app.main:configured_app --factory --host 127.0.0.1 --port 8000
```

Trong terminal thứ hai:

```powershell
cd frontend
npm.cmd ci
npm.cmd run dev
```

Mở `http://127.0.0.1:5173`. Vite chuyển `/api` về backend cổng 8000. Health báo chưa có model; chấm điểm/metrics trả 503 có hướng dẫn. Giao diện không tự sinh điểm. Không cần Kaggle đang chạy để mở web.

## Chạy checkpoint thật đã kiểm tra

Artifact hiện có trên máy này, không cần Kaggle đang mở. Tại thư mục gốc:

```powershell
$env:PROMPTALIGN_CLIP = '.cache/models/clip-vit-base-patch32/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
$env:PROMPTALIGN_TRAINING = '.cache/reviewed-training-20261020T174855207187Z/training-seed42'
$env:PROMPTALIGN_ALLOW_DEVELOPMENT = '0'
$env:HF_HUB_OFFLINE = '1'
.\.venv\Scripts\python.exe -X utf8 -m uvicorn backend.app.main:configured_app --factory --host 127.0.0.1 --port 8000
```

Terminal thứ hai chạy `cd frontend` rồi `npm.cmd run dev`; mở http://127.0.0.1:5173. Dashboard hiển thị số liệu validation thật, gồm M1 cosine và pairwise accuracy. N/A của MAE/RMSE cosine không phải lỗi. Chưa đánh giá test holdout/release.

Máy mới cần nhận ZIP training và CLIP đúng revision, rồi chạy `scripts/review_training.py --archive <ZIP> --output <thư_mục_mới> --report <JSON_mới>` trước khi trỏ PROMPTALIGN_TRAINING vào `training-seed42`. Receipt được tạo sau khi kiểm checkpoint và tính lại validation trên CPU; không tự viết receipt bỏ qua kiểm tra.

## Checkpoint tổng hợp (chỉ development)

Trước khi mở backend, có thể đặt:

```powershell
$env:PROMPTALIGN_CLIP = '.cache/models/clip-vit-base-patch32/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
$env:PROMPTALIGN_TRAINING = 'reports/results/G3_fixture_training'
$env:PROMPTALIGN_ALLOW_DEVELOPMENT = '1'
```

Model loader kiểm checksum head, fingerprint CLIP, kích thước feature; chỉ đọc file local và cache một lần khi khởi động. Checkpoint fixture không phải model sản phẩm; API/UI luôn ghi rõ phạm vi thử nghiệm. Khi chưa bật development, loader từ chối artifact chưa nghiệm thu. `scripts/check_predictor_api.py` kiểm CLIP thật + head fixture qua HTTP; đây không phải kiểm chất lượng dataset.

## API và phiên

- `/api/health`, `/api/metrics`: GET.
- `/api/evaluate`: multipart `prompt`, `image`; `/api/compare`: `prompt`, 2–6 `images`.
- `/api/sessions`: POST cấp session ID ngẫu nhiên. ID là bearer token của phiên local; frontend giữ trong bộ nhớ, không ghi log/token ra Git.
- `/api/uploads`: multipart `session_id`, 1–6 `files`, trả image IDs.
- `/api/chat`: JSON `session_id`, `action` (evaluate/compare/metrics/help/auto), `prompt`, `image_ids`, `text`. Follow-up có thể dùng prompt/ảnh trước; hỗ trợ câu “ảnh thứ hai thì sao”. Đây là router theo quy tắc, không phải chatbot LLM tổng quát.

Ảnh được kiểm bytes, định dạng JPG/PNG/WebP, 10 MB và 20 triệu pixel; EXIF orientation được áp dụng, filename không dùng làm đường dẫn. Body giới hạn 61 MB kể cả chunked. Dữ liệu phiên giữ trong RAM, tối đa 12 ảnh/phiên, tổng bytes nén 128 MB, 50 phiên; TTL 30 phút, dọn khi có truy cập. Restart xóa phiên. Chỉ chạy một worker local; chưa thiết kế cho public multiuser deployment.

So sánh giữ thứ tự ảnh, dùng dense rank, điểm bằng nhau cùng hạng. Lỗi có request ID và thông báo an toàn, không trả traceback. Metrics đọc từ report checkpoint, ghi đúng phạm vi fixture hoặc validation thật; `null` hiển thị N/A. Smoke CLI với checkpoint fixture cần thêm `--allow-development`.

## Kiểm thử

```powershell
.\.venv\Scripts\python.exe -X utf8 -m pytest tests/integration/test_api.py -q
cd frontend
$env:PLAYWRIGHT_BROWSERS_PATH = '..\.cache\playwright'
node node_modules/playwright/cli.js install chromium
npm.cmd run test:e2e
npm.cmd run build
```

Suite mặc định dùng API mock có nhãn `test_fixture`, không bật mock trong app thật. Suite `node node_modules/playwright/cli.js test --config playwright.real.config.js` chạy backend và CLIP/MLP thật từ artifact local, không mock API; đã PASS ngày 2026-10-26. Config real hiện dành cho Windows và đường dẫn artifact ghi trên. UAT và review toàn mốc vẫn còn chờ.

Benchmark local: `reports/results/G7_cpu_benchmark_20261026.json`, 20 request sau warmup mỗi batch, ảnh đỏ tổng hợp 224×224; p95 khoảng 243 ms (1 ảnh), 968 ms (6 ảnh), 5 request đồng thời ổn định. Đây là HTTP in-process, không gồm độ trễ mạng và không thay cho đánh giá ảnh thật; peak RAM chưa đo.

Tham chiếu triển khai: [FastAPI upload files](https://fastapi.tiangolo.com/tutorial/request-files/), [FastAPI testing](https://fastapi.tiangolo.com/tutorial/testing/).
