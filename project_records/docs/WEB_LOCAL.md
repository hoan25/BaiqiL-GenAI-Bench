# Web local — FastAPI + React

## Cài mới trên Windows CPU

Dùng CPython 3.13 x64 chuẩn và Node/npm. Trong thư mục source mới, chưa có `.venv`:

```powershell
py -3.13 -m venv ../.venv
..\.venv\Scripts\python.exe -X utf8 -m pip --isolated install -r requirements-web-win-cpu.txt --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu --only-binary=:all:
..\.venv\Scripts\python.exe -X utf8 -m pip check
cd frontend
npm.cmd ci
npm.cmd run build
cd ..
```

Nếu không có launcher `py`, thay bằng đường dẫn CPython chuẩn. Không dùng Python MSYS2. Requirements web gồm G0 CPU, FastAPI và PyArrow: predictor hiện dùng hàm checksum từ module QA có import PyArrow; chỉ cài requirements-api.txt vào Python trống không đủ để nạp model.

Cung cấp CLIP đúng revision và ZIP training đã có, rồi review artifact theo mục dưới. Không cần tải toàn bộ dataset hoặc huấn luyện lại. Đây là hướng dẫn Windows CPU, không dùng lock này cho Kaggle GPU. Gói model/artifact không nằm trong Git; clone repo riêng chưa đủ chạy suy luận.

Đã kiểm ngày 2026-10-27 trong venv và node_modules mới trên cùng máy: pip check PASS, 17 API/artifact tests PASS, build PASS và browser model thật PASS. Xem `reports/results/G7_clean_stack_20261027.json`; danh sách 45 package tại `G7_clean_packages_20261027.txt`. Tái dùng browser và artifact có sẵn; chưa phải kiểm trên máy vật lý khác hoặc xác minh lại quy trình tải artifact.

## Khởi động không có model

Tại thư mục gốc, cài `requirements-api.txt` vào `.venv` rồi chạy:

```powershell
..\.venv\Scripts\python.exe -X utf8 -m uvicorn backend.app.main:configured_app --factory --host 127.0.0.1 --port 8000
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
$env:PROMPTALIGN_CLIP = '../.cache/models/clip-vit-base-patch32/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
$env:PROMPTALIGN_TRAINING = '../.cache/reviewed-training-20261020T174855207187Z/training-seed42'
$env:PROMPTALIGN_ALLOW_DEVELOPMENT = '0'
$env:HF_HUB_OFFLINE = '1'
..\.venv\Scripts\python.exe -X utf8 -m uvicorn backend.app.main:configured_app --factory --host 127.0.0.1 --port 8000
```

Terminal thứ hai chạy `cd frontend` rồi `npm.cmd run dev`; mở http://127.0.0.1:5173. Dashboard hiển thị số liệu validation thật, gồm M1 cosine và pairwise accuracy. N/A của MAE/RMSE cosine không phải lỗi. Holdout đã có báo cáo riêng tại reports/results/G7_holdout_seed42_20261027.md; web giữ bảng validation. Chưa nghiệm thu release.

Máy mới cần nhận ZIP training và CLIP đúng revision, rồi chạy `scripts/review_training.py --archive <ZIP> --output <thư_mục_mới> --report <JSON_mới>` trước khi trỏ PROMPTALIGN_TRAINING vào `training-seed42`. Receipt được tạo sau khi kiểm checkpoint và tính lại validation trên CPU; không tự viết receipt bỏ qua kiểm tra.

## Checkpoint tổng hợp (chỉ development)

### Ensemble thử nghiệm được người dùng cho phép ngày2026-10-30

Giữ PROMPTALIGN_CLIP và PROMPTALIGN_TRAINING như mục checkpoint thật ở trên; thêm trước khi khởi động backend:

```powershell
$env:PROMPTALIGN_ENSEMBLE_CONFIG = 'configs/ensemble_trial.json'
$env:PROMPTALIGN_ALLOW_DEVELOPMENT = '1'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
```

Phiên bản `ensemble25-d5cf6e89a7ea`, dùng3MLP trong `.cache/synthetic-negatives-20261029/negative0.25-seed{42,43,44}/mlp.pt`. Cần báo cáo `reports/results/G7_synthetic_ensembles_20261029.json` đúng checksum trong config. Loader kiểm evidence/checkpoint hash, baseline receipt và fingerprint encoder; không ghi receipt nghiệm thu giả. Dashboard thêm ensemble25 và giữ các dòng baseline đối chứng. Clone source riêng chưa có3checkpoint này.

Quay lại model cũ: `Remove-Item Env:PROMPTALIGN_ENSEMBLE_CONFIG -ErrorAction SilentlyContinue`, đặt `PROMPTALIGN_ALLOW_DEVELOPMENT='0'`, restart backend. Trọng số cũ không đổi. Restart xóa session RAM.

Browser test dành riêng phiên bản mới: `node node_modules/playwright/cli.js test --config playwright.demo.config.js ensemble.spec.js` khi server đang chạy. File uat.spec.js cũ giữ các kỳ vọng baseline, không chạy chung để nghiệm thu ensemble.

Trước khi mở backend, có thể đặt:

```powershell
$env:PROMPTALIGN_CLIP = '../.cache/models/clip-vit-base-patch32/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
$env:PROMPTALIGN_TRAINING = '../project_records/reports/results/G3_fixture_training'
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
..\.venv\Scripts\python.exe -X utf8 -m pytest tests/integration/test_api.py -q
cd frontend
$env:PLAYWRIGHT_BROWSERS_PATH = '..\.cache\playwright'
node node_modules/playwright/cli.js install chromium
npm.cmd run test:e2e
npm.cmd run build
```

Suite mặc định dùng API mock có nhãn `test_fixture`, không bật mock trong app thật. Suite `node node_modules/playwright/cli.js test --config playwright.real.config.js` chạy backend và CLIP/MLP thật từ artifact local, không mock API; đã PASS ngày 2026-10-26. Config real hiện dành cho Windows và đường dẫn artifact ghi trên. UAT và review toàn mốc vẫn còn chờ.

Benchmark bổ sung RAM: `reports/results/G7_cpu_memory_benchmark_20261027.json`, 20 request sau warmup mỗi batch, ảnh đỏ tổng hợp 224×224; p95 khoảng 238 ms (1 ảnh), 857 ms (6 ảnh), 5 request đồng thời ổn định. Peak working set Python 1056,25 MiB, gồm nạp model/TestClient, không gồm browser. Đây là HTTP in-process, không phải SLA mạng hoặc đánh giá chất lượng ảnh thật.

Tham chiếu triển khai: [FastAPI upload files](https://fastapi.tiangolo.com/tutorial/request-files/), [FastAPI testing](https://fastapi.tiangolo.com/tutorial/testing/).
