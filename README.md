# PromptAlign AI

Hệ thống đánh giá prompt–image theo `PROJECT_PLAN.md` và `TEST_PLAN.md`.
Đã có code và kiểm chứng local cho dữ liệu, CLIP extraction và huấn luyện Ridge/MLP. Tiếp theo predictor/FastAPI; chưa có mô hình được nghiệm thu trên GenAI-Bench hoặc ứng dụng web.

## Tiếp tục công việc

Đọc `AGENTS.md` và [nhật ký tiến độ](docs/PROGRESS.md) trước khi thay đổi.
Giữ nguyên các phần đã có bằng chứng kiểm thử đạt; tiếp tục nhiệm vụ đầu tiên chưa nghiệm thu.

## Kiểm tra môi trường hiện tại

Chạy tại thư mục gốc bằng PowerShell:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/check_environment.py --output reports/results/environment-venv.json
.\.venv\Scripts\python.exe -X utf8 -m pytest tests/unit -q
node --version
npm.cmd --version
```

Script trả exit code 1 nếu thiếu thư viện hoặc `pip check` lỗi. Nó không tải model/dataset và không xác nhận inference CLIP.
Unit tests bootstrap dùng thư viện chuẩn, cũng có thể chạy bằng pytest sau khi cài môi trường.
Trên máy hiện tại, `npm.cmd` tránh lỗi ExecutionPolicy của `npm.ps1`.

## Môi trường Python của dự án

`.venv` được tạo bằng CPython 3.13.5 Windows x64 đã có trên máy. Python 3.11.9 trên PATH là MSYS2; không dùng nó để cài wheel PyTorch Windows chuẩn ([tài liệu MSYS2](https://www.msys2.org/docs/python/)).
Không cần activate môi trường hoặc đổi ExecutionPolicy: gọi trực tiếp `.venv\Scripts\python.exe`.
Dùng `-X utf8` khi chạy pip/Python để tránh lỗi ghi log với đường dẫn tiếng Việt.

Dependencies trực tiếp nằm trong `requirements-g0.in`. Trạng thái cài và kiểm chứng mới nhất nằm trong `docs/PROGRESS.md`; file requirements không tự chứng minh môi trường đã đạt.

Đã kiểm chứng import, 10 test và CLIP smoke trên `.venv-repro` sạch. Full lock: `requirements-g0-win-cpu.lock.txt` (31 package, Windows x64/CPython 3.13 CPU). Tái tạo bằng CPython chuẩn đã cài, ví dụ:

```powershell
# Thay đường dẫn bằng Python CPython chuẩn trên máy bạn.
& 'C:\Users\20146\AppData\Local\Programs\Python\Python313\python.exe' -m venv .venv-repro
.\.venv-repro\Scripts\python.exe -X utf8 -m pip --isolated install -r requirements-g0-win-cpu.lock.txt --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu --only-binary=:all: --progress-bar off
.\.venv-repro\Scripts\python.exe -X utf8 scripts/check_environment.py --output reports/results/environment-repro.json
.\.venv-repro\Scripts\python.exe -X utf8 -m pytest tests/unit -q
```

ENV-04 PASS: đã cài từ lock vào môi trường sạch, đối chiếu 31/31 package, import/pip check và 10 test đều đạt. Bằng chứng: `reports/results/G0_repro_lock_comparison.json` và `reports/results/G0_notebook_local/`. Không dùng lock Windows này cho môi trường Kaggle GPU.

## Smoke CLIP và Kaggle

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/smoke_clip.py --model .cache/models/clip-vit-base-patch32/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268 --output reports/results/G0_clip_cpu_current.json
```

Lệnh chỉ dùng checkpoint có sẵn trong cache hoặc thư mục được truyền qua `--model`; không tự tải trọng số.
Ảnh vuông đỏ tổng hợp chỉ kiểm tra pipeline CPU, không phải bằng chứng chất lượng chấm điểm.
Exit code: 0 PASS, 1 lỗi thực thi, 2 BLOCKED do không truy cập được tài nguyên local.

Đã tải CLIP theo quyền người dùng cấp; revision/checksum ở `reports/results/clip_download.json`. Dùng `--device auto` để chọn CUDA nếu có, CPU nếu không. `--device cuda` sẽ báo lỗi khi CUDA không khả dụng.

Luồng chính: **VS Code → [GitHub](https://github.com/hoan25/BaiqiL-GenAI-Bench) → Kaggle → artifact về VS Code**. Notebook clone source GitHub, tạo môi trường Kaggle riêng, tải CLIP và xuất `g0-<thời_gian>.zip` kể cả khi kiểm tra lỗi. Xem [hướng dẫn chạy và gửi kết quả](docs/G0_KAGGLE.md). Bundle 609 MB của phiên trước là phương án cũ; không cần upload cho luồng này.

Đã nhận kết quả Kaggle: 13 test PASS, CPU/CUDA CLIP PASS. Bản sửa bootstrap có 16 test local PASS và bootstrap thật local PASS. Theo lộ trình điều chỉnh, xác nhận bootstrap Kaggle sạch được thực hiện khi tích hợp GPU; hiện chưa cần chạy thêm notebook.

## Pipeline dữ liệu local

```powershell
.\.venv\Scripts\python.exe -X utf8 -m pytest tests/unit/test_data.py -q
.\.venv\Scripts\python.exe -X utf8 scripts/data_smoke.py --output reports/results/G1_fixture_smoke_new.json
```

Chọn đường dẫn output chưa tồn tại. CLI tạo ảnh tổng hợp trong bộ nhớ, kiểm tra 60 cặp và xuất thống kê split; không tải dataset. Contract đầu vào `src/data/manifest.py`: `shard`, `row`, `prompt`, `images[model]` dạng bytes và `ratings[model]` gồm ba số nguyên 1–5.

Đã có adapter Parquet và CLI `scripts/prepare_data.py`, hướng dẫn tại [pipeline dữ liệu](docs/DATA_PIPELINE.md). Toàn suite hiện có **34 test PASS**, bao gồm integration với file Parquet tổng hợp. QA trên 12 shard thật và khóa split vẫn chưa chạy; fixture không thay thế nghiệm thu GenAI-Bench.

G2 bổ sung **13 test PASS** cho metrics/extraction và kiểm chứng CLIP thật trên CPU với dữ liệu tổng hợp. Xem [hướng dẫn embeddings/baseline](docs/EMBEDDINGS.md). Các kết quả G0/G1 được giữ nguyên, không coi fixture là benchmark GenAI-Bench.

G3 thêm Ridge/MLP, early stopping và checkpoint có scaler; xem [hướng dẫn huấn luyện local](docs/TRAINING.md). CLI kiểm checksum đầu vào và reload checkpoint; chỉ chạy chế độ development trước khi dữ liệu thật được nghiệm thu.
