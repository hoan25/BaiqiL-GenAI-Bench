# Cấu trúc dự án sau sắp xếp — 2026-10-30

## Vị trí

```text
BTL/
├── code/                   # source, cấu hình, kiểm thử, notebook, requirements
│   ├── src/
│   ├── backend/
│   ├── frontend/
│   ├── scripts/
│   ├── tests/
│   ├── configs/
│   └── notebooks/
├── project_records/        # hồ sơ, không chứa source ứng dụng
│   ├── PROJECT_PLAN.md     # kế hoạch gốc, chỉ một bản chính
│   ├── TEST_PLAN.md
│   ├── docs/PROGRESS.md    # nhật ký chính
│   ├── docs/              # hướng dẫn và giải thích
│   ├── reports/           # báo cáo, JSON/log thực nghiệm, ảnh kết quả
│   ├── result/            # bộ tổng hợp báo cáo đã có của người dùng
│   └── artifacts/         # gói bàn giao cũ; giữ nguyên
├── data/                   # ảnh/prompt/nhãn; không đưa vào code
├── .cache/                 # checkpoint/embeddings/cache đã có
├── .venv/                  # Python local; không di chuyển
├── .venv-repro/            # môi trường kiểm tái lập cũ
├── .git/
├── .gitignore
├── AGENTS.md
└── README.md
```

Dependencies frontend/node_modules được giữ tại vị trí thông thường trong code/frontend, Git ignore; không phải source để upload. Tránh xóa môi trường/checkpoint chỉ vì tên cache: chúng chứa tài nguyên đã tải/huấn luyện.

## Chạy web mô hình cũ

Từ thư mục BTL, PowerShell:

```powershell
cd code
$env:PROMPTALIGN_CLIP = '../.cache/models/clip-vit-base-patch32/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
$env:PROMPTALIGN_TRAINING = '../.cache/reviewed-training-20261020T174855207187Z/training-seed42'
Remove-Item Env:PROMPTALIGN_ENSEMBLE_CONFIG -ErrorAction SilentlyContinue
$env:PROMPTALIGN_ALLOW_DEVELOPMENT = '0'
$env:HF_HUB_OFFLINE = '1'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
../.venv/Scripts/python.exe -X utf8 -m uvicorn backend.app.main:configured_app --factory --host 127.0.0.1 --port 8000
```

Terminal khác, từ BTL:

```powershell
cd code/frontend
npm.cmd run dev -- --host 127.0.0.1 --port 5173
```

Mở http://127.0.0.1:5173. Không cần Kaggle đang chạy.

Kiểm thử từ code/: `../.venv/Scripts/python.exe -X utf8 -m pytest tests -q`.
Browser từ code/frontend/: `PLAYWRIGHT_BROWSERS_PATH` trỏ `../../.cache/playwright`.

## Dữ liệu và đường dẫn lịch sử

Các JSON/log ghi kết quả cũ không sửa nội dung chỉ để thay đường dẫn: nhiều file có checksum và liên kết tới lần chạy cụ thể. Ánh xạ khi đọc: source cũ → code/source; reports cũ → project_records/reports; docs cũ → project_records/docs; result cũ → project_records/result. Đường dẫn tuyệt đối của ảnh/data/.cache vẫn không đổi.

README cũ được giữ tại docs/README_BEFORE_REORGANIZATION.md như tài liệu lịch sử. Hướng dẫn trong file đó và log cũ có thể dùng cấu trúc trước khi chuyển. Tài liệu này là chỉ dẫn đường dẫn hiện hành.

## Dọn dẹp

Chỉ xóa thư mục sinh tự động đã xác định: .pytest_cache, __pycache__ trong source, frontend/dist và frontend/test-results. Danh sách trong REORGANIZATION_CLEANUP.json; frontend build/tests sẽ tự tạo lại chúng khi cần. Không xóa ZIP người dùng gửi, ảnh, ratings, checkpoint, embeddings, nhật ký hay phần source GPU đang làm dở.

REORGANIZATION_MOVES.json ghi danh sách nguồn/đích. Thư mục result bị khóa khi rename nguyên thư mục; đã chuyển từng mục con thành công rồi xóa thư mục rỗng, không xóa nội dung.

Source và notebook cấu trúc mới hiện local; chưa commit/push. Notebook clone GitHub cần đúng phiên bản source mới trước khi chạy. Phần train GPU đã qua kiểm local nhưng CUDA thật còn NOT RUN; không coi việc sắp xếp là nghiệm thu GPU.
