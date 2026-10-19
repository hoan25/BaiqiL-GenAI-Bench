# G0 — Môi trường (cập nhật sau thiết lập .venv, chưa nghiệm thu toàn mốc)

- Ngày kiểm tra: 2026-10-19 (Asia/Saigon).
- Người thực hiện: Codex; người review/xác nhận: chưa có.
- Commit hash: đã khởi tạo Git, chưa có commit; không stage/commit/push.
- Dataset/split: chưa áp dụng; chỉ smoke pretrained CLIP, không train/evaluate dataset. Model: `openai/clip-vit-base-patch32`, revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`.
- Môi trường: Windows 10, CPython 3.13.5 x64 trong `.venv`, Node v24.11.1, npm 11.6.2. Python MSYS2 trên PATH không dùng cho dự án.
- Lệnh test mới nhất: `.venv-repro\Scripts\python.exe -X utf8 -m pytest tests/unit -q -p no:cacheprovider`, qua runner G0 và mã cell notebook.
- Unit tests: tổng 10, PASS 10, FAIL 0, SKIP 0; [pytest log](G0_notebook_local/pytest.log).
- Kiểm tra môi trường: `.venv\Scripts\python.exe -X utf8 scripts/check_environment.py --output reports/results/environment-venv.json` trả exit 0; torch 2.8.0+cpu, transformers 4.57.6, huggingface-hub 0.36.0, pytest 8.4.2; import CLIPModel/CLIPProcessor và pip check PASS.
- Bằng chứng hiện tại: [environment JSON](environment-venv.json), [31 package versions](environment.txt), [Git hygiene](G0_git_hygiene.json), [CLIP smoke](G0_clip_cpu.json), [smoke exit code](G0_clip_cpu_process.json).
- Bằng chứng lịch sử trước khi cài: [bootstrap tests](G0_bootstrap_tests.txt), [môi trường MSYS2 thiếu thư viện](environment.json). Không ghi đè bằng chứng cũ.
- Bằng chứng phiên tải CLIP: [revision/checksums](clip_download.json), [CPU smoke PASS](G0_clip_cpu_pass.json), [31/31 package tái tạo](G0_repro_lock_comparison.json), [lệnh chạy môi trường sạch](G0_notebook_local/commands.json), [auto fallback](G0_notebook_local/clip_auto.json), [checkpoint hygiene](G0_checkpoint_hygiene.json), [bundle](G0_bundle.json).

| Test case theo TEST_PLAN | Trạng thái | Lý do |
|---|---|---|
| ENV-01 | PASS | Import thư viện, CLIPModel/CLIPProcessor, pip check thành công trong .venv |
| ENV-02 | PASS | Checkpoint thật chạy offline CPU; text/image embedding [1,512], finite |
| ENV-03 | NOT RUN | CPU fallback đã PASS local; chưa chạy trên Kaggle, không tuyên bố GPU PASS |
| ENV-04 | PASS | Môi trường sạch từ lock khớp 31/31 package; import, pip check, 10 test đạt |
| ENV-05 | PASS | 10 đường dẫn mẫu được ignore; git ls-files rỗng; chưa stage/commit |
| ENV-06 | NOT RUN | Mã cell chạy được trong Python process mới; chưa có Jupyter/Kaggle clean kernel run |

- Quyết định: **BLOCKED** ở gate G0; không chuyển G1.
- Lỗi đã xử lý: chọn CPython chuẩn thay MSYS2; dùng quyền mạng để tải thư viện; pip lỗi encoding đường dẫn tiếng Việt được khắc phục với `-X utf8`.
- Bước tiếp theo: người dùng chạy bundle và notebook trên Kaggle theo `docs/G0_KAGGLE.md`, gửi notebook có output cùng log; review chéo trước đóng G0. Không làm lại phần local đã PASS nếu không thay đổi môi trường/code liên quan.
