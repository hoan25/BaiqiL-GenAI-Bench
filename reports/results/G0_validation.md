# G0 — Môi trường (cập nhật sau thiết lập .venv, chưa nghiệm thu toàn mốc)

- Ngày kiểm tra: 2026-10-20 (Asia/Saigon).
- Người thực hiện: Codex; người review/xác nhận: chưa có.
- Commit triển khai: `970e304`, đã push nhánh main lên `https://github.com/hoan25/BaiqiL-GenAI-Bench`. Commit tài liệu tiếp theo cập nhật checkpoint, không thay logic đã kiểm thử.
- Dataset/split: chưa áp dụng; chỉ smoke pretrained CLIP, không train/evaluate dataset. Model: `openai/clip-vit-base-patch32`, revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`.
- Môi trường: Windows 10, CPython 3.13.5 x64 trong `.venv`, Node v24.11.1, npm 11.6.2. Python MSYS2 trên PATH không dùng cho dự án.
- Lệnh test mới nhất: `.venv-repro\Scripts\python.exe -X utf8 -m pytest tests/unit -q -p no:cacheprovider`, qua runner G0 và mã cell notebook.
- Unit tests môi trường sạch: tổng 10, PASS 10, FAIL 0, SKIP 0; [pytest log](G0_notebook_local/pytest.log). Bộ test sau bổ sung luồng GitHub/Kaggle: **13 PASS, 0 FAIL, 0 SKIP**, [log](G0_github_workflow_tests.txt). Không chạy lại test chỉ vì phiên bị ngắt quota.
- Kiểm tra môi trường: `.venv\Scripts\python.exe -X utf8 scripts/check_environment.py --output reports/results/environment-venv.json` trả exit 0; torch 2.8.0+cpu, transformers 4.57.6, huggingface-hub 0.36.0, pytest 8.4.2; import CLIPModel/CLIPProcessor và pip check PASS.
- Bằng chứng hiện tại: [environment JSON](environment-venv.json), [31 package versions](environment.txt), [Git hygiene](G0_git_hygiene.json), [CLIP smoke](G0_clip_cpu.json), [smoke exit code](G0_clip_cpu_process.json).
- Bằng chứng lịch sử trước khi cài: [bootstrap tests](G0_bootstrap_tests.txt), [môi trường MSYS2 thiếu thư viện](environment.json). Không ghi đè bằng chứng cũ.
- Bằng chứng phiên tải CLIP: [revision/checksums](clip_download.json), [CPU smoke PASS](G0_clip_cpu_pass.json), [31/31 package tái tạo](G0_repro_lock_comparison.json), [lệnh chạy môi trường sạch](G0_notebook_local/commands.json), [auto fallback](G0_notebook_local/clip_auto.json), [checkpoint hygiene](G0_checkpoint_hygiene.json), [bundle](G0_bundle.json).

| Test case theo TEST_PLAN | Trạng thái | Lý do |
|---|---|---|
| ENV-01 | PASS | Import thư viện, CLIPModel/CLIPProcessor, pip check thành công trong .venv |
| ENV-02 | PASS | Checkpoint thật chạy offline CPU; text/image embedding [1,512], finite |
| ENV-03 | PASS | ZIP Kaggle commit f48f135: cuda_available=true, device=cuda; CPU smoke cũng PASS; notebook liệt kê Tesla T4 |
| ENV-04 | PASS | Môi trường sạch từ lock khớp 31/31 package; import, pip check, 10 test đạt |
| ENV-05 | PASS | Các đường dẫn nhạy cảm/checkpoint/môi trường được ignore; đã commit/push source và bằng chứng nhỏ, không upload dataset/checkpoint |
| ENV-06 | BLOCKED | Notebook gửi lại chạy counts 1–5 nhưng thêm bootstrap ngoài repo; code sửa cần xác nhận bằng 00b_bootstrap_check.ipynb |

- Quyết định: **BLOCKED** ở gate G0; không chuyển G1.
- Lỗi đã xử lý: chọn CPython chuẩn thay MSYS2; dùng quyền mạng để tải thư viện; pip lỗi encoding đường dẫn tiếng Việt được khắc phục với `-X utf8`.
- Bằng chứng mới: `G0_kaggle_review.json` xác nhận ZIP CRC, 8 checksum CLIP, commit, 13 test và CPU/CUDA; `G0_kaggle_packages.txt` lưu versions. Log có startup error sitecustomize/wrapt, nên ENV-01 trên Kaggle chưa sạch dù import và pip dependency check thành công. ENV-01 local vẫn PASS.
- Bản sửa: 16 test PASS trong `G0_bootstrap_fix_tests.txt`; bootstrap thực local PASS sau sửa UTF-8, giữ log cả lần lỗi và lần thành công trong `G0_bootstrap_local_validation.json`.
- Bước tiếp theo: người dùng chạy **00b_bootstrap_check.ipynb**, gửi ZIP nhỏ và notebook có output; không lặp lại CPU/GPU inference đã PASS. Review chéo còn chờ.
