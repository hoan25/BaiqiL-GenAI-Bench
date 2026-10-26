# PromptAlign AI — Nhật ký tiến độ xuyên suốt

## Checkpoint hiện tại

- Ngày cập nhật: **2026-10-26**, múi giờ Asia/Saigon.
- Đã nhận và kiểm training Kaggle thật: **9.598 cặp, 3 seed PASS**, CPU replay 9 head PASS; MLP seed 42 được chọn, validation MAE 0,773777. Dữ liệu và quyết định loại/giữ đã được người dùng xác nhận; xem reports/DATA_DECISIONS.md.
- G4/G5 đã tích hợp checkpoint thật: smoke HTTP, 17 test liên quan và browser E2E PASS; benchmark CPU batch 1/6 và concurrency 5 PASS. Không cần chạy lại QA/training Kaggle. Tiếp theo cần nhóm UAT và ≥20 ảnh mới; peak RAM/holdout/release chưa nghiệm thu.
- **ENV-01/02/04/05 PASS local**. `.venv-repro` CPython 3.13.5 Windows x64 tái tạo đúng 31 package; torch 2.8.0+cpu, transformers 4.57.6, huggingface-hub 0.36.0, pytest 8.4.2; import CLIP và pip check đạt. **10 pytest PASS**, CPU smoke và auto fallback PASS.
- Luồng mới: VS Code → GitHub `hoan25/BaiqiL-GenAI-Bench` → Kaggle → artifact về máy. Notebook clone GitHub và xuất `g0-<thời_gian>.zip`; không dùng bundle upload trong luồng chính.
- Người dùng đã gửi ZIP và notebook Kaggle chạy commit `f48f135`: 13 test PASS, CPU/CUDA PASS, đủ checksum. Bản sửa bootstrap có 16 test local PASS và phép thử venv thật PASS. ENV-06 được hoãn đến tích hợp GPU; hiện không yêu cầu người dùng chạy 00b.
- Checkpoint thật local: `.cache/reviewed-training-20261020T174855207187Z/training-seed42`. Model/hash/tokenizer và receipt được kiểm khi load. Các mục nhật ký cũ bên dưới giữ nguyên để truy vết; checkpoint này thay thế chỉ dẫn tiếp tục cũ.

## Quy tắc khôi phục sau hết quota / phiên mới

1. Đọc đầy đủ `PROJECT_PLAN.md`, `TEST_PLAN.md`, file này và `AGENTS.md`.
2. Kiểm tra file thực tế, Git status/diff nếu đã khởi tạo Git, môi trường và bằng chứng trong `reports/results/`. Đối chiếu cả thay đổi chưa ghi vào nhật ký do phiên trước bị ngắt.
3. Xác định nhiệm vụ chưa nghiệm thu đầu tiên theo thứ tự phụ thuộc. Không suy ra hoàn thành chỉ từ việc file tồn tại hoặc kế hoạch ghi đã làm trên Kaggle.
4. Giữ nguyên phần đã hoàn thành và kiểm thử đạt. Chỉ kiểm thử lại khi có thay đổi liên quan, nghi ngờ hồi quy hoặc thiếu bằng chứng tương ứng với trạng thái hiện tại.
5. Sau mỗi nhóm nhiệm vụ, thêm mục nhật ký và cập nhật checkpoint, bằng chứng, blocker, nhiệm vụ tiếp theo. Ghi sớm trước khi dừng; không thể bảo đảm ghi thêm sau khi quota đã hết.

## Trạng thái các mốc

| Mốc | Trạng thái | Bằng chứng / phần còn thiếu |
|---|---|---|
| G0 | LOCAL/KAGGLE execution PASS; review toàn mốc còn chờ | Bootstrap sạch, GPU T4, pipeline training 30 test PASS |
| G1 | FULL-DATA checks PASS; quyết định dữ liệu đã xác nhận | 12 shard/1.600 prompt, 9.598 cặp; lock reviewer hoan; giải trình trong DATA_DECISIONS.md |
| G2 | FULL-DATA extraction PASS | 9.598×512 vectors, frozen fingerprint; baseline validation có số liệu thật |
| G3 | FULL-DATA training/CPU replay PASS | 3 seed, 9 head, scaler train-only; chọn MLP seed 42; holdout chưa đánh giá |
| G4 | REAL MODEL HTTP PASS; full acceptance pending | Chấm/so sánh/metrics qua CLIP + MLP thật; còn concurrency/performance/UAT |
| G5 | REAL MODEL E2E PASS; full acceptance pending | Browser upload/score/compare/dashboard với checkpoint thật PASS; còn UAT và ảnh ngoài dataset |
| G6 | NOT RUN | Optional; chỉ sau MVP |
| G7 | NOT RUN | Chưa đủ điều kiện release |

## 2026-10-19 — Khảo sát và bootstrap G0

### Trạng thái thực tế trước khi thay đổi

- Workspace chỉ chứa `PROJECT_PLAN.md` và `TEST_PLAN.md`; đã đọc đầy đủ cả hai.
- Chưa có README, source, test, PROGRESS, artifact, notebook hoặc `.git`. `git status --short` báo `not a git repository`.
- Các kết quả shard mẫu được nhắc trong kế hoạch không có artifact ở đây để kiểm chứng; không coi là PASS và không làm lại xử lý shard trong phiên này.
- Python 3.11.9 tại `C:\msys64\ucrt64\bin\python.exe`; Node v24.11.1; npm.cmd 11.6.2. `py -0p` không tìm thấy Python đã đăng ký.

### File tạo mới

- `AGENTS.md`: quy trình bắt buộc đọc kế hoạch/nhật ký và khôi phục tiến độ.
- `docs/PROGRESS.md`: checkpoint, trạng thái gate và nhật ký này.
- `README.md`: lệnh kiểm tra, phạm vi bootstrap và hạn chế môi trường.
- `.gitignore`: loại secrets, môi trường, cache, Parquet và checkpoint khỏi Git về sau.
- `scripts/check_environment.py`: kiểm tra import thư viện/symbol CLIP và pip check, lưu JSON; không tải model hoặc dữ liệu.
- `tests/unit/test_environment.py`: 4 test bootstrap dùng unittest, kiểm import thành công, thiếu thư viện, thiếu symbol và conflict pip.
- `reports/results/environment.json`, `reports/results/G0_bootstrap_tests.txt`, `reports/results/G0_validation.md`: bằng chứng và biên bản.

### Kiểm thử thực tế

| Lệnh / kiểm tra | Kết quả |
|---|---|
| `python -m pip check` | PASS — No broken requirements found; không chứng minh đã cài thư viện cần thiết |
| `python -m pytest --version` | FAIL — No module named pytest |
| `python scripts/check_environment.py --output reports/results/environment.json` | FAIL, exit 1 — thiếu torch, transformers, huggingface_hub, pytest; pip check PASS |
| `python -m unittest discover -s tests/unit -p "test_*.py" -v` | PASS — 4 passed, 0 failed, 0 skipped; log tại `reports/results/G0_bootstrap_tests.txt` |
| `npm.cmd --version` | PASS — 11.6.2 |

Test bootstrap xác minh script phát hiện lỗi đúng; không thay thế ENV-01–06 và không chứng minh model hoạt động.

### Vấn đề và quyết định

- `npm` gọi `npm.ps1` bị ExecutionPolicy chặn; dùng `npm.cmd` thành công, không thay policy hệ thống.
- Thiếu thư viện bắt buộc: ENV-01 FAIL; các ca inference CPU, môi trường sạch, Kaggle/notebook chưa chạy.
- ENV-05 mới có `.gitignore`; chưa có Git để kiểm chứng trạng thái tracking.
- Kế hoạch yêu cầu commit và người khác review cho DONE, trong khi yêu cầu người dùng không cho tự commit/push. Không tự commit, không giả lập review; ghi nhận phần này chưa nghiệm thu.
- PROJECT_PLAN yêu cầu import trên Kaggle ở gate G0; TEST_PLAN ghi GPU tùy chọn nhưng yêu cầu ENV-01–06. Chưa có bằng chứng Kaggle/clean notebook, nên giữ G0 BLOCKED, không bỏ qua gate.
- Chưa tải dataset/checkpoint, chưa huấn luyện, không có metric thực nghiệm.

### Thứ tự tiếp tục G0

1. ENV-01/04: chuẩn bị môi trường riêng và requirements khóa phiên bản, xác minh import và test runner; lưu phiên bản môi trường.
2. ENV-02: chuẩn bị fixture nhỏ và smoke CLIP CPU; ghi rõ tài nguyên model cần dùng trước khi tải checkpoint lớn.
3. ENV-03/06: chuẩn bị notebook tái sử dụng script; chạy clean kernel trên Kaggle khi có quyền truy cập và lưu log thực tế.
4. ENV-05: khởi tạo/kiểm tra Git và hygiene, không commit/push khi chưa được yêu cầu.
5. Tổng hợp ENV-01–06 và review chéo để xét gate G0; chỉ sau đó chuyển G1.

## 2026-10-19 — Tiếp tục ENV-01/04; dừng tại trở ngại ENV-02

- Đã đọc lại kế hoạch, test plan, AGENTS và nhật ký; kiểm tra file và Git thực tế. Chưa có Git repository.
- Python trên PATH là MSYS2 với platform tag `mingw_x86_64_ucrt`. Wheel C extension của MSYS2 không tương thích CPython chuẩn: [tài liệu MSYS2](https://www.msys2.org/docs/python/).
- Tìm thấy CPython 3.13.5 tại `C:\Users\20146\AppData\Local\Programs\Python\Python313\python.exe`, platform `win-amd64`; đã tạo `.venv` thành công. Dùng Python 3.13 có sẵn thay vì cài thêm Python 3.11; import thực tế đã đạt.
- Mạng trong sandbox báo `WinError 10013`. Kiểm tra/cài với quyền mạng thành công; đã cài PyTorch CPU từ `https://download.pytorch.org/whl/cpu` vào `.venv`. Chỉ tải thư viện, không tải checkpoint/dataset.
- Pip lần đầu gặp `UnicodeEncodeError` với đường dẫn tiếng Việt và kết thúc exit 1 dù packages đã cài. Chạy lại bằng `python -X utf8 -m pip ... --progress-bar off` thành công exit 0; không nâng dependency toàn cục.
- Thêm `requirements-g0.in` khóa dependencies trực tiếp, `requirements-g0-win-cpu.lock.txt` ghi đủ 31 package đã cài (kể cả pip), `pyproject.toml` cấu hình pytest. Lock này chỉ cho Windows CPU/CPython 3.13; ENV-04 vẫn NOT RUN vì chưa tái tạo môi trường thứ hai từ lock.
- Thêm `scripts/smoke_clip.py`: chỉ chạy inference CPU với checkpoint local/cache; fixture ảnh vuông đỏ ghi nhãn synthetic, không tạo metric chất lượng giả.
- Thêm `notebooks/00_environment.ipynb`: tái sử dụng script môi trường/test/smoke; đã kiểm tra cú pháp cell, chưa thực thi notebook. Không coi kiểm tra cú pháp là ENV-06 PASS.
- Khởi tạo Git, không stage/commit/push; `git ls-files` rỗng. Kiểm tra 10 đường dẫn mẫu nhạy cảm/dữ liệu/môi trường/checkpoint đều bị ignore; ENV-05 PASS cho trạng thái hiện tại.
- Cập nhật README với Python đúng, UTF-8, smoke và notebook; `.gitignore` thêm môi trường `.venv-*`.

### Lệnh và bằng chứng

| Lệnh / kiểm tra | Kết quả | Bằng chứng |
|---|---|---|
| `.venv\Scripts\python.exe -X utf8 scripts/check_environment.py --output reports/results/environment-venv.json` | PASS / exit 0; toàn bộ import và pip check đạt | `reports/results/environment-venv.json` |
| `.venv\Scripts\python.exe -X utf8 -m pytest tests/unit -q` | 4 PASS, 0 FAIL, 0 SKIP | `reports/results/G0_pytest.txt` |
| `.venv\Scripts\python.exe -X utf8 scripts/smoke_clip.py --output reports/results/G0_clip_cpu.json` | BLOCKED / exit 2; thiếu tài nguyên CLIP local, không load được processor | `reports/results/G0_clip_cpu.json` |
| `git check-ignore --no-index ...`, `git ls-files`, `git status --short` | PASS cho 10 đường dẫn mẫu; không track file nào | `reports/results/G0_git_hygiene.json` |
| `pip freeze --all` | Ghi 31 package vào lock | `requirements-g0-win-cpu.lock.txt`, `reports/results/environment.txt` |

Log cài: `reports/results/G0_torch_install.log`, `G0_dependencies_install.log` (lần lỗi encoding), `G0_dependencies_utf8_install.log` (exit 0). Không có tiến trình cài đặt còn chạy khi kết thúc phiên.

**Trở ngại cần giải quyết tiếp:** CLIP checkpoint/processor local chưa có; chưa có phiên Kaggle hoặc log Run All; chưa có review chéo. G0 chưa PASS, chưa bắt đầu G1. Không coi lock đã xuất là bằng chứng tái tạo ENV-04.

## 2026-10-19 — Tải CLIP được cho phép, tiếp tục G0

- Người dùng cho phép tải CLIP và tiếp tục G0; đã ghi ngoại lệ này trong AGENTS.md để không hỏi lại.
- Tải 8 file cần thiết từ `openai/clip-vit-base-patch32`, revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`, vào `.cache/models/clip-vit-base-patch32/<revision>/`. Trọng số `pytorch_model.bin` 605.247.071 byte. Không tải dataset hoặc bản TensorFlow/Flax.
- Revision, đường dẫn, kích thước và SHA-256: `reports/results/clip_download.json`. Script: `scripts/download_clip.py`.
- ENV-02 **PASS**, exit 0: `scripts/smoke_clip.py --model .cache/models/clip-vit-base-patch32/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268 --output reports/results/G0_clip_cpu_pass.json` bằng `.venv\Scripts\python.exe -X utf8`. Text/image embedding `[1,512]`, không NaN/Inf, fixture tổng hợp chỉ kiểm tra pipeline.
- Đã thêm lựa chọn thiết bị cpu/cuda/auto; auto fallback về CPU khi không có CUDA, yêu cầu CUDA tường minh phải báo lỗi nếu không có. 6 test mới PASS, 0 FAIL: `python -m pytest tests/unit/test_smoke_device.py -q`.
- Đã cài lock đầy đủ vào `.venv-repro` sạch, exit 0; log `reports/results/G0_repro_install.log`. ENV-04 **PASS**: đối chiếu đúng 31/31 package, không thiếu/thừa (`reports/results/G0_repro_lock_comparison.json`), import/pip check và pytest thực tế đạt.
- Chuẩn bị `scripts/run_g0_checks.py`, cập nhật notebook, thêm `scripts/package_g0.py` và `docs/G0_KAGGLE.md` để bàn giao chạy Kaggle. Notebook thực tế chưa chạy trên Kaggle.
- Chạy mã cell notebook bằng tiến trình Python mới trong `.venv-repro`: toàn bộ lệnh environment, pytest, clip_cpu, clip_auto, versions đều exit 0. **10 PASS, 0 FAIL, 0 SKIP**; CPU và auto smoke PASS, `cuda_available=false`, auto dùng CPU. Log: `reports/results/G0_notebook_local/`; biên bản process: `reports/results/G0_notebook_local_process.json`. Đây không phải Jupyter kernel run, không đánh dấu ENV-06 PASS.
- Bundle `artifacts/G0_kaggle_bundle.zip` đã tạo: 608.886.973 byte, 21 file, gồm source/test/notebook/checkpoint. Đã kiểm tra CRC ZIP và checksum checkpoint trước đóng gói; manifest/checksum ZIP: `reports/results/G0_bundle.json`.
- Kiểm tra ignore checkpoint `.bin`, bundle và `.venv-repro` PASS; `git ls-files` vẫn rỗng. Bằng chứng: `reports/results/G0_checkpoint_hygiene.json`. Không stage/commit/push hoặc upload tài khoản ngoài.
- README và `reports/results/G0_validation.md` đã cập nhật. Không có tiến trình download/cài đặt còn chạy khi kết thúc phiên.
- **Blocker còn lại:** chưa có quyền điều khiển phiên Kaggle hoặc log từ người dùng; cần notebook Run All trên kernel sạch và review chéo. Chưa chuyển G1.

## 2026-10-19 — Đồng bộ GitHub và chuẩn bị chạy Kaggle

- Người dùng yêu cầu commit/push lên `https://github.com/hoan25/BaiqiL-GenAI-Bench.git`; đã ghi quyền này trong AGENTS.md.
- `git ls-remote` thành công, không có refs: remote hiện trống; local có identity và credential manager.
- Notebook đổi sang clone/fetch GitHub, bảo toàn dirty checkout; hỗ trợ private repository bằng Kaggle Secrets, không lưu token vào URL.
- Thêm `scripts/kaggle_g0.py`: venv Kaggle riêng, PyTorch CUDA/CPU, tải CLIP pinned revision, kiểm tra, lưu provenance và ZIP log kể cả lỗi. Linux dependencies chưa được nghiệm thu.
- Downloader thêm destination/output và pin revision. Runner thêm `--require-cuda` để tránh báo GPU PASS khi chỉ dùng CPU.
- Cập nhật hướng dẫn và README; ignore verbose install logs và trọng số `.bin`.
- Kiểm thử: **13 PASS, 0 FAIL, 0 SKIP**; có ca lỗi setup vẫn tạo ZIP, clone mới/resume thành công và dirty checkout được giữ nguyên. Log: `reports/results/G0_github_workflow_tests.txt`.
- GitHub đang chuẩn bị commit/push, chưa khẳng định đã upload trước khi kiểm tra remote. ENV-03/06 vẫn NOT RUN trên Kaggle.

## 2026-10-19 — Khôi phục sau quota, hoàn tất cập nhật GitHub

- Đã đọc lại PROJECT_PLAN.md, TEST_PLAN.md, nhật ký; kiểm tra index, working tree, remote và lịch sử. Trước khi tiếp tục: nhánh main chưa có commit, 46 file đã stage; hướng dẫn và notebook còn bản sửa Unicode chưa stage.
- Lần commit ở phiên trước bị bộ xét duyệt tự động chặn do hết quota, không phải kết luận thao tác không an toàn; lệnh đó chưa thực thi. Phiên này đã thực thi thành công với quyền đã được cấp.
- Stage bản sửa Unicode, kiểm tra diff; giữ hai Markdown hard break có sẵn trong PROJECT_PLAN.md. Không thay đổi logic hoặc chạy lại phần đã có bằng chứng PASS (13 test trong G0_github_workflow_tests.txt).
- Tạo commit `970e304` (46 file), push `main` lên origin thành công, thiết lập tracking origin/main. Không đưa .venv, checkpoint, dataset hoặc bundle ZIP lên GitHub.
- Cập nhật checkpoint và biên bản G0 để phản ánh GitHub đã hoàn tất. Nhiệm vụ đầu tiên chưa nghiệm thu vẫn ENV-03/06: người dùng chạy Kaggle và gửi `g0-*.zip` cùng notebook có output. Review chéo còn chờ; chưa chuyển G1.

## 2026-10-20 — Kiểm tra kết quả Kaggle và sửa bootstrap

- Đã đọc kế hoạch/nhật ký và kiểm tra Git; người dùng bổ sung `reports/results/kaggle/g0-20261019T175231Z.zip` cùng `genai-bench-data-preparation.ipynb`. Giữ nguyên file gốc, không thực thi code notebook do người dùng gửi. File gốc được ignore khỏi Git, chỉ commit báo cáo kiểm chứng.
- ZIP CRC PASS, đủ 5 lệnh exit 0, commit `f48f135752e769fa5025a39896fcbb0475900e05`; revision và SHA-256 của 8 file CLIP khớp local. Notebook có execution counts 1–5, không có error output; phiên chạy 00:52 ngày 20/10 giờ Việt Nam.
- Kaggle Python 3.13.15, torch 2.8.0+cu126, 2 Tesla T4 được liệt kê. **13 test PASS**, CLIP CPU và CUDA thực sự PASS, embedding `[1,512]`. **ENV-03 PASS**. Báo cáo: `reports/results/G0_kaggle_review.json`; versions thực tế: `G0_kaggle_packages.txt`.
- Phát hiện notebook người dùng thêm cell tạo venv `--without-pip` rồi dùng pip gốc `--python` để cài pip, trong khi code GitHub cũ dùng ensurepip mặc định. Ngoài ra 9 log có `Error in sitecustomize / No module named wrapt` dù exit 0; không coi toàn bộ G0 sạch chỉ vì result.json ghi PASS.
- Sửa `scripts/kaggle_g0.py`: tạo venv không phụ thuộc ensurepip, cài pip 26.2.1 bằng pip gốc, loại PYTHONPATH/PYTHONHOME kế thừa ở subprocess, dùng isolated startup probe, phát hiện lỗi startup kể cả exit 0; không xóa venv/checkpoint cũ. Nguyên nhân sitecustomize từ đường dẫn kế thừa là giả thuyết cần xác nhận Kaggle; không tự cài wrapt để che lỗi.
- Thêm `--bootstrap-only`: venv mới có tên riêng, chỉ kiểm startup và pip check, không cài torch/tải CLIP/inference. Notebook `00b_bootstrap_check.ipynb` dùng chế độ này để chỉ nghiệm thu phần thay đổi.
- Sửa `check_environment.py` không báo PASS khi pip check chứa lỗi sitecustomize/usercustomize. Thêm regression tests cho bootstrap không dùng ensurepip, lọc env và zero-exit startup errors.
- **16 test PASS, 0 FAIL, 0 SKIP** (`G0_bootstrap_fix_tests.txt`). Phép thử bootstrap thật local lần đầu phát hiện lỗi encoding đường dẫn tiếng Việt; đã thêm UTF-8 cho subprocess. Lần sau **PASS**, venv tách biệt và pip check sạch (`G0_bootstrap_local_validation.json` lưu cả lần lỗi và lần sửa đạt).
- Không chạy lại CLIP/GPU đã đạt. **G0 vẫn BLOCKED**: cần xác nhận bootstrap sửa trên Kaggle và review; chưa bắt đầu G1.

## 2026-10-20 — Điều chỉnh lộ trình và bắt đầu G1-local

- Đọc kế hoạch/test/nhật ký, đối chiếu Git sạch tại `c61f3d7`; chưa có `src/`. Người dùng yêu cầu phát triển trong VS Code trước, tránh lặp sửa notebook.
- Sửa `PROJECT_PLAN.md`, `TEST_PLAN.md`, `AGENTS.md`, README và hướng dẫn Kaggle: tách cổng local với nghiệm thu thực nghiệm; giữ yêu cầu full-data QA trước train và toàn bộ gate trước release. Không hạ ENV-06 thành PASS.
- Thêm `src/data/validate.py`, `manifest.py`, `split.py`, `scripts/data_smoke.py`, `tests/unit/test_data.py`. Contract chuẩn hóa độc lập với Parquet; giữ ratings gốc/mean/std, nguồn shard-row-model, checksum ảnh, lý do loại mẫu. Feature input chỉ prompt + image.
- Split chuẩn hóa Unicode NFKC/casefold/khoảng trắng, nhóm cùng prompt cùng split, seed ổn định không phụ thuộc thứ tự hàng. Tối thiểu 3 nhóm; tỷ lệ trên fixture nhỏ có làm tròn, thống kê thực được ghi ra.
- Lệnh `.venv\Scripts\python.exe -X utf8 -m pytest tests/unit/test_data.py -q`: **13 PASS, 0 FAIL**, exit 0, 0.44 giây. Bằng chứng: `reports/results/G1_local_tests.txt`.
- Lệnh `.venv\Scripts\python.exe -X utf8 scripts/data_smoke.py --output reports/results/G1_fixture_smoke.json`: exit 0, 10 hàng/60 cặp tổng hợp, train/val/test = 42/6/12. Output từ chối ghi đè. Không phải số liệu GenAI-Bench hoặc metric mô hình.
- Không thay dependency hoặc chạy lại CLIP/GPU. Các test G0 đã đạt được giữ nguyên.
- **Còn thiếu:** adapter Parquet xác minh schema nguồn thật; integration test file nhỏ; báo cáo exact/near-duplicate và provenance; DATA-01–12 trên toàn bộ dữ liệu; review chéo. Nhiệm vụ tiếp theo là adapter và QA local, chưa yêu cầu chạy Kaggle.

## 2026-10-20 — Adapter Parquet và QA G1-local

- Đọc kế hoạch/test/nhật ký và kiểm tra repo sạch tại `ea6b5b1` (đã push ở phiên trước). Giữ các kiểm chứng CLIP đã đạt, không tải lại model/dataset.
- Đối chiếu schema công bố trên `https://huggingface.co/datasets/BaiqiL/GenAI-Bench/blob/main/README.md`: Prompt, HumanRatings, sáu cột ảnh. Chưa đọc file Parquet thật; không coi tài liệu nguồn là nghiệm thu dataset.
- Cài `pyarrow==21.0.0` vào `.venv`, ghi `requirements-data.in`; không thay lock G0 lịch sử. `pip check`: **PASS**, No broken requirements found.
- Thêm `src/data/ingest.py`, `qa.py`, `scripts/prepare_data.py`: đọc batch nhỏ, ID tương đối gồm shard/hàng/model, checksum file/manifest, provenance Git/package, thống kê lỗi/count/split, exact/normalized/near-duplicate report. Thiếu cột FAIL; ảnh/nhãn lỗi loại có log; run cần review trả exit 2. Không tải dataset và không ghi đè output.
- Thêm `tests/integration/test_parquet_pipeline.py`: nhiều shard/row group, prompt trùng xuyên shard, manifest tái lập, ảnh hỏng, cột thiếu, path ảnh ngoài, CLI và bảo toàn output. Lần đầu unit data + integration **18 PASS**; sau thêm provenance chạy toàn suite do thêm dependency: **34 PASS, 0 FAIL, 0 SKIP**, 3.60 giây, exit 0. Log tự lưu: `reports/results/G1_parquet_tests.txt`.
- Hướng dẫn: `docs/DATA_PIPELINE.md`. Các file Parquet kiểm thử được tạo trong thư mục tạm; không commit dữ liệu ảnh/Parquet.
- **G1-local checks PASS; toàn mốc chưa DONE:** thiếu full-data QA, review, xử lý ứng viên gần trùng và khóa split. ENV-06 sạch vẫn chờ tích hợp Kaggle. Near-duplicate là heuristic từ vựng, không chứng minh tương đồng ngữ nghĩa.
- Tiếp theo G2-local theo kế hoạch local-first: implement metrics (Spearman constant = N/A, pairwise cùng prompt), mean baseline và frozen encoder/extraction với fixture; không công bố metric thật trước QA đầy đủ.

## 2026-10-20 — G2-local metrics, encoder và extraction

- Đọc kế hoạch/test/nhật ký, xác nhận Git sạch tại `8f99c81` đã push. Không đổi dependency, không tải thêm model/dataset.
- Thêm `src/evaluation/metrics.py`, `baselines.py`: MAE/RMSE, Spearman average-rank ties/constant null, pairwise chỉ cùng prompt; M0 fit train, M1 cosine thô, validation không đọc nhãn test; chặn prompt leakage.
- Thêm frozen `src/models/clip_encoder.py` và `src/training/extract_embeddings.py`: CPU/CUDA rõ ràng, model local, batch, text truncation, đối chiếu nguồn/checksum, giữ thứ tự pair ID, không đưa ratings/generator vào encoder.
- Thêm CLI `scripts/extract_embeddings.py` yêu cầu `--development`, lưu NPZ (Git ignore), result/checksum, fingerprint model trước/sau; từ chối ghi đè run. Entry point thực nghiệm chính thức vẫn chưa có.
- 13 test mới PASS: metric tham chiếu, ties, input lỗi, leakage, nguồn stale, thứ tự manifest và batch cuối thiếu; log `reports/results/G2_local_tests.txt` (0.54 giây, exit 0). Không chạy lại suite G0/G1 vì không thay code/dependency liên quan.
- `scripts/check_encoder.py --model .cache/models/clip-vit-base-patch32/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268 --output reports/results/G2_encoder_checks.json`: PASS, batch/single max delta 1.7881393432617188e-07, [2,512], unit norm, truncation, frozen weights/no gradients. Đây là test module mới, không lặp lại G0 environment smoke.
- Tạo fixture Parquet 3 prompt/18 cặp trong `.cache/g2-pipeline-fixture`, QA CHECKS_PASS ở scope fixture. Chạy extraction CPU batch 4 từ manifest vào `reports/results/G2_fixture_extraction`: PASS, 18x512, 6.95 giây. M0 MAE=0 do toàn nhãn tổng hợp bằng 2; không dùng làm chất lượng mô hình. File NPZ giữ local và được ignore.
- Hướng dẫn `docs/EMBEDDINGS.md`; biên bản `reports/results/G2_validation.md`. **G2-local checks PASS**, full-data/GPU/review còn chờ. Thiếu provenance Git/config/tokenizer cho thực nghiệm chính thức được ghi rõ.
- Tiếp theo G3-local: scaler/Ridge/MLP, fit chỉ train, early stopping theo val, save/load và overfit fixture. Không chạm test holdout.

## 2026-10-20 — G3-local Ridge/MLP và checkpoint

- Đã đọc kế hoạch/test/nhật ký; tiếp tục từ `73b354a`. Khi người dùng nhắc tiếp tục, có ba file training chưa track/chưa kiểm thử; đã đối chiếu và hoàn thiện, không viết lại G0–G2.
- Thêm `src/models/regression_head.py`, `src/training/train_regression.py`, `scripts/train_regression.py`: nối features text/image, scaler chỉ train, Ridge L2/intercept không penalty, MLP ReLU/sigmoid đầu ra 1–5; Adam full-batch CPU, seed, chọn best epoch theo val MAE, early stopping. Mean/Ridge clamp [1,5]; chọn mean/Ridge/MLP theo cùng val, không ép MLP thắng.
- Checkpoint chứa config/scaler/weights, load CPU `weights_only=True`, không ghi đè. CLI xác minh checksum manifest/embedding từ G2, lưu config/Git/versions/history/checksum/reload delta; chỉ cho chạy `--development`.
- `tests/unit/test_regression.py`: **9 PASS**, 9.63 giây. Overfit 32 mẫu tổng hợp đạt ngưỡng test; Ridge so lời giải tham chiếu; scaler không fit val, không đọc test target/features; seed/best-epoch/reload/range/invalid config đạt.
- CLI dùng lại embeddings G2, không chạy lại CLIP: `python scripts/train_regression.py --manifest .cache/g2-pipeline-fixture/qa/manifest.jsonl --extraction reports/results/G2_fixture_extraction --output reports/results/G3_fixture_training --epochs 30 --patience 5 --development`: **PASS**, exit 0. Cả ba reload delta = 0; mean được chọn trên fixture toàn nhãn 2, không phải metric GenAI-Bench.
- Thêm CLI integration kiểm checksum sai → FAIL, train/reload thành công và output không bị ghi đè. Unit + integration **10 PASS, 0 FAIL, 0 SKIP**, 22.41 giây, exit 0; log `reports/results/G3_local_tests.txt`. Không đổi dependency hoặc chạy lại các test G0–G2 đã đạt.
- Hướng dẫn `docs/TRAINING.md`, biên bản `reports/results/G3_validation.md`. Chưa full-data train, review/multi-seed trên dataset; không đánh dấu toàn mốc DONE. Tiếp theo predictor/HTTP contract G4-local; checkpoint thật và gate Kaggle vẫn chờ.

## 2026-10-20 — Checkpoint đang triển khai G4/G5-local

- Người dùng yêu cầu tiếp tục tự động đến khi cần họ hoặc hết quota. Repo bắt đầu sạch tại `3a1310b`; đã đọc lại kế hoạch/test/nhật ký.
- Thêm predictor nối CLIP/head, checksum/fingerprint, chỉ cho checkpoint fixture khi bật development rõ ràng. API mặc định không cấu hình model trả 503, không fake score.
- Thêm backend routes health/evaluate/compare/metrics/sessions/uploads/chat; session riêng TTL/giới hạn RAM, validate bytes/dimensions, request size kể cả chunked, lỗi an toàn. Cài API deps vào .venv, `pip check` PASS.
- API test lần đầu có lỗi pytest setup do parameter ID chứa 10 MB payload làm vượt giới hạn biến môi trường Windows; sửa ID ngắn. Sau sửa **10 test PASS**; thêm test chunked đang chờ chạy cùng suite cuối. Có một DeprecationWarning Starlette/AnyIO, không ảnh hưởng kết quả.
- `check_predictor_api.py` dùng CLIP thật và checkpoint G3 tổng hợp: **PASS**, evidence `reports/results/G4_predictor_http.json`. Không coi đây là model/dataset acceptance.
- Thêm frontend React/Vite, upload/preview/chat/ranking/metrics/error/empty states; build **PASS**. Cài browser trong `.cache/playwright`; lệnh đầu fail do PowerShell Get-Location không có Parent và npx.cmd không tồn tại, đã đổi sang Split-Path + Node CLI. E2E desktop/mobile đang chuẩn bị chạy, chưa đánh dấu PASS.
- Hướng dẫn local: `docs/WEB_LOCAL.md`. Chưa commit/push nhóm G4/G5 tại checkpoint này; kiểm Git thực tế trước khi resume.

## 2026-10-20 — Hoàn tất kiểm thử G4/G5-local và chuẩn bị handoff dữ liệu

- Toàn suite Python sau thêm API dependencies: **68 PASS**, 40.66 giây, exit 0, 1 cảnh báo deprecation Starlette/AnyIO (`G4_python_tests.txt`). Sau chỉnh follow-up giữ index ảnh thứ hai và thêm runner QA: **14 test liên quan PASS**, 1.55 giây (`G4_handoff_tests.txt`).
- React build PASS; Chromium E2E **10 PASS**, 15.2 giây (`G5_browser_tests.txt`). Test desktop/mobile dùng HTTP mock chỉ trong test. Đã xem ảnh `reports/figures/G5-desktop.png`, `G5-mobile.png`; giao diện không tràn ngang. Vite cần quyền ngoài sandbox để esbuild đọc parent; không phải lỗi source.
- Thêm `scripts/download_dataset.py` chỉ tải khi có --allow-download; pin revision từ HF trước download. Agent chưa tải toàn bộ dataset. Người dùng chạy cell DOWNLOAD_DATASET=True mới thực hiện download ở Kaggle.
- `scripts/kaggle_data.py` dựng môi trường CPU riêng chỉ cho data, startup/pip check/tests, QA, đóng gói ZIP cả khi FAIL; REVIEW_REQUIRED không bị hiểu nhầm là nghiệm thu/train PASS. Không đóng gói raw images/Parquet/checkpoint; không chạy lại CLIP.
- `notebooks/01_data_qa.ipynb` tái sử dụng cell GitHub đã kiểm thử, cell sau chỉ gọi runner. Cú pháp cell được compile local; notebook chưa chạy trên Kaggle. Hướng dẫn `docs/KAGGLE_DATA_QA.md`.
- Runner unit tests 3 PASS; `Kaggle_data_runner_tests.txt`. Test venv thật local lần đầu gặp ownership Git (đổi user do escalation); lần sau bootstrap/install/pip check đạt nhưng pytest temp chung bị từ chối quyền. Sửa dùng --basetemp riêng dưới venv và tắt pytest cache; không xóa/ghi đè run cũ. Đang kiểm chứng lại bản sửa, chưa coi Linux/Kaggle đã PASS.
- G4/G5 toàn mốc còn thiếu artifact nghiệm thu, full-stack E2E/model thật, concurrency/performance sâu và review/UAT. Hướng dẫn web `docs/WEB_LOCAL.md`; biên bản `reports/results/G4_G5_validation.md`.
- Kiểm chứng runner sau sửa: venv sạch thứ ba chạy **thành công**, exit 0; 18 data tests PASS, pip check/startup sạch, QA fixture trả REVIEW_REQUIRED do subset. ZIP CRC PASS, không chứa Parquet. Evidence lưu cả hai lần lỗi và lần sửa trong `reports/results/Kaggle_data_local_validation.json`; không ghi đè bằng chứng cũ. Không coi lần local này là ENV-06 Kaggle PASS.

## 2026-10-20 — Kiểm tra notebook Kaggle lỗi bootstrap_pip

- Đọc lại kế hoạch/test/nhật ký, repo sạch tại `76276cd`. Đọc JSON notebook người dùng ở `C:/Users/20146/Downloads/genai-bench-data-preparation.ipynb`; không thực thi code đính kèm, không sửa file gốc.
- Notebook chạy hai code cell, clone/fetch đúng `76276cd32977db185ec874920924239521a92fcd`. Runner in create_venv rồi bootstrap_pip, FAIL, exit 1; ZIP `data-qa-20261020T060426867085Z.zip`. Chưa đến dependencies/data download/QA.
- Output notebook không có chi tiết bootstrap_pip.log. Kiểm thư mục Downloads và reports/results/kaggle không thấy ZIP lần này. **Nguyên nhân gốc UNKNOWN**, không suy đoán là pip version/Internet/sitecustomize khi chưa có log.
- Sửa lỗi quan sát: `scripts/kaggle_data.py` in stage, returncode, tail log (giới hạn 6000 ký tự, che credential phổ biến) ngay ra notebook; lưu các trường tương ứng trong result.json. ZIP vẫn giữ log đầy đủ, không che lỗi startup exit 0.
- Regression tests: **6 PASS**, 0.64 giây, exit 0; `reports/results/Kaggle_data_diagnostic_tests.txt`. Có ca bootstrap pip fail, startup error exit 0 và redaction/giới hạn log. Chỉ chạy test runner liên quan.
- Evidence hash/nhận xét notebook: `reports/results/Kaggle_bootstrap_notebook_review.json`. Chưa đánh dấu Kaggle bootstrap PASS. Cần người dùng gửi ZIP đã tạo (không cần chạy lại để lấy lỗi cũ), hoặc chạy Run All để code main mới hiện lỗi trực tiếp.

## 2026-10-20 — Sửa lỗi startup chung cho QA Kaggle và G0 từ ZIP thật

- Repo bắt đầu sạch tại `52248a1`; đối chiếu PROJECT_PLAN.md, TEST_PLAN.md, nhật ký và source thực tế. Đọc ZIP người dùng `C:/Users/20146/Downloads/data-qa-20261020T060426867085Z.zip` bằng zipfile, CRC PASS; không thực thi nội dung đính kèm.
- **Nguyên nhân đã có bằng chứng:** bootstrap_pip exit 0, pip 26.2.1 cài thành công, nhưng log có `Error in sitecustomize` / thiếu `wrapt`. Runner từ chối đúng lỗi startup; chưa đến dependencies/download/QA. ZIP không ghi đường dẫn hook nên không xác định được file hook cụ thể. Thay kết luận UNKNOWN trước đó bằng chẩn đoán này, giữ nguyên bằng chứng lịch sử.
- Thêm `scripts/clean_python.py`: interpreter `-I -S`, gọi site.main để giữ đường dẫn venv và package, bỏ hai customization hook. Bootstrap chỉ nạp package pip của interpreter notebook vào interpreter đích, không thêm toàn bộ site-packages cha và không dùng pip --python tái khởi động thiếu cờ cách ly. Dùng chung trong kaggle_data/kaggle_g0 và subprocess G0/check_environment; CLI integration dùng cùng launcher.
- Thêm `requirements-kaggle-data.in` gom đủ dependency CPU QA; runner kiểm import PyArrow/Pillow/HF Hub/pytest sau pip check, lưu versions trước test/download. Không cần torch/CUDA/API/frontend cho notebook QA. Hướng dẫn cập nhật ở docs/KAGGLE_DATA_QA.md; notebook hiện có vẫn fetch main khi Run All.
- Lệnh `.venv/Scripts/python.exe -X utf8 -m pytest tests/unit/test_clean_python.py tests/unit/test_kaggle_data.py tests/unit/test_kaggle_export.py tests/unit/test_environment.py tests/integration/test_parquet_pipeline.py -q -p no:cacheprovider --basetemp .cache/pytest-bootstrap-fix`: **21 PASS, 6.43 giây**, exit 0. Ca mới tạo venv thật và hook thiếu wrapt: Python thường báo lỗi dù exit 0; launcher sạch, giữ đúng sys.prefix/site-packages, import package và chạy pip bootstrap được.
- `python scripts/kaggle_data.py --workdir .cache/data-bootstrap-fix --data-root .cache/g2-pipeline-fixture`: exit 0; venv mới cài dependency/import/pip check PASS; **18 data tests PASS, 2.09 giây**. QA fixture REVIEW_REQUIRED do subset, không coi là full-data acceptance. ZIP CRC PASS, path `.cache/data-bootstrap-fix/data-qa-20261020T164710807775Z.zip`.
- `python scripts/kaggle_g0.py --workdir .cache/g0-bootstrap-fix --bootstrap-only`: **PASS**, exit 0; `.cache/g0-bootstrap-fix/g0-20261020T164745206619Z.zip`, CRC PASS. Không tải CLIP hoặc chạy lại inference/GPU.
- `.venv/Scripts/python.exe -X utf8 -I -S scripts/clean_python.py scripts/check_environment.py --output reports/results/Kaggle_clean_launcher_ml_imports.json`: **PASS**, import torch/CLIPModel/CLIPProcessor/HF Hub/pytest và pip check qua launcher trên môi trường local hiện có. Không chứng minh CUDA mới.
- Bằng chứng tổng hợp/hash ZIP/source/logs: `reports/results/Kaggle_bootstrap_fix_validation.json`. Runs dùng working tree bản sửa trên base 52248a1; report ghi rõ base commit chưa chứa fix, kèm source hashes. `git diff --check` PASS.
- **Kaggle bản sửa: NOT RUN; full-data QA: NOT RUN.** Nhiệm vụ chưa nghiệm thu đầu tiên: người dùng Run All notebook QA hiện có sau khi source được push, gửi ZIP mới; kiểm startup sạch và QA 12 shard, xử lý near-duplicate/rejections trước khóa split/extraction/train chính thức. Agent chưa tải dataset đầy đủ. Source/docs/evidence chuẩn bị commit/push về đúng origin được người dùng cho phép.

## 2026-10-21 — Nhận full-data QA Kaggle thành công và sửa hiển thị tải ZIP

- Đọc kế hoạch/test/nhật ký và AGENTS.md; repo source tại `7b621d1`. Có các ZIP/notebook người dùng mới lưu trong G0_notebook_local và G3_fixture_training; giữ nguyên, không stage chúng hoặc checkpoint fixture cũ.
- Đọc `reports/results/G3_fixture_training/data-qa-20261020T165712283760Z.zip` và notebook kèm output bằng JSON/zipfile, không chạy code đính kèm. CRC PASS. Kaggle dùng đúng 7b621d1, working tree sạch; Python 3.13.15; bootstrap/dependencies/import/pip check đều PASS, không còn lỗi customization hook. **18 data tests PASS, 0.66 giây**.
- Đọc đủ **12 shard / 1.600 prompt** revision `94e543b0801feade1419873335191de90b68d164`. Giữ **9.598 cặp**, loại 2 do rating không hợp lệ. Hai ứng viên prompt gần trùng đều ở train; normalized leakage 0. REVIEW_REQUIRED là chờ review dữ liệu, không phải lỗi thực thi.
- Kiểm độc lập qua `.venv/Scripts/python.exe -X utf8 .cache/review_received_qa.py`: exit 0; CRC/hash manifest/ID/mapping/ratings/mean/std/counts và split đảo thứ tự seed 42 **PASS**. 0 nhóm ảnh trùng checksum. Train/val/test = 6.718/1.440/1.440 cặp. Không đọc lại raw dataset, chạy CLIP/training hoặc tính holdout metrics.
- Biên bản `reports/results/G1_kaggle_full_qa_review.md` và JSON cùng tên lưu provenance/logs/kết quả audit. Đề xuất giữ loại hai nhãn lỗi; giữ prompt head/paw khác nghĩa và hai prompt chỉ khác dấu phẩy trong cùng train; giữ manifest hash `4ed1e3c7909c81c239beaa2890f4adbcaef7545217e1d309a8b023e084182ca2` làm ứng viên khóa. Không tự nhận đã có review chéo.
- 404 được xác nhận từ HTML output: FileLink tạo href đường dẫn filesystem `/kaggle/working/...`. Sửa `notebooks/01_data_qa.ipynb` bỏ link đó, in tên/kích thước ZIP và tải qua Output / Files; cập nhật docs/KAGGLE_DATA_QA.md. Hai code cell compile **PASS**; không chạy lại pipeline đã đạt chỉ vì đổi hiển thị. UI Kaggle bản sửa NOT RUN.
- **Tiếp theo:** review quyết định loại mẫu/near-duplicate, khóa manifest chính thức rồi tích hợp runner GPU extraction/training. Không cần người dùng chạy lại QA hiện có. Full G0/G1 còn review; chưa có kết quả training dữ liệu thật. Commit/push chỉ source/docs/biên bản, không ZIP/notebook output/manifest raw.

## 2026-10-21 — Checkpoint tích hợp extraction/training Kaggle

- Người dùng yêu cầu tiếp tục đến khi cần họ/quota. Repo base `a6e33e0`, chỉ có các ZIP/notebook người dùng chưa track; giữ nguyên. Đọc lại kế hoạch/test/nhật ký, tiếp tục từ review/split và tích hợp ML thật.
- Thêm `configs/genai_bench_review.json` chốt checksum ZIP/manifest/shards/revision và ba quyết định dữ liệu từ biên bản. `src/data/acceptance.py` đọc đúng archive đã review, cần tên reviewer rõ ràng mới tạo lock; kiểm manifest/spec/shards không đổi, không extract đường dẫn ZIP tùy ý. Chưa tạo approval chính thức thay người review.
- CLI extraction/training thêm chế độ `--data-lock` tách với `--development`; extraction kiểm checksum shard và lưu hash các file model/tokenizer, Git; training kiểm lock trùng provenance extraction. Dataset downloader hỗ trợ revision đã chốt.
- `scripts/kaggle_train.py` dựng venv sạch theo launcher đã kiểm chứng; kiểm import/GPU/tests, tái dùng cache dataset hoặc opt-in download revision cố định, CLIP frozen, 3 seed head CPU; seed 42 là run chính cố định, 43/44 báo ổn định, không chọn seed theo kết quả đẹp. ZIP chứa logs/manifest/embeddings/head/checksums, không chứa dataset hoặc CLIP weights; lỗi cũng xuất ZIP.
- Thêm `requirements-train.txt` và `notebooks/02_train.ipynb` chỉ gọi script. Notebook mặc định chưa xác nhận review; cần người review đọc biên bản, điền tên và bật xác nhận trước khi chạy. Clone cell tái sử dụng logic đã kiểm chứng; không dùng FileLink filesystem.
- 16 test acceptance/runner/training CLI/extraction **PASS, 13.68 giây**, exit 0. `read_bundle` kiểm ZIP thật khớp spec **PASS**, không tạo approval. Notebook code compile **PASS**. `pip install --dry-run --no-index -r requirements-train.txt` **PASS** với môi trường local hiện có, chưa phải môi trường GPU mới.
- Đang kiểm tra extraction CLI vừa sửa bằng CLIP cache và fixture mới dưới `.cache/handoff-fixture-extraction`, chưa nghiệm thu kết quả ở checkpoint này. Tiếp theo bổ sung kiểm gate CLI, ghi bằng chứng/hướng dẫn, commit/push rồi bàn giao user chạy GPU. Chưa chạy experiment dữ liệu thật, chưa đánh dấu G1/G2/G3 DONE.

## 2026-10-21 — Hoàn tất handoff training local, chờ reviewer và GPU

- Extraction CLI cập nhật chạy CLIP thật trên fixture: **PASS**, 18×512, frozen fingerprint không đổi, 9.71 giây; `.cache/handoff-fixture-extraction/result.json`. Training nối kết quả này qua clean launcher: **PASS**, 5 epoch fixture, cả ba head reload max delta 0; `.cache/handoff-fixture-training/result.json`. Metric tổng hợp không phải chất lượng GenAI-Bench.
- Gate CLI chính thức với lock chưa duyệt: **2 PASS, 11.44 giây**. Bổ sung nhận QA ZIP hoặc thư mục giải nén, kiểm từng member theo hash gốc; tránh phụ thuộc việc upload Kaggle còn giữ đuôi ZIP hay không. Bộ acceptance/runner/CLI gate sau bổ sung: **13 PASS, 12.06 giây**, `Kaggle_training_gate_tests.txt`.
- Spec lock hash dùng JSON canonical để Git LF/CRLF giữa Windows/Linux không làm mất khả năng xác minh cùng nội dung; dataset/manifest vẫn dùng byte hash nguyên bản. Test acceptance sau chỉnh portability: **8 PASS, 0.27 giây**, `Kaggle_lock_portability_tests.txt`; thay nội dung spec vẫn bị từ chối.
- `reports/results/Kaggle_training_handoff_validation.json` lưu kết quả thật, lệnh/chỉ số test, source hashes và giới hạn. Đọc ZIP QA thật qua gate mới **PASS**, chưa tạo lock mang tên người review. Notebook code compile PASS; `git diff --check` PASS. Không tải dataset/model mới hoặc cài GPU local.
- Cập nhật trạng thái đầu PROJECT_PLAN.md/TEST_PLAN.md và hướng dẫn `docs/KAGGLE_TRAINING.md`; `notebooks/02_train.ipynb` dành cho GPU. Các source/bằng chứng chuẩn bị commit/push về origin đã cho phép; không stage artifact người dùng, NPZ hoặc checkpoint.
- **Cần người dùng tiếp theo:** review ba quyết định cụ thể ở `G1_kaggle_full_qa_review.md`; nếu đồng ý điền REVIEWER và ACCEPT_DATA_REVIEW=True trong notebook mới, bật Internet/GPU, cung cấp QA ZIP/thư mục và Run All. Tải `training-*.zip` về để kiểm checkpoint/metrics và tích hợp web. Cần xác nhận này vì PROJECT_PLAN.md yêu cầu review trước khóa split/thực nghiệm; không tự nhận agent đã được người khác review.
- **NOT RUN:** runner mới trên Kaggle GPU, full-data extraction/training, acceptance checkpoint thật/release/holdout. G0/G1 toàn mốc không tự chuyển DONE; QA thành công đã giữ nguyên, không yêu cầu chạy lại.

## 2026-10-26 — Khôi phục phần review training và tích hợp model thật

- Repo base `2b48cca`; phiên trước để lại source review_training/predictor/API/UI và hai bằng chứng chưa commit. Đã đối chiếu file thực, đọc kế hoạch/test/nhật ký. Người dùng đã đồng ý quyết định dữ liệu; không hỏi lại và không chạy lại QA/training.
- `reports/DATA_DECISIONS.md` ghi rõ lý do loại 2 nhãn sai, giữ head/paw khác nghĩa và dấu phẩy cùng train; có hạn chế tăng trọng số nội dung trùng, không tự sửa nhãn, không tuyên bố loại hết tương đồng ngữ nghĩa.
- ZIP training thật PASS trên Tesla T4, 30 pipeline tests PASS; source `2b48cca`. `G3_full_training_review.json` chứng minh CPU replay 9 head, hash/scaler/metrics/early stopping PASS. MLP seed 42 MAE 0,773777; 3-seed MAE mean±population std 0,771642±0,002564. Không chọn lại seed theo điểm; holdout NOT EVALUATED.
- Artifact local tại `.cache/reviewed-training-20261020T174855207187Z`. `G4_full_model_http.json` phiên trước đã PASS qua chấm/so sánh/metrics thật; không chạy lại smoke giống hệt khi source không đổi.
- Hoàn thiện receipt cho loader, cảnh báo đúng scope validation, dashboard thêm M1 cosine và pairwise. `pytest tests/unit/test_artifact_review.py tests/integration/test_api.py -q -p no:cacheprovider --basetemp .cache/pytest-real-artifact-20261026`: **17 PASS, 34.81 giây**, 1 deprecation warning AnyIO/Starlette.
- Build frontend trong sandbox FAIL do esbuild không được đọc thư mục cha; chạy ngoài sandbox **PASS**, 1.45 giây; không thay môi trường. Thêm `frontend/playwright.real.config.js` và `tests-real/model.spec.js`, dùng backend/CLIP/head thật, offline HF, không mock API.
- E2E lần đầu FAIL do test quên xóa ảnh cũ trước khi chọn 2 ảnh mới (UI cộng dồn đúng, nhận 4 kết quả lịch sử thay vì 3). Đã sửa test thao tác xóa, đang chạy lại tại checkpoint này; chưa ghi PASS trước kết quả.
- Đã cập nhật checkpoint đầu trang, kế hoạch và hướng dẫn docs/WEB_LOCAL.md. Tiếp theo: chốt E2E thật, commit/push source/docs/evidence; còn hiệu năng/concurrency, ảnh mới ngoài dataset, UAT và holdout sau chốt release.

## 2026-10-26 — E2E thật PASS, tiếp tục kiểm hiệu năng local

- E2E lần hai phát hiện header bảng được accessibility tree nhận là cell; bổ sung `scope=col`. E2E lần ba **1 PASS, 17.9 giây** (test 1.9 giây); build cuối **PASS, 1.19 giây**. Không mock API; server tự dừng sau test.
- Bằng chứng `reports/results/G4_G5_real_model_validation.json`; screenshot `reports/figures/G5-real-model-20261026.png` đã xem, hiển thị điểm model thật, các metrics M0/M1/Ridge/MLP và cảnh báo chưa test holdout. Synthetic ảnh đỏ chỉ là kiểm luồng, không đánh giá chất lượng ngoài dataset.
- Cập nhật `docs/WEB_LOCAL.md` lệnh chạy model thật không cần development override. Thêm benchmark HTTP in-process với model CPU thật: batch 1/6 mỗi loại 20 request sau warmup, 5 request đồng thời; đang chạy, chưa ghi PASS trước kết quả. Peak RAM chưa đo.
- Tiếp theo cần UAT và tối thiểu 20 ảnh mới ngoài dataset do nhóm lựa chọn; không đánh giá holdout trước chốt model/release. Chưa tự chuyển toàn bộ mốc thành DONE.

## 2026-10-26 — Checkpoint bàn giao demo local

- `scripts/benchmark_local_api.py` chạy **PASS**, exit 0; bằng chứng `reports/results/G7_cpu_benchmark_20261026.json`. 20 request sau 1 warmup cho từng batch; p95 batch 1 = khoảng 242,5 ms, batch 6 = khoảng 967,9 ms. 5 request đồng thời: score delta <1e-5 (chi tiết trong JSON). Đây là ảnh tổng hợp 224×224, HTTP in-process, CPU local với OMP/MKL 4 threads; không coi là network SLA hoặc chất lượng ảnh thật. Peak RAM NOT MEASURED.
- Source/docs/evidence hoàn tất để commit/push đúng origin; không stage ZIP/notebook người dùng, raw manifest/embeddings/checkpoint. Bảng tiến độ đầu trang đã sửa để phiên tiếp theo không quay lại bootstrap/training.
- **Cần nhóm tiếp theo:** chạy demo theo docs/WEB_LOCAL.md, UAT luồng chấm/so sánh/chat/dashboard và cung cấp ≥20 ảnh mới kèm prompt để phân tích lỗi. Giữ MLP seed 42 theo chính sách đã chốt, chưa đánh giá test holdout hoặc gọi toàn bộ release DONE. Không cần thêm phiên Kaggle lúc này.

## Mẫu ghi cho nhóm công việc tiếp theo

- Ngày / giai đoạn / nhiệm vụ:
- Trạng thái repository trước khi làm:
- File tạo hoặc sửa:
- Lệnh chạy và kết quả thực (PASS/FAIL/SKIP, exit code):
- Bằng chứng:
- Lỗi và cách xử lý:
- Trạng thái nghiệm thu (PASS / FAIL / BLOCKED / NOT RUN):
- Review còn thiếu:
- Nhiệm vụ chưa nghiệm thu đầu tiên và bước tiếp theo:
