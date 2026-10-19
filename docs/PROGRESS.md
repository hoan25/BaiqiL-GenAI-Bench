# PromptAlign AI — Nhật ký tiến độ xuyên suốt

## Checkpoint hiện tại

- Ngày cập nhật: **2026-10-20**, múi giờ Asia/Saigon.
- Mốc phát triển hiện tại: **G1-local — pipeline dữ liệu** theo lộ trình điều chỉnh được người dùng yêu cầu ngày 2026-10-20. G0-local đủ điều kiện phát triển; G0 toàn mốc vẫn chờ ENV-06 Kaggle và review.
- Nhiệm vụ chưa nghiệm thu đầu tiên trong lộ trình local: **adapter Parquet + integration test + báo cáo QA/near-duplicate**. Contract, validation và split đã có 13 test PASS; chưa nghiệm thu G1 toàn mốc.
- **ENV-01/02/04/05 PASS local**. `.venv-repro` CPython 3.13.5 Windows x64 tái tạo đúng 31 package; torch 2.8.0+cpu, transformers 4.57.6, huggingface-hub 0.36.0, pytest 8.4.2; import CLIP và pip check đạt. **10 pytest PASS**, CPU smoke và auto fallback PASS.
- Luồng mới: VS Code → GitHub `hoan25/BaiqiL-GenAI-Bench` → Kaggle → artifact về máy. Notebook clone GitHub và xuất `g0-<thời_gian>.zip`; không dùng bundle upload trong luồng chính.
- Người dùng đã gửi ZIP và notebook Kaggle chạy commit `f48f135`: 13 test PASS, CPU/CUDA PASS, đủ checksum. Bản sửa bootstrap có 16 test local PASS và phép thử venv thật PASS. ENV-06 được hoãn đến tích hợp GPU; hiện không yêu cầu người dùng chạy 00b.
- Chưa có mô hình huấn luyện/API/UI được nghiệm thu. Không dùng fixture để công bố chất lượng thực nghiệm. Các mục nhật ký cũ bên dưới giữ nguyên để truy vết; checkpoint này thay thế chỉ dẫn tiếp tục cũ.

## Quy tắc khôi phục sau hết quota / phiên mới

1. Đọc đầy đủ `PROJECT_PLAN.md`, `TEST_PLAN.md`, file này và `AGENTS.md`.
2. Kiểm tra file thực tế, Git status/diff nếu đã khởi tạo Git, môi trường và bằng chứng trong `reports/results/`. Đối chiếu cả thay đổi chưa ghi vào nhật ký do phiên trước bị ngắt.
3. Xác định nhiệm vụ chưa nghiệm thu đầu tiên theo thứ tự phụ thuộc. Không suy ra hoàn thành chỉ từ việc file tồn tại hoặc kế hoạch ghi đã làm trên Kaggle.
4. Giữ nguyên phần đã hoàn thành và kiểm thử đạt. Chỉ kiểm thử lại khi có thay đổi liên quan, nghi ngờ hồi quy hoặc thiếu bằng chứng tương ứng với trạng thái hiện tại.
5. Sau mỗi nhóm nhiệm vụ, thêm mục nhật ký và cập nhật checkpoint, bằng chứng, blocker, nhiệm vụ tiếp theo. Ghi sớm trước khi dừng; không thể bảo đảm ghi thêm sau khi quota đã hết.

## Trạng thái các mốc

| Mốc | Trạng thái | Bằng chứng / phần còn thiếu |
|---|---|---|
| G0 | LOCAL READY; nghiệm thu toàn mốc còn chờ | Giữ CPU/CUDA PASS; ENV-06 và review chuyển sang tích hợp GPU |
| G1 | IN PROGRESS local; full-data NOT RUN | Contract/validation/split 13 test PASS; CLI 60 cặp tổng hợp; còn adapter, QA report, dữ liệu thật và review |
| G2 | NOT RUN | Chưa có baseline, encoder hoặc metrics |
| G3 | NOT RUN | Chưa có training/checkpoint |
| G4 | NOT RUN | Chưa có backend |
| G5 | NOT RUN | Chưa có frontend |
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
