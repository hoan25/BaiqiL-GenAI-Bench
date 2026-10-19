# VS Code → GitHub → Kaggle → kết quả về máy

Source: https://github.com/hoan25/BaiqiL-GenAI-Bench

Code/test phát triển trong VS Code, đồng bộ qua GitHub; Kaggle thực thi phần GPU và trả artifact về máy. G0 chỉ kiểm tra môi trường/CLIP, chưa huấn luyện trên GenAI-Bench. Các mốc sau sẽ tải checkpoint về để tích hợp FastAPI + React local.

## 1. Chuẩn bị Kaggle

**Cập nhật 20/10:** kết quả CPU/CUDA và 13 test bạn gửi đã được kiểm chứng. Bước hiện tại chỉ cần tải **`notebooks/00b_bootstrap_check.ipynb`** từ GitHub, bật Internet (không cần GPU), restart kernel rồi Run All. Gửi ZIP mới và notebook có output. Notebook này không tải CLIP hoặc chạy lại inference; nó kiểm chứng cách tạo venv không dùng ensurepip và việc hết lỗi sitecustomize. Không dùng notebook cũ pin commit `f48f135` cho phép thử sửa này; notebook mới mặc định lấy `main`.

1. Tải `notebooks/00_environment.ipynb` từ GitHub hoặc workspace, rồi import vào notebook Kaggle.
2. Trong Settings bật Internet và chọn GPU được cấp (T4/P100). Nếu không có GPU, đổi `DEVICE = "cpu"`; kết quả ghi rõ CPU.
3. Giữ REPO_URL trỏ tới repository trên. `GIT_REF = "main"` lấy bản hiện tại; có thể thay bằng commit hash để tái lập chính xác.
4. Nếu repo private, thêm Kaggle Secret `GITHUB_TOKEN` có quyền đọc repo, cho phép notebook dùng secret, rồi đặt `PRIVATE_REPO = True`. Không dán token vào cell/URL. Repo public không cần token.

## 2. Chạy từ đầu

Restart kernel rồi Run All. Cell đầu clone/fetch source vào `/kaggle/working/promptalign-ai`, checkout revision và in commit hash. Nếu checkout có thay đổi, cell dừng để giữ nguyên chúng.

Cell thứ hai gọi `scripts/kaggle_g0.py`: tạo venv riêng, cài PyTorch 2.8.0 CUDA 12.6 (hoặc CPU), cài dependencies trực tiếp đã pin, tải đúng revision CLIP và chạy import/pip check, pytest, CLIP CPU/GPU.

Bootstrap dùng `venv --without-pip` và pip gốc với `--python`, theo [tài liệu pip](https://pip.pypa.io/en/stable/topics/python-option/). Chế độ bootstrap-only tạo venv mới, lưu `create_venv.log`, `bootstrap_pip.log`, `startup.log`, `pip_check.log`, `provenance.json` và `result.json`; không yêu cầu có các file checks/CLIP của lần full run.

Lần đầu cần thời gian tải thư viện/model. Không cần upload bundle 609 MB của cách cũ. Không tải dataset GenAI-Bench trong G0.

CUDA dependencies theo [hướng dẫn PyTorch 2.8.0](https://pytorch.org/get-started/previous-versions/). Không dùng lock Windows trên Kaggle; phiên bản Linux thực tế được ghi vào `checks/versions.log` để kiểm chứng.

## 3. Hai file cần gửi lại

Script in `RESULT: PASS` hoặc `FAIL`, tạo `/kaggle/working/g0-<thời_gian>.zip`, kể cả khi setup/test lỗi sau khi script khởi động.

1. **g0-<thời_gian>.zip**: tải bằng liên kết cuối notebook hoặc mục Output sau khi Save Version.
2. **Notebook .ipynb có output**: lưu phiên bản chạy từ kernel sạch rồi tải về.

ZIP chứa commit/thiết bị (`provenance.json`), revision/checksum CLIP, log cài thư viện, `checks/environment.json`, `checks/pytest.log`, `checks/commands.json`, `checks/clip_cpu.json`, `checks/clip_auto.json`, `checks/versions.log` và `result.json`. Khi yêu cầu CUDA, file `clip_auto.json` phải ghi `device: cuda`. Nếu lỗi có thêm `error.txt`.

Nếu clone lỗi trước khi tạo ZIP, gửi traceback cell đó; che token nếu bạn tự thêm thông tin xác thực. Không tự nâng hàng loạt thư viện khi lỗi chưa được phân tích.

## 4. Sau khi nhận kết quả

Lưu ZIP/notebook vào `reports/results/kaggle/`, đối chiếu ENV-03/06 và review để đóng G0, cập nhật nhật ký rồi mới chuyển G1. Các mốc sau giữ luồng VS Code → GitHub → Kaggle → artifact về VS Code.

Tham khảo: [Kaggle Notebooks documentation](https://www.kaggle.com/docs/notebooks).
