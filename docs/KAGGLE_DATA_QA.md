# Bước cần chạy trên Kaggle: QA đủ dữ liệu

Code dữ liệu, embeddings, training head và web đã được phát triển/kiểm thử local. Chưa có QA trên 12 shard thật nên chưa được chạy thực nghiệm chính thức. Đây là bước lấy báo cáo dữ liệu, không phải chạy lại notebook G0 hoặc huấn luyện ngay.

1. Lấy `notebooks/01_data_qa.ipynb` từ [repository GitHub](https://github.com/hoan25/BaiqiL-GenAI-Bench/blob/main/notebooks/01_data_qa.ipynb), import vào Kaggle.
2. Bật **Internet**, chọn **CPU / không cần accelerator GPU** để tiết kiệm quota.
3. Cell đầu lấy source từ GitHub. Cell sau gọi `scripts/kaggle_data.py`; không sửa logic Python trong notebook.
4. Mặc định `DOWNLOAD_DATASET = True`: khi bạn chạy cell, script tải khoảng **4,76 GB** GenAI-Bench vào Kaggle, chốt revision và tái sử dụng cache theo revision. Nếu đã có **đủ 12 shard** trong Kaggle Input, đặt False và sửa `DATA_ROOT` đúng thư mục chứa các file đó.
5. **Run All**. Không cần cài thư viện thủ công hoặc chạy 00b trước. Script tạo venv riêng, kiểm startup/pip/tests, QA dữ liệu và luôn đóng gói báo cáo kể cả lỗi. Không tải CLIP, không train model.
6. Tải `data-qa-*.zip` từ `/kaggle/working` cùng notebook có output, lưu vào `reports/results/kaggle/` trên máy rồi báo đã lưu.

ZIP gồm logs, versions, dataset revision (nếu tải bằng script), source checksum, candidate manifest và báo cáo QA/near-duplicate. Không kèm ảnh/Parquet/checkpoint. **REVIEW_REQUIRED là trạng thái dự kiến**, không đồng nghĩa lỗi thực thi: cần xem dữ liệu bị loại, count và prompt gần trùng trước khi khóa split. Nếu FAIL, gửi nguyên ZIP để xác định lỗi; không sửa nhiều cell thử ngẫu nhiên.

Chỉ cần import notebook, bật Internet và Run All; không upload cả repository hoặc cài thư viện thủ công. Cell đầu luôn lấy code mới từ `main`. Khi đã import đúng notebook này, có thể chạy lại Run All để lấy bản sửa script mà không import lại. Bản runner mới in stage/returncode và phần cuối log lỗi ngay trong output; ZIP vẫn là bằng chứng đầy đủ. Notebook lỗi cũ chỉ ghi FAIL ở bootstrap_pip thì cần ZIP để biết nguyên nhân, không thể kết luận chỉ từ tên bước.

Sau khi kiểm tra ZIP, sẽ xử lý/giải trình vấn đề dữ liệu rồi khóa split và nối cùng source scripts cho extraction/training GPU. Chưa khẳng định runner đã PASS trên Kaggle; kiểm chứng local và thực tế Kaggle là hai bằng chứng khác nhau.

Runner đã được kiểm chứng end-to-end bằng venv sạch Windows trên fixture nhỏ: 18 test dữ liệu PASS, startup/pip check đạt, xuất ZIP CRC hợp lệ. Bằng chứng: `reports/results/Kaggle_data_local_validation.json`. Môi trường Linux/Kaggle và dữ liệu đầy đủ cần kết quả từ phiên của bạn.

## Sửa bootstrap từ ZIP ngày 2026-10-20

ZIP `data-qa-20261020T060426867085Z.zip` cho thấy pip 26.2.1 đã cài thành công nhưng Python báo `Error in sitecustomize: ModuleNotFoundError: No module named 'wrapt'`. Runner dừng đúng ở startup dù pip trả exit 0; chưa cài thư viện QA hoặc tải dữ liệu. Đây là cùng dạng lỗi startup đã gặp ở G0.

QA và G0 hiện dùng chung `scripts/clean_python.py`: chạy Python với `-I -S`, khởi tạo đường dẫn venv bằng `site.main()` nhưng bỏ hai hook `sitecustomize`/`usercustomize`. Bootstrap chạy pip ngay trong interpreter đích, tránh `pip --python` tạo lại tiến trình không giữ cờ cách ly. Không che log lỗi hay thêm `wrapt` chỉ để thỏa hook của môi trường bên ngoài. Cơ chế `-S` và `site.main()` được mô tả trong [tài liệu Python](https://docs.python.org/3.13/library/site.html).

| Phạm vi | Thư viện và cách kiểm |
|---|---|
| Notebook QA hiện tại, CPU | `requirements-kaggle-data.in`: PyArrow 21.0.0, Pillow 12.0.0, huggingface-hub 0.36.0, pytest 8.4.2 cùng dependency bắc cầu do pip cài. Kiểm `pip check`, import API cần dùng, lưu `versions.log`, chạy 18 test trước đọc dữ liệu. |
| G0 / CLIP, GPU | Runner G0 cài PyTorch 2.8.0 từ index cu126 và `requirements-g0.in`, kiểm import CLIP và inference. Bootstrap dùng cùng bản sửa. Không cần chạy G0 riêng để chạy notebook QA. |
| Extraction/training sau QA | Cần kết hợp môi trường ML và dữ liệu; chỉ thực nghiệm chính thức sau QA đầy đủ, khóa split và kiểm GPU sạch. Notebook QA hiện tại chưa thực hiện bước này. |

Kiểm chứng bản sửa local: 21 regression test PASS (có tái hiện hook lỗi trong venv thật); G0 bootstrap-only PASS; QA trên venv mới PASS các bước cài/import/tests và xuất REVIEW_REQUIRED đúng cho fixture thiếu dữ liệu. Bằng chứng: `reports/results/Kaggle_bootstrap_fix_validation.json`. Chưa nghiệm thu bản sửa trên Kaggle.

**Bước tiếp theo:** mở notebook hiện có, bật Internet và Run All từ cell đầu để lấy code mới trên `main`. Không cần upload lại notebook hoặc tự thêm cell cài pip/wrapt. Gửi ZIP mới để kiểm tra kết quả trên Kaggle và full-data QA.
