# PROMPTALIGN AI — KẾ HOẠCH KIỂM THỬ CHI TIẾT

> Đây là **test specification**. Trạng thái thực tế và bằng chứng nằm trong `docs/PROGRESS.md` và `reports/results/`; không suy ra PASS từ danh mục test.

## Điều chỉnh 2026-10-20: kiểm thử local trước

- G0-local đã có bằng chứng môi trường/CPU; CUDA Kaggle cũng đã được kiểm chứng. ENV-06 bootstrap sạch còn chờ và được kiểm tra lúc tích hợp GPU, không chặn phát triển module local.
- G1-local kiểm contract chuẩn hóa bằng ảnh tổng hợp: mapping sáu model, nhãn, ảnh hỏng, nguồn trùng, split tái lập và chống rò rỉ. Chạy `python -m pytest tests/unit/test_data.py -q` và `python scripts/data_smoke.py --output reports/results/G1_fixture_smoke.json` (chọn tên output mới nếu đã tồn tại).
- G1-local chưa hoàn tất cho đến khi adapter Parquet và QA report cũng có integration test. DATA-01–12 trên 12 shard vẫn NOT RUN cho đến khi có dữ liệu thật; fixture không xác nhận schema nguồn, near-duplicate hay số lượng dataset.
- G2/G3 được kiểm thử bằng fixture sau khi contract dữ liệu local đạt; G4/G5 được phát triển theo contract với mock có nhãn. Không công bố chất lượng mô hình từ fixture.
- Trước train chính thức: ENV-06 sạch, full-data QA, split khóa và provenance phải đạt. Trước release: toàn bộ exit gate bắt buộc, checkpoint thật, review và UAT vẫn phải đạt.

## 0. Chiến lược test và công cụ

| Tầng | Công cụ | Khi chạy | Điều kiện đạt |
|---|---|---|---|
| Static/format | Ruff, Black (Python); ESLint (frontend) | mỗi PR | 0 lỗi cấu hình kiểm tra |
| Unit backend/ML | pytest | mỗi PR | 100% ca bắt buộc pass |
| API integration | pytest + FastAPI TestClient/httpx | mỗi PR có thay API | 100% ca bắt buộc pass |
| Unit component | Vitest + React Testing Library | mỗi PR frontend | 100% ca bắt buộc pass |
| Browser E2E | Playwright | trước merge/release | 100% luồng MVP pass |
| Data QA | script chạy độc lập + pytest | mỗi khi thay dataset/split | integrity/leakage pass |
| Model evaluation | notebook/script có seed và artifact | mỗi lần đổi model | metric tái lập, không chạm test lúc tuning |
| Smoke CPU/GPU | script inference + file fixtures | khi đóng gói/release | kết quả hữu hạn, trong [1,5] |

**Lệnh tham khảo (sau khi đã thêm config và test files):**

```bash
# Chạy trong VS Code, tại thư mục gốc
python -m pip check
python -m pytest tests/unit -q
python -m pytest tests/integration -q
python -m pytest tests -q

# Với frontend ở thư mục frontend/
npm run lint
npm run test -- --run
npx playwright test
```

Có thể cấu hình CI để chạy các test CPU, không tải toàn bộ 4,76 GB. Kaggle thực hiện model/full-data tests cần GPU; CI chỉ dùng fixture nhỏ 2–3 ảnh có thể chia sẻ hợp pháp.

## 1. G0 — Môi trường và khả năng tái lập

| ID | Test case | Cách kiểm tra | Pass nếu |
|---|---|---|---|
| ENV-01 | Đúng Python, PyTorch, Transformers, HF Hub | in versions, `pip check`, import CLIPModel/CLIPProcessor | Không còn lỗi dependency, không exception |
| ENV-02 | Khởi tạo CLIP trên CPU | load processor/model và 1 cặp fixture | Đầu ra hữu hạn, không crash |
| ENV-03 | GPU Kaggle tùy chọn | kiểm tra CUDA, chạy 1 batch | GPU dùng được nếu bật; CPU fallback nếu không |
| ENV-04 | Khóa phiên bản thư viện | tạo môi trường sạch từ requirements | chạy được import và tests |
| ENV-05 | Secret/data hygiene | `git status`, `.gitignore`, kiểm key giả | không track `.env`, ảnh gốc/Parquet, checkpoint lớn |
| ENV-06 | Notebook clean run | restart kernel, chạy lần lượt từ đầu | không phụ thuộc biến ẩn từ lần chạy trước |

**Exit gate G0:** ENV-01–06 đạt; ghi `environment.txt` và cách tái tạo môi trường vào README.

## 2. G1 — Chất lượng dữ liệu và chống rò rỉ

| ID | Test case | Cách kiểm tra | Pass nếu |
|---|---|---|---|
| DATA-01 | Đọc tất cả 12 shard | liệt kê shard, số bản ghi, checksum/size tùy khả năng | đọc được và ghi tổng số mẫu thực |
| DATA-02 | Mapping 6 mô hình | kiểm cột `DALLE_3`, `DeepFloyd_I_XL_v1`, `Midjourney_6`, `SDXL_2_1`, `SDXL_Base`, `SDXL_Turbo` | tên cột thống nhất, không ghép nhầm rating với ảnh |
| DATA-03 | Điểm đúng | 3 rating/model/ảnh, kiểm số nguyên trong [1,5] (hoặc ghi ngoại lệ) | mọi sample dùng train có rating hợp lệ |
| DATA-04 | Nhãn trung bình | fixture `[5,5,5]` -> `5.0`; `[2,4,3]` -> `3.0` | sai số < 1e-8 |
| DATA-05 | Liên kết tới ảnh | mở tối thiểu một ảnh của từng model/từng shard; kiểm RGB & kích thước | 0 ảnh không truy xuất được ở tập hợp lệ |
| DATA-06 | ID duy nhất | unique key `(shard,row,image_model)` | không có trùng key |
| DATA-07 | Chia theo prompt | `train_prompt ∩ val_prompt` và các giao khác | tất cả giao rỗng, kể cả normalized text |
| DATA-08 | Ổn định split | chạy hai lần cùng seed/config | danh sách prompt/split identical |
| DATA-09 | Không làm lộ nhãn | pipeline feature input chỉ ảnh + prompt | `HumanRatings` không được đưa vào inference feature |
| DATA-10 | Ảnh hỏng/missing | tạo bản ghi giả bị thiếu ảnh/rating | bị loại có log, không crash toàn quá trình |
| DATA-11 | Kích thước dataset | so số cặp thực với số kỳ vọng `prompts × 6` | khác biệt được báo và giải thích, không âm thầm bỏ qua |
| DATA-12 | Prompt gần trùng | kiểm trùng exact + normalized, báo near-duplicate | không để trùng rõ ràng xuyên split |

**Kiểm tra hồi quy từ mẫu đã xem:** shard 0 có 134 hàng, 804 cặp hợp lệ, ví dụ prompt thợ làm bánh và rating `[5,5,5]`/`[2,4,3]`. Không hardcode 134/804 thành quy tắc cho full dataset.

**Exit gate G1:** DATA-01–12 đạt, có data report và `split_manifest` bản chính thức. Khóa split; chỉ tái tạo nếu có lỗi dữ liệu được ghi nhận.

## 3. G2 — Baseline và CLIP frozen embeddings

| ID | Test case | Cách kiểm tra | Pass nếu |
|---|---|---|---|
| ML-01 | Processor ảnh + văn bản | 2 ảnh RGB và prompt, tokenizer truncation | batch tensor có shape đúng, không lỗi |
| ML-02 | Embedding đúng | chạy CLIP `get_text_features`, `get_image_features` hoặc output embed được kiểm | cùng kích thước chiều cuối, không NaN/Inf |
| ML-03 | Cosine an toàn | tính cosine sau chuẩn hóa | nằm trong [-1,1] (cho sai số số học nhỏ) |
| ML-04 | Single vs batch | cùng cặp chạy từng cái và chạy batch | lệch < 1e-4 (hoặc tolerance có căn cứ) |
| ML-05 | Số embedding | so count với manifest | đúng 1 embedding-pair cho mỗi cặp hợp lệ |
| ML-06 | Không fit trên val/test | kiểm code & input fit scaler/encoder | fit chỉ train, val/test dùng transform |
| ML-07 | Mean baseline | đo MAE trên val bằng trung bình train | metric finite, tái lập |
| ML-08 | Spearman | cặp mẫu xếp hạng tự tạo; constant predictor | Spearman đúng trường hợp chuẩn; constant được xử lý `N/A` |
| ML-09 | Model gốc không cập nhật | so hash/state_dict trước và sau extract | model frozen không thay đổi |

**Exit gate G2:** log số vector, MAE M0, Spearman M1, timing và fingerprint mô hình, không lỗi biến đổi/trích đặc trưng.

## 4. G3 — MLP/Ridge, kiểm thử mô hình và đánh giá

| ID | Test case | Cách kiểm tra | Pass nếu |
|---|---|---|---|
| REG-01 | Ridge fit trên train | kiểm StandardScaler & Ridge | val/test không xuất hiện trong fit |
| REG-02 | MLP forward shape | batch B vector text/image -> điểm | shape `[B]`, không NaN/Inf |
| REG-03 | Điểm hợp lệ | nhiều ảnh biên/đầu vào lạ | returned score trong [1,5], có quy tắc clamp/sigmoid rõ |
| REG-04 | Loss giảm ở mẫu nhỏ | overfit 16–32 cặp để kiểm pipeline | loss giảm rõ qua nhiều bước, không bug optimizer |
| REG-05 | Validation và early stopping | log epoch/val loss | checkpoint là epoch tốt theo val, không theo test |
| REG-06 | Model save/load | lưu checkpoint+scaler+config, reload | điểm fixture thay đổi < 1e-5 CPU cùng môi trường |
| REG-07 | Metric đầy đủ | tính MAE, RMSE, Spearman trên val | tất cả là phép đo thật; no fake values |
| REG-08 | So sánh có căn cứ | M0/M1/M2 cùng val/split | chênh lệch được báo, không cherry-pick |
| REG-09 | Seed và lặp lại | lặp 2–3 seeds nếu nguồn lực cho phép | có mean±std hoặc nêu rõ giới hạn |
| REG-10 | Không lộ generator label vào inference | review input features | chỉ dùng prompt, image (không dùng model nguồn làm shortcut nếu web không có) |

**Exit gate G3:** Có checkpoint load được và một model được chọn theo metric Validation. Nếu MLP kém Ridge/M0, giữ baseline tốt hơn làm sản phẩm và báo cáo kết quả thất bại của thí nghiệm trung thực.

## 5. G6 (tùy chọn) — Pairwise Ranking/Contrastive

| ID | Test case | Cách kiểm tra | Pass nếu |
|---|---|---|---|
| RANK-01 | Pair cùng prompt | tạo pair A,B | không có pair khác prompt |
| RANK-02 | Ties/near-ties | chọn ngưỡng chênh điểm, ví dụ >= 0.5 | pair không rõ ưu tiên bị bỏ/đánh dấu |
| RANK-03 | Gradient và loss | backward trên batch bé | gradient hữu hạn, loss thực sự cập nhật |
| RANK-04 | Ranking accuracy | synthetic score/rank có lời giải biết trước | tính đúng, không dùng ảnh khác prompt |
| RANK-05 | Ablation | so M2 vs M3 trên cùng val | công bố metric cải thiện/không cải thiện |
| RANK-06 | Không tụt MVP | chạy API fixture bằng model mới | shape/range/inference contract giữ nguyên |

**Exit gate G6:** Báo cáo khoa học, không bắt buộc M3 phải thắng; nếu kém, không thay model đang phục vụ web.

## 6. G4 — Kiểm thử backend FastAPI

| ID | Test case | Request | Kết quả mong đợi |
|---|---|---|---|
| API-01 | Health | GET `/api/health` | 200, model_ready, version |
| API-02 | Chấm đúng | POST prompt + PNG hợp lệ | 200, điểm trong [1,5], model_version |
| API-03 | Prompt rỗng | POST image, prompt `""` | 422/400 có message rõ |
| API-04 | Không có ảnh | POST prompt, no file | 422/400 |
| API-05 | File giả ảnh | POST text file đuôi `.png` | 415/400, kiểm bytes/MIME thực |
| API-06 | Ảnh quá lớn | POST > 10 MB/file hoặc quá giới hạn pixel | 413/400, không crash |
| API-07 | So sánh đúng | POST prompt + 2..6 ảnh | 200, đúng N item và rank rõ |
| API-08 | So sánh sai N | POST 1 ảnh hoặc 7 ảnh | 422/400 |
| API-09 | Ranking ties | 2 ảnh có score bằng nhau | logic tie được quy định nhất quán |
| API-10 | Chat intent | các câu "chấm ảnh", "so sánh", "xem biểu đồ" | route đúng tính năng |
| API-11 | Chat follow-up | prompt trước -> "ảnh thứ hai thì sao" | dùng prompt/session đúng, không lẫn session khác |
| API-12 | Metrics provenance | GET `/api/metrics` | đọc từ artifact metric thật; có dataset/split |
| API-13 | Lỗi inference | inject lỗi mô hình | 5xx có request_id, không leak stacktrace/token |
| API-14 | Upload an toàn | filename `../../...`, EXIF, đa request | không path traversal, không tràn bộ nhớ |
| API-15 | Upload ảnh và phân quyền theo phiên | POST `/api/uploads`, dùng image_id sai session/hết hạn | image_id hợp lệ dùng được; sai session/hết hạn bị từ chối |

**Exit gate G4:** Toàn bộ API bắt buộc pass (API-01–15) với mock predictor **trong test** và smoke với checkpoint **thật**; mock không được dùng trên giao diện release.

## 7. G5 — Frontend + hội thoại, tương tác thực tế

| ID | Test case | Cách thử | Pass nếu |
|---|---|---|---|
| UI-01 | Trang chat mở lần đầu | vào `/` | có hướng dẫn & 4 câu gợi ý chức năng |
| UI-02 | Nhập prompt | gõ, xóa, paste Unicode tiếng Việt | không mất nội dung, xử lý lỗi rõ |
| UI-03 | Tải 1 ảnh | chọn file, xem preview, xóa | preview đúng, có trạng thái tải |
| UI-04 | Tải 2–6 ảnh | chọn nhiều ảnh | hiện đủ thumbnail, thứ tự ổn định |
| UI-05 | Vượt giới hạn | >6 ảnh hoặc file không hỗ trợ | chặn trước khi gọi API |
| UI-06 | Chấm ảnh | prompt + 1 ảnh, nhấn gửi | hiển thị điểm, model version, giải thích giới hạn |
| UI-07 | Xếp hạng | prompt + 3 ảnh | rank nhất quán với API, không hoán đổi ảnh |
| UI-08 | Dashboard | vào từ chat/nút | metric từ API, có đơn vị, labels, N/A xử lý đúng |
| UI-09 | Loading & lỗi mạng | mô phỏng slow/timeout/500 | có spinner, nút thử lại, không treo |
| UI-10 | Ngữ cảnh chat | 2–3 lượt hỏi nối tiếp | vẫn giữ đúng prompt/ảnh trong cùng phiên |
| UI-11 | Responsive/accessibility | 375px/768px/desktop + keyboard | không tràn, có label/alt, dùng được bằng bàn phím |
| UI-12 | Không phóng đại kết quả | kiểm copy UI | không biến 4/5 thành "80% chính xác" |

**Exit gate G5:** Playwright E2E pass ít nhất 4 luồng: một ảnh, nhiều ảnh, dashboard, lỗi file. Không dừng ở kiểm tra ảnh chụp tĩnh.

## 8. Captioning và gợi ý prompt (mở rộng)

| ID | Test case | Cách kiểm tra | Pass nếu |
|---|---|---|---|
| CAP-01 | Caption ảnh cơ bản | 10–20 ảnh đa dạng | sinh text không rỗng, runtime/memory được đo |
| CAP-02 | Ảnh khó | nhiều người, vật thể nhỏ, chữ, phong cách nghệ thuật | không crash, có cảnh báo caption có thể sai |
| CAP-03 | Prompt template | caption + phong cách/từ khóa | output mạch lạc, không chứa chi tiết không được xác nhận một cách quả quyết |
| CAP-04 | Chấm prompt sinh tự động | đưa prompt gợi ý vào evaluate | phân biệt prompt gợi ý với ground truth và không tự tuyên bố đã khớp tuyệt đối |
| CAP-05 | CPU memory | tải CLIP + VLM/BLIP theo yêu cầu | không OOM trên máy demo; nếu OOM dùng lazy loading/fallback |

**Exit gate:** CAP có thể tắt bằng feature flag mà MVP vẫn chạy đủ chức năng. Không dùng GenAI-Bench làm bằng chứng chất lượng caption nếu chưa có nhãn mô tả tương ứng.

## 9. G7 — Acceptance / hiệu năng / an toàn / tính tái lập

| ID | Test case | Cách thử | Pass nếu |
|---|---|---|---|
| E2E-01 | Chạy máy sạch | clone repo, cài theo README, tải artifact có chỉ dẫn | chạy app được không cần notebook online |
| E2E-02 | Ảnh mới ngoài dataset | ít nhất 20 ảnh do nhóm tự chọn | không crash; phân tích đúng/sai có ví dụ |
| E2E-03 | Đánh giá holdout | sau khi khóa model, chạy 1 lần trên Test | metric ghi thật, không tuning lại vì test |
| E2E-04 | Đo CPU latency | 20 request warm, batch 1 và 6 | báo p50/p95, RAM peak, cấu hình CPU; tốc độ có chấp nhận hay chưa |
| E2E-05 | Ổn định đa request | 3–5 yêu cầu liên tục/đồng thời | không lẫn session, không corrupt model |
| E2E-06 | Restart server | restart FastAPI, request lại fixture | output tương đương, artifacts load ổn |
| E2E-07 | Offline Kaggle | tắt phiên Kaggle, chạy web local | không phụ thuộc Kaggle để suy luận |
| E2E-08 | Minh bạch | kiểm README, model card, Dashboard | có nguồn dataset, license, giới hạn ngôn ngữ/AI-generated, version |
| E2E-09 | Trình diễn 5 phút | trình tự chấm -> so sánh -> dashboard -> caption (nếu có) | không crash; dữ liệu demo đã chuẩn bị |

**Exit gate G7:** G0–G5 bắt buộc đều pass; G6 có thể bỏ. Tất cả lỗi P0/P1 phải được sửa trước demo; lỗi P2 nhỏ phải ghi vào known issues.

## 10. Quy trình báo cáo khi một mục hoàn thành

Tạo biên bản cho từng milestone tại `reports/results/Gx_validation.md`:

```markdown
# Gx — [Tên mốc]
- Ngày kiểm tra:
- Người thực hiện / Người review:
- Commit hash:
- Dataset + split + seed + model version:
- Môi trường: Python/torch/transformers/GPU hoặc CPU:
- Lệnh đã chạy:
- Tổng test case: __ ; Pass: __ ; Fail: __ ; Skipped: __
- Kết quả/metric cụ thể:
- Bằng chứng: link log/screenshot/artifact:
- Lỗi tồn đọng:
- Quyết định: PASS / BLOCKED / PASS_WITH_LIMITATIONS
- Người xác nhận:
```

Trạng thái công việc chỉ chuyển `DONE` khi kết quả là `PASS` hoặc, với phần optional, `PASS_WITH_LIMITATIONS` đã có giải trình và không ảnh hưởng MVP.

## 11. Lịch chạy kiểm thử khuyến nghị

- **Sau mỗi commit:** unit tests liên quan và lint.
- **Trước mỗi merge vào main:** tất cả unit, integration, frontend tests; review chéo.
- **Mỗi tối hoặc sau milestone:** E2E với fixture nhỏ (không cần GPU).
- **Sau mỗi lần train trên Kaggle:** lưu checkpoint + config + metric, chạy đánh giá validation, reload inference smoke.
- **Trước demo:** thử trên máy demo thực, không online Kaggle; chạy E2E/UAT; lưu màn hình và log cuối.

## 12. Definition of Done cho một task riêng

- [ ] Code và cấu hình đã commit.
- [ ] Unit test pass.
- [ ] Test case lỗi/biên pass.
- [ ] API/UI integration pass nếu liên quan.
- [ ] Có log, metric hoặc screenshot chứng minh.
- [ ] Mọi thay đổi score/model được benchmark trên split phù hợp.
- [ ] README hoặc tài liệu kỹ thuật được cập nhật.
- [ ] Một thành viên khác review.
