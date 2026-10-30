# G7 — Kịch bản nghiệm thu và bàn giao

Trạng thái 2026-10-27: chưa nghiệm thu. Không điền PASS trước khi thực hiện.

| Hạng mục | Bằng chứng hiện có | Phần còn thiếu |
|---|---|---|
| Model thật/API/browser | G4_G5_real_model_validation.json | Người dùng review thao tác |
| 20 ảnh mới | G7_external20_evaluation_20261027.json/.md | Review kết quả yếu và giới hạn sử dụng |
| CPU | G7_cpu_benchmark_20261026.json | Bổ sung peak RAM trong run mới |
| Môi trường sạch | G7_clean_stack_20261027.json: venv/npm ci mới, 17 test + real browser PASS | Chưa thử máy vật lý thứ hai; tái dùng browser/CLIP/artifact đã review |
| Test holdout | G7_holdout_seed42_20261027.json/.md, ứng viên/hash/giao thức đã khóa | Review kết quả; không tuning theo test |
| Minh bạch | reports/model_card.md | Kiểm nguồn giấy phép và người review |
| Demo 5 phút | Kịch bản dưới đây | Người thực hiện/ngày/kết quả thực tế |

## Kịch bản UAT 5 phút

1. Theo WEB_LOCAL.md khởi động model thật; kiểm health và model_version. Kaggle không cần mở.
2. Tải một ảnh, nhập prompt, chấm điểm; kiểm ảnh/prompt/điểm hiển thị đúng. Điểm 4/5 không diễn giải thành 80% chính xác.
3. Dùng 2–6 ảnh cho cùng một prompt, kiểm đủ ảnh, thứ tự và thứ hạng. Không dùng các prompt khác nhau để tạo ranking giả.
4. Thử follow-up, mở bảng metrics, kiểm nhãn validation và N/A đúng.
5. Thử file không phải ảnh, prompt rỗng, giới hạn số ảnh; phải báo lỗi dễ hiểu.
6. Trình bày một lỗi đã biết như external20 ảnh20, giải thích vì sao cần người kiểm lại kết quả.

Ghi người thử, ngày, model_version, từng bước PASS/FAIL và đường dẫn screenshot/log vào biên bản mới. Agent không ký thay người dùng. Các đề xuất sửa model là công việc mới, không được âm thầm thay checkpoint rồi sử dụng lại holdout hoặc external20 như kiểm chứng độc lập.
