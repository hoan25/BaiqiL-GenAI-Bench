# G4/G5-local — API và giao diện

- Ngày: 2026-10-20; thực hiện: Codex; review/UAT: chưa có.
- Python suite sau thêm API dependencies: **68 PASS**, 40.66 giây, một DeprecationWarning Starlette/AnyIO. Log `G4_python_tests.txt`. Không có dependency conflict (`pip check` PASS).
- Predictor nối CLIP thật + head tổng hợp, HTTP evaluate/compare: **PASS**, `G4_predictor_http.json`; không phải chất lượng trên dataset thật.
- React/Vite production build: **PASS**. Browser E2E với API mock: **10 PASS**, desktop/mobile, 15.2 giây; `G5_browser_tests.txt`. Đã xem ảnh chụp desktop/mobile để kiểm bố cục.
- Một lỗi test setup ban đầu do pytest ID chứa payload 10 MB trên Windows; sửa IDs ngắn, không thay rule validate API. E2E bị sandbox chặn esbuild đọc parent; chạy ngoài sandbox thành công.
- Có route health/evaluate/compare/metrics/sessions/uploads/chat, isolation/TTL, giới hạn file/pixel/body và phiên, sanitize error, dense rank ties. Frontend hiển thị scope thử nghiệm và không fake score khi chưa có model.
- G4/G5 toàn mốc **chưa nghiệm thu**: thiếu checkpoint train trên dataset thật, E2E nối frontend/backend/model thật, kiểm concurrent/performance sâu, review và UAT. Router chat MVP theo quy tắc, không phải LLM.
