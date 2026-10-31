# G7 — Cổng BLIP-ITM trên full GenAI-Bench train/validation

Ngày 2026-10-30. **Trạng thái: ứng viên development, chưa holdout, chưa thay web, chưa review.** Chạy local RTX 3050 Ti 4GB trong clone mới.

## Dữ liệu và tái lập baseline

- 12 shard revision `94e543b0801feade1419873335191de90b68d164` tải lại (sha256 trong `G7_itm_gate_full_20261030/checksums.txt`). Manifest dev dựng bằng `build_manifest` + `assign_splits(seed=42)`: 9.598 cặp hợp lệ, 2 loại, train 6.718 / val 1.440 / test 1.440 — trùng số liệu bản khóa. Hash file khác bản lock (định dạng), không thay thế lock.
- CLIP ViT-B/32 revision pinned, `extract_embeddings --development --device cuda`, `train_regression --seed 42`. MLP tái lập **đúng số model card**: val MAE 0,773777, Spearman 0,480619, pairwise 0,660825 (model card 0,661340 do cách tính min_gap cũ/mới; cùng checkpoint logic). M0 0,9379, Ridge 0,8678. Pipeline local tương đương pipeline Kaggle.
- BLIP `Salesforce/blip-itm-base-coco`, P(match) từ ITM head. Script: `code/scripts/experiment_itm_gate.py`, luật trong `code/src/evaluation/itm_gate.py`, test `tests/unit/test_itm_gate.py` (7 PASS).

## Giao thức chọn luật

Mỗi cặp chấm với prompt gốc và prompt tráo từ nhóm khác (seed 1701). Luật fusion **chọn trên train** (6.718 cặp), sau đó xác nhận trên val một lần. Holdout không mở.

| Train 6.718 | MAE | Spearman | Pairwise | Tráo prompt gốc thắng |
|---|---:|---:|---:|---:|
| MLP42 | 0,5927 | 0,7384 | 0,7509 | 3.674/6.718 (55%) |
| gate_multiply `1+(s−1)·itm` | 0,6189 | 0,7210 | 0,7728 | 6.714 |
| gate_min | 0,6067 | 0,7274 | 0,7596 | 6.713 |
| **gate_sqrt `1+(s−1)·√itm`** | **0,5951** | **0,7357** | **0,7700** | **6.714** |
| gate_soft25 | 0,5961 | 0,7312 | 0,7710 | 6.626 |
| match_only `1+4·itm` | 1,4262 | 0,2325 | 0,6995 | 6.715 |

Chọn `gate_sqrt`: giữ MAE/Spearman ngang MLP, tăng pairwise, sửa tráo prompt. Lý do mềm hóa: 102 cặp val có ITM<0,5 nhưng nhãn trung bình 2,58 (không phải 1); cổng cứng kéo quá tay.

## Xác nhận trên validation 1.440

| Val | MAE | Spearman | Pairwise (1.940 cặp) | Tráo prompt |
|---|---:|---:|---:|---:|
| MLP42 (web hiện tại) | 0,7737 | 0,4806 | 0,6608 | 815/1.440 (57%) |
| **gate_sqrt** | **0,7670** | **0,5191** | **0,6825** | **1.439/1.440**, drop 2,23 |
| gate_multiply | 0,7849 | 0,5224 | 0,6825 | 1.439/1.440 |
| gate_soft25 | 0,7624 | 0,5218 | 0,6825 | 1.427/1.440 |

ITM trên val: cặp thật median 0,997, phân vị 10% 0,70; cặp tráo phân vị 99% 0,008. Một cặp tráo ITM 0,936 (prompt tráo gần nghĩa).

## Kết luận và giới hạn

- MLP42 gần như không nhìn prompt (57% tráo prompt đúng, trên cả train lẫn val). Đây là nguyên nhân web chấm cao ảnh sai prompt; không phải lỗi UI hay seed.
- `gate_sqrt` không cần train lại, thêm BLIP base 224M (CPU được), cải thiện mọi chỉ số val và sửa mismatch gần tuyệt đối. Chi phí: thêm một lần chạy BLIP mỗi ảnh.
- Chưa kiểm: holdout (phải chạy đúng một lần sau review), external20 và bộ v2 (ảnh không có trong clone), lỗi đếm/trạng thái (ITM không đảm bảo), latency CPU, tiếng Việt. gate_soft25 có MAE thấp hơn trên val nhưng không chọn vì train đã chốt gate_sqrt; không đổi luật sau khi xem val.
- Không đụng checkpoint reviewed; MLP ở đây là bản train lại local, cùng số liệu nhưng khác file. Khi tích hợp web phải dùng checkpoint reviewed `mlp-b8dba09efa0e` + BLIP với checksum ghi lại.

## Bằng chứng

`G7_itm_gate_full_20261030/`: `val_result.json`, `val_scores.csv` (mlp/itm gốc và tráo từng cặp), `train_result.json`, `clip_full_training_seed42_result.json`, `checksums.txt`. Output thô trong `.cache/itm-gate-full-{train,val}`, `.cache/clip-full-*`. Lệnh:

```text
code> ..\.venv\Scripts\python.exe scripts\experiment_itm_gate.py --manifest ..\.cache\genai-bench-smoke\manifest_full.jsonl --data-root ..\.cache\genai-bench-smoke\data --clip-model ..\.cache\models\clip-vit-base-patch32\3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268 --training ..\.cache\clip-full-training-seed42 --output ..\.cache\itm-gate-full-val --device cuda
(thêm --splits train cho bảng train)
```

## Bước tiếp theo

1. Review báo cáo này; nếu đồng ý, chạy holdout **một lần** với `--splits test` cho MLP42 và gate_sqrt, ghi kết quả, không chỉnh gì sau đó.
2. Chạy external20 + v2 trên máy có ảnh (script nhận ảnh rời cần thêm; hiện chỉ manifest).
3. Tích hợp predictor: `score = 1 + (mlp − 1)·√itm`, hiển thị cả mlp, itm; artifact review cho BLIP (checksum, revision).
4. VQAScore trên Kaggle (VQASCORE.md) cho lỗi đếm/trạng thái; so với gate_sqrt trước khi chọn backbone dài hạn.
