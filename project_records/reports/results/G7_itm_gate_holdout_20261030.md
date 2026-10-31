# G7 — Holdout (test split) một lần cho cổng BLIP-ITM `gate_sqrt`

Ngày chạy 2026-10-30 (ghi báo cáo 2026-10-31). **Trạng thái: PASS (thực thi đúng giao thức, assert đạt). Không phải phê duyệt release; chưa thay web; chưa review chéo.** Chạy local RTX 3050 Ti 4GB, 224,9 s, exit 0.

## Giao thức đã đăng ký trước

- Người dùng cho phép chạy holdout **đúng một lần** sau báo cáo `G7_itm_gate_full_20261030.md`. So sánh chính thức: `mlp` (MLP42) với `gate_sqrt` (`score = 1 + (mlp − 1)·√itm`), luật đã chốt trên train và xác nhận trên val trước khi mở test. Các luật khác chỉ là đối chứng; không chọn lại luật, không đổi tham số sau khi xem test.
- Tập test: 1.440 cặp, 240 nhóm prompt theo `code/configs/holdout_seed42_20261027.json` (min_label_gap 0,5, prediction tie credit 0). Cùng manifest dev `manifest_full.jsonl` đã dùng cho train/val (hash giữ nguyên).
- Prompt tráo seed 1701 lấy từ nhóm khác trong cùng split test. Lệnh (chạy từ `code/`, `HF_HUB_OFFLINE=1`, `HF_HOME=E:\ai-cache\huggingface`):

```text
..\.venv\Scripts\python.exe scripts\experiment_itm_gate.py --manifest ..\.cache\genai-bench-smoke\manifest_full.jsonl --data-root ..\.cache\genai-bench-smoke\data --clip-model ..\.cache\models\clip-vit-base-patch32\3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268 --training ..\.cache\clip-full-training-seed42 --output ..\.cache\itm-gate-full-test --device cuda --splits test
```

- Trước khi chạy đã xác nhận `.cache/itm-gate-full-test` chưa tồn tại (script dùng `exist_ok=False`, không ghi đè). Checksum ghi trước và sau khi chạy, không đổi:

| File | sha256 |
|---|---|
| `code/scripts/experiment_itm_gate.py` | `b1efad42214310ee2c71290f4e5e33078a4902952c8ad350eba7406a8b53eed7` |
| `code/src/evaluation/itm_gate.py` | `308b07d10e1acf4e8ecbd8bd18552fd9bfb85b5a0bdf46f2d34af96fcc2e5d44` |
| `.cache/genai-bench-smoke/manifest_full.jsonl` | `b218ad7dffe15bfbb1e2bcee5cbb5f999fb10f0821f409197fbefa3c23f17160` |
| `.cache/clip-full-training-seed42/mlp.pt` (MLP train lại local) | `a575abfed14dc3aeacf05b55af53c9f5a201dd497792d8fb371c6ee7ec670ef1` |
| `code/configs/holdout_seed42_20261027.json` | `e604106fcd789e70c84f316037964635075a6aecf1a8157591920f122b0a4e5b` |
| BLIP `Salesforce/blip-itm-base-coco` (cache HF offline, refs/main) | revision `bed8ad38cb2d04a5a4bdf2d071b3c3c0a4aa724c` |

## Assert

- `result.json`: `count` = 1.440; `scores.csv` 1.440 dòng, tất cả `split = test`; manifest có đúng 1.440 bản ghi test. **Đạt.**
- Số nhóm prompt riêng biệt (join `pair_id` với `prompt_group` trong manifest): **240. Đạt.**
- 1.440 `pair_id` trùng hoàn toàn với 1.440 `pair_id` trong holdout chính thức `G7_holdout_seed42_20261027.json`.
- Pairwise cùng prompt: 1.930 cặp đủ chênh nhãn ≥0,5, loại 1.670 cặp tie — trùng số của holdout chính thức.

## Kết quả chính thức: `mlp` so với `gate_sqrt` (test 1.440)

| Test | MAE | RMSE | Spearman | Pairwise (1.930 cặp) | Tráo prompt gốc thắng | Drop TB khi tráo |
|---|---:|---:|---:|---:|---:|---:|
| MLP42 (local retrain) | 0,791966 | 0,966937 | 0,436959 | 0,670984 (1.295) | 788/1.440 (54,7%) | 0,081 |
| **gate_sqrt** | **0,790387** | 0,977287 | **0,471556** | **0,691192 (1.334)** | **1.438/1.440 (99,9%)** | 2,125 |

`gate_sqrt` thắng MLP42 ở MAE (chênh nhỏ, −0,0016), Spearman (+0,035), pairwise (+2,0 điểm, +39 cặp) và tráo prompt (từ 55% lên 99,9%). **Thua ở RMSE** (+0,010): cổng kéo mạnh các cặp ITM thấp nên sai số lớn ở số ít cặp tăng. Kết luận trên test nhất quán với val (val: MAE 0,7670 vs 0,7737, Spearman 0,5191 vs 0,4806, pairwise 0,6825 vs 0,6608); riêng RMSE trên val chưa được nêu trong báo cáo trước nên không có đối chiếu.

## Đối chứng (báo cáo đủ, không dùng để chọn)

| Test | MAE | RMSE | Spearman | Pairwise | Tráo prompt | Drop TB |
|---|---:|---:|---:|---:|---:|---:|
| gate_multiply `1+(s−1)·itm` | 0,812358 | 1,018757 | 0,468440 | 0,699482 (1.350) | 1.438/1.440 | 2,074 |
| gate_min | 0,799171 | 0,986093 | 0,458919 | 0,682902 (1.318) | 1.438/1.440 | 2,177 |
| gate_soft25 | 0,788529 | 0,981431 | 0,472091 | 0,695855 (1.343) | 1.420/1.440 | 1,576 |
| match_only `1+4·itm` | 1,484256 | 1,780351 | 0,245305 | 0,665803 (1.285) | 1.438/1.440 | 3,626 |

Quan sát sau đánh giá: `gate_soft25` nhỉnh hơn `gate_sqrt` ở MAE, Spearman, pairwise nhưng kém ở tráo prompt (20 cặp tráo vẫn thắng); `gate_multiply` pairwise cao nhất nhưng MAE/RMSE xấu nhất trong nhóm gate. Đây là quan sát, không phải lý do đổi luật; giữ `gate_sqrt` theo đăng ký.

## Phân vị ITM trên test

| | min | 10% | 50% | 90% | max |
|---|---:|---:|---:|---:|---:|
| Cặp thật | 0,0019 | 0,6632 | 0,9967 | 0,9999 | 1,0 |
| Cặp tráo | 0,0 | 0,0 | 0,0 | 0,0002 | 0,4833 |

101/1.440 cặp thật có ITM < 0,5 (val: 102). Không cặp tráo nào có ITM > 0,5 (val có 1 cặp 0,936); 2 cặp tráo thắng ở các luật gate là do mlp của prompt tráo cao hơn chứ không phải ITM.

## Đối chiếu MLP42 local với holdout chính thức ngày 2026-10-27

| MLP42 trên test | MAE | RMSE | Spearman | Pairwise |
|---|---:|---:|---:|---:|
| Chính thức, checkpoint reviewed `mlp-b8dba09efa0e` | 0,791899 | 0,966854 | 0,437097 | 0,670466 (1.294/1.930) |
| Local retrain `mlp-a575abfed14d` (bản này) | 0,791966 | 0,966937 | 0,436959 | 0,670984 (1.295/1.930) |

Khớp đến 3 chữ số thập phân ở mọi chỉ số; chênh lệch dự đoán từng cặp tối đa 0,0135, trung bình 0,0029 (1.440/1.440 pair_id trùng). **Nói rõ: MLP dùng ở đây là bản train lại local cùng seed/pipeline, đã tái lập đúng số val, nhưng không phải file checkpoint đã review.** Vì vậy cột `mlp` bảng trên là tái lập, không thay thế số holdout chính thức; khi tích hợp web phải dùng checkpoint reviewed + BLIP với checksum ghi lại.

## Giới hạn

- Một lần chạy, một seed tráo (1701); không có khoảng tin cậy. Chênh MAE 0,0016 nằm trong mức nhiễu; bằng chứng mạnh là tráo prompt và Spearman/pairwise.
- `gate_sqrt` không tốt hơn ở RMSE; cần cân nhắc khi báo cáo "cải thiện mọi chỉ số" (chỉ đúng trên val với bộ chỉ số đã báo cáo, không đúng RMSE test).
- BLIP tải từ cache HF offline, chưa có artifact review riêng (checksum từng file, revision pinned trong code). Cache có hai snapshot (`bed8ad…` refs/main với `pytorch_model.bin`, `284ad9…` với `model.safetensors`); cần pin revision khi tích hợp.
- Chưa kiểm external20, bộ v2, lỗi đếm/trạng thái, latency CPU, prompt tiếng Việt. Test này cùng phân phối với train/val (GenAI-Bench), không nói lên tổng quát hóa sang bộ ảnh khác.
- Không chỉnh code, không chỉnh `PROGRESS.md` trong bước này.

## Bằng chứng

`project_records/reports/results/G7_itm_gate_holdout_20261030/`: `result.json`, `scores.csv` (mlp/itm gốc và tráo từng cặp), `run_log.txt`, `checksums.txt` (sha256 của ba file trên và các file đầu vào). Output thô: `.cache/itm-gate-full-test/`.
