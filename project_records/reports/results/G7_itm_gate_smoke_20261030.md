# G7 — Thử BLIP-ITM làm cổng khớp prompt (smoke trên 1/12 shard)

Ngày 2026-10-30. **Phạm vi: development smoke trên clone mới, máy local RTX 3050 Ti 4GB.** Không phải manifest đã khóa, không phải full data, không mở holdout, không đổi model web. Số liệu chỉ để quyết định có chạy full data hay không.

## Dữ liệu và môi trường

- Tải 1 shard `data/train-00000-of-00012.parquet` revision `94e543b0801feade1419873335191de90b68d164` (sha256 trong `G7_itm_gate_smoke_20261030/checksums.txt`). 804 cặp hợp lệ, 0 loại. Manifest dev tạo bằng `build_manifest` + `assign_splits(seed=42)` trên shard này: train 558 / val 120 / test 126. Vì chỉ có 134 nhóm prompt nên cắt split khác bản khóa; không so trực tiếp với số full-data.
- `.venv` mới: torch 2.14.1+cu126, transformers 5.22.1, numpy 2.5.4. CLIP ViT-B/32 revision pinned tải lại bằng `scripts/download_clip.py`. BLIP `Salesforce/blip-itm-base-coco` và `blip-itm-large-coco` từ Hugging Face.
- `t2v-metrics` (VQAScore) **không cài được** vào venv này: build dependency lỗi `No module named 'pkg_resources'`, và gói pin torch==2.5.1, transformers==4.49.0, numpy==1.26.4 nên sẽ phá môi trường. CLIP-FlanT5-XL cũng không vừa GPU 4GB. VQAScore thật phải chạy trên Kaggle với venv riêng.

## Kết quả trên val 120 cặp (177 cặp cùng prompt đủ chênh nhãn)

| Scorer | MAE | Spearman | Pairwise cùng prompt | Tráo prompt: gốc thắng |
|---|---:|---:|---:|---:|
| M0 trung bình train | 0,864 | – | – | – |
| CLIP cosine thô | – | 0,023 | 0,627 | 120/120 |
| BLIP-ITM base thô (zero-shot) | – | 0,130 | 0,785 | 120/120, drop 0,96 |
| BLIP-ITM large thô | – | 0,107 | 0,751 | – |
| MLP nối CLIP, train trên shard, seed42 | 0,651 | 0,627 | 0,785 | **56/120, drop −0,017** |
| MLP nối + ITM + cosine làm feature (3 seed) | 0,669 | 0,576 | 0,800 | 67% |
| MLP nối + negatives tráo prompt 25% (3 seed) | 0,825 | 0,498 | 0,723 | 56% |
| **Fusion `1 + (mlp − 1) × itm`**, seed42 | **0,648** | **0,650** | **0,791** | **120/120, drop 2,25** |
| Fusion `min(mlp, 1 + 4·itm)` | 0,654 | 0,642 | 0,785 | 120/120 |

Tráo prompt: mỗi ảnh val chấm với prompt gốc và prompt của nhóm khác (seed 0/1701). Đây là kiểm độ nhạy, không phải nhãn người.

## Đọc kết quả

- Tái hiện đúng lỗi người dùng thấy trên web: head nối CLIP train như hiện tại **không phân biệt prompt đúng/sai** (56/120, gần tung đồng xu). Lý do cấu trúc: nhãn GenAI-Bench toàn cặp ảnh sinh từ chính prompt, head chỉ cần học prior ảnh.
- BLIP-ITM zero-shot phát hiện mismatch gần tuyệt đối (ITM median cặp thật 0,999; cặp tráo ≈ 0) nhưng không có thang chung giữa các prompt nên Spearman thấp.
- Đưa ITM làm feature trong MLP không đủ (67%): 1 chiều bị chìm trong 1024. Negatives trên 558 cặp train làm hỏng MAE.
- **Nhân cổng** `1 + (mlp − 1) × itm` giữ nguyên chất lượng trong miền (MAE/Spearman/pairwise bằng hoặc hơn MLP) và sửa hoàn toàn ca tráo prompt. Đây là ứng viên rẻ nhất: không train lại, BLIP base 224M chạy được CPU.
- Chưa kiểm: lỗi đếm/trạng thái (2 cam vs 3 cam, ô đóng/mở). ITM của BLIP có thể không bắt được; cần VQAScore hoặc VLM cho loại lỗi này.

## Giới hạn

N nhỏ, một shard, split dev, chỉ 3 seed cho head và 1 seed cho fusion. Luật fusion chọn sau khi xem val nên val này không còn độc lập; phải xác nhận trên full validation với manifest khóa rồi mới holdout một lần. Không có external20 trong clone (ảnh bị Git ignore) nên chưa chạy lại ca ô tô/xe đạp thật.

## Bằng chứng

Thư mục `G7_itm_gate_smoke_20261030/`: script dev (`build_smoke_manifest.py`, `hybrid_itm.py`, `swap_check.py`, `neg_hybrid.py`), result.json của ba lần chạy, checksums. Output thô ở `.cache/blip-itm-smoke-20261030`, `.cache/blip-itm-large-smoke-20261030`, `.cache/clip-smoke-extraction`, `.cache/clip-smoke-training` (Git ignore). Lệnh chính:

```text
code> ..\.venv\Scripts\python.exe scripts\score_vqascore.py --manifest ..\.cache\genai-bench-smoke\manifest_dev.jsonl --data-root ..\.cache\genai-bench-smoke\data --output ..\.cache\blip-itm-smoke-20261030 --scorer blip-itm --model Salesforce/blip-itm-base-coco --device cuda --batch-size 16
code> ..\.venv\Scripts\python.exe scripts\extract_embeddings.py ... --development ; scripts\train_regression.py ... --development --seed 42
```

## Bước tiếp theo đề xuất

1. Tải đủ 12 shard (≈4,6GB), chạy BLIP-ITM base + CLIP trên full train/val với manifest khóa (đối chiếu sha256 manifest với lock). GPU local đủ: ITM ≈ 10 cặp/giây.
2. Cố định luật fusion `1 + (mlp − 1) × itm` với MLP seed42 đã khóa, tính val đầy đủ; nếu giữ MAE/pairwise và sửa mismatch, chạy holdout một lần theo protocol cũ.
3. Chạy external20 và bộ v2 thật trên máy gốc có ảnh.
4. VQAScore trên Kaggle như VQASCORE.md để xử lý đếm/trạng thái; so với fusion ITM trước khi chọn.
