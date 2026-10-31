# Thử nghiệm VQAScore zero-shot — hướng dẫn chạy trên Kaggle

Ngày tạo: 2026-10-30. Trạng thái: **NOT RUN** trên dữ liệu thật; mới có unit test với scorer giả.

## Vì sao

Baseline MLP seed42 nối embedding CLIP ViT-B/32 rồi hồi quy. Chẩn đoán ngày 29/10 cho thấy head gần như bỏ qua prompt (đổi 4 prompt trên cùng ảnh, điểm chỉ lệch 0,08) và CLIP B/32 yếu về đếm, vị trí, phủ định. VQAScore (Lin et al. 2024, cùng nhóm tác giả GenAI-Bench) hỏi mô hình VQA "Does this figure show '{prompt}'?" và lấy P(yes); đây là phương pháp tương quan với người cao nhất trên chính GenAI-Bench. Thử nghiệm này đo xem VQAScore có vượt MLP42 trên validation không, trước khi quyết định thay backbone.

Ngưỡng so sánh cố định trước khi chạy (MLP42, validation GenAI-Bench, model_card.md): Spearman 0,480619; pairwise 0,661340; MAE 0,773777. Chỉ coi VQAScore là ứng viên nếu Spearman và pairwise thô trên validation vượt hai số đầu. MAE chỉ so sau khi calibration fit trên train.

## Script

- `code/src/evaluation/vqascore.py`: ghép cặp từ manifest + Parquet (kiểm checksum ảnh/prompt như extraction), gọi scorer theo batch, tính Spearman/pairwise thô theo split, fit calibration tuyến tính và isotonic **chỉ trên train**, báo MAE trên từng split.
- `code/scripts/score_vqascore.py`: CLI, nạp `t2v_metrics.VQAScore`. Mặc định chỉ chấm train + val; test holdout chỉ chạy khi có `--include-test`. Tùy chọn `--external20` chấm 20 ảnh Muse AI theo trung bình 3 người.
- `code/tests/unit/test_vqascore.py`: 7 test, scorer giả; PASS 2026-10-30.

## Chạy trên Kaggle (GPU T4)

Cần manifest đã khóa (`data/manifest.jsonl` trong ZIP training `reviewed-training-20261020T174855207187Z`, hoặc `qa/manifest.jsonl` trong ZIP data QA) và 12 shard Parquet (`/kaggle/input/genai-bench` hoặc tải lại bằng `scripts/download_dataset.py`).

```bash
cd /kaggle/working/promptalign-ai/code
pip install -q t2v-metrics
python scripts/score_vqascore.py \
  --manifest /kaggle/working/training/data/manifest.jsonl \
  --data-root /kaggle/input/genai-bench/data \
  --output /kaggle/working/vqascore-flant5xl \
  --model clip-flant5-xl --device cuda --batch-size 8
```

Smoke trước khi chạy đủ: thêm `--limit 50` (50 cặp mỗi split), output sang thư mục khác. Model `clip-flant5-xl` khoảng 3B tham số, fp16 vừa T4 16GB; batch 8 an toàn. Nếu thiếu RAM GPU giảm `--batch-size 4`. `clip-flant5-xxl` (11B) không vừa T4 nếu không quantize.

Ước lượng thời gian: 8.158 cặp train+val, khoảng 0,15–0,3 giây/cặp trên T4, tức 20–40 phút. Tải checkpoint lần đầu thêm vài phút.

## Đầu ra và cách đọc

- `scores.csv`: pair_id, split, target, score thô. Lưu file này; calibration và phân tích thêm chạy lại được trên CPU local không cần GPU.
- `result.json`: `genai_bench.splits.<split>.raw_spearman`, `raw_pairwise.accuracy`; `genai_bench.calibration.{linear,isotonic}.metrics.val.mae`; tham chiếu MLP42 kèm sẵn.
- `external20_scores.csv` và `external20` trong result.json khi dùng `--external20`: Spearman thô so với −0,307534 của MLP42; MAE sau calibration so với 1,456520.

Status `PASS_EXECUTION_NOT_MODEL_QUALITY` chỉ nghĩa là chạy xong; kết luận chất lượng phải so với ngưỡng ở trên. Không mở holdout để chọn model. External20 đã được dùng chẩn đoán baseline nên không là tập độc lập; nếu VQAScore thắng, vẫn cần bộ ảnh/prompt mới có nhãn người trước khi thay web.

## Bước tiếp nếu VQAScore thắng

1. Ghi kết quả vào `project_records/reports/results/G7_vqascore_<ngày>.md`, giữ nguyên baseline.
2. Quyết định kiến trúc web: thay hẳn head bằng VQAScore + calibration, hoặc đưa VQAScore làm feature thêm vào head. Phương án thay hẳn đơn giản hơn, không cần train.
3. Đo latency/RAM CPU local của CLIP-FlanT5-XL; nếu không chạy nổi CPU, cân nhắc phục vụ trên GPU hoặc model nhỏ hơn.
4. Bộ kiểm độc lập mới có nhãn người, rồi mới UAT và thay checkpoint.

## Chấm ảnh rời với cổng ITM

`code/scripts/score_external_itm_gate.py` chấm một bộ ảnh rời (không phải Parquet GenAI-Bench) bằng head MLP trên CLIP đã khóa, cộng xác suất khớp BLIP-ITM, rồi áp mọi quy tắc trong `src/evaluation/itm_gate.py` (quy tắc chính thức là `gate_sqrt`). Script kiểm tra sha256 của `mlp.pt` và fingerprint CLIP so với `result.json` của thư mục training giống `experiment_itm_gate.py`; lệch là dừng.

Đầu vào:

- `--prompts`: CSV có cột `image_path` hoặc `image_id` và cột `prompt` (chấp nhận `sample_id`, `evaluation_prompt_en` như `data/external20/samples.csv`). Đường dẫn ảnh được tìm trong `--images-dir`, theo đường dẫn tương đối rồi theo tên file.
- `--ratings` (tùy chọn): CSV dạng dài `image_id,rater,score` (hoặc `sample_id`, `score_1_to_5`); target là trung bình các rater của mỗi ảnh. `01` và `01.png` được coi là cùng một ảnh.
- `--training`: thư mục output của `train_regression.py`; `--clip-model`: thư mục CLIP local; `--itm-model` mặc định `Salesforce/blip-itm-base-coco`; `--output` phải chưa tồn tại.

```bash
cd E:\BaiqiL-GenAI-Bench\code
set HF_HUB_OFFLINE=1
set HF_HOME=E:\ai-cache\huggingface
..\.venv\Scripts\python.exe scripts\score_external_itm_gate.py ^
  --prompts ..\data\image_eval_dataset_v2_20_anh\prompts.csv ^
  --images-dir ..\data\image_eval_dataset_v2_20_anh ^
  --ratings ..\data\image_eval_dataset_v2_20_anh\ratings_human_blank.csv ^
  --clip-model ..\.cache\models\clip-vit-base-patch32\3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268 ^
  --training ..\.cache\clip-full-training-seed42 ^
  --output ..\.cache\external-v2-itm-gate ^
  --device cpu
```

Thay `ratings_human_blank.csv` bằng file đã điền điểm thật; file blank chỉ là mẫu. Ảnh của hai bộ ngoài hiện không có trong clone này nên lệnh trên chưa chạy được trên dữ liệu thật.

Đầu ra: `scores.csv` (image_id, prompt, mlp, itm, target nếu có ratings, rồi một cột cho mỗi quy tắc) và `result.json` (sha256 của từng ảnh, của prompts/ratings và checkpoint; MAE/RMSE/Spearman theo từng quy tắc khi có ratings; phân vị ITM). Status `PASS_EXECUTION_NOT_MODEL_QUALITY` chỉ nghĩa là chạy xong, không phải kết luận chất lượng. Kiểm thử end-to-end với ảnh giả và stub CLIP/BLIP nằm ở `code/tests/unit/test_score_external.py`.
