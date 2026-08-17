# framme-extracting

Repo production độc lập để cắt keyframe từ shot boundaries TransNetV2 và encode
Jina CLIP v2 trên Modal. Pipeline không chạy lại TransNetV2 và không còn stage eval.

## Pipeline production

```text
216 video L21-L25 + shot JSON
  -> tạo đúng 627.000 candidate
  -> decode, quality rescue và lọc frame lỗi
  -> Jina CLIP v2 1024d fp16 theo mini-batch
  -> semantic/local dedup và canonical selection
  -> ghi lossless WebP + metadata
  -> gather lại vector đã encode, không encode ảnh lần hai
  -> đóng băng flat index và CURRENT.json
```

GPU encode được cấu hình ramp lên **10 container A10 đồng thời** và không vượt quá
10 GPU. Mỗi video là một job; trong mỗi job ảnh được encode theo mini-batch, mặc
định 32 ảnh. Các lệnh `sync/status` không giữ GPU chạy nền.

Pha candidate và canonical selection được phép scale tối đa 100 CPU container.
Encoder được giới hạn riêng ở 10 GPU container.

## Cài đặt

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[test,modal]'
modal profile activate duy-nguyencse
```

Modal cần có:

- `aic-data-vol/video/<video_id>.mp4`;
- `aic-framme-vol` để ghi output;
- `hf-cache` để cache model;
- thư mục local chứa đúng 216 file shot JSON của L21-L25.

## Chạy toàn bộ

```bash
modal run modal_app.py \
  --stage run \
  --run-id l21-l25-prod-v1 \
  --image-batch-size 32 \
  --boundaries /absolute/path/to/shot-json
```

Một lệnh trên tự chạy bốn pha: candidate CPU, encode 10 GPU, materialize WebP và
freeze index. Ngay khi một GPU encode xong một video, job CPU của video đó ghi WebP
trong khi các GPU còn lại tiếp tục encode. Nếu tiến trình bị ngắt, chạy lại đúng
lệnh và `run-id`; checkpoint hợp lệ sẽ được bỏ qua. Nếu thay code/config, dùng
`run-id` mới.

Chỉ đồng bộ boundary mà chưa chạy production:

```bash
modal run modal_app.py --stage sync --boundaries /absolute/path/to/shot-json
```

Xem tiến độ:

```bash
modal run modal_app.py --stage status --run-id l21-l25-prod-v1
```

## Frame nằm ở đâu?

Ảnh chỉ xuất hiện sau pha canonical selection, tại volume `aic-framme-vol`:

```text
/runs/l21-l25-prod-v1/dataset/<video_id>/images/000001.webp
/runs/l21-l25-prod-v1/dataset/<video_id>/metadata.csv
/runs/l21-l25-prod-v1/dataset/<video_id>/discovery_vectors.npy
/runs/l21-l25-prod-v1/index/emb.npy
/CURRENT.json
```

Kiểm tra trực tiếp:

```bash
modal volume ls aic-framme-vol /runs/l21-l25-prod-v1/dataset/L21_V001/images
modal volume get aic-framme-vol \
  /runs/l21-l25-prod-v1/dataset/L21_V001/images/000001.webp \
  ./000001.webp
```

Lần chạy trước là `sample-eval`, vì vậy chỉ sinh report và candidate tạm; nó không
đi đến pha canonical selection nên không có WebP trong `dataset/.../images`.

## Sau khi chạy xong

- Không encode lại ảnh bằng Jina: `emb.npy` đã chứa vector 1024d canonical.
- Search 512d: lấy 512 chiều đầu rồi L2-normalize, không encode ảnh lại.
- Chạy OCR và object detection/caption trên bộ WebP mới.
- ASR không cần transcribe lại nếu video không đổi; chỉ remap theo `frame_idx/fps`.
- Query text được encode lúc truy vấn bằng đúng revision Jina đã pin.
- VLM/crop rerank chỉ chạy trên shortlist lúc truy vấn.

Chỉ encode lại toàn bộ ảnh khi đổi model, revision, preprocessing hoặc pixel nguồn.

## Kiểm tra cục bộ

```bash
python -m py_compile modal_app.py src/framme_extracting/*.py
pytest -q
modal run modal_app.py --help
```
