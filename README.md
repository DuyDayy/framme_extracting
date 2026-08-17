# framme-extracting

Pipeline dùng trực tiếp shot boundaries TransNetV2 đã có. Nó không chạy lại
TransNetV2 và không tự nới shot để che các frame chưa được gán.

## Luồng dữ liệu

```text
video + shot JSON
  -> probe và kiểm biên
  -> 613.256 locator tối thiểu cho mọi cửa sổ 10 frame
  -> 13.744 slot boundary/stable/change/rescue (tổng đúng 627.000)
  -> candidate union theo (video_id, frame_idx)
  -> lọc frame lỗi/transition, quality rescue, local dedup
  -> Candidate Embedding Store: Jina CLIP v2, 1024d, fp16, L2 norm
  -> discovery view + periodic localization view
  -> materialize WebP và gather lại đúng vector đã encode
  -> kiểm hash, manifest, freeze
```

Vector được định danh bằng `video_id + frame_idx + pixel_sha256 +
encoder_fingerprint`. Final vector phải giống từng byte với hàng tương ứng trong
Candidate Embedding Store. Search 512d phải cắt 1024d rồi normalize lại.

## Cài đặt độc lập

```bash
git clone <repository-url>
cd framme_extracting
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[test,modal]'
pytest
```

Đầu vào cần có:

- Modal volume `aic-data-vol`: video ở `/video/<video_id>.mp4`;
- một thư mục local chứa `<video_id>.json` với metadata video và `shots` từ
  TransNetV2;
- Modal profile có quyền đọc `aic-data-vol` và ghi `aic-framme-vol`.

## Thứ tự chạy bắt buộc

```bash
# Dùng đúng Modal profile của dự án
modal profile activate duy-nguyencse

# 1. Kiểm 216 boundary JSON tại máy, chưa ghi hay chạy GPU
modal run modal_app.py --stage preflight --boundaries /path/to/shots

# 1b. Eval CPU 3 video thật; chỉ upload 3 boundary vào namespace pilot riêng
modal run modal_app.py --stage sample-eval --run-id sample-01 --limit 3 \
  --boundaries /path/to/shots

# 2. Upload boundary; video L21-L25 đã nằm ở /video trên aic-data-vol
modal run modal_app.py --stage sync --boundaries /path/to/shots

# 3. Pilot CPU 3 video; full scan chỉ được mở nếu dự phóng thời gian đạt gate
modal run modal_app.py --stage cpu-pilot --max-cpu-wall-hours 4 \
  --boundaries /path/to/shots

# 4. Quét CPU toàn bộ candidate và tạo báo cáo chi tiết
modal run modal_app.py --stage candidates --boundaries /path/to/shots
modal run modal_app.py --stage evaluate --boundaries /path/to/shots

# 5. Chỉ khi eval_gate.json PASS: pilot GPU 3 video để đo FPS/chi phí thật
modal run modal_app.py --stage pilot --max-usd 12 --boundaries /path/to/shots

# 6. Chỉ khi cả eval gate và pilot gate PASS
modal run modal_app.py --stage encode --boundaries /path/to/shots
modal run modal_app.py --stage select --boundaries /path/to/shots
modal run modal_app.py --stage freeze --boundaries /path/to/shots
```

`--stage full` bị vô hiệu có chủ ý để không thể bỏ qua bước đọc báo cáo. Kết quả nằm
trong volume `aic-framme-vol` tại `/runs/<run_id>`. Chỉ khi đủ 216 video và mọi hash
hợp lệ, `freeze` mới cập nhật `/CURRENT.json`.

Với L21–L25 hiện tại, preflight cho 613.256 locator và 13.744 frame đa nguồn,
tổng đúng 627.000 vector 1024d fp16 (khoảng 1,196 GiB). OCR và object cũ dùng hệ `n`
cũ nên phải chạy lại trên metadata/ảnh mới trước khi bật hai nguồn đó; ASR có thể ánh
xạ lại theo `frame_idx/fps` trong `metadata.csv`.

## Eval gate kiểm gì

- đúng số video, shot hợp lệ, không overlap/out-of-range;
- giữ riêng mọi transition gap, không chọn candidate trong gap;
- mỗi shot có locator; mọi cửa sổ 10 frame trong shot chứa ít nhất một locator (100% hình học);
- coverage time-weighted chính xác cho cửa sổ 5/7/9/11/21 frame;
- số candidate và dung lượng vector 1024d dự kiến không vượt cap;
- vector đúng shape/dtype/norm, pixel hash không đổi;
- pilot đo throughput thật và chi phí dự kiến không vượt `--max-usd`;
- vector discovery/locator là phép gather byte-identical từ kho candidate.

Chạy unit test cục bộ:

```bash
pytest -q
```

## Sau khi pipeline chạy xong cần encode gì?

Không cần encode lại ảnh bằng Jina/CLIP. `freeze` đã tạo vector Jina 1024d cho
canonical frames bằng cách gather nguyên byte từ Candidate Embedding Store. Search
512d chỉ cần cắt `vector[:, :512]` rồi L2-normalize lại.

Cần xử lý tiếp các nguồn có ID phụ thuộc bộ keyframe mới:

- **OCR:** chạy lại trên ảnh canonical và ghi theo `(video_id, n)` mới;
- **object detection/caption:** chạy lại vì `n` và frame đã đổi;
- **ASR:** không cần transcribe lại nếu video không đổi; chỉ remap transcript hiện có
  sang keyframe mới bằng `frame_idx / fps` trong `metadata.csv`;
- **query text:** encode mỗi query bằng text tower của đúng revision Jina đã pin. Đây
  là encode truy vấn, không phải encode lại dataset;
- **VLM/crop rerank:** chạy lúc có query trên shortlist, không chạy offline cho toàn
  bộ 627k frame.

Chỉ phải encode lại toàn bộ ảnh khi model/revision/preprocessing hoặc pixel đầu vào
thay đổi. Đổi search từ 1024d sang 512d không cần encode lại.
