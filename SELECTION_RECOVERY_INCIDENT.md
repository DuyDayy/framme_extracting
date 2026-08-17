# Sự cố selection và phương án recovery

## Tóm tắt

Run `l21-l25-prod-v1` đã hoàn thành candidate extraction và đang encode Jina thì
các job `select_video_remote` lỗi. App cũ đã được dừng thủ công. Tại thời điểm
kiểm tra, **82/216 video** có vector checkpoint hợp lệ.

Các vector đã encode không bị mất và không cần encode lại. Recovery hiện tại chỉ
chạy selection bằng CPU trên 82 video đó.

## Lỗi chính

```text
NameError: name 'run_manifest' is not defined
```

Trong `select_video_remote`, manifest được đọc vào biến `manifest`:

```python
manifest = _read_manifest(run, config)
```

Nhưng khi tạo `selection.done.json`, code cũ lại dùng biến không tồn tại:

```python
"gpu_type": run_manifest.get("gpu_type")
```

Sửa đúng:

```python
"gpu_type": manifest.get("gpu_type")
```

Bản sửa có trong commit `fd1d5a5`.

## Ảnh hưởng đến dữ liệu

Không mất các output đã có checkpoint hợp lệ:

```text
/runs/l21-l25-prod-v1/candidates/<video_id>/candidate.done.json
/runs/l21-l25-prod-v1/vectors/<video_id>/vector.done.json
```

Candidate và vector được ghi atomic, kèm SHA-256. Recovery chỉ nhận vector khi:

- có `candidate_vectors.npy` và `vector_rows.jsonl`;
- fingerprint config và encoder khớp manifest;
- hash vector và metadata khớp checkpoint.

Một số thư mục `dataset/<video_id>` có thể chứa WebP hoặc metadata dở dang. Vì
lỗi xảy ra trước khi ghi `selection.done.json`, các thư mục này không được coi là
hoàn tất. Selection chạy lại sẽ tạo và xác minh output trước khi ghi checkpoint.

Flat index và `CURRENT.json` chưa được freeze cho đến khi đủ toàn bộ 216 video.

## Recovery đang dùng

Function `recover_selection_remote` thực hiện:

1. Đọc `RUN_MANIFEST.json` của run cũ.
2. Tìm mọi `vector.done.json` hiện có.
3. Xác minh fingerprint và SHA-256.
4. Chạy `select_video_remote` bằng CPU cho từng vector hợp lệ.
5. Ghi báo cáo vào:

```text
/runs/l21-l25-prod-v1/SELECTION_RECOVERY.json
```

Function này không khởi tạo `CandidateEncoder`, vì vậy không encode lại và không
dùng GPU.

Lệnh chạy:

```bash
MODAL_PROFILE=duy-nguyencse python recover_selection.py \
  --run-id l21-l25-prod-v1
```

Function call hiện tại:

```text
fc-01M07VAF4FFYPT4FQW3E9WZRWE
```

## Phần còn thiếu

App cũ bị dừng khi mới có 82/216 vector. Sau khi selection-only hoàn thành:

1. Encode bổ sung 134 video chưa có vector checkpoint.
2. Không encode lại 82 video đã hợp lệ.
3. Chạy selection cho các vector mới.
4. Kiểm tra đủ 216 `selection.done.json`.
5. Chạy `freeze_remote` để tạo WebP dataset hoàn chỉnh, flat index và
   `CURRENT.json`.

Không được chạy hai job selection hoặc hai encoder cùng ghi vào một `run-id` tại
cùng thời điểm.

## Tiêu chí hoàn tất

- Có đúng 216 `vector.done.json` hợp lệ.
- Có đúng 216 `selection.done.json` hợp lệ.
- `SELECTION_RECOVERY.json` không còn failure.
- `FROZEN_MANIFEST.json` có `status: pass`.
- Số dòng của `index/emb.npy`, `ids.npy`, `frame_idx.npy` và metadata khớp nhau.
- `CURRENT.json` trỏ đúng tới `l21-l25-prod-v1`.

## Các commit liên quan

- `fd1d5a5`: sửa biến manifest trong selection và budget reconciliation.
- `54fc685`: thêm selection-only recovery từ vector checkpoint.
