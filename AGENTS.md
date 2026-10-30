# AGENTS.md — Retail ALS trên Delta Lake, Spark và GPU

Tài liệu thực thi cho agent (Claude Code, Codex). Đọc hết trước khi sửa code hoặc chạy pipeline.
Đề cương nội dung bài tiểu luận nằm ở `de_cuong_tieu_luan.md`; file này mô tả **hệ thống đang chạy**
và các quy tắc phải giữ.

## 1. Mục tiêu

Notebook `tieu_luan_retail_als.ipynb` vừa là bài tiểu luận vừa là pipeline batch:

```text
T-ECD Small Retail (Parquet)
  → Bronze Delta → Silver Delta → Gold DWH (Star Schema) → Gold aggregates → Spark SQL / dashboard
                                        └→ Gold ML mart (split thời gian, rút gọn cặp) → GPU Implicit ALS + Popularity + RepeatPop
                                                                                      → E2, E1, gold.recommendations
```

Tất cả bảng là Delta trên MinIO (`s3a://bigdata/retail_als/<RUN_ID>`). Spark (PySpark 4.2, CPU) làm ETL
và SQL; ALS và chấm Top-10 chạy trên NVIDIA CUDA bằng CuPy (`tools/cuda_als.py`, `tools/gpu_als.py`).

## 2. Quy tắc bắt buộc

1. **Logic nằm trong notebook.** Không tạo module `.py` mới cho pipeline. `tools/` chỉ chứa runtime,
   backend GPU, runner và script kiểm tra đã có; chỉ sửa khi thay đổi đúng phạm vi đó.
2. **Không đọc/in `.env`.** File chứa mật khẩu MinIO. Chỉ đọc từng khóa cần thiết (như
   `run_notebook.sh` đang làm); không `cat`, `source` hay in giá trị khóa ra log/output.
3. **Không dùng test để chọn.** Tham số, trọng số, ngưỡng rút gọn chỉ dựa trên train/validation.
   Lựa chọn E2 bị khóa trước E1; không xóa lock để chạy lại test.
4. **Giao thức gắn với RUN_ID.** Đổi bất kỳ khóa nào trong `PROTOCOL_KEYS` (nguồn, cửa sổ, trọng số,
   `ml_reduction`, lưới ALS, k, seed, …) phải dùng `--run-id` mới; checkpoint cũ sẽ từ chối giao thức khác.
5. **Một writer cho một RUN_ID.** Notebook giữ `fcntl` lock `results/<RUN_ID>/.writer.lock`.
6. **Bộ nhớ có giới hạn.** Không bỏ `systemd-run` scope; không `collect()`/`toPandas()` bảng lớn;
   không tạo ma trận dense user × item.
7. **Không bịa kết quả.** Markdown kết luận chỉ dựa trên output đã chạy. Lượt chạy thử phải dùng RUN_ID
   riêng và được dọn sau khi kiểm tra (mục 12).
8. Viết code theo phong cách hiện có của notebook (tên ngắn, comment tiếng Việt khi cần, hàm nhỏ).

## 3. Bản đồ file

| File | Vai trò |
|---|---|
| `tieu_luan_retail_als.ipynb` | Bài tiểu luận + toàn bộ pipeline (99 cell) |
| `run_notebook.sh` | Chạy từ terminal: giới hạn RAM/swap, MinIO, log, resume, `status`, `log` |
| `de_cuong_tieu_luan.md` | Đề cương để hoàn thiện notebook thành bài tiểu luận |
| `architecture.svg` | Sơ đồ kiến trúc (phần 2 notebook) |
| `docker-compose.yml`, `docker/minio/` | MinIO build từ mã nguồn, tạo bucket `bigdata` |
| `.env.example` | Mẫu cấu hình (không chứa bí mật thật) |
| `requirements-notebook.txt`, `requirements-gpu.txt` | Thư viện đã khóa |
| `tools/runtime.py` | Nạp `.env`, JDK 21, `systemd-run` scope (`MemoryMax`, `MemorySwapMax`) |
| `tools/cuda_als.py` | CUDA Implicit ALS (CG, constant L2) |
| `tools/gpu_als.py` | CSR từ Spark, fit/lưu/nạp model, Top-K theo tile, population đánh giá |
| `tools/notebook_runner.py`, `tools/execute_notebook.py` | Chạy từng cell, stream log, lưu bản đã chạy |
| `tools/gpu_kernel.py`, `tools/setup_gpu_kernel.py` | Kernel Jupyter `retail-als-gpu` có giới hạn bộ nhớ |
| `tools/check_cuda_solver.py`, `tools/check_gpu.py` | Kiểm tra hồi quy CUDA |
| `tools/prepare_small_dataset.py` | Tạo cohort user nhỏ |
| `tools/profile_recommendation_design.py` | Hồ sơ nguồn → `results/design_review/source_profile.json` |
| `t_ecd_data/` | Dữ liệu nguồn và cohort (không sửa) |

## 4. Môi trường

Máy đã kiểm tra: RTX 4050 Laptop 6 GiB VRAM, khoảng 15 GiB RAM, Linux. Stack: Python 3.14 (`.venv`),
Java 21, PySpark 4.2.0, Delta Spark 4.4.1, Hadoop/S3A 3.5.0, Scala 2.13, pandas 2.3.3, CuPy CUDA 13.

```bash
python3.14 -m venv .venv
uv pip install --python .venv/bin/python -r requirements-gpu.txt   # gồm requirements-notebook.txt
.venv/bin/python tools/setup_gpu_kernel.py      # kernel "Retail ALS GPU (CUDA / bounded swap)"
.venv/bin/python tools/check_cuda_solver.py     # CUDA solver so với NumPy
.venv/bin/python tools/check_gpu.py             # fit, lưu/nạp, Top-K khi hòa điểm, lọc item đã thấy, population
```

MinIO (lần đầu cần mạng để build):

```bash
docker compose up -d --build    # server minio + tác vụ minio-init tạo bucket bigdata
docker compose ps -a
```

| Cấu hình | Mặc định |
|---|---|
| S3 API / console | `http://127.0.0.1:19000` / `http://127.0.0.1:19001` (chỉ loopback) |
| Bucket, lake root | `bigdata`, `s3a://bigdata/retail_als` |
| Khóa | `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY` trong `.env` |
| Dữ liệu | volume `retail-lake_minio-data`; `docker compose down -v` sẽ xóa dữ liệu |

Biến môi trường chính (`.env.example`): `MINIO_ENDPOINT`, `RETAIL_LAKE_ROOT`, `RETAIL_DATA_REVISION`,
`RETAIL_JOB_MEMORY_MAX=7G`, `RETAIL_JOB_SWAP_MAX=4G`, `SPARK_MASTER=local[2]`, `SPARK_DRIVER_MEMORY=3g`,
`RETAIL_GPU_HOST_BUDGET_GB=2.5`, `RETAIL_GPU_USER_BATCH=64`, `RETAIL_GPU_ITEM_BATCH=8192`, `RETAIL_GPU_CG_STEPS=5`.

## 5. Chạy

```bash
./run_notebook.sh                                   # toàn bộ Retail, RUN_ID retail_star_full_v1
./run_notebook.sh --source small                    # cohort 2,5% user, RUN_ID retail_star_small_v1
./run_notebook.sh --run-id ID --min-views N         # RUN_ID riêng, ngưỡng rút gọn khác (1 = không rút gọn)
./run_notebook.sh status [--run-id ID]              # trạng thái checkpoint, không khởi động Spark
./run_notebook.sh log                               # tail -F log mới nhất
```

Tùy chọn khác: `--memory 7G`, `--swap 4G`, `--heartbeat 60`, `--source PATH` (thư mục có `events/` và `items.pq`),
`--kill-stale`.

Trước mỗi lượt chạy, script từ chối nếu đang có `run_notebook.sh` khác, rồi quét tiến trình Python/Java còn sót
của dự án (cwd là thư mục dự án; runner, Spark JVM, pyspark worker, và kernel chỉ khi đang giữ Spark JVM con
hoặc khóa `.writer.lock`; kernel rảnh của VS Code được bỏ qua): hỏi trước khi dừng (SIGTERM, sau 15 giây
SIGKILL), hoặc dừng luôn với `--kill-stale`; chạy không tương tác mà không có cờ thì dừng script. Sau đó in RAM khả
dụng, VRAM đang dùng và cảnh báo nếu tiến trình khác giữ hơn 256 MiB VRAM. Script không đụng tiến trình ngoài dự án.

- Log: `results/<RUN_ID>/logs/run_*.log` (`latest.log`); dòng `STAGE_START/DONE/FAILED/SKIP`,
  `HEARTBEAT` (RAM, swap, VRAM, stage Spark đang chạy), `CELL_START/PASSED/FAILED`.
- Notebook đã chạy: `results/<RUN_ID>/execution/latest.ipynb` (ghi nguyên tử sau mỗi cell).
- Kết thúc thành công: `NOTEBOOK_ALL_CELLS_PASSED` và `HOÀN TẤT`. Notebook dừng sau bước phát hành và tổng kết;
  cell cuối gọi `spark.stop()` để JVM tự dọn `.spark-local/<RUN_ID>` trước khi kernel tắt (ipykernel gửi SIGTERM
  cho tiến trình con khi shutdown). Muốn truy vấn tiếp trong VS Code thì chạy lại từ đầu (stage đã xong được bỏ qua).
- Trong VS Code/Jupyter: chọn kernel **Retail ALS GPU (CUDA / bounded swap)** rồi Run All (cùng checkpoint).

Thời gian đo trên cohort nhỏ (7,9 triệu sự kiện): khoảng 25 phút, Bronze chiếm hơn một nửa.

## 6. Checkpoint và resume

- `results/<RUN_ID>/checkpoint/state.json`, ghi nguyên tử (file tạm → fsync → rename → fsync thư mục).
- Stage theo thứ tự: `source_survey`, `bronze`, `delta_checks`, `silver`, `warehouse`, `gold_aggregates`,
  `ml_split`, `ml_mart`, `popularity_validation`, `e2`, `e1`, `publish`, `repeat_baseline`. Stage `done` được bỏ qua khi chạy lại.
- Tiến độ trong stage: từng ngày Bronze, từng bảng Gold/ML, từng cấu hình E2, từng giao thức/mô hình E1.
- Delta đọc theo version đã ghim trong checkpoint (`read_delta`), không đọc bản mới nhất.
- Giá trị ghi một lần (`lock_value`): `manifest`, `protocol`, `selection`.
- Mỗi stage lưu `status`, `attempts`, `seconds` (lần hoàn tất), `interrupted_seconds`, đỉnh RAM/swap/VRAM,
  số stage Spark, shuffle read/write, spill bộ nhớ/đĩa. Ctrl+C ghi `interrupted`; lỗi ghi `failed` kèm `error`.
- Sau mất điện: khởi động MinIO (script tự làm nếu health check lỗi) rồi chạy lại đúng lệnh cũ.

## 7. Dữ liệu đã kiểm tra

Nguồn `t_ecd_data/dataset/small/retail`, revision `79bcf78cd4699267f561ad3fd51fe7b52af53083`
(chỉ miền Retail của T-ECD Small; dữ liệu đã ẩn danh và gây nhiễu có kiểm soát).
Bằng chứng: `results/design_review/source_profile.json`.

| Đối tượng | Toàn bộ Retail | Cohort `retail_users025` |
|---|---:|---:|
| File sự kiện / ngày tương đối | 227 / 1082–1308 | 227 / 1082–1308 |
| Sự kiện | 267.043.972 | 7.903.187 |
| User / item có sự kiện | 140.062 / 257.327 | 3.640 / 142.026 |
| Catalog | 250.856 item, `item_id` duy nhất | như nguồn, bỏ `embedding` |
| Parquet nguồn | 2,13 GB | 79,7 MB |

- Schema sự kiện: `timestamp: duration[us]`, `user_id: uint64`, `item_id`, `subdomain`, `action_type`, `os`
  (string). Không có null; ngày trong timestamp luôn khớp tên file.
- Catalog: `item_id`, `brand_id: uint64`, `category`, `subcategory`, `price: double`, `embedding[300]`.
- Hành vi: `view` 253,5 triệu, `order` 7,43 triệu, `added_to_cart` 3,85 triệu, `click` 2,29 triệu.
  Subdomain `search`, `catalog`, `other`, `i2i`, `main`; OS `android`, `ios`, `other`.
- 8.521 item của sự kiện thiếu trong catalog (14.097 sự kiện) → member Unknown riêng cho từng item.
  9.615 dòng catalog thiếu category/subcategory; 248.431 price âm; 725 price null → giữ `price_raw` và cờ, không tính doanh thu.
- Không có event_id, order_id, quantity: `order` là sự kiện mua, không phải số đơn hay doanh thu.
- Thời gian là tương đối: dùng `event_us`, `day_index`; không tạo ngày lịch, thứ hay mùa.
- Cohort: user có `SplitMix64(user_id) % 40 == 0`, giữ mọi sự kiện của user được chọn
  (`t_ecd_data/reduced/retail_users025/cohort.json`). Cohort khác:
  `tools/prepare_small_dataset.py --modulus N --destination t_ecd_data/reduced/<tên>` rồi `--source <thư mục>`.

## 8. Yêu cầu nghiệm thu

| ID | Yêu cầu | Điều kiện |
|---|---|---|
| R01 | Gợi ý batch tối đa 10 item/user | Không lặp item; có score, rank, `model_version`, `cutoff_us`, `source`; ít hơn 10 khi thiếu ứng viên |
| R02 | Implicit ALS trên GPU | Strength từ hành vi; ghi backend, solver, rank, regularization, alpha, seed, iterations; lỗi CUDA/VRAM báo rõ, không fallback CPU |
| R03 | Popularity đối chứng và fallback | Chỉ dùng lịch sử được phép; user không có lịch sử (hoặc mất hết cặp sau rút gọn) nhận fallback, báo riêng |
| R04 | Giao thức thời gian | Validation 14 ngày, test 14 ngày, gap 12 giờ; chia thời gian trước tổng hợp; khóa trước test |
| R05 | Chính sách ứng viên | Giao thức chính cho mua lại; giao thức phụ loại item đã thấy (lịch sử đầy đủ) khỏi ứng viên và nhãn; báo population, coverage, cold-start |
| R06 | Chất lượng và tái lập | Recall@10, NDCG@10 cùng population/nhãn với baseline; lưu snapshot, Delta version, mapping, config, model |
| R07 | Spark ETL trên Delta/MinIO | Bronze giữ fidelity và lineage; Silver chuẩn hóa/quarantine; nạp lại không nhân đôi; có history/version |
| R08 | Star Schema | Fact grain = một sự kiện; khóa dimension duy nhất; join không làm tăng/mất sự kiện; giữ os, subdomain, thuộc tính sản phẩm |
| R09 | Spark SQL và dashboard | Theo ngày, hành vi, category, brand, OS, subdomain; long-tail; thiếu catalog; ghi phạm vi dữ liệu; không gọi revenue/conversion |
| R10 | EDA phục vụ train | Hoạt động user/item, sparsity, hành vi, ngưỡng rút gọn; thống kê dùng cho quyết định chỉ lấy train |
| R11 | Ưu tiên RAM | RAM → Spark spill SSD (`.spark-local/<RUN_ID>`) → swap OS có hạn mức; đo đỉnh |
| R12 | Hiệu năng | Một lượt chạy toàn bộ Retail; mỗi stage ghi thời gian, đỉnh RAM/swap/VRAM, shuffle/spill vào `results.run_performance` |
| R13 | Vận hành và phục hồi | Model + mapping + cutoff phát hành cùng version; chạy lại idempotent; checkpoint, log lỗi, resume |

## 9. Bảng dữ liệu

**Bronze/Silver**

| Bảng | Nội dung |
|---|---|
| `bronze.retail_events` | Sự kiện gốc, partition theo ngày, ghi `replaceWhere` từng ngày; `_event_record_key=(day<<40)+id`, `_batch_id`, `_source_revision` |
| `bronze.retail_products` | Catalog gốc |
| `silver.events` / `silver.events_quarantine` | Sự kiện sạch (`event_us`, `day_index`, `os`/`subdomain` rỗng → `unknown`) / dòng bị loại kèm lý do |
| `silver.products` / `silver.products_quarantine` | Một dòng/`item_id`, cờ thiếu brand/category/subcategory/price, giá âm/nonfinite / dòng thiếu hoặc trùng ID |

Kiểm tra Delta (stage `delta_checks`): nạp lại một ngày không nhân đôi, `versionAsOf`, `DESCRIBE HISTORY`,
ghi sai schema bị từ chối, khóa bản ghi ổn định khi nạp lại.

**Gold DWH (Star Schema)**

| Bảng | Grain và cột |
|---|---|
| `gold.fact_interactions` | Một sự kiện; `event_record_key`, `user_key`, `product_key`, `day_key`, `action_key`, `context_key`, `event_us`, `event_count=1`, `source_batch_id`; partition `day_key` |
| `gold.dim_user` | `user_key`, `user_id` |
| `gold.dim_product` | `product_key`, `item_id`, `brand_id`, `category`, `subcategory`, `price_raw`, `catalog_present`, cờ thiếu, `catalog_snapshot_id`; mỗi item thiếu catalog có member Unknown riêng |
| `gold.dim_day` | `day_key`, `day_index` (ngày tương đối) |
| `gold.dim_action` | `action_key`, `action_type`, `is_purchase` |
| `gold.dim_context` | `context_key`, `os`, `subdomain` |
| `gold.agg_*` | 10 bảng Spark SQL: overview, daily, daily_action, category_action, brand_action, context_action, catalog_coverage, user_activity, item_activity, item_long_tail |

Kiểm tra: khóa và khóa tự nhiên của mỗi dimension duy nhất; số dòng fact = số dòng Silver; số member Unknown = số item thiếu catalog.

**Gold ML mart** (đọc fact + dimension Gold ở grain sự kiện, lọc `event_us` theo cutoff trước khi tổng hợp; không đọc `gold.agg_*`)

| Bảng | Nội dung |
|---|---|
| `gold_ml.validation_labels`, `gold_ml.test_labels` | Cặp `user_id`, `item_id` có `order` trong cửa sổ |
| `gold_ml.{train,refit}_pair_actions` | Một cặp user–item trước cutoff: `view_events`, `click_events`, `added_to_cart_events`, `order_events`, `kept` |
| `gold_ml.{train,refit}_{user,item}_index` | Index Int32 từ 0, chỉ gồm thực thể còn cặp `kept`, sắp theo ID gốc |
| `gold_ml.train_user_item_strength_<profile>` | `user_index`, `item_index`, `interaction_strength` = Σ trọng số × số sự kiện, chỉ cặp `kept` |
| `gold_ml.refit_user_item_strength_<profile đã chọn>` | Tạo trong E1 |
| `gold_ml.{train,refit}_item_popularity` | Số sự kiện `order` mỗi item |
| `gold_ml.train_eda_*` | Spark SQL chỉ trên train: actions, category, context, user_activity, item_activity, reduction |

**Kết quả**: `results.validation_popularity_recs`, `results.e2_recs_<config>`, `results.e2_validation`,
`results.e1_<protocol>_<model>_recs`, `results.e1_cold_start_popularity_recs`, `results.e1_test`,
`results.validation_repeatpop_recs`, `results.e1_main_repurchase_repeatpop_recs`, `results.repeat_baseline`,
`gold.recommendations`, `results.model_registry`, `results.model_coverage`, `results.run_performance`.
Artifact: `results/<RUN_ID>/artifacts/<model_version>/` gồm `model/`, `user_index.parquet`,
`item_index.parquet`, `manifest.json`; `model_version = <RUN_ID>.<config_id>.<protocol_sha[:8]>`.

## 10. Mô hình và giao thức

**Cấu hình (`CFG`, đều thuộc giao thức)**

| Khóa | Giá trị |
|---|---|
| `validation_days`, `test_days`, `gap_hours` | 14, 14, 12 |
| `purchase_action` | `order` |
| `weight_profiles` | `uniform` (1,1,1,1); `purchase_focused` (view 1, click 2, added_to_cart 3, order 5) |
| `ml_reduction` | `{'action':'view','min_events':2}`; env `RETAIL_ML_MIN_VIEW_EVENTS` hoặc `--min-views` |
| `als_grid` | rank {32, 64} × regParam {0,05; 0,1} × alpha {10, 40} → 16 cấu hình với 2 profile |
| `k`, `max_iter`, `seed`, `cg_steps` | 10, 10, 42, 5 |

**Chia thời gian**: test = 14 ngày cuối, validation = 14 ngày trước đó. Lịch sử train kết thúc 12 giờ trước
đầu validation; lịch sử refit kết thúc 12 giờ trước đầu test. Nhãn = cặp user–item có `order` trong cửa sổ.

**Rút gọn ở Gold ML**: Bronze/Silver/Gold DWH giữ đủ sự kiện. Gold ML bỏ cặp chỉ có `view` và ít hơn
`min_events` lần; cặp có click/added_to_cart/order luôn giữ. Lý do: thời gian fit ALS tỉ lệ với số cặp khác 0,
không với số dòng sự kiện. Trên train toàn bộ Retail: 49,8 triệu → 20,8 triệu cặp (41,8%), mất 4,5% user
và 9,2% item của train. So sánh trên cohort (ngưỡng 1 với 2): cùng cấu hình được chọn, NDCG@10 E1 giao thức
chính 0,1138 với 0,1135, fit CUDA E2 giảm 32%, thêm 156/3.640 user nhận fallback.

**ALS GPU**: confidence `1 + alpha * strength`, regularization constant L2, CG 5 bước/lần cập nhật, 10 vòng.
CSR float32/int32 dựng từ Parquet theo batch 65.536 dòng; kiểm tra ngân sách RAM host và VRAM trước fit.
Top-10 theo tile 64 user × 8192 item; hòa điểm xếp `item_index` tăng dần.

**Popularity**: điểm = số sự kiện `order` trong lịch sử; hòa điểm theo `item_index`.

**RepeatPop** (stage `repeat_baseline`, mục 11.6, thêm sau khi đã biết E1): item user đã `order` trước cutoff, xếp
số order ↓ → tổng sự kiện của cặp ↓ → `item_index` ↑, tối đa 10; chỗ còn lại lấp Popularity cùng lịch sử (score 0).
Không có tham số; định nghĩa khóa bằng `lock_value('repeat_baseline')` trước test. Chỉ giao thức chính, vì ở
`novel_items` RepeatPop trùng Popularity theo định nghĩa. Không đổi E2/E1/`gold.recommendations`.

**E2**: fit 16 cấu hình trên train, chọn theo NDCG@10 validation (giao thức chính), hòa → Recall@10 → `config_id`.
**E1**: refit một lần trên lịch sử refit với cấu hình đã khóa; đánh giá test theo giao thức chính
(`main_repurchase`) và phụ (`novel_items`, loại item đã thấy theo lịch sử refit đầy đủ, gồm cả cặp bị rút gọn);
báo riêng user cold-start (Popularity) và `users_reduced_out`.
**Chỉ số**: Recall@10, NDCG@10 macro theo user, relevance nhị phân; user đủ điều kiện có nhãn và có trong index.
**Phát hành**: mọi user của `gold.dim_user` nhận Top-10: `als` nếu có trong index refit, `popularity_fallback` nếu không.

## 11. Hiệu năng

Phần 12 notebook đọc checkpoint, lưu `results.run_performance` và `run_performance.json`: thời gian từng stage,
số lần thử, thời gian bị ngắt, số dòng đầu ra chính, đỉnh RAM/swap (cgroup, mẫu 2 giây), VRAM (10 giây),
shuffle/spill (REST API Spark UI, đọc mỗi 10 giây). Đây là một lần đo; không suy rộng sang quy mô khác.
Event log Spark ở `results/<RUN_ID>/spark-events` (dùng History Server). Ảnh minh chứng: `results/<RUN_ID>/spark_ui.png`.

Giới hạn tài nguyên: `MemoryMax=7G`, `MemorySwapMax=4G`; Spark `local[2]`, heap 3 GiB, direct memory/metaspace
512 MiB; S3A upload qua đĩa; shuffle 64 partition (cohort 16). Máy thiếu RAM chung thì job vẫn có thể bị swap
trong hạn mức; ghi nhận, không coi là lỗi.

## 12. Kiểm tra trước khi báo xong

1. Mọi code cell của notebook parse được bằng `ast.parse`; id cell không trùng.
2. `tools/check_gpu.py` in `ALL_GPU_CHECKS_PASSED` khi sửa `tools/gpu_als.py` hoặc `tools/cuda_als.py`.
3. Thay đổi pipeline: chạy cohort với RUN_ID riêng, ví dụ
   `./run_notebook.sh --source small --run-id retail_star_<việc>_check`, đến `NOTEBOOK_ALL_CELLS_PASSED`.
   Chạy lại cùng lệnh phải `STAGE_SKIP` đủ 13 stage.
4. Dọn lượt chạy thử (không chạm RUN_ID thật):

```bash
docker exec retail-lake-minio-1 sh -c 'mc alias set local http://127.0.0.1:9000 "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD" >/dev/null && mc rm --recursive --force local/bigdata/retail_als/<RUN_ID>'
rm -rf results/<RUN_ID> .spark-local/<RUN_ID>
```

5. Báo kết quả đúng như output: lỗi thì kèm log, bước chưa chạy thì nói rõ.
