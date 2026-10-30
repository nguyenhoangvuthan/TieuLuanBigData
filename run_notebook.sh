#!/usr/bin/env bash
# Chạy tieu_luan_retail_als.ipynb từ terminal, ghi log để trace và tự resume từ checkpoint.
#
#   ./run_notebook.sh [run|status|log] [--source full|small|PATH] [--run-id ID]
#                     [--memory 7G] [--swap 4G] [--heartbeat 60] [--min-views 2] [--kill-stale]
#
#   run     (mặc định) chạy notebook; stage đã xong được bỏ qua, stage dở dang chạy tiếp.
#   status  in trạng thái checkpoint, không khởi động Spark.
#   log     theo dõi log mới nhất (tail -F).
#   --min-views  Gold ML bỏ cặp user–item chỉ có view và ít hơn N lần (1 = không rút gọn);
#                đổi giá trị thì dùng --run-id mới.
#   --kill-stale Dừng kernel/runner/Spark JVM còn sót của dự án mà không hỏi (mặc định hỏi trước).
#                Trước mỗi lượt chạy, script quét tiến trình sót và in RAM/VRAM còn trống.
#
# Sau khi tắt máy đột ngột: chạy lại đúng lệnh cũ (cùng --source/--run-id).
set -euo pipefail

project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$project"

command=run
source_name=full
run_id=''
memory=''
swap=''
heartbeat=60
kill_stale=0
while (($#)); do
    case "$1" in
        run|status|log) command=$1 ;;
        --source) source_name=$2; shift ;;
        --run-id) run_id=$2; shift ;;
        --memory) memory=$2; shift ;;
        --swap) swap=$2; shift ;;
        --heartbeat) heartbeat=$2; shift ;;
        --min-views) export RETAIL_ML_MIN_VIEW_EVENTS=$2; shift ;;
        --kill-stale) kill_stale=1 ;;
        -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
        *) echo "Tham số không hợp lệ: $1" >&2; exit 2 ;;
    esac
    shift
done

env_value() {  # đọc một khóa từ .env mà không source cả file (có mật khẩu)
    [[ -f .env ]] && grep -E "^(export )?$1=" .env | tail -1 | cut -d= -f2- | tr -d "\"'" || true
}

case "$source_name" in
    full|small) default_run="retail_star_${source_name}_v1" ;;
    *) default_run=retail_star_custom_v1 ;;
esac
run_id=${run_id:-${RETAIL_STAR_RUN_ID:-$default_run}}
memory=${memory:-${RETAIL_JOB_MEMORY_MAX:-$(env_value RETAIL_JOB_MEMORY_MAX)}}
swap=${swap:-${RETAIL_JOB_SWAP_MAX:-$(env_value RETAIL_JOB_SWAP_MAX)}}
export RETAIL_SOURCE=$source_name
export RETAIL_STAR_RUN_ID=$run_id
export RETAIL_JOB_MEMORY_MAX=${memory:-7G}
export RETAIL_JOB_SWAP_MAX=${swap:-4G}
export RETAIL_HEARTBEAT_SECONDS=$heartbeat
if [[ $source_name == small ]]; then export SPARK_SHUFFLE_PARTITIONS=${SPARK_SHUFFLE_PARTITIONS:-16}; fi

run_dir="results/$run_id"
log_dir="$run_dir/logs"
state="$run_dir/checkpoint/state.json"
python=.venv/bin/python

show_status() {
    if [[ ! -f $state ]]; then
        echo "Chưa có checkpoint cho $run_id ($state)."
        return
    fi
    "$python" - "$state" <<'EOF'
import json, sys
state = json.load(open(sys.argv[1]))
print(f"RUN_ID {state['run_id']} | cập nhật {state.get('updated_at')} | bảng Delta đã ghim: {len(state['delta'])}")
for name, stage in state['stages'].items():
    peak = stage.get('ram_peak_bytes')
    print(f"  {name:<22} {stage.get('status', '?'):<12} {stage.get('seconds', ''):>10} s  "
          f"RAM peak {peak / 1024**3:.2f}G" if peak else f"  {name:<22} {stage.get('status', '?'):<12}")
progress = state['progress']
print('Tiến độ:', f"Bronze {len(progress.get('bronze_days', []))} ngày;",
      f"E2 {len(progress.get('e2_rows', []))} cấu hình;", f"E1 {len(progress.get('e1_rows', []))} dòng")
if 'selection' in state['locks']:
    print('Lựa chọn đã khóa:', state['locks']['selection']['config_id'])
EOF
}

case "$command" in
    status) show_status; exit 0 ;;
    log)
        [[ -e $log_dir/latest.log ]] || { echo "Chưa có log trong $log_dir" >&2; exit 1; }
        exec tail -n 200 -F "$log_dir/latest.log" ;;
esac

# --- Kiểm tra trước khi chạy ---
[[ -x $python ]] || { echo "Thiếu $python; cài môi trường theo AGENTS.md (mục Môi trường)" >&2; exit 1; }
command -v systemd-run >/dev/null || { echo 'Cần systemd-run --user để giới hạn RAM/swap.' >&2; exit 1; }
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader \
    || { echo 'Không thấy GPU NVIDIA (nvidia-smi lỗi).' >&2; exit 1; }

# --- Giải phóng RAM/VRAM: kernel, runner, Spark JVM còn sót của dự án (cwd = thư mục dự án) ---
# Subshell/pipeline của chính script là bash con có cùng dòng lệnh: bỏ qua $$ và mọi tiến trình con của nó.
other_runs=''
for pid in $(pgrep -u "$USER" -f '^(/usr)?(/bin/)?bash .*run_notebook\.sh' || true); do
    [[ $pid == "$$" || $(awk '{print $4}' "/proc/$pid/stat" 2>/dev/null) == "$$" ]] && continue
    [[ -e /proc/$pid ]] && other_runs+="$pid "
done
[[ -z $other_runs ]] || { echo "Đang có lượt run_notebook.sh khác (PID $other_runs); dừng nó trước." >&2; exit 1; }
stale_pids() {
    local pid argv0 ancestors=" " parent=$$
    while ((parent > 1)); do  # không chạm vào chính script và các shell cha
        ancestors+="$parent "
        parent=$(awk '{print $4}' "/proc/$parent/stat" 2>/dev/null || echo 1)
    done
    for pid in $(pgrep -u "$USER" -f 'java|python' || true); do
        [[ $ancestors == *" $pid "* ]] && continue
        [[ $(readlink "/proc/$pid/cwd" 2>/dev/null) == "$project" ]] || continue
        IFS= read -r -d '' argv0 < "/proc/$pid/cmdline" 2>/dev/null || true
        case "${argv0##*/}" in python*|java) ;; *) continue ;; esac
        case "$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)" in
            *deploy.history.HistoryServer*) ;;  # History Server chỉ đọc spark-events, không giữ RUN_ID
            *execute_notebook.py*|*SparkSubmit*|*pyspark*) echo "$pid" ;;
            *gpu_kernel.py*|*ipykernel*)  # kernel rảnh (VS Code vừa mở) không tính; chỉ kernel giữ Spark hoặc khóa RUN_ID
                if pgrep -P "$pid" -x java >/dev/null || ls -l "/proc/$pid/fd" 2>/dev/null | grep -q '\.writer\.lock'; then
                    echo "$pid"
                fi ;;
        esac
    done
}
stale=$(stale_pids)
if [[ -n $stale ]]; then
    echo 'Tiến trình còn sót của dự án (giữ RAM/VRAM hoặc khóa RUN_ID):'
    ps -o pid=,rss=,etime=,args= -p "${stale//$'\n'/,}" | awk '{printf "  PID %-7s %6.0f MiB  %-10s %s\n",$1,$2/1024,$3,substr($0,index($0,$4),90)}'
    answer=n
    if ((kill_stale)); then answer=y
    elif [[ -t 0 ]]; then read -r -p 'Dừng các tiến trình này? [y/N] ' answer
    fi
    [[ $answer == [yY] ]] || { echo 'Giữ nguyên; đóng kernel/notebook đó rồi chạy lại (hoặc dùng --kill-stale).' >&2; exit 1; }
    kill -TERM $stale 2>/dev/null || true
    for _ in $(seq 1 15); do
        alive=$(for pid in $stale; do kill -0 "$pid" 2>/dev/null && echo "$pid"; done || true)
        [[ -z $alive ]] && break
        sleep 1
    done
    [[ -z ${alive:-} ]] || kill -KILL $alive 2>/dev/null || true
    echo 'Đã dừng tiến trình còn sót.'
fi
available=$(awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo)
vram=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | tr -d ' ')
echo "RAM khả dụng: ${available} GiB | VRAM đang dùng: ${vram%,*}/${vram#*,} MiB"
gpu_apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | awk -F', ' '$3+0 > 256')
[[ -z $gpu_apps ]] || echo "Cảnh báo: tiến trình khác đang giữ VRAM (> 256 MiB):"$'\n'"$gpu_apps"

minio_port=$(env_value MINIO_API_PORT)
minio_url="http://127.0.0.1:${minio_port:-19000}/minio/health/ready"
if ! curl -fsS --max-time 5 "$minio_url" >/dev/null; then
    echo "MinIO chưa sẵn sàng; khởi động bằng docker compose..."
    docker compose up -d minio minio-init
    for _ in $(seq 1 30); do
        curl -fsS --max-time 5 "$minio_url" >/dev/null && break
        sleep 2
    done
    curl -fsS --max-time 5 "$minio_url" >/dev/null || { echo "MinIO không phản hồi: $minio_url" >&2; exit 1; }
fi

mkdir -p "$log_dir"
log_file="$log_dir/run_$(date +%Y%m%d_%H%M%S).log"
ln -sfn "$(basename "$log_file")" "$log_dir/latest.log"

{
    echo "=== $(date '+%F %T') | RUN_ID=$run_id | source=$source_name"
    echo "=== Giới hạn job: MemoryMax=$RETAIL_JOB_MEMORY_MAX MemorySwapMax=$RETAIL_JOB_SWAP_MAX | heartbeat=${heartbeat}s | min-views=${RETAIL_ML_MIN_VIEW_EVENTS:-2}"
    echo "=== Log: $log_file"
    show_status
} | tee -a "$log_file"

interrupted() {
    echo "=== $(date '+%F %T') Đã nhận Ctrl+C; checkpoint giữ nguyên. Chạy lại cùng lệnh để resume." | tee -a "$log_file"
}
trap interrupted INT TERM

# execute_notebook.py tự tạo scope systemd (MemoryMax/MemorySwapMax) rồi chạy từng cell và stream output.
# tee -i: vẫn ghi log khi Ctrl+C để giữ thông báo dừng của notebook.
set +e
"$python" -u tools/execute_notebook.py 2>&1 | tee -i -a "$log_file"
status=${PIPESTATUS[0]}
set -e

{
    if ((status == 0)); then
        echo "=== $(date '+%F %T') HOÀN TẤT. Notebook đã chạy: $run_dir/execution/latest.ipynb"
    else
        echo "=== $(date '+%F %T') DỪNG với mã $status. Xem CELL_FAILED/STAGE_FAILED ở trên."
        echo "=== Resume: ./run_notebook.sh --source $source_name --run-id $run_id"
    fi
    show_status
} | tee -a "$log_file"
exit "$status"
