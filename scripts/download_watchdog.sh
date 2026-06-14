#!/bin/bash
# ============================================================================
# OPD-TTT 数据下载看门狗
# ============================================================================
# 每10分钟检查一次，发现网络掉线则重启下载进程
#
# 用法:
#   bash scripts/download_watchdog.sh start   # 启动看门狗
#   bash scripts/download_watchdog.sh stop    # 停止看门狗
#   bash scripts/download_watchdog.sh status   # 查看状态
# ============================================================================

# 获取项目根目录（脚本所在目录的父目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

WATCHDOG_PID_FILE="$PROJECT_ROOT/.cache/opdttt_download_watchdog.pid"
LOG_DIR="$PROJECT_ROOT/data"
CHECK_INTERVAL=600  # 10分钟（秒）
RESTART_DELAY=600   # 重启前等待时间（秒）

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[WATCHDOG]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warn() {
    echo -e "${YELLOW}[WATCHDOG]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[WATCHDOG]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# 获取最新的日志文件
get_latest_log() {
    ls -t ${LOG_DIR}/data_process_*.log 2>/dev/null | head -1
}

# 检查数据处理是否完成
check_data_processing_complete() {
    local completion_marker="$PROJECT_ROOT/.cache/data_processing_complete"
    if [ -f "$completion_marker" ]; then
        return 0  # 数据处理已完成
    fi
    return 1  # 数据处理未完成
}

# 检查日志中是否有网络错误
check_network_error() {
    local log_file="$1"
    if [ ! -f "$log_file" ]; then
        return 1
    fi

    # 检查最近5分钟的日志是否有网络错误
    local error_count=$(tail -n 100 "$log_file" | grep -c -E "(timeout|broken pipe|connection|network|disconnected|incompleteread|Got disconnected)")

    if [ "$error_count" -gt 3 ]; then
        return 0  # 发现网络错误
    fi
    return 1  # 无网络错误
}

# 获取下载进程PID
get_download_pid() {
    pgrep -f "prepare_pretrain_data.py"
}

# 杀掉下载进程
kill_download() {
    local pid=$(get_download_pid)
    if [ -n "$pid" ]; then
        log_warn "杀掉下载进程 (PID: $pid)"
        kill $pid
        sleep 5
        # 如果还在运行，强制杀死
        if kill -0 $pid 2>/dev/null; then
            log_warn "强制杀掉下载进程"
            kill -9 $pid
        fi
    fi
}

# 启动下载
start_download() {
    log_info "启动下载进程..."
    cd "$PROJECT_ROOT"
    nohup bash scripts/data_process.sh full >> ${LOG_DIR}/data_watchdog_$(date +%Y%m%d_%H%M%S).log 2>&1 &
    log_info "下载进程已启动 (PID: $!)"
}

# 检查进程是否卡死（CPU不活动超过阈值时间）
check_process_stuck() {
    local pid=$1
    local stuck_threshold=300  # 5分钟无活动视为卡死

    if [ -z "$pid" ]; then
        return 1
    fi

    # 检查进程是否存在
    if ! kill -0 "$pid" 2>/dev/null; then
        return 1
    fi

    # 检查进程状态（是否在睡眠/等待IO）
    local proc_state=$(ps -o state= -p "$pid" 2>/dev/null | tr -d ' ')
    if [ "$proc_state" = "D" ]; then
        # 进程在不可中断睡眠，可能卡在IO
        log_warn "进程 $pid 处于不可中断睡眠状态 (D)，可能卡死"
        return 0
    fi

    # 检查CPU使用率（持续低CPU可能表示卡死）
    # 获取进程的累计CPU时间
    local prev_cpu=$(ps -o time= -p "$pid" 2>/dev/null | tr -d ' ')
    sleep 30
    local curr_cpu=$(ps -o time= -p "$pid" 2>/dev/null | tr -d ' ')

    if [ "$prev_cpu" = "$curr_cpu" ]; then
        log_warn "进程 $pid CPU时间未变化 (30秒内)，可能卡死"
        return 0
    fi

    return 1
}

# 强制终止卡死的进程
kill_stuck_process() {
    local pid=$1
    log_warn "强制终止卡死的进程 (PID: $pid)"
    kill -9 "$pid" 2>/dev/null
    sleep 2
    # 确保进程被杀死
    if kill -0 "$pid" 2>/dev/null; then
        log_error "无法终止进程 $pid"
        return 1
    fi
    log_info "进程 $pid 已终止"
    return 0
}

# 看门狗主循环
watchdog_loop() {
    log_info "看门狗启动 (PID: $$), 检查间隔: ${CHECK_INTERVAL}秒"

    # 首次检查：是否已有下载进程
    local download_pid=$(get_download_pid)
    if [ -n "$download_pid" ]; then
        log_info "检测到已有下载进程运行 (PID: $download_pid)，将监控现有进程"
    else
        log_info "未检测到下载进程，将立即启动下载"
        start_download
    fi

    while true; do
        # 检查数据处理是否已完成
        if check_data_processing_complete; then
            log_info "检测到数据处理已完成，看门狗将停止"
            log_info "完成标记文件: $PROJECT_ROOT/.cache/data_processing_complete"

            # 停止下载进程（如果还在运行）
            download_pid=$(get_download_pid)
            if [ -n "$download_pid" ]; then
                log_info "停止下载进程 (PID: $download_pid)"
                kill_download
            fi

            # 删除看门狗PID文件并退出
            rm -f "$WATCHDOG_PID_FILE"
            log_info "看门狗已停止"
            exit 0
        fi

        # 检查是否有下载进程在运行
        download_pid=$(get_download_pid)

        if [ -z "$download_pid" ]; then
            log_info "没有下载进程在运行，启动下载..."
            start_download
        fi

        # 检查最新日志是否有网络错误
        local latest_log=$(get_latest_log)
        if [ -n "$latest_log" ]; then
            if check_network_error "$latest_log"; then
                log_warn "检测到网络错误，准备重启下载..."

                # 杀掉当前进程
                kill_download

                # 等待一段时间再重启
                log_info "等待 ${RESTART_DELAY} 秒后重启..."
                sleep $RESTART_DELAY

                # 重启下载
                start_download
            fi
        fi

        # 检查进程是否卡死（新增）
        download_pid=$(get_download_pid)
        if check_process_stuck "$download_pid"; then
            log_warn "检测到进程卡死，准备强制重启..."

            # 强制终止卡死的进程
            if kill_stuck_process "$download_pid"; then
                # 等待一段时间再重启
                log_info "等待 ${RESTART_DELAY} 秒后重启..."
                sleep $RESTART_DELAY

                # 重启下载
                start_download
            else
                log_error "无法终止卡死的进程，尝试使用常规方法..."
                kill_download
                sleep 10
                start_download
            fi
        fi

        # 等待下一次检查
        sleep $CHECK_INTERVAL
    done
}

# 启动看门狗
start_watchdog() {
    if [ -f "$WATCHDOG_PID_FILE" ]; then
        local pid=$(cat "$WATCHDOG_PID_FILE")
        if kill -0 $pid 2>/dev/null; then
            log_info "看门狗已在运行 (PID: $pid)"
            return 0
        fi
    fi

    # 检查是否已有下载进程在运行
    local existing_download=$(get_download_pid)
    if [ -n "$existing_download" ]; then
        log_info "检测到下载进程已运行 (PID: $existing_download)"
        log_info "看门狗将监控现有进程，不会重复启动"
    else
        log_info "未检测到下载进程，看门狗启动后将自动开始下载"
    fi

    # 后台运行看门狗
    nohup bash "$0" --daemon-loop >/dev/null 2>&1 &
    local pid=$!
    echo $pid > "$WATCHDOG_PID_FILE"

    log_info "看门狗已启动 (PID: $pid)"
    log_info "日志将输出到: nohup.out"
}

# 停止看门狗
stop_watchdog() {
    if [ ! -f "$WATCHDOG_PID_FILE" ]; then
        log_info "看门狗未运行"
        return 0
    fi

    local pid=$(cat "$WATCHDOG_PID_FILE")
    if kill -0 $pid 2>/dev/null; then
        log_info "停止看门狗 (PID: $pid)"
        kill $pid
        rm -f "$WATCHDOG_PID_FILE"
        log_info "看门狗已停止"
    else
        log_info "看门狗进程不存在，清理PID文件"
        rm -f "$WATCHDOG_PID_FILE"
    fi
}

# 查看状态
status_watchdog() {
    echo "看门狗状态:"
    if [ -f "$WATCHDOG_PID_FILE" ]; then
        local pid=$(cat "$WATCHDOG_PID_FILE")
        if kill -0 $pid 2>/dev/null; then
            echo "  状态: 运行中 (PID: $pid)"
        else
            echo "  状态: 已停止 (PID文件残留)"
        fi
    else
        echo "  状态: 未运行"
    fi

    echo ""
    echo "下载进程状态:"
    local download_pid=$(get_download_pid)
    if [ -n "$download_pid" ]; then
        echo "  状态: 运行中 (PID: $download_pid)"
    else
        echo "  状态: 未运行"
    fi

    echo ""
    echo "最新日志:"
    local latest_log=$(get_latest_log)
    if [ -n "$latest_log" ]; then
        echo "  文件: $latest_log"
        echo "  最后10行:"
        tail -n 10 "$latest_log" | sed 's/^/    /'
    else
        echo "  未找到日志文件"
    fi
}

# 主函数
main() {
    case "${1:-start}" in
        start)
            start_watchdog
            ;;
        stop)
            stop_watchdog
            ;;
        restart)
            stop_watchdog
            sleep 2
            start_watchdog
            ;;
        status)
            status_watchdog
            ;;
        --daemon-loop)
            watchdog_loop
            ;;
        *)
            echo "用法: $0 {start|stop|restart|status}"
            exit 1
            ;;
    esac
}

main "$@"
