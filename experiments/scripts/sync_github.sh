#!/bin/bash
# ============================================
# LLM-SRAF GitHub Sync Script
# 自动同步实验代码到 GitHub
# ============================================

REPO_URL="https://github.com/LinkDry/ARIS.git"
BRANCH="main"
SYNC_INTERVAL=${SYNC_INTERVAL:-30}  # 分钟

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查是否在 git 仓库中
check_git_repo() {
    if [ ! -d ".git" ]; then
        log_info "Initializing git repository..."
        git init
        git remote add origin $REPO_URL
    fi
}

# 同步到 GitHub
sync_to_github() {
    local commit_msg="$1"

    log_info "Starting GitHub sync..."

    # 添加要同步的文件
    git add experiments/configs/
    git add experiments/scripts/
    git add experiments/src/
    git add experiments/tests/
    git add requirements.txt
    git add README.md
    git add *.md 2>/dev/null || true

    # 检查是否有变更
    if git diff --cached --quiet; then
        log_info "No changes to sync"
        return 0
    fi

    # 提交
    git commit -m "$commit_msg" --author="ARIS Auto-Sync <aris@auto-sync>"

    # 推送 (最多重试3次)
    for i in 1 2 3; do
        if git push origin $BRANCH; then
            log_info "Successfully synced to GitHub"
            return 0
        else
            log_warn "Push failed, attempt $i/3, retrying..."
            sleep 5
        fi
    done

    log_error "Failed to sync to GitHub after 3 attempts"
    return 1
}

# 初始化同步
init_sync() {
    check_git_repo

    # 创建 .gitignore
    cat > .gitignore << 'EOF'
# 不同步的内容
paper/
data/raw/
data/processed/
*.pt
*.pth
*.ckpt
logs/
results/*.csv
.env
__pycache__/
*.pyc
.ipynb_checkpoints/
.idea/
.vscode/
*.log
EOF

    log_info "Git sync initialized"
}

# 定时同步
start_periodic_sync() {
    log_info "Starting periodic sync (interval: ${SYNC_INTERVAL} minutes)"

    while true; do
        sleep $(($SYNC_INTERVAL * 60))

        TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
        sync_to_github "Auto-sync: $TIMESTAMP"
    done
}

# 单次同步
single_sync() {
    local msg="${1:-Auto-sync: $(date '+%Y-%m-%d %H:%M:%S')}"
    sync_to_github "$msg"
}

# 主函数
main() {
    case "$1" in
        init)
            init_sync
            ;;
        start)
            start_periodic_sync
            ;;
        once)
            single_sync "$2"
            ;;
        status)
            git status
            ;;
        *)
            echo "Usage: $0 {init|start|once|status}"
            echo "  init   - Initialize git sync"
            echo "  start  - Start periodic sync"
            echo "  once   - Single sync with optional commit message"
            echo "  status - Show git status"
            exit 1
            ;;
    esac
}

main "$@"