#!/bin/bash
# SatFault-LLM GitHub 同步脚本
# 用法: ./scripts/sync_github.sh [commit_message]

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# 检查是否有变更
if [ -z "$(git status --porcelain)" ]; then
    echo "✅ 没有需要提交的变更"
    exit 0
fi

# 生成默认提交消息
if [ -z "$1" ]; then
    TIMESTAMP=$(date "+%Y-%m-%d %H:%M")
    COMMIT_MSG="chore: 自动同步实验进度 ($TIMESTAMP)"
else
    COMMIT_MSG="$1"
fi

echo "📦 正在提交变更..."
git add -A
git commit -m "$COMMIT_MSG"

echo "🚀 正在推送到 GitHub..."
git push origin main

echo "✅ 同步完成！"
echo "   提交消息: $COMMIT_MSG"