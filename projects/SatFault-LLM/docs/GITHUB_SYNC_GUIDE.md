# GitHub 同步指南

## 快速同步

```bash
cd projects/SatFault-LLM
./scripts/sync_github.sh "你的提交消息"
```

## 定期同步建议

### 手动同步时机

| 时机 | 操作 |
|------|------|
| 完成一个实验模块 | `./scripts/sync_github.sh "feat: 完成XX模块"` |
| 获得重要实验结果 | `./scripts/sync_github.sh "results: XX实验结果"` |
| 修改研究方案 | `./scripts/sync_github.sh "docs: 更新研究方案"` |
| 每日工作结束 | `./scripts/sync_github.sh "chore: 日常进度同步"` |

### 自动同步（可选）

如果想设置自动同步，可以使用 Cron 任务：

```bash
# 每天晚上10点自动同步
crontab -e

# 添加以下行
0 22 * * * cd /path/to/AI4Sci/projects/SatFault-LLM && ./scripts/sync_github.sh >> /tmp/sync.log 2>&1
```

## 同步规则

### ✅ 需要同步

```
- 实验代码 (experiments/src/)
- 配置文件 (experiments/configs/)
- 实验结果报告 (experiments/results/*.md)
- 文档 (docs/)
- README 等项目说明
```

### ❌ 不同步（已在 .gitignore 中）

```
- 模型权重 (*.pt, *.pth, *.bin)
- 大型数据文件 (data/raw/, data/processed/)
- 日志文件 (logs/)
- 敏感配置 (.env)
```

## 同步检查清单

在每次同步前，确认：

- [ ] 代码可正常运行
- [ ] 敏感信息已移除
- [ ] 提交消息清晰
- [ ] 大文件已排除

---

*最后更新: 2026-03-22*