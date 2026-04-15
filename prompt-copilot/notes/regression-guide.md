# 回归说明

`real-world-prompt-regression.json` 用来模拟更接近真实中文创作场景的 Prompt 输入。

建议调优流程：

1. 先用 `--stub` 模式检查流程和报告结构
2. 再切到 live 模式比较真实模型表现
3. 每次模板或策略调整后都跑同一批样本，观察是否稳定跑赢两个基线

常用命令：

```powershell
python scripts/run_prompt_copilot_regression.py --stub --limit 5
python scripts/run_prompt_copilot_regression.py --strategy conversion
```
