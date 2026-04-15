# Prompt Copilot

中文提示词理解助手的领域目录。

这里存放：

- `datasets/`：固定中文创作评测集和真实回归样本
- `rubric/`：评分维度和场景权重
- `notes/`：后续实验记录、回归观察和版本说明
- `reports/`：回归脚本产出的结果报告

第一版聚焦中文文案与脚本提示词优化，不做模型微调。

## Datasets

- `datasets/creative-eval-set.json`
  用于内部预筛场景权重和基础案例映射。

- `datasets/real-world-prompt-regression.json`
  面向真实中文创作 Prompt 的回归集，适合在模板调整后批量复跑。

## Regression

运行回归：

```powershell
python backend/scripts/run_prompt_copilot_regression.py
```

只跑前 5 条：

```powershell
python backend/scripts/run_prompt_copilot_regression.py --limit 5
```

指定输出文件：

```powershell
python backend/scripts/run_prompt_copilot_regression.py --output prompt-copilot/reports/manual-check.json
```

开发期走确定性 fallback：

```powershell
python backend/scripts/run_prompt_copilot_regression.py --stub --limit 5
```
