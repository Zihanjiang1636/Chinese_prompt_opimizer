# Chinese Prompt Optimizer

面向中文内容创作者的提示词理解与优化工具。

它不是一个简单的“提示词美化器”，而是一个偏产品化的中文 Prompt 助手：

- 先解析原提示词的语义核心、文采倾向和歧义风险
- 再生成多路候选改写
- 用 `原提示词基线` 和 `单轮直改基线` 做内部比较
- 只把跑赢基线的那版交给用户

第一版聚焦这些中文创作场景：

- 短视频开场
- 种草文案
- 广告口播
- 标题优化
- 卖点提炼
- 情绪共鸣

## Features

- 本地优先的 Prompt 优化工作台
- FastAPI 接口：`/api/prompt-copilot/*`
- 轻量 Web 界面：输入原提示词和任务目标即可使用
- 历史记录与反馈沉淀
- 真实中文回归集与批量回归脚本

## Quick Start

1. 安装依赖

```powershell
pip install -r requirements.txt
```

2. 复制环境变量

```powershell
Copy-Item .env.example .env
```

3. 启动服务

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8008
```

4. 打开页面

浏览器访问 [http://127.0.0.1:8008](http://127.0.0.1:8008)

## LLM Mode

- 如果 `.env` 里没有配置 `LLM_API_KEY`，工具会自动进入 `stub` 模式，使用确定性 fallback 逻辑
- 如果配置了兼容 OpenAI 的模型接口，会进入 `live` 模式

## Regression

运行真实回归集：

```powershell
python scripts/run_prompt_copilot_regression.py
```

开发阶段建议先跑确定性版本：

```powershell
python scripts/run_prompt_copilot_regression.py --stub --limit 5
```

回归报告默认输出到：

```text
prompt-copilot/reports/
```

## Project Layout

```text
backend/
  main.py
  config.py
  core/
  services/
prompt-copilot/
  datasets/
  rubric/
  notes/
  reports/
scripts/
tests/
web/
```

## Tests

```powershell
python -m pytest
```

## Notes

这个仓库是从一个更大的多功能项目里收口出来的独立工具版，目标是让“中文提示词理解 + 内部筛选 + 回归验证”这条链路可以单独演进。
