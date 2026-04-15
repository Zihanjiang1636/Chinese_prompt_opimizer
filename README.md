# Chinese Prompt Optimizer

面向中文内容创作者的提示词理解与优化工具。

它不是一个简单的“提示词美化器”，而是一条更产品化的中文 Prompt 工作流：

- 先分析原提示词的语义核心、文采线索和潜在歧义
- 再生成多路候选改写
- 用 `原提示词基线` 和 `单轮直改基线` 做内部比较
- 只把真正跑赢基线的版本交给用户

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
- 轻量 Web 界面：输入原提示词和目标任务即可使用
- 历史记录、反馈闭环、回归摘要
- 策略切换：`平衡 / 文采优先 / 转化优先`
- 模型接入预设：`OpenAI / OpenAI-compatible / Ollama / LM Studio / DashScope / DeepSeek`

## Design Notes

这版实现借鉴了一些开源产品的设计思路，但没有照搬业务逻辑：

- Promptfoo：把“提示词回归”和“基线比较”做成默认能力
- Langfuse：把模板版本和运行配置显式暴露出来
- DSPy：强调多候选生成和筛选，而不是一次性润色
- Open WebUI：借鉴本地优先和 OpenAI-compatible 接入预设的思路

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
- 如果配置了兼容 OpenAI 的接口，就会进入 `live` 模式

## Regression

运行真实回归集：

```powershell
python scripts/run_prompt_copilot_regression.py
```

按策略跑回归：

```powershell
python scripts/run_prompt_copilot_regression.py --stub --strategy literary --limit 5
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
  prompt-templates/
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
