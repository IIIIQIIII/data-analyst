# Data Collection Guide

This guide explains how to use CoPaw Trajectory Collector to generate training datasets for CoPaw-Flash models.

## Overview

The collector uses a dual-AI architecture:

1. **User Agent**: Simulates a non-technical user asking questions about data
2. **Analyst Agent**: A professional data analyst that uses tools to explore and analyze data

Both agents are powered by the same LLM (configurable via environment variables).

## Prerequisites

### 1. Install Bun Runtime

```bash
curl -fsSL https://bun.sh/install | bash
```

### 2. Install Project Dependencies

```bash
cd copaw-trajectory-collector
bun install
```

### 3. Setup Python Environment

```bash
# Create virtual environment with uv
uv venv .venv-analysis --python 3.12

# Install data analysis packages
uv pip install pandas numpy matplotlib seaborn scipy scikit-learn --python .venv-analysis/bin/python
```

### 4. Configure API Access

You need an API key from a supported provider:

| Provider | Base URL | Example Model |
|----------|----------|---------------|
| OpenRouter | `https://openrouter.ai/api/v1` | `qwen/qwen3.6-plus:free` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| Ollama (local) | `http://localhost:11434/v1` | `llama3.3` |

## Running Data Collection

### Basic Usage

```bash
CLAUDE_CODE_USE_OPENAI=1 \
OPENAI_API_KEY=your-api-key \
OPENAI_BASE_URL=https://openrouter.ai/api/v1 \
OPENAI_MODEL=qwen/qwen3.6-plus:free \
bun run collect \
  --dataset-dir ./your_data \
  --dataset-title "Your Dataset Name" \
  --max-turns 3
```

### Parameters Explained

| Parameter | Description |
|-----------|-------------|
| `--dataset-dir` | Directory containing your data files (CSV, JSON, Excel) |
| `--dataset-title` | Human-readable name for the dataset |
| `--dataset-desc` | Optional description to help the AI understand the data |
| `--output-dir` | Where to save the trajectory files |
| `--max-turns` | Number of conversation turns (user question + AI response pairs) |
| `--workspace-dir` | Directory for AI-generated scripts and charts |

### Example: Sales Data Analysis

```bash
# Prepare your data
mkdir -p ./sales_data
cp your_sales.csv ./sales_data/

# Run collection
CLAUDE_CODE_USE_OPENAI=1 \
OPENAI_API_KEY=sk-or-v1-xxx \
OPENAI_BASE_URL=https://openrouter.ai/api/v1 \
OPENAI_MODEL=qwen/qwen3.6-plus:free \
bun run collect \
  --dataset-dir ./sales_data \
  --dataset-title "Q1 2024 Sales Report" \
  --dataset-desc "Monthly sales data including product, region, and revenue" \
  --output-dir ./trajectories \
  --max-turns 3
```

## Understanding the Output

### Output Files

After running, you'll find two files in the output directory:

1. **`trajectories_<timestamp>.jsonl`** - Training format (use this for ms-swift)
2. **`trajectories_full_<timestamp>.json`** - Full metadata (for debugging)

### JSONL Format Structure

```json
{
  "messages": [
    {"role": "user", "content": "Initial user question..."},
    {"role": "system", "content": "System prompt with tools..."},
    {"role": "assistant", "content": "Response with optional <tool_call>..."},
    {"role": "user", "content": "<tool_response>...</tool_response>"},
    {"role": "assistant", "content": "Analysis explanation..."}
  ]
}
```

### Message Roles

| Role | Content |
|------|---------|
| `user` | User questions OR tool responses wrapped in `<tool_response>` tags |
| `system` | System prompt (appears once at the beginning) |
| `assistant` | AI responses, may contain `<tool_call>` blocks |

## Batch Collection

To collect multiple trajectories, create a shell script:

```bash
#!/bin/bash

DATASETS=(
  "sales_data:Sales Analysis:Monthly sales figures"
  "customer_data:Customer Segmentation:Customer demographics and purchases"
  "inventory:Inventory Report:Stock levels and turnover"
)

for entry in "${DATASETS[@]}"; do
  IFS=':' read -r dir title desc <<< "$entry"

  CLAUDE_CODE_USE_OPENAI=1 \
  OPENAI_API_KEY=$OPENAI_API_KEY \
  OPENAI_BASE_URL=https://openrouter.ai/api/v1 \
  OPENAI_MODEL=qwen/qwen3.6-plus:free \
  bun run collect \
    --dataset-dir "./data/$dir" \
    --dataset-title "$title" \
    --dataset-desc "$desc" \
    --output-dir ./trajectories \
    --max-turns 3

  echo "Completed: $title"
  sleep 5  # Rate limiting
done
```

## Quality Guidelines

### Good Datasets for Collection

- Clean CSV/JSON files with clear column names
- Numeric data suitable for analysis (sales, metrics, counts)
- Multiple dimensions for cross-analysis (date, category, region)
- 100-10,000 rows (not too small, not overwhelming)

### Adjusting Conversation Quality

1. **More turns = deeper analysis**: Increase `--max-turns` for more comprehensive trajectories
2. **Better descriptions**: Provide detailed `--dataset-desc` to guide the AI
3. **Diverse datasets**: Use varied data types for training diversity

## Troubleshooting

### "ModuleNotFoundError: No module named 'pandas'"

Your Python environment isn't configured correctly:

```bash
# Check which Python is being used
which python3

# Ensure packages are in the right venv
uv pip install pandas numpy --python .venv-analysis/bin/python
```

### "API error: 429 Too Many Requests"

You're hitting rate limits. Add delays between runs:

```bash
sleep 60  # Wait 60 seconds between trajectories
```

### "Tool call not parsed correctly"

The AI's response format may be inconsistent. This is model-dependent. Try:
- Using a different model (GPT-4o, Claude models tend to be more consistent)
- Adjusting the system prompt in `src/entrypoints/trajectoryCollect.tsx`

## Next Steps

After collecting trajectories:

1. **Validate**: Check the JSONL files for completeness
2. **Filter**: Remove failed or low-quality trajectories
3. **Merge**: Combine multiple JSONL files: `cat *.jsonl > combined.jsonl`
4. **Train**: Use with ms-swift for CoPaw-Flash fine-tuning

See [Output Format Specification](output-format.md) for detailed format documentation.
