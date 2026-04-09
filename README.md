# CoPaw Trajectory Collector

Agent Trajectory Collector for generating CoPaw-Flash training datasets. Uses a dual-AI architecture where both user and assistant roles are played by AI models.

## Overview

This tool collects agent trajectories in the CoPaw-Flash format for fine-tuning language models on tool-use and data analysis tasks. It leverages the claude-code-clean framework for stable agent execution.

### Key Features

- **Dual-AI Architecture**: User Agent simulates non-technical users, Analyst Agent performs data analysis
- **CoPaw-Flash Format**: Output compatible with ms-swift training pipeline
- **OpenAI-Compatible API**: Works with OpenRouter, DeepSeek, Ollama, and other providers
- **Real Tool Execution**: Bash, Read, Write, Glob, Grep, Edit tools with actual execution
- **Python Data Analysis**: Pre-configured uv venv with pandas, numpy, matplotlib, seaborn
- **Batch Collection**: Process multiple datasets with parallel workers and API key rotation
- **Multi-Model Support**: Adapters for different LLM tool call formats (JSON, XML)

## Quick Start

### Prerequisites

- [Bun](https://bun.sh) (v1.3+)
- [uv](https://github.com/astral-sh/uv) (for Python environment)
- Linux/macOS/WSL2

### Installation

```bash
# Install dependencies
bun install

# Setup Python analysis environment
uv venv .venv-analysis --python 3.12
uv pip install pandas numpy matplotlib seaborn scipy scikit-learn --python .venv-analysis/bin/python
```

### Environment Setup

Create a `.env` file with your API keys:

```bash
# .env
OPENROUTER_API_KEYS=sk-or-v1-xxx,sk-or-v1-yyy,sk-or-v1-zzz
```

> **IMPORTANT**: Never commit the `.env` file to git. It contains sensitive API keys.

### Run Single Trajectory Collection

```bash
CLAUDE_CODE_USE_OPENAI=1 \
OPENAI_API_KEY=your-api-key \
OPENAI_BASE_URL=https://openrouter.ai/api/v1 \
OPENAI_MODEL=qwen/qwen3.6-plus:free \
bun run collect \
  --dataset-dir ./test_data \
  --dataset-title "Sales Data Analysis" \
  --output-dir ./trajectory_output \
  --max-turns 3
```

## Batch Collection

For collecting trajectories from multiple datasets at scale.

### Batch Collection with Qwen (JSON format)

```bash
# Load environment variables and run
source .env
bun run src/entrypoints/batchCollect.tsx \
  --kaggle-dir ./kaggle-top1000 \
  --output-dir ./copaw-trajectories \
  --max-turns 3 \
  --checkpoint ./checkpoint.json
```

### Batch Collection with Stepfun (XML format)

Stepfun uses a different tool call format (XML-like). Use the dedicated adapter:

```bash
source .env
bun run src/entrypoints/batchCollectStepfun.tsx \
  --kaggle-dir ./kaggle-top1000 \
  --output-dir ./copaw-trajectories-stepfun \
  --max-turns 2 \
  --checkpoint ./checkpoint-stepfun.json
```

### Batch Collection Features

- **Parallel Workers**: Multiple API keys run concurrent workers
- **Checkpoint/Resume**: Automatically saves progress, can resume after interruption
- **API Key Rotation**: Distributes load across multiple API keys to avoid rate limits
- **Progress Tracking**: Real-time progress and statistics

### Command Line Options (Batch)

| Option | Description | Default |
|--------|-------------|---------|
| `--kaggle-dir` | Directory containing dataset folders | `./kaggle-top1000` |
| `--output-dir` | Directory for output trajectories | `./copaw-trajectories` |
| `--max-turns` | Maximum conversation turns | `3` |
| `--checkpoint` | Checkpoint file for resume | `./checkpoint.json` |
| `--start-index` | Start from specific index | `0` |
| `--limit` | Limit number of datasets (0=all) | `0` |
| `--delay` | Delay between datasets (ms) | `5000` |

## Output Format

### JSONL (Training Format)

```json
{"messages":[
  {"role":"user","content":"Help me analyze this sales data..."},
  {"role":"system","content":"You are a professional Data Analyst..."},
  {"role":"assistant","content":"<tool_call>\n{\"name\": \"Glob\", \"arguments\": {\"pattern\": \"*.csv\"}}\n</tool_call>"},
  {"role":"user","content":"<tool_response>\nsales_data.csv\n</tool_response>"},
  {"role":"assistant","content":"I found the data file. Let me read it..."}
]}
```

### Tool Call Formats

**JSON Format (Qwen, GPT, etc.)**
```xml
<tool_call>
{"name": "Bash", "arguments": {"command": "python analysis.py"}}
</tool_call>
```

**XML Format (Stepfun)**
```xml
<tool_call>
<function=Bash>
<parameter=command>python analysis.py</parameter>
</function>
</tool_call>
```

### Tool Response Format

```xml
<tool_response>
Analysis complete. Total revenue: $162,270.15
</tool_response>
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Trajectory Collector                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐         ┌──────────────┐                  │
│  │  User Agent  │ ──────> │Analyst Agent │                  │
│  │ (Simulates   │         │ (Data        │                  │
│  │  non-tech    │ <────── │  Analysis)   │                  │
│  │  user)       │         │              │                  │
│  └──────────────┘         └──────┬───────┘                  │
│                                  │                           │
│                           ┌──────▼───────┐                  │
│                           │ Tool Executor │                  │
│                           │ Bash/Read/    │                  │
│                           │ Write/Glob/   │                  │
│                           │ Grep/Edit     │                  │
│                           └──────┬───────┘                  │
│                                  │                           │
│                           ┌──────▼───────┐                  │
│                           │   Python     │                  │
│                           │ Environment  │                  │
│                           │ (uv venv)    │                  │
│                           └──────────────┘                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ Output Files    │
                    │ - .jsonl        │
                    │ - .json (full)  │
                    └─────────────────┘
```

## Performance

### Qwen (qwen/qwen3.6-plus:free)

| Metric | Value |
|--------|-------|
| Time per trajectory | ~4 minutes |
| Input tokens | ~50,000 |
| Output tokens | ~5,000 |
| Messages generated | ~20 |
| Tool calls | ~10-15 |
| Success rate | ~52% |

### Stepfun (stepfun/step-3.5-flash:free)

| Metric | Value |
|--------|-------|
| Time per trajectory | ~1-2 minutes |
| Input tokens | ~60,000 |
| Output tokens | ~6,000 |
| Messages generated | ~25 |
| Tool calls | ~10-12 |
| Success rate | ~99% |

## Available Tools

| Tool | Description |
|------|-------------|
| **Bash** | Execute shell commands, run Python scripts |
| **Read** | Read file contents with line numbers |
| **Write** | Create/overwrite files |
| **Glob** | Find files by pattern |
| **Grep** | Search for patterns in files |
| **Edit** | Edit existing files (find & replace) |

## Project Structure

```
copaw-trajectory-collector/
├── src/
│   ├── entrypoints/
│   │   ├── trajectoryCollect.tsx   # Single trajectory collection
│   │   ├── batchCollect.tsx        # Batch collection (Qwen/JSON)
│   │   └── batchCollectStepfun.tsx # Batch collection (Stepfun/XML)
│   └── trajectory/
│       ├── collector.ts            # Trajectory recording
│       ├── dualAgent.ts            # Dual-AI coordination
│       └── stepfunAdapter.ts       # Stepfun XML format adapter
├── test_data/                      # Sample datasets
│   └── sales_data.csv
├── .venv-analysis/                 # Python environment
├── docs/                           # Documentation
├── .env                            # API keys (DO NOT COMMIT)
└── README.md
```

## Model Adapters

### Adding Support for New Models

If a model uses a different tool call format, create an adapter in `src/trajectory/`:

1. Implement parser function for the tool call format
2. Implement `hasToolCalls()` detection function
3. Create system prompt generator with tool definitions
4. Create dedicated batch collection entry point

See `stepfunAdapter.ts` as an example.

## Documentation

See the `docs/` directory for detailed documentation:

- [Data Collection Guide](docs/data-collection-guide.md) - How to collect trajectories
- [Output Format Specification](docs/output-format.md) - Dataset format details
- [API Configuration](docs/api-configuration.md) - Setting up different LLM providers

## HuggingFace Datasets

Collected trajectories are available on HuggingFace:

- [copaw-data-analysis-trajectories](https://huggingface.co/datasets/LocoreMind/copaw-data-analysis-trajectories) - Qwen model trajectories
- [copaw-trajectories-stepfun](https://huggingface.co/datasets/LocoreMind/copaw-trajectories-stepfun) - Stepfun model trajectories

## Based On

This project is based on [claude-code-clean](https://github.com/IIIIQIIII/claude-code-clean), a privacy-focused fork of Anthropic's Claude Code with all telemetry removed.

## License

MIT License - See LICENSE file for details.

---

**Version:** 1.1.0
**Last Updated:** 2026-04-06
