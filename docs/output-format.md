# Output Format Specification

This document describes the trajectory output format used by CoPaw Trajectory Collector.

## Overview

The collector generates two output files:

| File | Format | Purpose |
|------|--------|---------|
| `trajectories_<timestamp>.jsonl` | JSONL | Training data for ms-swift |
| `trajectories_full_<timestamp>.json` | JSON | Debugging and analysis |

## JSONL Training Format

### Structure

Each line is a complete JSON object representing one conversation trajectory:

```json
{"messages": [<message>, <message>, ...]}
```

### Message Schema

```typescript
interface Message {
  role: "user" | "assistant" | "system";
  content: string;
}
```

### Message Order

1. First message: `user` - Initial question from the (simulated) user
2. Second message: `system` - Full system prompt with tool definitions
3. Subsequent messages: Alternating `assistant` and `user` (tool responses)

### Example Trajectory

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Hi! I need to analyze the \"Sales Data\" dataset. Can you help me understand what's in it?"
    },
    {
      "role": "system",
      "content": "You are a professional Data Analyst...\n\nAvailable tools:\n- Bash: Execute shell commands...\n- Read: Read file contents...\n..."
    },
    {
      "role": "assistant",
      "content": "I'd be happy to help! Let me first see what files are available.\n\n<tool_call>\n{\"name\": \"Glob\", \"arguments\": {\"pattern\": \"/path/to/data/*\"}}\n</tool_call>"
    },
    {
      "role": "user",
      "content": "<tool_response>\nsales_data.csv\nmetadata.json\n</tool_response>"
    },
    {
      "role": "assistant",
      "content": "I found two files. Let me read the CSV file to understand its structure.\n\n<tool_call>\n{\"name\": \"Read\", \"arguments\": {\"file_path\": \"/path/to/data/sales_data.csv\"}}\n</tool_call>"
    },
    {
      "role": "user",
      "content": "<tool_response>\n1\tdate,product,revenue\n2\t2024-01-01,Widget A,1000\n3\t2024-01-02,Widget B,1500\n</tool_response>"
    },
    {
      "role": "assistant",
      "content": "The dataset contains sales data with columns: date, product, and revenue. It appears to track daily sales by product."
    }
  ]
}
```

## Tool Call Format

Tool calls are embedded in assistant messages using XML-like tags:

```xml
<tool_call>
{"name": "ToolName", "arguments": {"param1": "value1", "param2": "value2"}}
</tool_call>
```

### Supported Tools

| Tool | Arguments | Description |
|------|-----------|-------------|
| `Bash` | `{"command": "..."}` | Execute shell command |
| `Read` | `{"file_path": "..."}` | Read file contents |
| `Write` | `{"file_path": "...", "content": "..."}` | Write to file |
| `Glob` | `{"pattern": "..."}` | Find files by pattern |
| `Grep` | `{"pattern": "...", "path": "..."}` | Search in files |
| `Edit` | `{"file_path": "...", "old_string": "...", "new_string": "..."}` | Edit file |

### Multiple Tool Calls

An assistant message may contain multiple tool calls:

```xml
Let me check the files and read the main data.

<tool_call>
{"name": "Glob", "arguments": {"pattern": "*.csv"}}
</tool_call>

<tool_call>
{"name": "Read", "arguments": {"file_path": "config.json"}}
</tool_call>
```

## Tool Response Format

Tool responses appear as user messages with special wrapping:

```xml
<tool_response>
[output from tool execution]
</tool_response>
```

### Error Responses

When a tool fails, the error is included in the response:

```xml
<tool_response>
Error: Command failed: python script.py
Traceback (most recent call last):
  File "script.py", line 1, in <module>
    import pandas
ModuleNotFoundError: No module named 'pandas'
</tool_response>
```

## Full JSON Format (Debugging)

The full JSON file includes metadata:

```json
[
  {
    "conversation_id": "conv_abc123_xyz789",
    "messages": [...],
    "metadata": {
      "timestamp": "2026-04-05T01:48:58.084Z",
      "model": "qwen/qwen3.6-plus:free",
      "total_input_tokens": 50392,
      "total_output_tokens": 4956,
      "turns": 10,
      "dataset": "Sales Data Analysis"
    }
  }
]
```

### Metadata Fields

| Field | Type | Description |
|-------|------|-------------|
| `conversation_id` | string | Unique identifier for the trajectory |
| `timestamp` | ISO 8601 | When collection started |
| `model` | string | LLM model used |
| `total_input_tokens` | number | Total input tokens consumed |
| `total_output_tokens` | number | Total output tokens generated |
| `turns` | number | Number of assistant turns |
| `dataset` | string | Dataset title used for collection |

## Compatibility

### ms-swift Training

The JSONL format is directly compatible with ms-swift's conversation format:

```bash
# Use directly with ms-swift
swift sft \
  --model CoPaw-Flash-9B \
  --dataset ./trajectories.jsonl \
  --output_dir ./output
```

### CoPaw-Flash chat_template.jinja

The tool call format aligns with CoPaw-Flash's expected format. The model will recognize:

- `<tool_call>` tags for generating tool invocations
- `<tool_response>` tags for processing tool outputs
- Standard JSON format for tool arguments

## Statistics

After collection, you can analyze the output:

```bash
# Count trajectories
wc -l trajectories.jsonl

# Check average message count
jq -r '.messages | length' trajectories.jsonl | awk '{sum+=$1} END {print "Avg messages:", sum/NR}'

# Count tool calls
grep -o '<tool_call>' trajectories.jsonl | wc -l
```

## Validation

To validate the output format:

```python
import json

with open('trajectories.jsonl', 'r') as f:
    for i, line in enumerate(f):
        try:
            data = json.loads(line)
            assert 'messages' in data
            assert len(data['messages']) >= 2
            assert data['messages'][0]['role'] == 'user'
            print(f"Line {i+1}: OK ({len(data['messages'])} messages)")
        except Exception as e:
            print(f"Line {i+1}: ERROR - {e}")
```
