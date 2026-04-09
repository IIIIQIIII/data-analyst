# API Configuration Guide

This guide explains how to configure different LLM providers for the CoPaw Trajectory Collector.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `CLAUDE_CODE_USE_OPENAI` | Yes | Set to `1` to enable OpenAI-compatible mode |
| `OPENAI_API_KEY` | Yes* | API key (* optional for local models) |
| `OPENAI_BASE_URL` | No | API endpoint (default: `https://api.openai.com/v1`) |
| `OPENAI_MODEL` | No | Model identifier (default: `gpt-4o`) |

## Provider Configurations

### OpenRouter (Recommended for Cost)

Access to 200+ models including free options:

```bash
CLAUDE_CODE_USE_OPENAI=1 \
OPENAI_API_KEY=sk-or-v1-your-key \
OPENAI_BASE_URL=https://openrouter.ai/api/v1 \
OPENAI_MODEL=qwen/qwen3.6-plus:free \
bun run collect --dataset-dir ./data --max-turns 3
```

**Free Models on OpenRouter:**
- `qwen/qwen3.6-plus:free` - Good for data analysis
- `google/gemma-2-9b-it:free` - Decent instruction following
- `meta-llama/llama-3.2-3b-instruct:free` - Lightweight option

**Paid Models (Better Quality):**
- `openai/gpt-4o-mini` - Fast and affordable
- `anthropic/claude-3.5-sonnet` - High quality
- `google/gemini-2.0-flash-001` - Good balance

### OpenAI Direct

```bash
CLAUDE_CODE_USE_OPENAI=1 \
OPENAI_API_KEY=sk-your-openai-key \
OPENAI_MODEL=gpt-4o-mini \
bun run collect --dataset-dir ./data --max-turns 3
```

**Recommended Models:**
- `gpt-4o-mini` - Best cost/performance ratio
- `gpt-4o` - Highest quality
- `gpt-3.5-turbo` - Budget option

### DeepSeek

```bash
CLAUDE_CODE_USE_OPENAI=1 \
OPENAI_API_KEY=your-deepseek-key \
OPENAI_BASE_URL=https://api.deepseek.com/v1 \
OPENAI_MODEL=deepseek-chat \
bun run collect --dataset-dir ./data --max-turns 3
```

### Ollama (Local)

Run models locally with no API costs:

```bash
# Start Ollama server first
ollama serve

# Pull a model
ollama pull llama3.3

# Run collection
CLAUDE_CODE_USE_OPENAI=1 \
OPENAI_BASE_URL=http://localhost:11434/v1 \
OPENAI_MODEL=llama3.3 \
bun run collect --dataset-dir ./data --max-turns 3
```

**Recommended Local Models:**
- `llama3.3` - Best quality for local
- `qwen2.5:14b` - Good Chinese support
- `codellama:13b` - Code-focused

### LM Studio (Local)

```bash
# Start LM Studio server first (default port 1234)

CLAUDE_CODE_USE_OPENAI=1 \
OPENAI_BASE_URL=http://localhost:1234/v1 \
OPENAI_MODEL=your-loaded-model \
bun run collect --dataset-dir ./data --max-turns 3
```

### Together AI

```bash
CLAUDE_CODE_USE_OPENAI=1 \
OPENAI_API_KEY=your-together-key \
OPENAI_BASE_URL=https://api.together.xyz/v1 \
OPENAI_MODEL=meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo \
bun run collect --dataset-dir ./data --max-turns 3
```

### Groq (Fast Inference)

```bash
CLAUDE_CODE_USE_OPENAI=1 \
OPENAI_API_KEY=your-groq-key \
OPENAI_BASE_URL=https://api.groq.com/openai/v1 \
OPENAI_MODEL=llama-3.3-70b-versatile \
bun run collect --dataset-dir ./data --max-turns 3
```

### Azure OpenAI

```bash
CLAUDE_CODE_USE_OPENAI=1 \
OPENAI_API_KEY=your-azure-key \
OPENAI_BASE_URL=https://your-resource.openai.azure.com/openai/deployments/your-deployment \
OPENAI_MODEL=gpt-4o \
AZURE_OPENAI_API_VERSION=2024-12-01-preview \
bun run collect --dataset-dir ./data --max-turns 3
```

## Cost Estimation

| Provider | Model | Input (per 1M) | Output (per 1M) | Per Trajectory* |
|----------|-------|----------------|-----------------|-----------------|
| OpenRouter | qwen/qwen3.6-plus:free | $0 | $0 | $0 |
| OpenAI | gpt-4o-mini | $0.15 | $0.60 | ~$0.01 |
| OpenAI | gpt-4o | $2.50 | $10.00 | ~$0.18 |
| DeepSeek | deepseek-chat | $0.14 | $0.28 | ~$0.02 |
| Together | Llama-3.1-70B | $0.88 | $0.88 | ~$0.05 |
| Groq | llama-3.3-70b | Free tier | Free tier | $0 (rate limited) |

*Based on ~50K input + 5K output tokens per trajectory

## Rate Limits

Different providers have different rate limits:

| Provider | Free Tier | Paid Tier |
|----------|-----------|-----------|
| OpenRouter | 20 req/min | 60+ req/min |
| OpenAI | 3 req/min (gpt-4) | 500+ req/min |
| DeepSeek | 5 req/min | 60 req/min |
| Ollama | No limit | No limit |
| Groq | 30 req/min | 6000 req/day |

**Handling Rate Limits:**

```bash
# Add delay between trajectories
for dataset in data/*; do
  bun run collect --dataset-dir "$dataset" --max-turns 3
  sleep 30  # Wait 30 seconds
done
```

## Troubleshooting

### "401 Unauthorized"

Check your API key:
```bash
# Test with curl
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
  "$OPENAI_BASE_URL/models"
```

### "429 Too Many Requests"

You're hitting rate limits. Solutions:
1. Add delays between requests
2. Upgrade to a paid tier
3. Use a different provider

### "Model not found"

The model name might be different. Check the provider's documentation:
```bash
# List available models
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
  "$OPENAI_BASE_URL/models" | jq '.data[].id'
```

### Slow Responses

For local models, ensure:
- Sufficient GPU VRAM
- Model fits in memory
- No other processes using GPU

### Poor Quality Output

Try:
1. Using a larger/better model
2. Adjusting the temperature (if supported)
3. Increasing max tokens in the API call

## Best Practices

1. **Start with free models** for testing
2. **Use paid models** for final dataset generation
3. **Monitor costs** with provider dashboards
4. **Cache responses** when possible (avoid re-running same data)
5. **Batch processing** during off-peak hours for better rate limits
