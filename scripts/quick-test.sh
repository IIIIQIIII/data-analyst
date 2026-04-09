#!/bin/bash
# Quick test with HR Employee Attrition dataset
# Uses the running vLLM server with agent-lora

set -e

cd /home/shadeform/copaw-trajectory-collector

export PATH="$HOME/.bun/bin:$PATH"
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_MODEL="agent-lora"

DATASET="itssuru_hr-employee-attrition"
OUTPUT_DIR="./quick-test-output"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=============================================="
echo "Quick Test: CoPaw-Flash-9B Agent"
echo "=============================================="
echo "Time: $(date)"
echo "API: ${OPENAI_BASE_URL}"
echo "Model: ${OPENAI_MODEL}"
echo "Dataset: ${DATASET}"
echo ""

# Clean previous test
rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Run test
echo "Starting trajectory collection..."
echo ""

bun run src/entrypoints/batchCollectLocal.tsx \
  --kaggle-dir "/home/shadeform/clean-test-datasets" \
  --output-dir "${OUTPUT_DIR}" \
  --max-tool-iterations 25 \
  --checkpoint "${OUTPUT_DIR}/checkpoint.json" \
  --limit 1

echo ""
echo "=============================================="
echo "Test Complete"
echo "=============================================="

# Show results
if [ -f "${OUTPUT_DIR}/trajectories.jsonl" ]; then
    echo ""
    echo "Results:"

    # Parse with node
    node -e "
    const fs = require('fs');
    const line = fs.readFileSync('${OUTPUT_DIR}/trajectories.jsonl', 'utf-8').trim();
    const data = JSON.parse(line);

    console.log('  Dataset:', data.metadata.dataset_slug);
    console.log('  Tool iterations:', data.metadata.tool_iterations);
    console.log('  Input tokens:', data.metadata.total_input_tokens.toLocaleString());
    console.log('  Output tokens:', data.metadata.total_output_tokens.toLocaleString());
    console.log('  Messages:', data.messages.length);
    console.log('');

    // Count tool calls
    let toolCalls = 0;
    let errors = 0;
    for (const msg of data.messages) {
        if (msg.role === 'assistant' && msg.content.includes('<tool_call>')) {
            const matches = msg.content.match(/<tool_call>/g);
            if (matches) toolCalls += matches.length;
        }
        if (msg.role === 'user' && msg.content.includes('Error:')) {
            errors++;
        }
    }
    console.log('  Total tool calls:', toolCalls);
    console.log('  Tool errors:', errors);
    console.log('');

    // Show last assistant message (summary)
    const lastAssistant = data.messages.filter(m => m.role === 'assistant').pop();
    if (lastAssistant && !lastAssistant.content.includes('<tool_call>')) {
        console.log('Final Summary (truncated):');
        console.log('---');
        console.log(lastAssistant.content.slice(0, 1000));
        if (lastAssistant.content.length > 1000) console.log('...');
    }
    "
else
    echo "No output generated!"
fi
