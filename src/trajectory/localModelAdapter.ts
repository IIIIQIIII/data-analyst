/**
 * Local Model Adapter
 *
 * Adapter for local vLLM-served models (e.g., CoPaw-Flash-9B).
 * Uses the same XML-like tool call format as Stepfun for compatibility.
 *
 * This adapter is separate from stepfunAdapter to avoid modifying
 * the original Stepfun integration.
 */

import {
  parseStepfunToolCalls,
  hasStepfunToolCalls,
  type StepfunToolCall,
} from './stepfunAdapter.js'

// Re-export tool call types and parsing functions (same format)
export type LocalToolCall = StepfunToolCall
export const parseToolCalls = parseStepfunToolCalls
export const hasToolCalls = hasStepfunToolCalls

/**
 * Format tool response for local model
 */
export function formatToolResponse(results: Array<{ tool: string; result: string }>): string {
  const formatted = results.map(r => `[${r.tool}] ${r.result}`).join('\n---\n')
  return `<tool_response>\n${formatted}\n</tool_response>`
}

/**
 * Local API client configuration
 */
export interface LocalApiConfig {
  baseUrl: string
  model: string
  apiKey?: string
}

/**
 * Create default config from environment
 */
export function getLocalApiConfig(): LocalApiConfig {
  return {
    baseUrl: process.env.OPENAI_BASE_URL || 'http://localhost:8000/v1',
    model: process.env.OPENAI_MODEL || 'agent-lora',
    apiKey: process.env.OPENAI_API_KEY,
  }
}

/**
 * Call local LLM API
 */
export async function callLocalLLM(
  config: LocalApiConfig,
  messages: Array<{ role: string; content: string }>,
): Promise<{ content: string; inputTokens: number; outputTokens: number }> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  }

  // Add API key if provided (some local setups may require it)
  if (config.apiKey) {
    headers['Authorization'] = `Bearer ${config.apiKey}`
  }

  const response = await fetch(`${config.baseUrl}/chat/completions`, {
    method: 'POST',
    headers,
    body: JSON.stringify({
      model: config.model,
      messages,
      stream: false,
    }),
  })

  if (!response.ok) {
    const text = await response.text()
    throw new Error(`API error ${response.status}: ${text}`)
  }

  const data = await response.json() as {
    choices: Array<{ message: { content: string } }>
    usage?: { prompt_tokens?: number; completion_tokens?: number }
  }

  return {
    content: data.choices[0]?.message?.content || '',
    inputTokens: data.usage?.prompt_tokens || 0,
    outputTokens: data.usage?.completion_tokens || 0,
  }
}

/**
 * Generate system prompt for autonomous data analysis
 */
export function getAutonomousSystemPrompt(
  datasetDir: string,
  workspaceDir: string,
  pythonPath: string,
): string {
  return `You are an autonomous data analyst AI. Your task is to perform comprehensive data analysis.

<context>
Dataset directory: ${datasetDir}
Workspace: ${workspaceDir}
Python: ${pythonPath}
</context>

<tools>
Available tools with EXACT parameter names:

1. Bash - Execute shell commands
   Parameters:
   - command (required): The shell command to execute

   Example:
   <tool_call>
   <function=Bash>
   <parameter=command>ls -la ${datasetDir}</parameter>
   </function>
   </tool_call>

2. Read - Read file contents
   Parameters:
   - file_path (required): Path to the file to read

   Example:
   <tool_call>
   <function=Read>
   <parameter=file_path>${datasetDir}/data.csv</parameter>
   </function>
   </tool_call>

3. Write - Create files in workspace
   Parameters:
   - file_path (required): Path where to write the file
   - content (required): Content to write

   Example:
   <tool_call>
   <function=Write>
   <parameter=file_path>${workspaceDir}/analysis.py</parameter>
   <parameter=content>import pandas as pd
print("hello")</parameter>
   </function>
   </tool_call>

4. Glob - Find files matching pattern
   Parameters:
   - pattern (required): Glob pattern (e.g., "*.csv")

   Example:
   <tool_call>
   <function=Glob>
   <parameter=pattern>*.csv</parameter>
   </function>
   </tool_call>

5. Grep - Search file contents
   Parameters:
   - pattern (required): Search pattern
   - path (optional): Path to search in

   Example:
   <tool_call>
   <function=Grep>
   <parameter=pattern>import</parameter>
   <parameter=path>${datasetDir}</parameter>
   </function>
   </tool_call>

6. Edit - Edit existing files
   Parameters:
   - file_path (required): Path to the file
   - old_string (required): String to replace
   - new_string (required): Replacement string

CRITICAL: Use EXACT parameter names as shown in examples above!
</tools>

<task>
Perform a COMPLETE, AUTONOMOUS data analysis:
1. Explore the dataset (find data files, understand structure)
2. Load and examine the data (columns, types, missing values, statistics)
3. Identify key patterns and relationships
4. Create meaningful visualizations (save as PNG to workspace)
5. Provide actionable insights and conclusions

Work independently - do NOT wait for user input. Complete the entire analysis.
</task>

<guidelines>
- Use pandas for data manipulation
- Create visualizations with matplotlib/seaborn
- Save all plots to workspace with plt.savefig()
- Explain findings in clear, business-friendly language
- Focus on actionable insights
</guidelines>

Begin your autonomous analysis now.`
}
