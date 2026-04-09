/**
 * Stepfun Model Adapter
 *
 * Stepfun uses a different tool call format than the JSON format we use for other models.
 * This adapter provides functions to:
 * 1. Generate system prompts with Stepfun-compatible tool definitions
 * 2. Parse Stepfun's XML-like tool call format
 *
 * Stepfun tool call format:
 * <tool_call>
 * <function=function_name>
 * <parameter=param1>value1</parameter>
 * <parameter=param2>value2</parameter>
 * </function>
 * </tool_call>
 */

export interface StepfunToolCall {
  name: string
  arguments: Record<string, string>
}

/**
 * Parse Stepfun's XML-like tool call format
 */
export function parseStepfunToolCalls(response: string): StepfunToolCall[] {
  const toolCalls: StepfunToolCall[] = []

  // Match <tool_call>...</tool_call> blocks
  const toolCallRegex = /<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/g
  const matches = [...response.matchAll(toolCallRegex)]

  for (const match of matches) {
    const toolCallContent = match[1]

    // Extract function name: <function=name>
    const functionMatch = toolCallContent.match(/<function=([^>]+)>/)
    if (!functionMatch) continue

    const functionName = functionMatch[1].trim()
    const args: Record<string, string> = {}

    // Extract parameters: <parameter=name>value</parameter>
    const paramRegex = /<parameter=([^>]+)>\s*([\s\S]*?)\s*<\/parameter>/g
    const paramMatches = [...toolCallContent.matchAll(paramRegex)]

    for (const paramMatch of paramMatches) {
      const paramName = paramMatch[1].trim()
      const paramValue = paramMatch[2].trim()
      args[paramName] = paramValue
    }

    toolCalls.push({
      name: functionName,
      arguments: args,
    })
  }

  return toolCalls
}

/**
 * Check if response contains tool calls in Stepfun format
 */
export function hasStepfunToolCalls(response: string): boolean {
  return /<tool_call>[\s\S]*?<function=/.test(response)
}

/**
 * Format tool response for Stepfun model
 */
export function formatStepfunToolResponse(toolName: string, result: string): string {
  return `<tool_response>
[${toolName}] ${result}
</tool_response>`
}

/**
 * Generate Stepfun-compatible system prompt for data analyst
 */
export function getStepfunAnalystSystemPrompt(
  datasetDir: string,
  workspaceDir: string,
  pythonPath: string,
): string {
  return `You are a professional Data Analyst helping a non-technical user explore and analyze datasets.

<data-analysis-context>
You are helping a non-technical user analyze a dataset.

Working context:
- Dataset directory: ${datasetDir}
- Workspace for outputs: ${workspaceDir}
- The user is a beginner - explain everything in simple terms
- Always show your work: run code, show outputs, then explain

Python environment:
- ALWAYS use the specified Python path to execute Python code
- Available libraries: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn
- When creating plots, ALWAYS use plt.savefig() to save to workspace
- NEVER use plt.show() - save plots instead

Your approach:
1. First explore the data structure and content
2. Provide clear, jargon-free explanations
3. Create visualizations when helpful
4. Focus on actionable insights the user can share with their manager
</data-analysis-context>

IMPORTANT ENVIRONMENT INFO:
- Python executable: ${pythonPath}
- Dataset directory: ${datasetDir}
- Workspace for scripts: ${workspaceDir}

You have access to the following tools. Use them to analyze data.

Tool definitions:

1. Bash - Execute shell commands
   Parameters:
   - command (required): The shell command to execute
   Example: Run Python scripts with ${pythonPath} script.py

2. Read - Read file contents
   Parameters:
   - file_path (required): Path to the file to read

3. Write - Write content to a file
   Parameters:
   - file_path (required): Path to the file to write
   - content (required): Content to write

4. Glob - Find files matching a pattern
   Parameters:
   - pattern (required): Glob pattern to match files

5. Grep - Search for patterns in files
   Parameters:
   - pattern (required): Regex pattern to search
   - path (required): Path to search in

6. Edit - Edit existing files
   Parameters:
   - file_path (required): Path to the file
   - old_string (required): String to replace
   - new_string (required): Replacement string

To call a tool, use this EXACT format:
<tool_call>
<function=ToolName>
<parameter=param_name>
parameter_value
</parameter>
</function>
</tool_call>

Example - List files:
<tool_call>
<function=Bash>
<parameter=command>
ls -la ${datasetDir}
</parameter>
</function>
</tool_call>

Example - Read a CSV file:
<tool_call>
<function=Read>
<parameter=file_path>
${datasetDir}/data.csv
</parameter>
</function>
</tool_call>

Example - Write a Python script:
<tool_call>
<function=Write>
<parameter=file_path>
${workspaceDir}/analysis.py
</parameter>
<parameter=content>
import pandas as pd
df = pd.read_csv('${datasetDir}/data.csv')
print(df.head())
</parameter>
</function>
</tool_call>

Example - Run Python script:
<tool_call>
<function=Bash>
<parameter=command>
${pythonPath} ${workspaceDir}/analysis.py
</parameter>
</function>
</tool_call>

When analyzing data:
1. First use Glob to find data files
2. Use Read to examine file contents
3. Write Python scripts to workspace directory
4. Execute scripts with: ${pythonPath} <script_path>
5. Explain results in simple, non-technical terms`
}

/**
 * Generate Stepfun-compatible User Agent system prompt
 */
export function getStepfunUserAgentPrompt(
  datasetTitle: string,
  datasetDesc: string,
): { systemPrompt: string } {
  const systemPrompt = `You are simulating a data science beginner who works at a company.
Your manager asked you to analyze a dataset called "${datasetTitle}".
Description: ${datasetDesc}

You have NO coding skills and need help from a data analyst.
You communicate in natural, casual language.

Your behavior:
- Ask questions about the data in plain, simple language
- Be curious about patterns, trends, and insights
- Ask follow-up questions based on what the analyst tells you
- Request visualizations ("Can you show me a chart of ...?")
- You should NOT write or suggest any code
- Keep each message concise (2-3 sentences max)
- Output ONLY your question text, nothing else
- Be creative and natural - don't use generic phrases
- Your questions should reflect genuine curiosity about THIS specific dataset`

  return { systemPrompt }
}

export {
  parseStepfunToolCalls as parseToolCalls,
  hasStepfunToolCalls as hasToolCalls,
  formatStepfunToolResponse as formatToolResponse,
  getStepfunAnalystSystemPrompt as getAnalystSystemPrompt,
  getStepfunUserAgentPrompt as getUserAgentPrompt,
}
