#!/usr/bin/env bun
/**
 * Trajectory Collection Mode Entry Point
 *
 * This script runs claude-code-clean in a special mode that:
 * 1. Uses a User Agent (AI) to simulate user questions
 * 2. Uses the Analyst Agent (claude-code-clean) to perform data analysis
 * 3. Records the full trajectory in CoPaw-Flash format
 *
 * Usage:
 *   CLAUDE_CODE_USE_OPENAI=1 \
 *   OPENAI_API_KEY=sk-or-v1-xxx \
 *   OPENAI_BASE_URL=https://openrouter.ai/api/v1 \
 *   OPENAI_MODEL=qwen/qwen3.6-plus:free \
 *   bun run src/entrypoints/trajectoryCollect.tsx \
 *     --dataset-dir ./data \
 *     --dataset-title "Sales Data" \
 *     --output-dir ./trajectory_output \
 *     --max-turns 3
 */

import { parseArgs } from 'util'
import { existsSync, mkdirSync, readdirSync, writeFileSync } from 'fs'
import { join, resolve } from 'path'
import {
  trajectoryCollector,
  formatToolCall,
  formatToolResponse,
  formatThinking,
} from '../trajectory/collector.js'
import {
  getUserAgentPrompt,
  getAnalystSystemPromptExtension,
  callUserAgent,
  extractUserQuestion,
} from '../trajectory/dualAgent.js'

// Parse command line arguments
const { values: args } = parseArgs({
  options: {
    'dataset-dir': { type: 'string', default: './data' },
    'dataset-title': { type: 'string', default: 'Dataset' },
    'dataset-desc': { type: 'string', default: 'A dataset for analysis' },
    'output-dir': { type: 'string', default: './trajectory_output' },
    'max-turns': { type: 'string', default: '3' },
    'workspace-dir': { type: 'string' },
  },
})

const config = {
  apiKey: process.env.OPENAI_API_KEY || '',
  baseUrl: process.env.OPENAI_BASE_URL || 'https://openrouter.ai/api/v1',
  model: process.env.OPENAI_MODEL || 'qwen/qwen3.6-plus:free',
  datasetDir: resolve(args['dataset-dir'] || './data'),
  datasetTitle: args['dataset-title'] || 'Dataset',
  datasetDesc: args['dataset-desc'] || 'A dataset for analysis',
  outputDir: resolve(args['output-dir'] || './trajectory_output'),
  maxTurns: parseInt(args['max-turns'] || '3', 10),
  workspaceDir: args['workspace-dir']
    ? resolve(args['workspace-dir'])
    : resolve(args['dataset-dir'] || './data', 'workspace'),
}

// Ensure directories exist
if (!existsSync(config.outputDir)) {
  mkdirSync(config.outputDir, { recursive: true })
}
if (!existsSync(config.workspaceDir)) {
  mkdirSync(config.workspaceDir, { recursive: true })
}

console.log('='
.repeat(70))
console.log('TRAJECTORY COLLECTION MODE')
console.log('='.repeat(70))
console.log(`Model: ${config.model}`)
console.log(`Dataset: ${config.datasetTitle}`)
console.log(`Dataset Dir: ${config.datasetDir}`)
console.log(`Output Dir: ${config.outputDir}`)
console.log(`Max Turns: ${config.maxTurns}`)
console.log('='.repeat(70))

// Check API key
if (!config.apiKey) {
  console.error('Error: OPENAI_API_KEY environment variable not set')
  process.exit(1)
}

// Check dataset directory
if (!existsSync(config.datasetDir)) {
  console.error(`Error: Dataset directory not found: ${config.datasetDir}`)
  process.exit(1)
}

// List files in dataset directory
const datasetFiles = readdirSync(config.datasetDir).filter(
  (f) => f.endsWith('.csv') || f.endsWith('.json') || f.endsWith('.xlsx'),
)
console.log(`Dataset files: ${datasetFiles.join(', ') || '(none)'}`)

// Initialize trajectory collector
trajectoryCollector.enable()
const convId = trajectoryCollector.startConversation({
  model: config.model,
  dataset: config.datasetTitle,
})
console.log(`Started conversation: ${convId}`)

// Get User Agent prompt
const userAgentConfig = getUserAgentPrompt(config.datasetTitle, config.datasetDesc)

// Analyst system prompt extension
const analystExtension = getAnalystSystemPromptExtension(
  config.datasetDir,
  config.workspaceDir,
)

// Conversation history for User Agent context
const userAgentHistory: Array<{ role: string; content: string }> = []

// Initial question from User Agent
let userQuestion = userAgentConfig.initialQuestion

// Python environment path for data analysis
const PYTHON_PATH = resolve(__dirname, '../../.venv-analysis/bin/python')

// Full analyst system prompt
const analystSystemPrompt = `You are a professional Data Analyst helping a non-technical user explore and analyze datasets.

${analystExtension}

IMPORTANT ENVIRONMENT INFO:
- Python executable with pandas/numpy/matplotlib/seaborn: ${PYTHON_PATH}
- Always use this Python path for running scripts: ${PYTHON_PATH} script.py
- Dataset directory: ${config.datasetDir}
- Workspace for scripts: ${config.workspaceDir}

IMPORTANT: You have access to tools for data analysis.
When you need to execute code or read files, use the tools.

Tool call format (use this EXACT format):
<tool_call>
{"name": "ToolName", "arguments": {"param": "value"}}
</tool_call>

Available tools:
- Bash: Execute shell commands. Args: {"command": "..."}
  Use this to run Python scripts for data analysis.
  IMPORTANT: Always use ${PYTHON_PATH} to run Python scripts.

- Read: Read file contents. Args: {"file_path": "..."}
  Use this to examine data files (CSV, JSON, etc.)

- Write: Write content to file. Args: {"file_path": "...", "content": "..."}
  Use this to create Python analysis scripts in the workspace.

- Glob: Find files matching pattern. Args: {"pattern": "..."}
  Use this to discover what data files are available.

- Grep: Search for patterns in files. Args: {"pattern": "...", "path": "..."}
  Use this to find specific content in files.

- Edit: Edit existing files. Args: {"file_path": "...", "old_string": "...", "new_string": "..."}
  Use this to modify existing scripts.

When analyzing data:
1. First use Glob and Read to understand available data files
2. Write Python scripts using Write tool (save to workspace directory)
3. Execute scripts with: ${PYTHON_PATH} <script_path>
4. Explain results in simple, non-technical terms

Example for running Python:
<tool_call>
{"name": "Bash", "arguments": {"command": "${PYTHON_PATH} ${config.workspaceDir}/analysis.py"}}
</tool_call>`

// Record system prompt at conversation start
let systemPromptRecorded = false

/**
 * Call the Analyst Agent (claude-code-clean via OpenAI-compatible API).
 * This is a simplified version - in full integration, this would use
 * the actual claude-code-clean agent loop with tool execution.
 */
async function callAnalystAgent(userMessage: string): Promise<string> {
  // Record system prompt to trajectory (only once)
  if (!systemPromptRecorded) {
    trajectoryCollector.addSystemMessage(analystSystemPrompt)
    systemPromptRecorded = true
  }

  // Build messages with analyst context
  const messages = [
    {
      role: 'system',
      content: analystSystemPrompt,
    },
    ...trajectoryCollector.getCurrentMessages().filter(m => m.role !== 'system').map((m) => ({
      role: m.role,
      content: m.content,
    })),
    { role: 'user', content: userMessage },
  ]

  const response = await fetch(`${config.baseUrl}/chat/completions`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${config.apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: config.model,
      messages,
      stream: false,
    }),
  })

  if (!response.ok) {
    const error = await response.text()
    throw new Error(`Analyst Agent API error: ${response.status} - ${error}`)
  }

  const data = (await response.json()) as {
    choices: Array<{ message: { content: string } }>
    usage?: { prompt_tokens?: number; completion_tokens?: number }
  }

  // Update token counts
  if (data.usage) {
    trajectoryCollector.updateTokens(
      data.usage.prompt_tokens || 0,
      data.usage.completion_tokens || 0,
    )
  }

  return data.choices[0]?.message?.content || ''
}

/**
 * Parse and execute tool calls from analyst response.
 * Returns the response with tool results appended.
 */
async function processToolCalls(response: string): Promise<string> {
  // Extract tool calls using regex
  const toolCallRegex = /<tool_call>\s*(\{[\s\S]*?\})\s*<\/tool_call>/g
  const matches = [...response.matchAll(toolCallRegex)]

  if (matches.length === 0) {
    return response
  }

  let fullResponse = response
  const toolResults: string[] = []

  for (const match of matches) {
    try {
      const toolCall = JSON.parse(match[1]) as {
        name: string
        arguments: Record<string, string>
      }

      console.log(`  [Tool] ${toolCall.name}`)

      let result = ''

      // Execute tool based on name
      switch (toolCall.name) {
        case 'Bash': {
          const { execSync } = await import('child_process')
          try {
            result = execSync(toolCall.arguments.command, {
              cwd: config.datasetDir,
              encoding: 'utf-8',
              timeout: 60000,
              maxBuffer: 10 * 1024 * 1024,
            })
          } catch (e: unknown) {
            const error = e as { stdout?: string; stderr?: string; message?: string }
            result = `Error: ${error.message || 'Command failed'}\n${error.stdout || ''}${error.stderr || ''}`
          }
          break
        }

        case 'Read': {
          const { readFileSync } = await import('fs')
          const filePath = toolCall.arguments.file_path
          const fullPath = filePath.startsWith('/')
            ? filePath
            : join(config.datasetDir, filePath)
          try {
            const content = readFileSync(fullPath, 'utf-8')
            // Add line numbers
            const lines = content.split('\n')
            result = lines
              .slice(0, 100)
              .map((line, i) => `${i + 1}\t${line}`)
              .join('\n')
            if (lines.length > 100) {
              result += `\n... (${lines.length - 100} more lines)`
            }
          } catch (e: unknown) {
            const error = e as { message?: string }
            result = `Error reading file: ${error.message || 'Unknown error'}`
          }
          break
        }

        case 'Write': {
          const { writeFileSync, mkdirSync } = await import('fs')
          const { dirname } = await import('path')
          const filePath = toolCall.arguments.file_path
          const fullPath = filePath.startsWith('/')
            ? filePath
            : join(config.workspaceDir, filePath)
          try {
            mkdirSync(dirname(fullPath), { recursive: true })
            writeFileSync(fullPath, toolCall.arguments.content, 'utf-8')
            result = `Successfully wrote to ${filePath}`
          } catch (e: unknown) {
            const error = e as { message?: string }
            result = `Error writing file: ${error.message || 'Unknown error'}`
          }
          break
        }

        case 'Glob': {
          const { globSync } = await import('glob')
          try {
            const matches = globSync(toolCall.arguments.pattern, {
              cwd: config.datasetDir,
            })
            result = matches.slice(0, 50).join('\n') || 'No files found'
          } catch (e: unknown) {
            const error = e as { message?: string }
            result = `Error: ${error.message || 'Glob failed'}`
          }
          break
        }

        case 'Grep': {
          const { execSync } = await import('child_process')
          try {
            const pattern = toolCall.arguments.pattern
            const searchPath = toolCall.arguments.path || config.datasetDir
            const fullPath = searchPath.startsWith('/') ? searchPath : join(config.datasetDir, searchPath)
            result = execSync(`grep -rn "${pattern}" "${fullPath}" 2>/dev/null | head -50`, {
              encoding: 'utf-8',
              timeout: 30000,
            })
            if (!result.trim()) {
              result = 'No matches found'
            }
          } catch (e: unknown) {
            const error = e as { stdout?: string; message?: string }
            result = error.stdout || 'No matches found'
          }
          break
        }

        case 'Edit': {
          const { readFileSync, writeFileSync } = await import('fs')
          try {
            const filePath = toolCall.arguments.file_path
            const fullPath = filePath.startsWith('/') ? filePath : join(config.workspaceDir, filePath)
            const content = readFileSync(fullPath, 'utf-8')
            const oldStr = toolCall.arguments.old_string
            const newStr = toolCall.arguments.new_string
            if (!content.includes(oldStr)) {
              result = `Error: old_string not found in file`
            } else {
              const newContent = content.replace(oldStr, newStr)
              writeFileSync(fullPath, newContent, 'utf-8')
              result = `Successfully edited ${filePath}`
            }
          } catch (e: unknown) {
            const error = e as { message?: string }
            result = `Error editing file: ${error.message || 'Unknown error'}`
          }
          break
        }

        default:
          result = `Unknown tool: ${toolCall.name}`
      }

      toolResults.push(result)
      console.log(`  [Result] ${result.slice(0, 100)}...`)
    } catch (e) {
      console.error(`  [Error] Failed to parse tool call: ${e}`)
      toolResults.push(`Error parsing tool call: ${e}`)
    }
  }

  // Return tool results for next iteration
  return toolResults.join('\n---\n')
}

// Main conversation loop
async function runConversation(): Promise<void> {
  for (let turn = 1; turn <= config.maxTurns; turn++) {
    console.log(`\n${'─'.repeat(70)}`)
    console.log(`Turn ${turn}/${config.maxTurns}`)
    console.log('─'.repeat(70))

    // Record user message
    console.log(`[User] ${userQuestion.slice(0, 100)}...`)
    trajectoryCollector.addUserMessage(userQuestion)

    // Get analyst response
    let analystResponse = await callAnalystAgent(userQuestion)
    console.log(`[Analyst] ${analystResponse.slice(0, 200)}...`)

    // Process tool calls if present (up to 5 iterations)
    let toolIterations = 0
    while (
      analystResponse.includes('<tool_call>') &&
      toolIterations < 5
    ) {
      toolIterations++

      // Record assistant message with tool calls
      trajectoryCollector.addAssistantMessage(analystResponse)

      // Execute tools and get results
      const toolResults = await processToolCalls(analystResponse)

      // Record tool response
      trajectoryCollector.addToolResponse(toolResults)

      // Continue conversation with tool results
      analystResponse = await callAnalystAgent(
        `Tool execution results:\n${toolResults}\n\nPlease continue your analysis based on these results.`,
      )
      console.log(`[Analyst] ${analystResponse.slice(0, 200)}...`)
    }

    // Record final assistant message (no tool calls)
    trajectoryCollector.addAssistantMessage(analystResponse)

    // Update User Agent history
    userAgentHistory.push(
      { role: 'user', content: userQuestion },
      { role: 'assistant', content: analystResponse },
    )

    // Check if we should continue
    if (turn >= config.maxTurns) {
      break
    }

    // Generate follow-up question from User Agent
    try {
      const userResponse = await callUserAgent(
        {
          apiKey: config.apiKey,
          baseUrl: config.baseUrl,
          model: config.model,
          maxTurns: config.maxTurns,
          datasetDir: config.datasetDir,
          datasetTitle: config.datasetTitle,
          datasetDesc: config.datasetDesc,
        },
        userAgentConfig.systemPrompt,
        userAgentHistory,
      )
      userQuestion = extractUserQuestion(userResponse)

      if (!userQuestion || userQuestion.length < 5) {
        console.log('[User Agent] No follow-up question generated, ending conversation')
        break
      }
    } catch (e) {
      console.error(`[User Agent] Error generating question: ${e}`)
      break
    }
  }
}

// Run and save
async function main(): Promise<void> {
  try {
    await runConversation()
  } catch (e) {
    console.error(`Error during conversation: ${e}`)
  }

  // End conversation and export
  trajectoryCollector.endConversation()

  const jsonlPath = trajectoryCollector.exportJSONL()
  const fullJsonPath = trajectoryCollector.exportFullJSON()

  const stats = trajectoryCollector.getStats()

  console.log('\n' + '='.repeat(70))
  console.log('COLLECTION COMPLETE')
  console.log('='.repeat(70))
  console.log(`Conversations: ${stats.conversations}`)
  console.log(`Total messages: ${stats.totalMessages}`)
  console.log(`Tool calls: ${stats.totalToolCalls}`)
  console.log(`Think blocks: ${stats.totalThinkBlocks}`)
  console.log(`\nOutput files:`)
  console.log(`  JSONL (training): ${jsonlPath}`)
  console.log(`  Full JSON: ${fullJsonPath}`)
  console.log('='.repeat(70))

  // Print sample of generated data
  const conv = trajectoryCollector['conversations'][0]
  if (conv) {
    console.log('\nSAMPLE TRAJECTORY:')
    console.log('─'.repeat(70))
    for (const msg of conv.messages.slice(0, 4)) {
      const preview =
        msg.content.length > 150
          ? msg.content.slice(0, 150) + '...'
          : msg.content
      console.log(`[${msg.role}] ${preview}`)
      console.log()
    }
    if (conv.messages.length > 4) {
      console.log(`... (${conv.messages.length - 4} more messages)`)
    }
  }
}

main().catch(console.error)
