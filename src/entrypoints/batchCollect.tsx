#!/usr/bin/env bun
/**
 * Batch Trajectory Collection for Kaggle Top 1000 Datasets
 *
 * Features:
 * - API key rotation across multiple OpenRouter accounts
 * - Checkpoint/resume support
 * - Rate limit handling with automatic retry
 * - Progress tracking and statistics
 *
 * Usage:
 *   bun run src/entrypoints/batchCollect.tsx \
 *     --kaggle-dir ./kaggle-top1000 \
 *     --output-dir ./copaw-trajectories \
 *     --max-turns 3 \
 *     --checkpoint ./checkpoint.json
 */

import { parseArgs } from 'util'
import {
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  writeFileSync,
  appendFileSync,
  statSync,
} from 'fs'
import { join, resolve, basename } from 'path'
import {
  trajectoryCollector,
  formatToolResponse,
} from '../trajectory/collector.js'
import {
  getUserAgentPrompt,
  generateInitialQuestion,
  getAnalystSystemPromptExtension,
  callUserAgent,
  extractUserQuestion,
} from '../trajectory/dualAgent.js'

// Parse command line arguments
const { values: args } = parseArgs({
  options: {
    'kaggle-dir': { type: 'string', default: './kaggle-top1000' },
    'output-dir': { type: 'string', default: './copaw-trajectories' },
    'max-turns': { type: 'string', default: '3' },
    'checkpoint': { type: 'string', default: './checkpoint.json' },
    'start-index': { type: 'string', default: '0' },
    'limit': { type: 'string', default: '0' },
    'delay': { type: 'string', default: '10000' },
  },
})

// Load API keys from environment variable
// Set OPENROUTER_API_KEYS in .env file (comma-separated)
function loadApiKeys(): string[] {
  const envKeys = process.env.OPENROUTER_API_KEYS
  if (!envKeys) {
    console.error('ERROR: OPENROUTER_API_KEYS environment variable not set!')
    console.error('Please create a .env file with your API keys.')
    console.error('Example: OPENROUTER_API_KEYS=sk-or-v1-xxx,sk-or-v1-yyy')
    process.exit(1)
  }
  const keys = envKeys.split(',').map(k => k.trim()).filter(k => k.length > 0)
  if (keys.length === 0) {
    console.error('ERROR: No valid API keys found in OPENROUTER_API_KEYS')
    process.exit(1)
  }
  return keys
}

const API_KEYS = loadApiKeys()

const config = {
  baseUrl: 'https://openrouter.ai/api/v1',
  model: 'qwen/qwen3.6-plus:free',
  kaggleDir: resolve(args['kaggle-dir'] || './kaggle-top1000'),
  outputDir: resolve(args['output-dir'] || './copaw-trajectories'),
  checkpointFile: resolve(args['checkpoint'] || './checkpoint.json'),
  maxTurns: parseInt(args['max-turns'] || '3', 10),
  startIndex: parseInt(args['start-index'] || '0', 10),
  limit: parseInt(args['limit'] || '0', 10),
  delayMs: parseInt(args['delay'] || '5000', 10),
}

// Python environment path
const PYTHON_PATH = resolve(__dirname, '../../.venv-analysis/bin/python')

// State tracking
interface Checkpoint {
  completedDatasets: string[]
  failedDatasets: { slug: string; error: string; attempts: number }[]
  currentKeyIndex: number
  totalProcessed: number
  totalSuccess: number
  totalFailed: number
  lastUpdated: string
}

let checkpoint: Checkpoint = {
  completedDatasets: [],
  failedDatasets: [],
  currentKeyIndex: 0,
  totalProcessed: 0,
  totalSuccess: 0,
  totalFailed: 0,
  lastUpdated: new Date().toISOString(),
}

let currentKeyIndex = 0

function getNextApiKey(): string {
  const key = API_KEYS[currentKeyIndex]
  currentKeyIndex = (currentKeyIndex + 1) % API_KEYS.length
  return key
}

function loadCheckpoint(): void {
  if (existsSync(config.checkpointFile)) {
    try {
      checkpoint = JSON.parse(readFileSync(config.checkpointFile, 'utf-8'))
      currentKeyIndex = checkpoint.currentKeyIndex
      console.log(`Loaded checkpoint: ${checkpoint.completedDatasets.length} completed, ${checkpoint.failedDatasets.length} failed`)
    } catch (e) {
      console.log('No valid checkpoint found, starting fresh')
    }
  }
}

function saveCheckpoint(): void {
  checkpoint.currentKeyIndex = currentKeyIndex
  checkpoint.lastUpdated = new Date().toISOString()
  writeFileSync(config.checkpointFile, JSON.stringify(checkpoint, null, 2))
}

function getDatasetTitle(slug: string): string {
  // Convert slug to readable title
  // e.g., "abcsds_pokemon" -> "Pokemon"
  const parts = slug.split('_')
  if (parts.length > 1) {
    return parts.slice(1).join(' ').replace(/-/g, ' ')
      .split(' ')
      .map(w => w.charAt(0).toUpperCase() + w.slice(1))
      .join(' ')
  }
  return slug
}

async function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

async function callAnalystWithRetry(
  apiKey: string,
  messages: Array<{ role: string; content: string }>,
  maxRetries: number = 3
): Promise<{ content: string; inputTokens: number; outputTokens: number }> {
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      const response = await fetch(`${config.baseUrl}/chat/completions`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: config.model,
          messages,
          stream: false,
        }),
      })

      if (response.status === 429) {
        // Rate limited - switch to next key and retry
        const waitTime = 60000 + Math.random() * 30000 // 60-90 seconds
        console.log(`  [Rate limited] Switching API key and waiting ${(waitTime/1000).toFixed(0)}s...`)
        apiKey = getNextApiKey()
        await sleep(waitTime)
        continue
      }

      if (!response.ok) {
        const error = await response.text()
        throw new Error(`API error ${response.status}: ${error}`)
      }

      const data = (await response.json()) as {
        choices: Array<{ message: { content: string } }>
        usage?: { prompt_tokens?: number; completion_tokens?: number }
      }

      return {
        content: data.choices[0]?.message?.content || '',
        inputTokens: data.usage?.prompt_tokens || 0,
        outputTokens: data.usage?.completion_tokens || 0,
      }
    } catch (e) {
      if (attempt < maxRetries - 1) {
        console.log(`  [Retry ${attempt + 1}] ${e}`)
        await sleep(5000 * (attempt + 1))
      } else {
        throw e
      }
    }
  }
  throw new Error('Max retries exceeded')
}

async function processToolCalls(
  response: string,
  datasetDir: string,
  workspaceDir: string
): Promise<string> {
  const toolCallRegex = /<tool_call>\s*(\{[\s\S]*?\})\s*<\/tool_call>/g
  const matches = [...response.matchAll(toolCallRegex)]

  if (matches.length === 0) {
    return response
  }

  const toolResults: string[] = []

  for (const match of matches) {
    try {
      const toolCall = JSON.parse(match[1]) as {
        name: string
        arguments: Record<string, string>
      }

      let result = ''

      switch (toolCall.name) {
        case 'Bash': {
          const { execSync } = await import('child_process')
          try {
            result = execSync(toolCall.arguments.command, {
              cwd: datasetDir,
              encoding: 'utf-8',
              timeout: 120000,
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
          const fullPath = filePath.startsWith('/') ? filePath : join(datasetDir, filePath)
          try {
            const content = readFileSync(fullPath, 'utf-8')
            const lines = content.split('\n')
            result = lines.slice(0, 100).map((line, i) => `${i + 1}\t${line}`).join('\n')
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
          const fullPath = filePath.startsWith('/') ? filePath : join(workspaceDir, filePath)
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
            const pattern = toolCall.arguments.pattern
            const searchPath = pattern.startsWith('/') ? pattern : join(datasetDir, pattern)
            const matches = globSync(searchPath)
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
            const searchPath = toolCall.arguments.path || datasetDir
            const fullPath = searchPath.startsWith('/') ? searchPath : join(datasetDir, searchPath)
            result = execSync(`grep -rn "${pattern}" "${fullPath}" 2>/dev/null | head -50`, {
              encoding: 'utf-8',
              timeout: 30000,
            })
            if (!result.trim()) result = 'No matches found'
          } catch (e: unknown) {
            const error = e as { stdout?: string }
            result = error.stdout || 'No matches found'
          }
          break
        }

        case 'Edit': {
          const { readFileSync, writeFileSync } = await import('fs')
          try {
            const filePath = toolCall.arguments.file_path
            const fullPath = filePath.startsWith('/') ? filePath : join(workspaceDir, filePath)
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
    } catch (e) {
      toolResults.push(`Error parsing tool call: ${e}`)
    }
  }

  return toolResults.join('\n---\n')
}

async function collectTrajectoryForDataset(
  datasetSlug: string,
  datasetDir: string,
  apiKey: string  // Now passed explicitly for parallel processing
): Promise<{ success: boolean; trajectory?: object; error?: string }> {
  const title = getDatasetTitle(datasetSlug)
  const workspaceDir = join(datasetDir, 'workspace')

  // Create workspace
  if (!existsSync(workspaceDir)) {
    mkdirSync(workspaceDir, { recursive: true })
  }

  // Build system prompt
  const analystExtension = getAnalystSystemPromptExtension(datasetDir, workspaceDir)
  const systemPrompt = `You are a professional Data Analyst helping a non-technical user explore and analyze datasets.

${analystExtension}

IMPORTANT ENVIRONMENT INFO:
- Python executable with pandas/numpy/matplotlib/seaborn: ${PYTHON_PATH}
- Always use this Python path for running scripts: ${PYTHON_PATH} script.py
- Dataset directory: ${datasetDir}
- Workspace for scripts: ${workspaceDir}

IMPORTANT: You have access to tools for data analysis.
When you need to execute code or read files, use the tools.

Tool call format (use this EXACT format):
<tool_call>
{"name": "ToolName", "arguments": {"param": "value"}}
</tool_call>

Available tools:
- Bash: Execute shell commands. Args: {"command": "..."}
  IMPORTANT: Always use ${PYTHON_PATH} to run Python scripts.

- Read: Read file contents. Args: {"file_path": "..."}

- Write: Write content to file. Args: {"file_path": "...", "content": "..."}

- Glob: Find files matching pattern. Args: {"pattern": "..."}

- Grep: Search for patterns in files. Args: {"pattern": "...", "path": "..."}

- Edit: Edit existing files. Args: {"file_path": "...", "old_string": "...", "new_string": "..."}

When analyzing data:
1. First use Glob and Read to understand available data files
2. Write Python scripts using Write tool (save to workspace directory)
3. Execute scripts with: ${PYTHON_PATH} <script_path>
4. Explain results in simple, non-technical terms`

  // Initialize trajectory
  const messages: Array<{ role: 'user' | 'assistant' | 'system'; content: string }> = []
  let totalInputTokens = 0
  let totalOutputTokens = 0

  // User Agent config
  const userAgentConfig = getUserAgentPrompt(title, `A dataset for analysis from Kaggle`)
  const userAgentHistory: Array<{ role: string; content: string }> = []

  // Generate initial question using AI (not templated!)
  let userQuestion = await generateInitialQuestion(
    {
      apiKey,
      baseUrl: config.baseUrl,
      model: config.model,
      maxTurns: config.maxTurns,
      datasetDir,
      datasetTitle: title,
      datasetDesc: 'Kaggle dataset',
    },
    userAgentConfig.systemPrompt
  )

  try {
    for (let turn = 1; turn <= config.maxTurns; turn++) {
      // Add user message
      messages.push({ role: 'user', content: userQuestion })

      // Add system prompt on first turn
      if (turn === 1) {
        messages.push({ role: 'system', content: systemPrompt })
      }

      // Build API messages
      const apiMessages = [
        { role: 'system', content: systemPrompt },
        ...messages.filter(m => m.role !== 'system').map(m => ({
          role: m.role,
          content: m.content,
        })),
      ]

      // Get analyst response
      let response = await callAnalystWithRetry(apiKey, apiMessages)
      totalInputTokens += response.inputTokens
      totalOutputTokens += response.outputTokens

      let analystContent = response.content

      // Process tool calls (up to 5 iterations)
      let toolIterations = 0
      while (analystContent.includes('<tool_call>') && toolIterations < 5) {
        toolIterations++

        // Add assistant message with tool call
        messages.push({ role: 'assistant', content: analystContent })

        // Execute tools
        const toolResults = await processToolCalls(analystContent, datasetDir, workspaceDir)

        // Add tool response
        messages.push({ role: 'user', content: formatToolResponse(toolResults) })

        // Continue conversation
        const continueMessages = [
          { role: 'system', content: systemPrompt },
          ...messages.filter(m => m.role !== 'system').map(m => ({
            role: m.role,
            content: m.content,
          })),
          { role: 'user', content: 'Please continue your analysis based on the tool results.' },
        ]

        response = await callAnalystWithRetry(apiKey, continueMessages)
        totalInputTokens += response.inputTokens
        totalOutputTokens += response.outputTokens
        analystContent = response.content
      }

      // Add final assistant message
      messages.push({ role: 'assistant', content: analystContent })

      // Update User Agent history
      userAgentHistory.push(
        { role: 'user', content: userQuestion },
        { role: 'assistant', content: analystContent }
      )

      // Generate follow-up question if not last turn
      if (turn < config.maxTurns) {
        try {
          const userResponse = await callUserAgent(
            {
              apiKey,
              baseUrl: config.baseUrl,
              model: config.model,
              maxTurns: config.maxTurns,
              datasetDir,
              datasetTitle: title,
              datasetDesc: 'Kaggle dataset',
            },
            userAgentConfig.systemPrompt,
            userAgentHistory
          )
          userQuestion = extractUserQuestion(userResponse)

          if (!userQuestion || userQuestion.length < 5) {
            break
          }
        } catch (e) {
          break
        }
      }
    }

    // Build final trajectory
    const trajectory = {
      messages,
      metadata: {
        timestamp: new Date().toISOString(),
        model: config.model,
        dataset_slug: datasetSlug,
        dataset_title: title,
        total_input_tokens: totalInputTokens,
        total_output_tokens: totalOutputTokens,
        turns: messages.filter(m => m.role === 'assistant').length,
      },
    }

    return { success: true, trajectory }
  } catch (e) {
    return { success: false, error: String(e) }
  }
}

// Mutex for thread-safe file writes
let writeLock = false
async function safeAppendFile(file: string, content: string): Promise<void> {
  while (writeLock) await sleep(10)
  writeLock = true
  try {
    appendFileSync(file, content)
  } finally {
    writeLock = false
  }
}

async function safeUpdateCheckpoint(
  update: (cp: Checkpoint) => void
): Promise<void> {
  while (writeLock) await sleep(10)
  writeLock = true
  try {
    update(checkpoint)
    saveCheckpoint()
  } finally {
    writeLock = false
  }
}

// Worker function - each key processes datasets from its queue
async function worker(
  workerId: number,
  apiKey: string,
  datasets: string[],
  outputFile: string,
  totalDatasets: number
): Promise<{ success: number; failed: number }> {
  let success = 0
  let failed = 0

  for (const datasetSlug of datasets) {
    const datasetDir = join(config.kaggleDir, datasetSlug)
    const startTime = Date.now()

    console.log(`[Worker ${workerId}] Processing: ${datasetSlug}`)

    const result = await collectTrajectoryForDataset(datasetSlug, datasetDir, apiKey)
    const duration = ((Date.now() - startTime) / 1000).toFixed(1)

    if (result.success && result.trajectory) {
      await safeAppendFile(outputFile, JSON.stringify(result.trajectory) + '\n')
      await safeUpdateCheckpoint(cp => {
        cp.completedDatasets.push(datasetSlug)
        cp.totalSuccess++
        cp.totalProcessed++
      })
      success++
      const progress = ((checkpoint.completedDatasets.length / totalDatasets) * 100).toFixed(1)
      console.log(`[Worker ${workerId}] ✓ ${datasetSlug} (${duration}s) - Progress: ${progress}%`)
    } else {
      await safeUpdateCheckpoint(cp => {
        const existingFail = cp.failedDatasets.find(f => f.slug === datasetSlug)
        if (existingFail) {
          existingFail.attempts++
          existingFail.error = result.error || 'Unknown error'
        } else {
          cp.failedDatasets.push({
            slug: datasetSlug,
            error: result.error || 'Unknown error',
            attempts: 1,
          })
        }
        cp.totalFailed++
        cp.totalProcessed++
      })
      failed++
      console.log(`[Worker ${workerId}] ✗ ${datasetSlug} (${duration}s): ${result.error?.slice(0, 80)}`)
    }

    // Small delay between requests for each worker
    await sleep(config.delayMs)
  }

  return { success, failed }
}

async function main(): Promise<void> {
  console.log('='.repeat(70))
  console.log('PARALLEL BATCH TRAJECTORY COLLECTION FOR KAGGLE TOP 1000')
  console.log('='.repeat(70))
  console.log(`Kaggle Dir: ${config.kaggleDir}`)
  console.log(`Output Dir: ${config.outputDir}`)
  console.log(`Max Turns: ${config.maxTurns}`)
  console.log(`Parallel Workers: ${API_KEYS.length}`)
  console.log(`Delay per worker: ${config.delayMs}ms`)
  console.log('='.repeat(70))

  // Ensure output directory exists
  if (!existsSync(config.outputDir)) {
    mkdirSync(config.outputDir, { recursive: true })
  }

  // Load checkpoint
  loadCheckpoint()

  // Get list of datasets
  const allDatasets = readdirSync(config.kaggleDir)
    .filter(d => {
      const fullPath = join(config.kaggleDir, d)
      return statSync(fullPath).isDirectory()
    })
    .sort()

  console.log(`\nTotal datasets: ${allDatasets.length}`)
  console.log(`Already completed: ${checkpoint.completedDatasets.length}`)
  console.log(`Already failed: ${checkpoint.failedDatasets.length}`)

  // Filter out completed datasets
  const pendingDatasets = allDatasets.filter(
    d => !checkpoint.completedDatasets.includes(d)
  )

  // Apply start index and limit
  let toProcess = pendingDatasets.slice(config.startIndex)
  if (config.limit > 0) {
    toProcess = toProcess.slice(0, config.limit)
  }

  console.log(`Datasets to process: ${toProcess.length}`)
  console.log('='.repeat(70))

  if (toProcess.length === 0) {
    console.log('No datasets to process!')
    return
  }

  // Output file for trajectories
  const outputFile = join(config.outputDir, 'trajectories.jsonl')
  const statsFile = join(config.outputDir, 'collection_stats.json')

  // Distribute datasets across workers (round-robin)
  const workerQueues: string[][] = API_KEYS.map(() => [])
  toProcess.forEach((dataset, i) => {
    workerQueues[i % API_KEYS.length].push(dataset)
  })

  console.log('\nWorker distribution:')
  workerQueues.forEach((queue, i) => {
    if (queue.length > 0) {
      console.log(`  Worker ${i}: ${queue.length} datasets`)
    }
  })
  console.log('')

  // Start all workers in parallel
  const startTime = Date.now()
  const workerPromises = API_KEYS.map((apiKey, i) => {
    if (workerQueues[i].length === 0) {
      return Promise.resolve({ success: 0, failed: 0 })
    }
    return worker(i, apiKey, workerQueues[i], outputFile, allDatasets.length)
  })

  // Wait for all workers to complete
  const results = await Promise.all(workerPromises)
  const totalDuration = ((Date.now() - startTime) / 1000 / 60).toFixed(1)

  // Aggregate results
  const totalSuccess = results.reduce((sum, r) => sum + r.success, 0)
  const totalFailed = results.reduce((sum, r) => sum + r.failed, 0)

  // Final stats
  console.log('\n' + '='.repeat(70))
  console.log('COLLECTION COMPLETE')
  console.log('='.repeat(70))
  console.log(`Duration: ${totalDuration} minutes`)
  console.log(`Total processed: ${totalSuccess + totalFailed}`)
  console.log(`Successful: ${totalSuccess}`)
  console.log(`Failed: ${totalFailed}`)
  if (totalSuccess + totalFailed > 0) {
    console.log(`Success rate: ${((totalSuccess / (totalSuccess + totalFailed)) * 100).toFixed(1)}%`)
  }
  console.log(`\nOutput: ${outputFile}`)
  console.log(`Checkpoint: ${config.checkpointFile}`)

  // Save final stats
  writeFileSync(statsFile, JSON.stringify({
    ...checkpoint,
    outputFile,
    config: {
      model: config.model,
      maxTurns: config.maxTurns,
      totalDatasets: allDatasets.length,
      parallelWorkers: API_KEYS.length,
    },
  }, null, 2))
}

main().catch(console.error)
