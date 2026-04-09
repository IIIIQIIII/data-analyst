#!/usr/bin/env bun
/**
 * Batch Trajectory Collection for Local Models
 *
 * This entrypoint is designed for locally-served models via vLLM or similar.
 * It runs autonomous data analysis tasks without intermediate user guidance.
 *
 * Environment variables:
 *   OPENAI_BASE_URL  - Local API endpoint (default: http://localhost:8000/v1)
 *   OPENAI_MODEL     - Model name/path (default: agent-lora)
 *   OPENAI_API_KEY   - Optional API key
 *
 * Usage:
 *   bun run src/entrypoints/batchCollectLocal.tsx \
 *     --kaggle-dir ./kaggle-datasets \
 *     --output-dir ./local-trajectories \
 *     --max-tool-iterations 30 \
 *     --limit 10
 */

import { parseArgs } from 'util'
import {
  existsSync,
  mkdirSync,
  readFileSync,
  writeFileSync,
  appendFileSync,
  readdirSync,
  statSync,
} from 'fs'
import { join, resolve, dirname } from 'path'
import { execSync } from 'child_process'
import { globSync } from 'glob'
import {
  parseToolCalls,
  hasToolCalls,
  formatToolResponse,
  callLocalLLM,
  getLocalApiConfig,
  getAutonomousSystemPrompt,
  type LocalToolCall,
} from '../trajectory/localModelAdapter.js'

const { values: args } = parseArgs({
  options: {
    'kaggle-dir': { type: 'string', default: '../claude-code-clean/datasets/kaggle-datasets-1001-1100' },
    'output-dir': { type: 'string', default: './local-trajectories' },
    'max-tool-iterations': { type: 'string', default: '30' },
    'checkpoint': { type: 'string', default: './checkpoint-local.json' },
    'limit': { type: 'string', default: '0' },
  },
})

const apiConfig = getLocalApiConfig()

const config = {
  ...apiConfig,
  kaggleDir: resolve(args['kaggle-dir']!),
  outputDir: resolve(args['output-dir']!),
  checkpointFile: resolve(args['checkpoint']!),
  maxToolIterations: parseInt(args['max-tool-iterations']!, 10),
  limit: parseInt(args['limit']!, 10),
}

const PYTHON_PATH = resolve(__dirname, '../../.venv-analysis/bin/python')

interface Checkpoint {
  completedDatasets: string[]
  failedDatasets: { slug: string; error: string }[]
  totalProcessed: number
  totalSuccess: number
  totalFailed: number
  lastUpdated: string
}

let checkpoint: Checkpoint = {
  completedDatasets: [],
  failedDatasets: [],
  totalProcessed: 0,
  totalSuccess: 0,
  totalFailed: 0,
  lastUpdated: new Date().toISOString(),
}

function loadCheckpoint(): void {
  if (existsSync(config.checkpointFile)) {
    try {
      checkpoint = JSON.parse(readFileSync(config.checkpointFile, 'utf-8'))
    } catch (e) {
      console.log('Starting fresh')
    }
  }
}

function saveCheckpoint(): void {
  checkpoint.lastUpdated = new Date().toISOString()
  writeFileSync(config.checkpointFile, JSON.stringify(checkpoint, null, 2))
}

function executeToolCall(
  toolCall: LocalToolCall,
  datasetDir: string,
  workspaceDir: string
): string {
  let result = ''

  switch (toolCall.name) {
    case 'Bash': {
      const command = toolCall.arguments.command || toolCall.arguments.param || toolCall.arguments.cmd
      if (!command) {
        result = 'Error: Missing command parameter. Use <parameter=command>...</parameter>'
        break
      }
      try {
        result = execSync(command, {
          cwd: datasetDir,
          encoding: 'utf-8',
          timeout: 120000,
          maxBuffer: 10 * 1024 * 1024,
        })
      } catch (e: any) {
        result = `Error: ${e.message}\n${e.stdout || ''}${e.stderr || ''}`
      }
      break
    }

    case 'Read': {
      const filePath = toolCall.arguments.file_path || toolCall.arguments.path || toolCall.arguments.filepath || ''
      if (!filePath) {
        result = 'Error: Missing file_path parameter. Use <parameter=file_path>...</parameter>'
        break
      }
      const fullPath = filePath.startsWith('/') ? filePath : join(datasetDir, filePath)
      try {
        const content = readFileSync(fullPath, 'utf-8')
        const lines = content.split('\n')
        result = lines.slice(0, 100).map((line, i) => `${i + 1}\t${line}`).join('\n')
        if (lines.length > 100) result += `\n... (${lines.length - 100} more lines)`
      } catch (e: any) {
        result = `Error: ${e.message}`
      }
      break
    }

    case 'Write': {
      const filePath = toolCall.arguments.file_path || toolCall.arguments.path || toolCall.arguments.filepath || ''
      if (!filePath) {
        result = 'Error: Missing file_path parameter. Use <parameter=file_path>...</parameter>'
        break
      }
      const content = toolCall.arguments.content || ''
      if (!content) {
        result = 'Error: Missing content parameter. Use <parameter=content>...</parameter>'
        break
      }
      const fullPath = filePath.startsWith('/') ? filePath : join(workspaceDir, filePath)
      try {
        mkdirSync(dirname(fullPath), { recursive: true })
        writeFileSync(fullPath, content, 'utf-8')
        result = `Successfully wrote to ${filePath}`
      } catch (e: any) {
        result = `Error: ${e.message}`
      }
      break
    }

    case 'Glob': {
      const pattern = toolCall.arguments.pattern
      if (!pattern) {
        result = 'Error: Missing pattern parameter. Use <parameter=pattern>...</parameter>'
        break
      }
      try {
        const searchPath = pattern.startsWith('/') ? pattern : join(datasetDir, pattern)
        const matches = globSync(searchPath)
        result = matches.slice(0, 50).join('\n') || 'No files found'
      } catch (e: any) {
        result = `Error: ${e.message}`
      }
      break
    }

    case 'Grep': {
      try {
        const pattern = toolCall.arguments.pattern
        const searchPath = toolCall.arguments.path || datasetDir
        const fullPath = searchPath.startsWith('/') ? searchPath : join(datasetDir, searchPath)
        result = execSync(`grep -rn "${pattern}" "${fullPath}" 2>/dev/null | head -50`, {
          encoding: 'utf-8',
          timeout: 30000,
        })
        if (!result.trim()) result = 'No matches found'
      } catch (e: any) {
        result = e.stdout || 'No matches found'
      }
      break
    }

    case 'Edit': {
      try {
        const filePath = toolCall.arguments.file_path || ''
        const fullPath = filePath.startsWith('/') ? filePath : join(workspaceDir, filePath)
        const content = readFileSync(fullPath, 'utf-8')
        const oldStr = toolCall.arguments.old_string
        const newStr = toolCall.arguments.new_string
        if (!content.includes(oldStr)) {
          result = 'Error: old_string not found'
        } else {
          writeFileSync(fullPath, content.replace(oldStr, newStr), 'utf-8')
          result = `Successfully edited ${filePath}`
        }
      } catch (e: any) {
        result = `Error: ${e.message}`
      }
      break
    }

    default:
      result = `Unknown tool: ${toolCall.name}`
  }

  return result
}

async function collectTrajectory(
  datasetSlug: string,
  datasetDir: string
): Promise<{ success: boolean; trajectory?: object; error?: string }> {
  const workspaceDir = join(datasetDir, 'workspace')
  if (!existsSync(workspaceDir)) {
    mkdirSync(workspaceDir, { recursive: true })
  }

  const systemPrompt = getAutonomousSystemPrompt(datasetDir, workspaceDir, PYTHON_PATH)
  const initialTask = `Analyze the dataset in ${datasetDir}. Find the data files, explore them, identify patterns, create visualizations, and provide insights. Work autonomously.`

  const messages: Array<{ role: 'user' | 'assistant' | 'system'; content: string }> = [
    { role: 'system', content: systemPrompt },
    { role: 'user', content: initialTask },
  ]

  let totalInputTokens = 0
  let totalOutputTokens = 0
  let toolIterations = 0

  try {
    while (toolIterations < config.maxToolIterations) {
      const apiMessages = messages.map(m => ({ role: m.role, content: m.content }))
      const response = await callLocalLLM(apiConfig, apiMessages)
      totalInputTokens += response.inputTokens
      totalOutputTokens += response.outputTokens

      const agentContent = response.content

      if (!hasToolCalls(agentContent)) {
        messages.push({ role: 'assistant', content: agentContent })
        break
      }

      messages.push({ role: 'assistant', content: agentContent })

      const toolCalls = parseToolCalls(agentContent)
      const results: Array<{ tool: string; result: string }> = []

      for (const toolCall of toolCalls) {
        const result = executeToolCall(toolCall, datasetDir, workspaceDir)
        results.push({ tool: toolCall.name, result: result.slice(0, 3000) })
      }

      const toolResponse = formatToolResponse(results)
      messages.push({ role: 'user', content: toolResponse })

      toolIterations++
    }

    const trajectory = {
      messages,
      metadata: {
        timestamp: new Date().toISOString(),
        model: config.model,
        dataset_slug: datasetSlug,
        total_input_tokens: totalInputTokens,
        total_output_tokens: totalOutputTokens,
        tool_iterations: toolIterations,
        mode: 'autonomous-local',
      },
    }

    return { success: true, trajectory }
  } catch (e) {
    return { success: false, error: String(e) }
  }
}

async function main(): Promise<void> {
  console.log('='.repeat(70))
  console.log('LOCAL MODEL TRAJECTORY COLLECTION')
  console.log('='.repeat(70))
  console.log(`API: ${config.baseUrl}`)
  console.log(`Model: ${config.model}`)
  console.log(`Max Tool Iterations: ${config.maxToolIterations}`)
  console.log('='.repeat(70))

  if (!existsSync(config.outputDir)) {
    mkdirSync(config.outputDir, { recursive: true })
  }

  loadCheckpoint()

  const allDatasets = readdirSync(config.kaggleDir)
    .filter(d => {
      const fullPath = join(config.kaggleDir, d)
      return statSync(fullPath).isDirectory() && !d.startsWith('.')
    })
    .sort()

  const pendingDatasets = allDatasets.filter(
    d => !checkpoint.completedDatasets.includes(d)
  )

  let toProcess = pendingDatasets
  if (config.limit > 0) {
    toProcess = toProcess.slice(0, config.limit)
  }

  console.log(`\nTo process: ${toProcess.length} datasets\n`)

  const outputFile = join(config.outputDir, 'trajectories.jsonl')

  for (const datasetSlug of toProcess) {
    const datasetDir = join(config.kaggleDir, datasetSlug)
    const taskStart = Date.now()

    console.log(`[${datasetSlug}] Starting...`)

    const result = await collectTrajectory(datasetSlug, datasetDir)
    const duration = ((Date.now() - taskStart) / 1000).toFixed(1)

    if (result.success && result.trajectory) {
      appendFileSync(outputFile, JSON.stringify(result.trajectory) + '\n')
      checkpoint.completedDatasets.push(datasetSlug)
      checkpoint.totalSuccess++
      checkpoint.totalProcessed++
      saveCheckpoint()

      console.log(`[${datasetSlug}] Success (${duration}s)\n`)
    } else {
      checkpoint.failedDatasets.push({ slug: datasetSlug, error: result.error || 'Unknown' })
      checkpoint.totalFailed++
      checkpoint.totalProcessed++
      saveCheckpoint()

      console.log(`[${datasetSlug}] Failed: ${result.error?.slice(0, 80)}\n`)
    }
  }

  console.log('='.repeat(70))
  console.log(`Complete: ${checkpoint.totalSuccess} success, ${checkpoint.totalFailed} failed`)
  console.log(`Output: ${outputFile}`)
}

main().catch(console.error)
