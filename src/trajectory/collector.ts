/**
 * TrajectoryCollector - Records agent conversations in CoPaw-Flash format.
 *
 * This module integrates with claude-code-clean to capture the full agent
 * trajectory including:
 * - User messages
 * - Assistant responses (with <think> tags if present)
 * - Tool calls (in <tool_call> XML format)
 * - Tool responses (in <tool_response> XML format)
 *
 * Output format matches CoPaw-Flash-9B training data:
 * {
 *   "messages": [
 *     {"role": "user", "content": "..."},
 *     {"role": "assistant", "content": "<think>...</think>\n\n<tool_call>...</tool_call>"},
 *     {"role": "user", "content": "<tool_response>...</tool_response>"},
 *     {"role": "assistant", "content": "..."}
 *   ]
 * }
 */

import { writeFileSync, mkdirSync, existsSync, appendFileSync } from 'fs'
import { join } from 'path'

export interface CoPawMessage {
  role: 'user' | 'assistant' | 'system'
  content: string
}

export interface CoPawTrajectory {
  conversation_id: string
  messages: CoPawMessage[]
  metadata: {
    timestamp: string
    model: string
    total_input_tokens: number
    total_output_tokens: number
    turns: number
    dataset?: string
  }
}

/**
 * Format a tool call in CoPaw-Flash XML format.
 */
export function formatToolCall(
  toolName: string,
  args: Record<string, unknown>,
): string {
  const paramLines = Object.entries(args)
    .map(([key, value]) => {
      const valueStr =
        typeof value === 'string' ? value : JSON.stringify(value)
      return `<parameter=${key}>\n${valueStr}\n</parameter>`
    })
    .join('\n')

  return `<tool_call>
<function=${toolName}>
${paramLines}
</function>
</tool_call>`
}

/**
 * Format a tool response in CoPaw-Flash XML format.
 */
export function formatToolResponse(content: string): string {
  return `<tool_response>\n${content}\n</tool_response>`
}

/**
 * Format thinking content in CoPaw-Flash XML format.
 */
export function formatThinking(content: string): string {
  return `<think>\n${content}\n</think>`
}

class TrajectoryCollector {
  private outputDir: string
  private conversations: CoPawTrajectory[] = []
  private current: CoPawTrajectory | null = null
  private enabled: boolean = false

  constructor(outputDir: string = './trajectory_output') {
    this.outputDir = outputDir
  }

  /**
   * Enable trajectory collection.
   */
  enable(): void {
    this.enabled = true
    if (!existsSync(this.outputDir)) {
      mkdirSync(this.outputDir, { recursive: true })
    }
  }

  /**
   * Disable trajectory collection.
   */
  disable(): void {
    this.enabled = false
  }

  /**
   * Check if collection is enabled.
   */
  isEnabled(): boolean {
    return this.enabled
  }

  /**
   * Start a new conversation trajectory.
   */
  startConversation(metadata: Partial<CoPawTrajectory['metadata']> = {}): string {
    if (!this.enabled) return ''

    const convId = `conv_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`

    this.current = {
      conversation_id: convId,
      messages: [],
      metadata: {
        timestamp: new Date().toISOString(),
        model: '',
        total_input_tokens: 0,
        total_output_tokens: 0,
        turns: 0,
        ...metadata,
      },
    }

    return convId
  }

  /**
   * Add a system message to the current trajectory.
   * This should be called first to record the system prompt.
   */
  addSystemMessage(content: string): void {
    if (!this.enabled || !this.current) return

    this.current.messages.push({
      role: 'system',
      content,
    })
  }

  /**
   * Add a user message to the current trajectory.
   */
  addUserMessage(content: string): void {
    if (!this.enabled || !this.current) return

    this.current.messages.push({
      role: 'user',
      content,
    })
  }

  /**
   * Add an assistant message to the current trajectory.
   * The content may include <think> and <tool_call> tags.
   */
  addAssistantMessage(content: string): void {
    if (!this.enabled || !this.current) return

    this.current.messages.push({
      role: 'assistant',
      content,
    })
    this.current.metadata.turns++
  }

  /**
   * Add a tool response as a user message with <tool_response> wrapper.
   */
  addToolResponse(content: string): void {
    if (!this.enabled || !this.current) return

    this.current.messages.push({
      role: 'user',
      content: formatToolResponse(content),
    })
  }

  /**
   * Update token counts.
   */
  updateTokens(inputTokens: number, outputTokens: number): void {
    if (!this.enabled || !this.current) return

    this.current.metadata.total_input_tokens += inputTokens
    this.current.metadata.total_output_tokens += outputTokens
  }

  /**
   * Set model name.
   */
  setModel(model: string): void {
    if (!this.enabled || !this.current) return
    this.current.metadata.model = model
  }

  /**
   * Get current conversation messages (for context).
   */
  getCurrentMessages(): CoPawMessage[] {
    return this.current?.messages || []
  }

  /**
   * End current conversation and save to collection.
   */
  endConversation(): CoPawTrajectory | null {
    if (!this.enabled || !this.current) return null

    const conv = this.current
    this.conversations.push(conv)
    this.current = null

    return conv
  }

  /**
   * Export all trajectories to JSONL file (CoPaw-Flash training format).
   * Only exports the messages field for each trajectory.
   */
  exportJSONL(filename?: string): string {
    const ts = new Date().toISOString().replace(/[:.]/g, '-')
    const fname = filename || `trajectories_${ts}.jsonl`
    const filepath = join(this.outputDir, fname)

    const lines = this.conversations.map((conv) => {
      // Only export messages for training
      return JSON.stringify({ messages: conv.messages })
    })

    writeFileSync(filepath, lines.join('\n') + '\n', 'utf-8')
    return filepath
  }

  /**
   * Export all trajectories with full metadata to JSON file.
   */
  exportFullJSON(filename?: string): string {
    const ts = new Date().toISOString().replace(/[:.]/g, '-')
    const fname = filename || `trajectories_full_${ts}.json`
    const filepath = join(this.outputDir, fname)

    writeFileSync(filepath, JSON.stringify(this.conversations, null, 2), 'utf-8')
    return filepath
  }

  /**
   * Append current trajectory to a JSONL file immediately (streaming mode).
   */
  appendToJSONL(filepath: string): void {
    if (!this.current) return

    const line = JSON.stringify({ messages: this.current.messages })
    appendFileSync(filepath, line + '\n', 'utf-8')
  }

  /**
   * Get statistics about collected data.
   */
  getStats(): {
    conversations: number
    totalMessages: number
    totalToolCalls: number
    totalThinkBlocks: number
    avgMessagesPerConv: number
  } {
    let totalMessages = 0
    let totalToolCalls = 0
    let totalThinkBlocks = 0

    for (const conv of this.conversations) {
      totalMessages += conv.messages.length
      for (const msg of conv.messages) {
        if (msg.content.includes('<tool_call>')) {
          totalToolCalls += (msg.content.match(/<tool_call>/g) || []).length
        }
        if (msg.content.includes('<think>')) {
          totalThinkBlocks++
        }
      }
    }

    return {
      conversations: this.conversations.length,
      totalMessages,
      totalToolCalls,
      totalThinkBlocks,
      avgMessagesPerConv:
        this.conversations.length > 0
          ? totalMessages / this.conversations.length
          : 0,
    }
  }

  /**
   * Clear all collected data.
   */
  clear(): void {
    this.conversations = []
    this.current = null
  }
}

// Global singleton instance
export const trajectoryCollector = new TrajectoryCollector()

export default TrajectoryCollector
