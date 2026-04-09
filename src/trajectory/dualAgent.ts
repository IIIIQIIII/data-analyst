/**
 * DualAgentOrchestrator - Coordinates two AI agents for trajectory generation.
 *
 * Architecture:
 * 1. User Agent - Simulates a non-technical user asking questions (no tools)
 * 2. Analyst Agent - Data analyst with full tool access (the actual claude-code-clean agent)
 *
 * The User Agent's questions drive the Analyst Agent to perform data analysis,
 * and the full trajectory is recorded in CoPaw-Flash format.
 */

import { trajectoryCollector, formatToolCall, formatThinking } from './collector.js'

export interface DualAgentConfig {
  apiKey: string
  baseUrl: string
  model: string
  maxTurns: number
  datasetDir: string
  datasetTitle: string
  datasetDesc: string
}

export interface UserAgentPrompt {
  systemPrompt: string
}

/**
 * Generate User Agent system prompt for simulating a non-technical user.
 */
export function getUserAgentPrompt(
  datasetTitle: string,
  datasetDesc: string,
): UserAgentPrompt {
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

/**
 * Generate the initial question using the User Agent AI.
 */
export async function generateInitialQuestion(
  config: DualAgentConfig,
  systemPrompt: string,
): Promise<string> {
  const response = await fetch(`${config.baseUrl}/chat/completions`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${config.apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: config.model,
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: 'Start the conversation by asking the data analyst your first question about this dataset. Be natural and specific to this dataset.' },
      ],
      stream: false,
    }),
  })

  if (!response.ok) {
    const error = await response.text()
    throw new Error(`User Agent initial question error: ${response.status} - ${error}`)
  }

  const data = await response.json() as {
    choices: Array<{ message: { content: string } }>
  }
  return extractUserQuestion(data.choices[0]?.message?.content || '')
}

/**
 * Generate Analyst Agent system prompt (data analysis focused).
 * This extends the base claude-code-clean system prompt with data analysis context.
 */
export function getAnalystSystemPromptExtension(
  datasetDir: string,
  workspaceDir: string,
): string {
  return `
<data-analysis-context>
You are helping a non-technical user analyze a dataset.

Working context:
- Dataset directory: ${datasetDir}
- Workspace for outputs: ${workspaceDir}
- The user is a beginner - explain everything in simple terms
- Always show your work: run code, show outputs, then explain

Python environment:
- ALWAYS use \`python\` or \`python3\` to execute Python code
- Available libraries: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn
- When creating plots, ALWAYS use plt.savefig() to save to workspace
- NEVER use plt.show() - save plots instead

Your approach:
1. First explore the data structure and content
2. Provide clear, jargon-free explanations
3. Create visualizations when helpful
4. Focus on actionable insights the user can share with their manager
</data-analysis-context>
`
}

/**
 * Parse assistant response to extract thinking and tool calls.
 * This is used to format the response in CoPaw-Flash format.
 */
export function parseAssistantResponse(response: {
  content: Array<{ type: string; text?: string; thinking?: string; name?: string; input?: unknown }>
}): string {
  const parts: string[] = []

  for (const block of response.content) {
    if (block.type === 'thinking' && block.thinking) {
      parts.push(formatThinking(block.thinking))
    } else if (block.type === 'text' && block.text) {
      parts.push(block.text)
    } else if (block.type === 'tool_use' && block.name) {
      parts.push(formatToolCall(block.name, (block.input || {}) as Record<string, unknown>))
    }
  }

  return parts.join('\n\n')
}

/**
 * Simple LLM call for the User Agent (no tools, just generates questions).
 */
export async function callUserAgent(
  config: DualAgentConfig,
  systemPrompt: string,
  conversationHistory: Array<{ role: string; content: string }>,
): Promise<string> {
  const response = await fetch(`${config.baseUrl}/chat/completions`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${config.apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: config.model,
      messages: [
        { role: 'system', content: systemPrompt },
        ...conversationHistory,
      ],
      stream: false,
    }),
  })

  if (!response.ok) {
    const error = await response.text()
    throw new Error(`User Agent API error: ${response.status} - ${error}`)
  }

  const data = await response.json() as {
    choices: Array<{ message: { content: string } }>
  }
  return data.choices[0]?.message?.content || ''
}

/**
 * Extract plain text question from User Agent response.
 */
export function extractUserQuestion(response: string): string {
  // Remove any XML tags
  let text = response.replace(/<[^>]+>/g, '').trim()
  // Remove markdown code blocks
  text = text.replace(/```\w*\n?/g, '').trim()
  return text
}

export default {
  getUserAgentPrompt,
  generateInitialQuestion,
  getAnalystSystemPromptExtension,
  parseAssistantResponse,
  callUserAgent,
  extractUserQuestion,
}
