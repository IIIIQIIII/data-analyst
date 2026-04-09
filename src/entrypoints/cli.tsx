/**
 * data-analyst CLI entry point.
 * Simplified from Claude Code — strips all feature flag fast paths,
 * keeps only the core TUI launch flow.
 */

// Force OpenAI-compatible mode for local model support
// eslint-disable-next-line custom-rules/no-top-level-side-effects
process.env.CLAUDE_CODE_USE_OPENAI ??= '1'

async function main(): Promise<void> {
  const args = process.argv.slice(2);

  // Fast-path for --version/-v
  if (args.length === 1 && (args[0] === '--version' || args[0] === '-v' || args[0] === '-V')) {
    // biome-ignore lint/suspicious/noConsole:: intentional console output
    console.log('0.1.0 (data-analyst)');
    return;
  }

  // Fast-path for --dump-system-prompt: output the rendered system prompt and exit.
  if (args[0] === '--dump-system-prompt') {
    const { getSystemPrompt } = await import('../constants/prompts.js');
    const prompt = await getSystemPrompt([], '');
    // biome-ignore lint/suspicious/noConsole:: intentional console output
    console.log(prompt.join('\n'));
    return;
  }

  // Load and run the full CLI
  const { startCapturingEarlyInput } = await import('../utils/earlyInput.js');
  startCapturingEarlyInput();
  const { main: cliMain } = await import('../main.js');
  await cliMain();
}

// eslint-disable-next-line custom-rules/no-top-level-side-effects
void main();
