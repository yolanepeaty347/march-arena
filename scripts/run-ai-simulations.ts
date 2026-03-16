/**
 * Run N full AI bracket simulations locally (no workflow, no API).
 *
 * Calls Claude Haiku directly for each game pick and saves completed brackets
 * to Redis. Each invocation runs brackets serially; launch multiple processes
 * in separate terminals for parallelism.
 *
 * Prerequisites:
 *   1. ANTHROPIC_API_KEY in .env.local.
 *   2. REDIS_URL in .env.local (for leaderboard persistence).
 *
 * Usage:
 *   pnpm ai-sim-batch --count 10
 *   pnpm ai-sim-batch 50
 */

import { readFileSync } from "fs";
import { resolve } from "path";

const envPath = resolve(__dirname, "../.env.local");
try {
  const envFile = readFileSync(envPath, "utf-8");
  for (const line of envFile.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const eqIdx = trimmed.indexOf("=");
    if (eqIdx === -1) continue;
    const key = trimmed.slice(0, eqIdx).trim();
    let value = trimmed.slice(eqIdx + 1).trim();
    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }
    if (!process.env[key]) process.env[key] = value;
  }
} catch {
  /* no .env.local */
}

import { BRACKET_2026 } from "../lib/bracket-data";
import { simulateBracketLocally } from "../lib/ai-pick";
import { saveSimulationResults } from "../lib/leaderboard";

function parseArgs() {
  const argv = process.argv.slice(2);
  let count = 10;
  let continueOnError = false;

  const positionalNums: number[] = [];

  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--continue-on-error") {
      continueOnError = true;
    } else if (a === "--count" && argv[i + 1]) {
      count = Math.max(1, parseInt(argv[++i], 10) || count);
    } else if (/^\d+$/.test(a)) {
      positionalNums.push(parseInt(a, 10));
    }
  }
  if (positionalNums[0] != null) count = Math.max(1, positionalNums[0]);

  return { count, continueOnError };
}

async function main() {
  const { count, continueOnError } = parseArgs();

  console.log(`AI simulation batch (local, no workflow)`);
  console.log(`  Total runs:  ${count}`);
  console.log(`  Model:       anthropic/claude-4-5-haiku-latest`);
  console.log(`  Redis:       ${process.env.REDIS_URL ? "configured" : "NOT SET — results won't persist"}`);
  console.log(`\nEach run simulates a full 67-game bracket via Claude Haiku.\n`);

  let completed = 0;
  let failed = 0;
  const t0 = Date.now();

  for (let run = 1; run <= count; run++) {
    const runStart = Date.now();
    console.log(`\n${"#".repeat(80)}`);
    console.log(`# RUN ${run}/${count}`);
    console.log(`${"#".repeat(80)}\n`);

    try {
      const result = await simulateBracketLocally(BRACKET_2026);

      console.log(`\n>>> Champion: ${result.winner?.name} (#${result.winner?.seed})`);

      await saveSimulationResults(result);
      completed++;

      const runMins = ((Date.now() - runStart) / 1000 / 60).toFixed(1);
      const totalMins = ((Date.now() - t0) / 1000 / 60).toFixed(1);
      console.log(
        `  Saved to leaderboard. Run took ${runMins} min. Progress: ${completed}/${count} done, ${failed} failed, ${totalMins} min elapsed.`
      );
    } catch (err) {
      failed++;
      console.error(`\n  Run ${run} failed:`, err instanceof Error ? err.message : err);
      if (!continueOnError) throw err;
    }
  }

  const mins = ((Date.now() - t0) / 1000 / 60).toFixed(1);
  console.log(`\n\nDone in ${mins} min. Completed: ${completed}, failed: ${failed}`);
  process.exit(failed > 0 && !continueOnError ? 1 : 0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
