import { anthropic } from "@ai-sdk/anthropic";
import { generateText, jsonSchema, Output } from "ai";
import {
  formatVenueLine,
  getFirstSecondRoundVenueForGameId,
  type Bracket,
  type Game,
  type RegionName,
  type SimulatedBracket,
  type SimulatedGame,
  type Team,
  type TournamentSchedule,
} from "@/lib/bracket-data";
import { FIRST_FOUR_SLOTS } from "@/lib/simulation-shared";
import { getMatchupKey, SEED_MATCHUP_STATS } from "@/lib/tournament-context";
import { generateMatchupAnalysis } from "@/lib/win-probability";

export const MODEL = anthropic("claude-haiku-4-5");

export const ROUND_NAMES = [
  "Round of 64",
  "Round of 32",
  "Sweet 16",
  "Elite Eight",
];

export const SIM_GAME_CONCURRENCY = Math.max(
  1,
  Math.min(64, Number(process.env.WORKFLOW_SIM_GAME_CONCURRENCY ?? "4"))
);

// ── Types ───────────────────────────────────────────────────────────

export interface TournamentContext {
  upsets: Array<{ winner: Team; loser: Team; round: string }>;
  eliminatedTeams: Team[];
  cinderellaTeams: Team[];
  chalkPicks: number;
  upsetPicks: number;
  gamesPlayed: number;
}

export interface GameContext {
  roundName?: string;
  location?: string;
  regionName?: string;
  tournamentContext?: TournamentContext;
  nextRegionalSite?: string;
}

type GamePick = {
  winner: "1" | "2";
  reasoning: string;
};

const GAME_PICK_SCHEMA = jsonSchema<GamePick>({
  $schema: "http://json-schema.org/draft-07/schema#",
  type: "object",
  properties: {
    winner: { type: "string", enum: ["1", "2"] },
    reasoning: { type: "string" },
  },
  required: ["winner", "reasoning"],
  additionalProperties: false,
});

// ── Team / venue data ───────────────────────────────────────────────

const TEAM_HOME_STATE: Record<string, string> = {
  Michigan: "MI",
  "Michigan St": "MI",
  Houston: "TX",
  Texas: "TX",
  "Texas A&M": "TX",
  "Texas Tech": "TX",
  TCU: "TX",
  Baylor: "TX",
  SMU: "TX",
  Duke: "NC",
  "North Carolina": "NC",
  "NC State": "NC",
  Kansas: "KS",
  "Iowa State": "IA",
  Iowa: "IA",
  Illinois: "IL",
  Northwestern: "IL",
  Purdue: "IN",
  Indiana: "IN",
  Arizona: "AZ",
  UCLA: "CA",
  USC: "CA",
  "Santa Clara": "CA",
  "San Diego St": "CA",
  California: "CA",
  "Saint Mary's": "CA",
  Gonzaga: "WA",
  Florida: "FL",
  "Miami FL": "FL",
  UCF: "FL",
  UConn: "CT",
  Kentucky: "KY",
  Louisville: "KY",
  Alabama: "AL",
  Auburn: "AL",
  Wisconsin: "WI",
  Nebraska: "NE",
  Georgia: "GA",
  Clemson: "SC",
  Virginia: "VA",
  "Virginia Tech": "VA",
  Villanova: "PA",
  Arkansas: "AR",
  Tennessee: "TN",
  Vanderbilt: "TN",
  BYU: "UT",
  "Ole Miss": "MS",
  LSU: "LA",
  "Oklahoma St": "OK",
  "New Mexico": "NM",
  Missouri: "MO",
  "Saint Louis": "MO",
  "Ohio St": "OH",
  Cincinnati: "OH",
  Akron: "OH",
  "Seton Hall": "NJ",
  "St John's": "NY",
  Maryland: "MD",
  McNeese: "LA",
  Troy: "AL",
  Penn: "PA",
  Siena: "NY",
  "Northern Iowa": "IA",
  "Cal Baptist": "CA",
  "South Florida": "FL",
  "North Dakota St": "ND",
  Furman: "SC",
  LIU: "NY",
  Hawaii: "HI",
  "Kennesaw St": "GA",
  Queens: "NC",
  Idaho: "ID",
  Hofstra: "NY",
  "Wright St": "OH",
  "Tennessee St": "TN",
  UMBC: "MD",
  Howard: "DC",
  "Prairie View": "TX",
  Lehigh: "PA",
  "Miami OH": "OH",
  "Mount St Marys": "MD",
  "Robert Morris": "PA",
  "Norfolk St": "VA",
  "Grand Canyon": "AZ",
  Drake: "IA",
  "UNC Wilmington": "NC",
  SIUE: "IL",
  "Saint Francis": "PA",
  American: "DC",
  Omaha: "NE",
  Bryant: "RI",
  Lipscomb: "TN",
  "UC San Diego": "CA",
  "Colorado St": "CO",
  Montana: "MT",
  "Mississippi St": "MS",
  Liberty: "VA",
  Oregon: "OR",
  Xavier: "OH",
  Wofford: "SC",
  Yale: "CT",
  Memphis: "TN",
  Marquette: "WI",
  Creighton: "NE",
  Oklahoma: "OK",
  Tulsa: "OK",
  Nevada: "NV",
  "Boise St": "ID",
  Washington: "WA",
};

const NEARBY_STATES: Record<string, string[]> = {
  TX: ["TX", "LA", "AR", "OK", "NM"],
  CA: ["CA", "AZ", "NV", "OR"],
  IL: ["IL", "IN", "WI", "IA", "MO", "MI", "KY"],
  IN: ["IN", "IL", "OH", "MI", "KY"],
  "D.C.": ["VA", "MD", "PA", "DE", "NJ", "DC"],
  OH: ["OH", "IN", "MI", "PA", "KY", "WV"],
  FL: ["FL", "GA", "AL", "SC", "TN", "NC"],
  NY: ["NY", "PA", "NJ", "CT", "MA", "VT"],
  MO: ["MO", "IL", "IA", "KS", "NE", "AR", "KY", "TN", "OK"],
  PA: ["PA", "NJ", "NY", "DE", "MD", "OH", "WV"],
  OK: ["OK", "TX", "KS", "MO", "AR", "NM", "LA", "NE"],
  SC: ["SC", "NC", "GA", "TN", "FL", "VA"],
  OR: ["OR", "WA", "ID", "CA", "NV", "UT"],
};

function teamCampusState(team: Team): string | undefined {
  return team.location?.state ?? TEAM_HOME_STATE[team.name];
}

// ── Prompt helpers ──────────────────────────────────────────────────

function buildGeographicAdvantage(
  team1: Team,
  team2: Team,
  venueLocation?: string
): string {
  if (!venueLocation) return "";

  const lines: string[] = [];

  const stateMatch = venueLocation.match(/,\s*([A-Za-z.]+)\s*$/);
  const venueState = stateMatch?.[1];

  if (venueState) {
    const team1State = teamCampusState(team1);
    const team2State = teamCampusState(team2);
    const nearbyStates = NEARBY_STATES[venueState] ?? [venueState];

    const team1Near = team1State ? nearbyStates.includes(team1State) : false;
    const team2Near = team2State ? nearbyStates.includes(team2State) : false;

    if (team1Near && !team2Near) {
      lines.push(
        `${team1.name} is likely to have the better crowd/travel edge (campus/home footprint ${team1State} is closer to this venue than ${team2.name}'s ${team2State ?? "region"}) — still a neutral site, use as a tiebreaker only`
      );
    } else if (team2Near && !team1Near) {
      lines.push(
        `${team2.name} is likely to have the better crowd/travel edge (campus/home footprint ${team2State} is closer to this venue than ${team1.name}'s ${team1State ?? "region"}) — still a neutral site, use as a tiebreaker only`
      );
    } else if (team1Near && team2Near) {
      lines.push(
        `Both teams draw from regions relatively close to this venue — expect a mixed crowd; fan-base strength may matter more than mileage`
      );
    }
  }

  const fb1 = team1.fanBaseStrength;
  const fb2 = team2.fanBaseStrength;
  const FAN_STRENGTH_ORDER = ["elite", "strong", "average", "weak"];

  if (fb1 && fb2 && fb1 !== fb2) {
    const rank1 = FAN_STRENGTH_ORDER.indexOf(fb1);
    const rank2 = FAN_STRENGTH_ORDER.indexOf(fb2);
    if (rank1 >= 0 && rank2 >= 0 && Math.abs(rank1 - rank2) > 0) {
      const stronger = rank1 < rank2 ? team1 : team2;
      const strongerFb = rank1 < rank2 ? fb1 : fb2;
      const weakerFb = rank1 < rank2 ? fb2 : fb1;
      lines.push(
        `${stronger.name}'s fan base travels at a ${strongerFb} level vs ${weakerFb} for ${rank1 < rank2 ? team2.name : team1.name} — can swing atmosphere in a toss-up`
      );
    }
  }

  if (lines.length === 0) return "";
  return `SITE & TRAVEL (neutral court — KenPom does not include location):\n${lines.map((l) => `- ${l}`).join("\n")}`;
}

function describeTeam(
  team: Team,
  context?: { isCinderella?: boolean }
): string {
  const parts: string[] = [`${team.name} (#${team.seed} seed)`];
  if (team.conference) parts.push(team.conference);
  if (team.teamTier === "blueblood") {
    parts.push("blueblood program");
  } else if (team.teamTier === "mid-major") {
    parts.push("mid-major");
  } else if (team.teamTier === "power") {
    parts.push("power conference");
  }
  if (team.stats) {
    parts.push(`KenPom #${team.stats.kenpomRank}`);
    parts.push(`${team.stats.record.wins}-${team.stats.record.losses}`);
  }
  const st = teamCampusState(team);
  if (st) parts.push(`home state ${st}`);
  if (context?.isCinderella) {
    parts.push("advanced via upset");
  }

  return parts.join(" | ");
}

export function getExpectedUpsets(gamesPlayed: number): number {
  if (gamesPlayed <= 4) return 0;
  if (gamesPlayed <= 36) return 10;
  if (gamesPlayed <= 52) return 14;
  if (gamesPlayed <= 60) return 16;
  if (gamesPlayed <= 64) return 17;
  return 17.5;
}

export function buildEnhancedPrompt(game: Game, context: GameContext): string {
  const matchupKey = getMatchupKey(game.team1.seed, game.team2.seed);
  const matchupStats = SEED_MATCHUP_STATS[matchupKey];
  const tourneyCtx = context.tournamentContext;
  const team1IsCinderella =
    tourneyCtx?.cinderellaTeams.some((t) => t.id === game.team1.id) ?? false;
  const team2IsCinderella =
    tourneyCtx?.cinderellaTeams.some((t) => t.id === game.team2.id) ?? false;
  const higherSeed =
    game.team1.seed <= game.team2.seed ? game.team1 : game.team2;
  const lowerSeed =
    game.team1.seed > game.team2.seed ? game.team1 : game.team2;

  let prompt = `You are an elite March Madness analyst picking NCAA Tournament games (2026 NCAA Division I field). `;
  prompt += `March Madness is DEFINED by upsets — a typical tournament produces 12-18 upsets across all rounds. `;
  prompt += `Do NOT default to always picking the higher seed or higher-ranked KenPom team. `;
  prompt += `In single-elimination, defensive efficiency, tempo control, tournament experience, and matchup-specific edges often matter more than raw efficiency margin.\n\n`;
  prompt += `Every game is at a neutral site — there is no true home court. Crowd edges can favor teams whose fans travel shorter distances or show in force; `;
  prompt += `use this only to break ties or nudge close calls.\n\n`;

  prompt += `=== ${context.roundName || "GAME"} ===`;
  if (context.regionName) {
    prompt += ` | ${context.regionName} Region`;
  }
  prompt += `\n\n`;

  if (context.location) {
    prompt += `GAME SITE:\n- ${context.location}\n`;
    if (context.nextRegionalSite) {
      prompt += `- Next stop for the winner (regional rounds): ${context.nextRegionalSite}\n`;
    }
    prompt += `\n`;
  }

  prompt += `MATCHUP:\n`;
  prompt += `Team 1: ${describeTeam(game.team1, { isCinderella: team1IsCinderella })}\n`;
  prompt += `Team 2: ${describeTeam(game.team2, { isCinderella: team2IsCinderella })}\n\n`;

  const matchupAnalysis = generateMatchupAnalysis(game.team1, game.team2);
  if (matchupAnalysis) {
    prompt += matchupAnalysis + `\n\n`;
  }

  const factors: string[] = [];
  const stats1 = game.team1.stats;
  const stats2 = game.team2.stats;

  if (higherSeed.teamTier === "blueblood") {
    factors.push(
      `${higherSeed.name} is a blueblood with deep tournament pedigree`
    );
  }
  if (
    lowerSeed.teamTier === "blueblood" &&
    higherSeed.teamTier !== "blueblood"
  ) {
    factors.push(
      `UPSET INDICATOR: ${lowerSeed.name} is a blueblood despite their lower seed -- they have more tournament DNA`
    );
  }

  if (stats1 && stats2) {
    const higherSeedStats =
      game.team1.seed <= game.team2.seed ? stats1 : stats2;
    const lowerSeedStats =
      game.team1.seed > game.team2.seed ? stats1 : stats2;

    // AdjEM gap — the single most predictive metric for game outcomes
    const emGap = higherSeedStats.adjEM - lowerSeedStats.adjEM;
    if (emGap >= 8) {
      factors.push(
        `CHALK INDICATOR: ${higherSeed.name} has a commanding AdjEM advantage (+${emGap.toFixed(1)} gap) — this is a dominant efficiency edge that wins ~75%+ of the time`
      );
    } else if (emGap >= 4) {
      factors.push(
        `${higherSeed.name} has a meaningful AdjEM edge (+${emGap.toFixed(1)} gap) — roughly a 65/35 game in their favor`
      );
    }

    if (lowerSeedStats.kenpomRank < higherSeedStats.kenpomRank) {
      factors.push(
        `UPSET INDICATOR: ${lowerSeed.name} (KenPom #${lowerSeedStats.kenpomRank}) is ranked HIGHER in KenPom than ${higherSeed.name} (KenPom #${higherSeedStats.kenpomRank}) -- the underdog may actually be the better team`
      );
    }

    // Luck penalties — apply to BOTH teams
    if (higherSeedStats.luck > 0.04) {
      factors.push(
        `UPSET INDICATOR: ${higherSeed.name} has been lucky (${higherSeedStats.luck > 0 ? "+" : ""}${higherSeedStats.luck.toFixed(3)}) -- their record may overstate their quality`
      );
    }
    if (lowerSeedStats.luck > 0.04) {
      factors.push(
        `CHALK INDICATOR: ${lowerSeed.name} has also been lucky (${lowerSeedStats.luck > 0 ? "+" : ""}${lowerSeedStats.luck.toFixed(3)}) -- their quality may not be as strong as their record suggests, making the upset less likely`
      );
    }
    // Unlucky teams are undervalued
    if (higherSeedStats.luck < -0.04) {
      factors.push(
        `CHALK INDICATOR: ${higherSeed.name} has been unlucky (${higherSeedStats.luck.toFixed(3)}) -- they are likely BETTER than their record/seed indicates`
      );
    }
    if (lowerSeedStats.luck < -0.04) {
      factors.push(
        `UPSET INDICATOR: ${lowerSeed.name} has been unlucky (${lowerSeedStats.luck.toFixed(3)}) -- they are likely better than their record/seed indicates`
      );
    }

    if (lowerSeedStats.sosEM > higherSeedStats.sosEM + 3) {
      factors.push(
        `UPSET INDICATOR: ${lowerSeed.name} played a significantly tougher schedule (SOS ${lowerSeedStats.sosEM > 0 ? "+" : ""}${lowerSeedStats.sosEM.toFixed(1)} vs ${higherSeedStats.sosEM > 0 ? "+" : ""}${higherSeedStats.sosEM.toFixed(1)})`
      );
    }
    if (higherSeedStats.sosEM > lowerSeedStats.sosEM + 3) {
      factors.push(
        `CHALK INDICATOR: ${higherSeed.name} played a significantly tougher schedule (SOS ${higherSeedStats.sosEM > 0 ? "+" : ""}${higherSeedStats.sosEM.toFixed(1)} vs ${lowerSeedStats.sosEM > 0 ? "+" : ""}${lowerSeedStats.sosEM.toFixed(1)})`
      );
    }
    if (lowerSeedStats.adjDRank <= 15 && higherSeedStats.adjDRank > 25) {
      factors.push(
        `UPSET INDICATOR: ${lowerSeed.name} has an elite defense (#${lowerSeedStats.adjDRank}) that can grind out a tournament win against ${higherSeed.name}'s weaker defense (#${higherSeedStats.adjDRank})`
      );
    }
    if (lowerSeed.teamTier === "mid-major" && lowerSeedStats.kenpomRank <= 50) {
      factors.push(
        `UPSET INDICATOR: ${lowerSeed.name} is a mid-major with a top-50 KenPom profile -- likely underseeded by the committee`
      );
    }
  }

  if (factors.length > 0) {
    prompt += `KEY MATCHUP FACTORS:\n`;
    factors.forEach((f) => (prompt += `- ${f}\n`));
    prompt += `\n`;
  }

  if (matchupStats) {
    const higherSeedWinRate = Math.round(
      (1 - matchupStats.upsetRate) * 100
    );
    const upsetPct = Math.round(matchupStats.upsetRate * 100);
    prompt += `HISTORICAL SEED DATA (${matchupKey.toUpperCase()}):\n`;
    prompt += `- Higher seed wins ${higherSeedWinRate}% / Lower seed wins ${upsetPct}%\n`;
    prompt += `- ${matchupStats.note}\n`;
    if (upsetPct >= 20) {
      prompt += `- ${matchupStats.upsetInstruction}\n`;
    }
    prompt += `\n`;
  } else {
    const seedDiff = Math.abs(game.team1.seed - game.team2.seed);
    prompt += `LATER ROUND CONTEXT:\n`;
    prompt += `- Both teams have proven themselves to reach this point.\n`;
    if (seedDiff <= 3) {
      prompt += `- Seeds are close (${seedDiff} apart) - this should be treated as a toss-up.\n`;
    } else {
      prompt += `- Lower seeds that advance this far are battle-tested and dangerous.\n`;
    }
    prompt += `\n`;
  }

  const geoAdvantage = buildGeographicAdvantage(
    game.team1,
    game.team2,
    context.location
  );
  if (geoAdvantage) {
    prompt += geoAdvantage + `\n\n`;
  } else if (
    context.location &&
    (context.roundName === "Sweet 16" || context.roundName === "Elite Eight")
  ) {
    prompt += `SITE & TRAVEL: Regional rounds draw fans from a wider radius; bluebloods and nearby campuses can still tilt the building slightly on a neutral floor.\n\n`;
  } else if (
    context.roundName === "Final Four" ||
    context.roundName === "National Championship"
  ) {
    prompt += `SITE & TRAVEL: Final Four and title game are at Lucas Oil Stadium, Indianapolis — expect massive neutral crowds; Big Ten / Midwest alumni bases (and national brands) may be slightly over-represented.\n\n`;
  }

  if (tourneyCtx) {
    const expected = getExpectedUpsets(tourneyCtx.gamesPlayed);
    const actual = tourneyCtx.upsetPicks;
    const totalGames = tourneyCtx.gamesPlayed;

    prompt += `TOURNAMENT SO FAR:\n`;
    prompt += `- Games played: ${totalGames} | Upsets: ${actual} | Chalk: ${tourneyCtx.chalkPicks}\n`;

    if (totalGames > 4) {
      prompt += `- Expected upsets at this point: ~${Math.round(expected)}\n`;
      if (actual < expected - 2) {
        prompt += `- WARNING: This tournament is running WELL BELOW the historical upset rate. Upsets are overdue.\n`;
      } else if (actual < expected) {
        prompt += `- This tournament is slightly below the historical upset rate.\n`;
      } else {
        prompt += `- This tournament is tracking near historical upset rates.\n`;
      }
    }

    if (tourneyCtx.upsets.length > 0) {
      const recentUpsets = tourneyCtx.upsets.slice(-3);
      prompt += `- Recent upsets: ${recentUpsets.map((u) => `${u.winner.name} (#${u.winner.seed}) over ${u.loser.name} (#${u.loser.seed})`).join(", ")}\n`;
    }

    if (tourneyCtx.cinderellaTeams.length > 0) {
      const relevantCinderellas = tourneyCtx.cinderellaTeams.filter(
        (t) => t.id === game.team1.id || t.id === game.team2.id
      );
      if (relevantCinderellas.length > 0) {
        prompt += `- ${relevantCinderellas[0].name} is on a Cinderella run!\n`;
      }
    }
    prompt += `\n`;
  }

  prompt += `YOUR PICK:\n`;
  prompt += `Analyze this like a March Madness expert filling out a bracket to WIN A POOL (picking all chalk never wins). Weigh these factors IN ORDER OF IMPORTANCE:\n`;
  prompt += `- EFFICIENCY MARGIN (most predictive): AdjEM is the best single predictor. A 5-point AdjEM gap = ~65/35 game. A 10+ point gap = ~75/25. Respect large gaps — they exist for a reason.\n`;
  prompt += `- LUCK REGRESSION: Teams with high luck ratings (>0.04) have overperformed their underlying quality and are due for regression. Teams with negative luck are UNDERVALUED — they're better than their record.\n`;
  prompt += `- STRENGTH OF SCHEDULE: Teams battle-tested against top competition (high SOS) handle tournament pressure better. Weak-SOS teams may be paper tigers.\n`;
  prompt += `- DEFENSE IN MARCH: Elite defenses (AdjD top 15) have a modest edge in single-elimination — but defense alone doesn't overcome large AdjEM gaps.\n`;
  prompt += `- STYLE MATCHUP: A great offense vs. a great defense is a genuine toss-up regardless of seed or overall KenPom rank.\n`;
  prompt += `- TOURNAMENT DNA: Bluebloods handle pressure and close games better, but this is a tiebreaker, not a primary factor.\n`;
  prompt += `- TEMPO CONTROL: Slow, disciplined teams can frustrate faster opponents, but tempo is already captured in efficiency metrics.\n`;

  if (tourneyCtx && tourneyCtx.gamesPlayed > 4) {
    const expected = getExpectedUpsets(tourneyCtx.gamesPlayed);
    const actual = tourneyCtx.upsetPicks;
    if (actual < expected - 2) {
      prompt += `\nCALIBRATION: Your picks have been too chalky (${actual} upsets vs ~${Math.round(expected)} expected at this point). Real tournaments have MORE upsets than this. If this game has ANY legitimate upset indicators above, pick the underdog.\n`;
    } else if (actual > expected + 3) {
      prompt += `\nCALIBRATION: Your picks have produced more upsets than typical (${actual} vs ~${Math.round(expected)} expected). Lean toward the higher seed unless the data strongly favors the underdog.\n`;
    }
  }

  prompt += `\nReturn JSON with:\n`;
  prompt += `- winner: "1" or "2"\n`;
  prompt += `- reasoning: 1-2 sentences explaining your pick like an analyst on TV\n`;
  prompt += `Mapping: 1 = ${game.team1.name} (${game.team1.seed} seed) | 2 = ${game.team2.name} (${game.team2.seed} seed)\n`;

  return prompt;
}

// ── Location helpers ────────────────────────────────────────────────

export function getLocationContext(
  schedule: TournamentSchedule | undefined,
  roundType:
    | "firstFour"
    | "firstRound"
    | "secondRound"
    | "sweet16"
    | "elite8"
    | "finalFour"
    | "championship",
  regionName?: RegionName,
  gameId?: string
): string {
  if (!schedule) return "";

  switch (roundType) {
    case "firstFour":
      return `Neutral site — ${formatVenueLine(schedule.firstFour.venue)} (First Four)`;
    case "firstRound":
    case "secondRound": {
      const v = gameId
        ? getFirstSecondRoundVenueForGameId(gameId, schedule)
        : undefined;
      return v
        ? `Neutral site — ${formatVenueLine(v)} (Round of 64/32)`
        : "";
    }
    case "sweet16":
    case "elite8":
      if (regionName && schedule.regionals[regionName]) {
        const regional = schedule.regionals[regionName];
        return `Neutral site — ${formatVenueLine(regional.venue)} (${regionName} Regional)`;
      }
      return "";
    case "finalFour":
      return `Neutral site — ${formatVenueLine(schedule.finalFour.venue)} (Final Four)`;
    case "championship":
      return `Neutral site — ${formatVenueLine(schedule.championship.venue)} (National Championship)`;
    default:
      return "";
  }
}

export function getNextRegionalLine(
  schedule: TournamentSchedule | undefined,
  regionName?: RegionName
): string | undefined {
  if (!schedule || !regionName || !schedule.regionals[regionName])
    return undefined;
  return `${formatVenueLine(schedule.regionals[regionName].venue)} — ${regionName} Sweet 16 / Elite Eight`;
}

// ── Core helpers ────────────────────────────────────────────────────

export function getWinner(game: Game, winner: 1 | 2): Team {
  return winner === 1 ? game.team1 : game.team2;
}

export function updateTournamentContext(
  ctx: TournamentContext,
  game: Game,
  result: SimulatedGame,
  roundName: string
): void {
  const winner = result.winner === 1 ? game.team1 : game.team2;
  const loser = result.winner === 1 ? game.team2 : game.team1;
  ctx.eliminatedTeams.push(loser);
  ctx.gamesPlayed++;
  if (winner.seed > loser.seed) {
    ctx.upsets.push({ winner, loser, round: roundName });
    ctx.upsetPicks++;
    if (
      winner.seed >= 10 &&
      !ctx.cinderellaTeams.some((t) => t.id === winner.id)
    ) {
      ctx.cinderellaTeams.push(winner);
    }
  } else {
    ctx.chalkPicks++;
  }
}

// ── AI game simulation ─────────────────────────────────────────────

export async function simulateGameWithAI(
  game: Game,
  context?: GameContext
): Promise<SimulatedGame> {
  const prompt = buildEnhancedPrompt(game, context ?? {});

  console.log(`\n${"=".repeat(80)}`);
  console.log(
    `GAME: ${game.team1.name} (#${game.team1.seed}) vs ${game.team2.name} (#${game.team2.seed})`
  );
  console.log(`${"=".repeat(80)}`);

  const result = await generateText({
    model: MODEL,
    prompt,
    temperature: 0.7,
    output: Output.object({
      schema: GAME_PICK_SCHEMA,
      name: "gameResult",
      description: "Winner selection with brief reasoning.",
    }),
  });

  const winner: 1 | 2 = result.output.winner === "1" ? 1 : 2;
  const winnerTeam = winner === 1 ? game.team1 : game.team2;

  console.log(`RESULT: ${winnerTeam.name} (#${winnerTeam.seed}) wins`);
  console.log(`REASONING: ${result.output.reasoning}`);

  return {
    ...game,
    status: "final",
    winner,
    reasoning: result.output.reasoning,
  };
}

// ── Full bracket simulation (no workflow) ───────────────────────────

export async function simulateBracketLocally(
  bracket: Bracket
): Promise<SimulatedBracket> {
  const tournamentContext: TournamentContext = {
    upsets: [],
    eliminatedTeams: [],
    cinderellaTeams: [],
    chalkPicks: 0,
    upsetPicks: 0,
    gamesPlayed: 0,
  };

  // 1. First Four
  const firstFourContext: GameContext = {
    roundName: "First Four",
    location: getLocationContext(bracket.schedule, "firstFour"),
    tournamentContext,
  };
  const firstFourResults: SimulatedGame[] = [];
  for (let i = 0; i < bracket.firstFour.length; i += SIM_GAME_CONCURRENCY) {
    const chunk = bracket.firstFour.slice(i, i + SIM_GAME_CONCURRENCY);
    const chunkRes = await Promise.all(
      chunk.map((game) => simulateGameWithAI(game, firstFourContext))
    );
    firstFourResults.push(...chunkRes);
  }
  bracket.firstFour.forEach((game, i) => {
    updateTournamentContext(
      tournamentContext,
      game,
      firstFourResults[i],
      "First Four"
    );
  });

  // 2. Build R64 with First Four winners injected
  let currentRoundGames: Game[][] = bracket.regions.map(
    (region, regionIdx) => {
      const games = region.rounds[0].map((g) => ({
        ...g,
        status: "scheduled" as const,
      }));
      FIRST_FOUR_SLOTS.forEach((slot, ffIdx) => {
        if (slot.region === regionIdx) {
          const game = games[slot.gameIndex];
          const winner = getWinner(
            bracket.firstFour[ffIdx],
            firstFourResults[ffIdx].winner!
          );
          if (slot.slot === 1) {
            games[slot.gameIndex] = { ...game, team1: winner };
          } else {
            games[slot.gameIndex] = { ...game, team2: winner };
          }
        }
      });
      return games;
    }
  );

  // 3. Simulate regional rounds (R64 → R32 → Sweet 16 → Elite 8)
  const allRegionRounds: SimulatedGame[][][] = [];
  let currentRoundResults: SimulatedGame[][] = [];

  type RegionalJob = { game: Game; regionIdx: number; context: GameContext };

  for (let roundIdx = 0; roundIdx <= 3; roundIdx++) {
    if (roundIdx > 0) {
      currentRoundGames = bracket.regions.map((region, regionIdx) => {
        const prevR = currentRoundResults[regionIdx];
        const roundSize = region.rounds[roundIdx].length;
        return Array.from({ length: roundSize }, (_, i) => {
          const game1 = currentRoundGames[regionIdx][2 * i];
          const game2 = currentRoundGames[regionIdx][2 * i + 1];
          const result1 = prevR[2 * i];
          const result2 = prevR[2 * i + 1];
          const template = region.rounds[roundIdx][i];
          return {
            ...template,
            id: template.id,
            status: "scheduled" as const,
            team1: getWinner(game1, result1.winner!),
            team2: getWinner(game2, result2.winner!),
          };
        });
      });
    }

    const roundType =
      roundIdx === 2 ? "sweet16" : roundIdx === 3 ? "elite8" : undefined;
    const openingRoundType =
      roundIdx <= 1
        ? roundIdx === 0
          ? "firstRound"
          : "secondRound"
        : undefined;

    const jobs: RegionalJob[] = bracket.regions.flatMap(
      (region, regionIdx) =>
        currentRoundGames[regionIdx].map((game) => ({
          game,
          regionIdx,
          context: {
            roundName: ROUND_NAMES[roundIdx],
            regionName: region.name,
            location: roundType
              ? getLocationContext(bracket.schedule, roundType, region.name)
              : openingRoundType
                ? getLocationContext(
                    bracket.schedule,
                    openingRoundType,
                    region.name,
                    game.id
                  )
                : undefined,
            nextRegionalSite:
              openingRoundType && bracket.schedule
                ? getNextRegionalLine(bracket.schedule, region.name)
                : undefined,
            tournamentContext,
          },
        }))
    );

    const resultsByGameId = new Map<string, SimulatedGame>();
    for (let i = 0; i < jobs.length; i += SIM_GAME_CONCURRENCY) {
      const chunk = jobs.slice(i, i + SIM_GAME_CONCURRENCY);
      const results = await Promise.all(
        chunk.map((j) => simulateGameWithAI(j.game, j.context))
      );
      chunk.forEach((job, idx) => {
        updateTournamentContext(
          tournamentContext,
          job.game,
          results[idx],
          ROUND_NAMES[roundIdx]
        );
        resultsByGameId.set(job.game.id, results[idx]);
      });
    }

    currentRoundResults = bracket.regions.map((_, regionIdx) =>
      currentRoundGames[regionIdx].map(
        (game) => resultsByGameId.get(game.id)!
      )
    );
    allRegionRounds.push(
      currentRoundResults.map((r) =>
        r.map((g) => ({ ...g, status: "final" as const }))
      )
    );
  }

  // 4. Region winners
  const regionWinners: Team[] = bracket.regions.map((_, regionIdx) =>
    getWinner(
      currentRoundGames[regionIdx][0],
      currentRoundResults[regionIdx][0].winner!
    )
  );

  // 5. Final Four (2026 NCAA): East(1) vs South(0), West(2) vs Midwest(3)
  const finalFourGames: Game[] = [
    {
      id: "ff-east-south",
      status: "scheduled",
      team1: regionWinners[1],
      team2: regionWinners[0],
    },
    {
      id: "ff-west-midwest",
      status: "scheduled",
      team1: regionWinners[2],
      team2: regionWinners[3],
    },
  ];

  const finalFourContext: GameContext = {
    roundName: "Final Four",
    location: getLocationContext(bracket.schedule, "finalFour"),
    tournamentContext,
  };

  const finalFourResults: SimulatedGame[] = [];
  for (const game of finalFourGames) {
    const result = await simulateGameWithAI(game, finalFourContext);
    updateTournamentContext(tournamentContext, game, result, "Final Four");
    finalFourResults.push(result);
  }

  // 6. Championship
  const championshipGame: Game = {
    id: "champ",
    status: "scheduled",
    team1: getWinner(finalFourGames[0], finalFourResults[0].winner!),
    team2: getWinner(finalFourGames[1], finalFourResults[1].winner!),
  };

  const championshipContext: GameContext = {
    roundName: "National Championship",
    location: getLocationContext(bracket.schedule, "championship"),
    tournamentContext,
  };

  const championshipResult = await simulateGameWithAI(
    championshipGame,
    championshipContext
  );
  const winner = getWinner(championshipGame, championshipResult.winner!);

  return {
    year: bracket.year,
    schedule: bracket.schedule,
    firstFour: firstFourResults,
    regions: bracket.regions.map((region, i) => ({
      ...region,
      rounds: allRegionRounds.map((roundResults) => roundResults[i]),
    })),
    finalFour: finalFourResults.map((r) => ({
      ...r,
      status: "final" as const,
    })),
    championship: { ...championshipResult, status: "final" as const },
    winner,
  };
}
