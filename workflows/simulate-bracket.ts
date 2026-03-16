import { getWritable } from "workflow";
import {
  type Bracket,
  type Game,
  type RegionName,
  type SimulatedBracket,
  type SimulatedGame,
  type Team,
} from "@/lib/bracket-data";
import {
  buildWorkflowStreamPayload,
  FIRST_FOUR_SLOTS,
  placeholderRoundsFrom,
  type SimulationProgress,
  type SimulationStage,
} from "@/lib/simulation-shared";
import { placeholderGame } from "@/lib/bracket-data";
import {
  type GameContext,
  type TournamentContext,
  ROUND_NAMES,
  SIM_GAME_CONCURRENCY,
  getLocationContext,
  getNextRegionalLine,
  getWinner,
  simulateGameWithAI,
  updateTournamentContext,
} from "@/lib/ai-pick";

function patchRoundById(round: Game[], id: string, next: Game): Game[] {
  return round.map((g) => (g.id === id ? next : g));
}

const FF_PLACEHOLDERS: Game[] = [
  placeholderGame("ff-south-west"),
  placeholderGame("ff-east-midwest"),
];
const CHAMP_PLACEHOLDER = placeholderGame("champ");

export async function simulateBracket(bracket: Bracket): Promise<SimulatedBracket> {
  "use workflow";

  const regionResults: Map<RegionName, SimulatedGame[][]> = new Map();
  const completedStages: SimulationStage[] = [];
  
  const tournamentContext: TournamentContext = {
    upsets: [],
    eliminatedTeams: [],
    cinderellaTeams: [],
    chalkPicks: 0,
    upsetPicks: 0,
    gamesPlayed: 0,
  };

  const cloneDisplay = (d: Game[][][]): Game[][][] =>
    d.map((rounds) => rounds.map((round) => round.map((g) => ({ ...g }))));

  // 1. First Four (stream in_progress / final like other rounds)
  const firstFourContext: GameContext = {
    roundName: "First Four",
    location: getLocationContext(bracket.schedule, "firstFour"),
    tournamentContext,
  };
  const regionDisplayDuringFF = bracket.regions.map((region) => [
    region.rounds[0].map((g) => ({ ...g, status: "scheduled" as const })),
    ...placeholderRoundsFrom(region, 1),
  ]);
  let firstFourDisplay: Game[] = bracket.firstFour.map((g) => ({
    ...g,
    status: "scheduled" as const,
  }));
  const firstFourResults: SimulatedGame[] = [];

  function buildFirstFourPayload(stage: SimulationStage): SimulationProgress {
    return buildWorkflowStreamPayload(
      bracket,
      firstFourDisplay.map((g) => ({ ...g })),
      cloneDisplay(regionDisplayDuringFF),
      FF_PLACEHOLDERS.map((g) => ({ ...g })),
      { ...CHAMP_PLACEHOLDER },
      stage,
      [...completedStages]
    );
  }

  for (let i = 0; i < bracket.firstFour.length; i += SIM_GAME_CONCURRENCY) {
    const chunk = bracket.firstFour.slice(i, i + SIM_GAME_CONCURRENCY);
    chunk.forEach((game, j) => {
      firstFourDisplay[i + j] = { ...game, status: "in_progress" };
    });
    const chunkRes = await Promise.all(
      chunk.map((game) =>
        simulateAndEmit(game, firstFourContext, buildFirstFourPayload("first-four"))
      )
    );
    chunkRes.forEach((res, j) => {
      firstFourDisplay[i + j] = { ...res, status: "final" as const };
      firstFourResults.push(res);
    });
    chunk.forEach((game, j) => {
      updateTournamentContext(
        tournamentContext,
        game,
        chunkRes[j]!,
        "First Four"
      );
    });
  }

  completedStages.push("first-four");

  const round0GamesByRegion: Map<RegionName, Game[]> = new Map();
  bracket.regions.forEach((region, regionIdx) => {
    const games = region.rounds[0].map((g) => ({ ...g, status: "scheduled" as const }));
    FIRST_FOUR_SLOTS.forEach((slot, ffIdx) => {
      if (slot.region === regionIdx) {
        const game = games[slot.gameIndex];
        const winner = getWinner(bracket.firstFour[ffIdx], firstFourResults[ffIdx].winner!);
        if (slot.slot === 1) {
          games[slot.gameIndex] = { ...game, team1: winner };
        } else {
          games[slot.gameIndex] = { ...game, team2: winner };
        }
      }
    });
    round0GamesByRegion.set(region.name, games);
  });

  const regionDisplay: Game[][][] = bracket.regions.map((region) => [
    round0GamesByRegion.get(region.name)!.map((g) => ({ ...g })),
    ...placeholderRoundsFrom(region, 1),
  ]);

  function buildRegionPayload(
    display: Game[][][],
    ff: Game[],
    champ: Game,
    stage: SimulationStage
  ): SimulationProgress {
    return buildWorkflowStreamPayload(
      bracket,
      firstFourResults,
      cloneDisplay(display),
      ff.map((g) => ({ ...g })),
      { ...champ },
      stage,
      [...completedStages]
    );
  }

  await emitProgress(buildRegionPayload(regionDisplay, FF_PLACEHOLDERS, CHAMP_PLACEHOLDER, "first-four"));

  const regionWinners: Team[] = [];
  let currentRoundGames: Game[][] = bracket.regions.map((r) =>
    round0GamesByRegion.get(r.name)!
  );
  let currentRoundResults: SimulatedGame[][] = [];

  type RegionalJob = { game: Game; regionIdx: number; context: GameContext };

  for (let roundIdx = 0; roundIdx <= 3; roundIdx++) {
    const stage: SimulationStage =
      roundIdx === 0 ? "south" : roundIdx >= 2 ? "midwest" : "east";

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
      bracket.regions.forEach((_, ri) => {
        regionDisplay[ri][roundIdx] = currentRoundGames[ri].map((g) => ({ ...g }));
      });
      await emitProgress(buildRegionPayload(regionDisplay, FF_PLACEHOLDERS, CHAMP_PLACEHOLDER, stage));
    }

    const roundType =
      roundIdx === 2 ? "sweet16" : roundIdx === 3 ? "elite8" : undefined;
    const openingRoundType = roundIdx <= 1 ? (roundIdx === 0 ? "firstRound" : "secondRound") : undefined;
    const jobs: RegionalJob[] = bracket.regions.flatMap((region, regionIdx) =>
      currentRoundGames[regionIdx].map((game) => ({
        game,
        regionIdx,
        context: {
          roundName: ROUND_NAMES[roundIdx],
          regionName: region.name,
          location: roundType
            ? getLocationContext(bracket.schedule, roundType, region.name)
            : openingRoundType
              ? getLocationContext(bracket.schedule, openingRoundType, region.name, game.id)
              : undefined,
          nextRegionalSite:
            openingRoundType && bracket.schedule
              ? getNextRegionalLine(bracket.schedule, region.name)
              : undefined,
          tournamentContext,
        },
      }))
    );

    for (let i = 0; i < jobs.length; i += SIM_GAME_CONCURRENCY) {
      const chunk = jobs.slice(i, i + SIM_GAME_CONCURRENCY);
      for (const job of chunk) {
        regionDisplay[job.regionIdx][roundIdx] = patchRoundById(
          regionDisplay[job.regionIdx][roundIdx],
          job.game.id,
          { ...job.game, status: "in_progress" }
        );
      }
      const progressPayload = buildRegionPayload(regionDisplay, FF_PLACEHOLDERS, CHAMP_PLACEHOLDER, stage);
      const results = await Promise.all(
        chunk.map((j) => simulateAndEmit(j.game, j.context, progressPayload))
      );
      chunk.forEach((job, j) => {
        regionDisplay[job.regionIdx][roundIdx] = patchRoundById(
          regionDisplay[job.regionIdx][roundIdx],
          job.game.id,
          { ...results[j], status: "final" as const }
        );
        updateTournamentContext(tournamentContext, job.game, results[j], ROUND_NAMES[roundIdx]);
      });
    }

    currentRoundResults = bracket.regions.map(
      (_, ri) => regionDisplay[ri][roundIdx] as SimulatedGame[]
    );
  }

  bracket.regions.forEach((region, regionIdx) => {
    regionWinners.push(
      getWinner(
        currentRoundGames[regionIdx][0],
        currentRoundResults[regionIdx][0].winner!
      )
    );
  });
  completedStages.push("south", "east", "west", "midwest");

  const finalFourGames: Game[] = [
    {
      id: "ff-south-west",
      status: "scheduled",
      team1: regionWinners[0],
      team2: regionWinners[2],
    },
    {
      id: "ff-east-midwest",
      status: "scheduled",
      team1: regionWinners[1],
      team2: regionWinners[3],
    },
  ];

  const finalFourContext: GameContext = {
    roundName: "Final Four",
    location: getLocationContext(bracket.schedule, "finalFour"),
    tournamentContext,
  };

  let ffDisplay: Game[] = finalFourGames.map((g) => ({ ...g }));
  await emitProgress(buildRegionPayload(regionDisplay, ffDisplay, CHAMP_PLACEHOLDER, "finals"));

  const finalFourResults: SimulatedGame[] = [];
  for (let fi = 0; fi < 2; fi++) {
    ffDisplay = ffDisplay.map((g, idx) =>
      idx === fi ? { ...g, status: "in_progress" as const } : g
    );
    const ffProgressPayload = buildRegionPayload(regionDisplay, ffDisplay, CHAMP_PLACEHOLDER, "finals");
    const res = await simulateAndEmit(finalFourGames[fi], finalFourContext, ffProgressPayload);
    finalFourResults.push(res);
    updateTournamentContext(tournamentContext, finalFourGames[fi], res, "Final Four");
    ffDisplay = ffDisplay.map((g, idx) =>
      idx === fi ? { ...res, status: "final" as const } : g
    );
  }
  const finalFourFinal = ffDisplay as SimulatedGame[];

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

  let champDisplay: Game = { ...championshipGame, status: "in_progress" };
  const champProgressPayload = buildRegionPayload(regionDisplay, finalFourFinal, champDisplay, "finals");
  const championshipResult = await simulateAndEmit(championshipGame, championshipContext, champProgressPayload);
  const winner = getWinner(championshipGame, championshipResult.winner!);
  const finalChampionship = { ...championshipResult, status: "final" as const };
  champDisplay = finalChampionship;
  completedStages.push("finals");
  await emitProgress({
    ...buildWorkflowStreamPayload(
      bracket,
      firstFourResults,
      cloneDisplay(regionDisplay),
      finalFourFinal.map((g) => ({ ...g })),
      { ...finalChampionship },
      "finals",
      completedStages
    ),
    winner,
  });
  await closeProgressStream();

  bracket.regions.forEach((region, ri) => {
    regionResults.set(
      region.name,
      regionDisplay[ri].map((round) =>
        round.map((g) => ({ ...g, status: "final" as const }))
      ) as SimulatedGame[][]
    );
  });

  return {
    year: bracket.year,
    schedule: bracket.schedule,
    firstFour: firstFourResults,
    regions: bracket.regions.map((region, ri) => ({
      ...region,
      rounds: regionResults.get(region.name) ?? [],
    })),
    finalFour: finalFourFinal,
    championship: finalChampionship,
    winner,
  };
}

async function simulateAndEmit(
  game: Game,
  context: GameContext | undefined,
  progressPayload: SimulationProgress
): Promise<SimulatedGame> {
  "use step";

  const writable = getWritable<string>();
  const writer = writable.getWriter();
  await writer.write(JSON.stringify(progressPayload) + "\n");
  writer.releaseLock();

  return simulateGameWithAI(game, context);
}

async function emitProgress(update: SimulationProgress): Promise<void> {
  "use step";

  const writable = getWritable<string>();
  const writer = writable.getWriter();
  await writer.write(JSON.stringify(update) + "\n");
  writer.releaseLock();
}

async function closeProgressStream(): Promise<void> {
  "use step";

  await getWritable<string>().close();
}
