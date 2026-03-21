"""ELO measurement framework — play against Stockfish at calibrated levels.

Supports ablation: toggle each layer on/off to measure its ELO contribution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import chess

from yami.engine import YamiEngine
from yami.oracle import StockfishOracle


@dataclass
class GameResult:
    """Result of a single game."""

    white_engine: str
    black_engine: str
    result: str  # "1-0", "0-1", "1/2-1/2"
    moves: int
    move_history: list[str] = field(default_factory=list)


@dataclass
class EloEstimate:
    """ELO estimate from a match."""

    elo: float
    games: int
    wins: int
    draws: int
    losses: int
    win_rate: float
    confidence_interval: tuple[float, float]


def play_game(
    yami: YamiEngine,
    oracle: StockfishOracle,
    yami_color: chess.Color = chess.WHITE,
    max_moves: int = 200,
) -> GameResult:
    """Play a full game between Yami and Stockfish."""
    board = chess.Board()
    yami.reset()
    yami.state.board = board
    move_sans: list[str] = []

    for _ in range(max_moves):
        if board.is_game_over():
            break

        if board.turn == yami_color:
            decision = yami.decide(board)
            move = decision.move
        else:
            move = oracle.best_move(board)

        if move is None:
            break

        move_sans.append(board.san(move))
        board.push(move)

    result = board.result()
    return GameResult(
        white_engine="yami" if yami_color == chess.WHITE else "stockfish",
        black_engine="stockfish" if yami_color == chess.WHITE else "yami",
        result=result,
        moves=len(move_sans),
        move_history=move_sans,
    )


def run_match(
    yami: YamiEngine,
    oracle: StockfishOracle,
    num_games: int = 10,
    max_moves_per_game: int = 200,
) -> list[GameResult]:
    """Run a match of num_games, alternating colors."""
    results = []
    for i in range(num_games):
        color = chess.WHITE if i % 2 == 0 else chess.BLACK
        result = play_game(yami, oracle, yami_color=color, max_moves=max_moves_per_game)
        results.append(result)
    return results


def estimate_elo(
    results: list[GameResult],
    opponent_elo: float,
    yami_name: str = "yami",
) -> EloEstimate:
    """Estimate ELO from game results against a known-ELO opponent."""
    wins = 0
    draws = 0
    losses = 0

    for r in results:
        if r.white_engine == yami_name:
            if r.result == "1-0":
                wins += 1
            elif r.result == "0-1":
                losses += 1
            else:
                draws += 1
        else:
            if r.result == "0-1":
                wins += 1
            elif r.result == "1-0":
                losses += 1
            else:
                draws += 1

    total = wins + draws + losses
    if total == 0:
        return EloEstimate(
            elo=opponent_elo, games=0, wins=0, draws=0, losses=0,
            win_rate=0.5, confidence_interval=(opponent_elo, opponent_elo),
        )

    score = (wins + 0.5 * draws) / total

    # Avoid division by zero in logit
    score = max(0.001, min(0.999, score))

    elo_diff = -400 * math.log10(1 / score - 1)
    elo = opponent_elo + elo_diff

    # Simple confidence interval (approximate)
    stderr = math.sqrt(score * (1 - score) / total) * 400
    ci = (elo - 2 * stderr, elo + 2 * stderr)

    return EloEstimate(
        elo=round(elo, 1),
        games=total,
        wins=wins,
        draws=draws,
        losses=losses,
        win_rate=round(score, 3),
        confidence_interval=(round(ci[0], 1), round(ci[1], 1)),
    )


@dataclass
class AblationResult:
    """Result of an ablation study — one layer toggled."""

    layer_name: str
    enabled: bool
    elo: EloEstimate


def run_ablation(
    oracle: StockfishOracle,
    opponent_elo: float,
    num_games: int = 10,
) -> list[AblationResult]:
    """Run ablation study — measure each layer's ELO contribution.

    Tests: full stack, then disabling each layer one at a time.
    """
    all_on = {
        "use_llm": True, "use_opening_book": True,
        "use_endgame_tables": True, "use_censors": True,
    }
    configs = [
        ("full_stack", {**all_on}),
        ("no_llm", {**all_on, "use_llm": False}),
        ("no_opening_book", {**all_on, "use_opening_book": False}),
        ("no_endgame_tables", {**all_on, "use_endgame_tables": False}),
        ("no_censors", {**all_on, "use_censors": False}),
        ("infrastructure_only", {**all_on, "use_llm": False}),
    ]

    results = []
    for name, kwargs in configs:
        engine = YamiEngine(**kwargs)
        game_results = run_match(engine, oracle, num_games=num_games)
        elo = estimate_elo(game_results, opponent_elo)
        results.append(AblationResult(layer_name=name, enabled=True, elo=elo))

    return results
