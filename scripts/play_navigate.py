"""Recognition → move, LIVE (§14.7): write the story over the very game we're playing, in the moment.

The principled play loop (user, 2026-07-14): we are IN a game; render the GAME-SO-FAR → understand/significance
gives the recognized arc + where we are in it (the load-bearing state) → each Scale-0 candidate is scored by
how it ADVANCES that story toward its climax (gated by the tactical floor: never hang material / allow mate-in-1)
→ navigate(outcome=+1) picks the move that best CARRIES THE STORY FORWARD. Not a static snapshot eval, not
search — understanding-in-the-moment, and only as good as the recognition of the game-so-far (which reads real
games correctly, §14). This script emits the history (to `understand`) + the scored candidates; the recognized
plan is passed back via --plan for the final pick."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import chess
import chess.pgn
from yami.datagen.game_renderer import render_game, render_move
from yami.tactical_scoper import scope_moves

# Recognized plan → the advancing verbs of its style-arc (from §14.5), weighted by how climactic.
PLAN_ARC = {
    "ATTACK_KING":      {"checks": 3, "exposes": 3, "sacrifices": 3, "infiltrates": 2, "forks": 2, "pins": 2},
    "EXPLOIT_WEAKNESS": {"infiltrates": 3, "occupies": 2, "isolates": 2, "weakens": 2, "pins": 2},
    "IMPROVE_PIECES":   {"develops": 2, "occupies": 3, "coordinates": 2, "infiltrates": 2},
    "PAWN_BREAK":       {"advances": 3, "isolates": 2, "weakens": 2},
    "PROPHYLAXIS":      {"restrains": 3, "defends": 2, "coordinates": 2},
    "FORTIFY":          {"defends": 3, "coordinates": 2, "restrains": 2},
    "SIMPLIFY":         {"captures": 2, "coordinates": 2},
}

# ── The AUTOMATED recognition seam (§2.1): understand-read → PlanType, no hand-passed --plan ──
# The plan a read calls for is NOT its highest-weight fact — "morphy become active" (generic ascendancy)
# outweighs "morphy storm king" (the actual idea). The plan is the most-COMMITTAL category the read
# supports: a real king-storm IS the move (mate > pressing a weakness). So we scan ALL significant facts
# (chain-improbability floats them out of the noise) and take the first category, in priority order, that
# any of them evidences — reading both action-verbs (storm/counter king) and become-STATES (rank→vulnerable).
_ATTACK_VERBS   = {"storm", "hunt", "overwhelm", "trap", "counter"}              # US aggression AT the king
_KING_STATES    = {"exposed", "cramped", "strangled", "immobilized", "hunted",   # the king as terminal target
                   "trapped", "stormed", "overwhelmed", "vulnerable", "shattered"}
_WEAK_SUBJ      = {"rank", "file", "pawn", "square", "structure", "diagonal"}    # a board-feature weakness…
_WEAK_STATES    = {"vulnerable", "exposed", "weak", "isolated", "backward", "overextended"}   # …to press
_RESTRAIN_SUBJ  = {"opponent", "piece", "knight", "bishop", "rook", "queen"}
_RESTRAIN_STATES = {"constrained", "restrained", "contained", "cramped", "neutralized"}       # bind mobility
_BREAK_STATES   = {"advanced", "passed", "broken"}
_ACTIVE_STATES  = {"active", "developed", "coordinated", "connected", "dominant", "ascendant"}
_HOLD_STATES    = {"balanced", "resilient", "unbreakable", "held", "impregnable", "solid"}
_HOLD_ARCS      = {"chess_draw", "chess_defender"}
_STATE_NOISE    = {"taken", "pressure", "safe"}   # capture/absorb/own-safety residue: real, not plan-bearing


def _facts(read: dict) -> list[tuple[str, str, str]]:
    """Significant derived facts as (subject, verb, object) — chain-improbability already floated the crux
    to the top; we keep all with weight>0 so a lower-ranked-but-committal fact (storm king) can still win."""
    out = []
    for s in read.get("significance", []):
        if s.get("weight", 0) <= 0 or s.get("noise"):
            continue
        parts = s.get("fact", "").replace("—", " ").replace("→", " ").split()
        if len(parts) >= 3 and parts[-1] not in _STATE_NOISE:
            out.append((parts[0], parts[1], parts[-1]))     # (subj, verb, obj/state)
    return out


def _us_arc(read: dict, us: str) -> str:
    """The pattern whose subject == US — the style-arc we are IN (our stance)."""
    for p in read.get("patterns", []):
        if p.get("subject", "").lower() == us.lower():
            return p.get("name", "")
    return ""


def recognized_plan(read: dict, us: str) -> tuple[str, str]:
    """Map an `understand` read → a PlanType (one of PLAN_ARC's keys). Returns (plan, why)."""
    facts = _facts(read)
    arc = _us_arc(read, us)
    tag = f" under {arc}(us)" if arc else ""

    def hit(pred):
        return next((f for f in facts if pred(f)), None)

    # priority: hunt the king ▸ press a weakness ▸ bind mobility ▸ pawn break ▸ improve ▸ hold
    f = hit(lambda f: f[2] == "king" and f[1] in _ATTACK_VERBS) \
        or hit(lambda f: f[1] == "become" and f[0] == "king" and f[2] in _KING_STATES)
    if f:
        return "ATTACK_KING", f"'{f[0]} {f[1]} {f[2]}'{tag}"
    f = hit(lambda f: f[1] == "become" and f[0] in _WEAK_SUBJ and f[2] in _WEAK_STATES)
    if f:
        return "EXPLOIT_WEAKNESS", f"'{f[0]} → {f[2]}'{tag}"
    f = hit(lambda f: f[1] == "become" and f[0] in _RESTRAIN_SUBJ and f[2] in _RESTRAIN_STATES)
    if f:
        return "PROPHYLAXIS", f"'{f[0]} → {f[2]}'{tag}"
    f = hit(lambda f: f[1] == "become" and f[0] == "pawn" and f[2] in _BREAK_STATES)
    if f:
        return "PAWN_BREAK", f"'{f[0]} → {f[2]}'{tag}"
    f = hit(lambda f: f[1] == "become" and f[2] in _ACTIVE_STATES)
    if f:
        return "IMPROVE_PIECES", f"'{f[0]} → {f[2]}'{tag}"
    if arc in _HOLD_ARCS or hit(lambda f: f[1] == "become" and f[2] in _HOLD_STATES):
        return "FORTIFY", f"the hold{tag}"
    return "IMPROVE_PIECES", f"(default: quiet improvement){tag}"


def render_history(game: chess.pgn.Game, upto: int, us: str, them: str) -> tuple[str, chess.Board]:
    """The game-so-far as a story (the recognition input) + the board at the live decision point."""
    board = game.board()
    sub = chess.pgn.Game()
    node = sub
    for m in list(game.mainline_moves())[:upto]:
        node = node.add_variation(m)
        board.push(m)
    return " ".join(render_game(sub, white=us, black=them)), board


def _allows_mate_in_1(board: chess.Board, move: chess.Move) -> bool:
    board.push(move)
    bad = any((board.push(m), board.is_checkmate(), board.pop())[1] for m in board.legal_moves)
    board.pop()
    return bad


def score_candidates(board: chess.Board, name: str, plan: str, top: int = 6) -> list[dict]:
    """Each candidate scored by how it ADVANCES the recognized arc (its climactic verbs), floor-gated."""
    arc = PLAN_ARC.get(plan, {})
    out = []
    for sm in scope_moves(board):
        mv = sm.move
        sound = sm.see_value >= -50 and not _allows_mate_in_1(board, mv)
        acts = [e.split(maxsplit=1)[1].rstrip(".") for e in render_move(board, mv, name)
                if len(e.split()) > 1 and e.split()[0] == name and e.split()[1] in arc]
        advance = sum(arc.get(a.split()[0], 0) for a in acts)
        score = (advance if sound else advance * 0.1) + sm.see_value / 400.0
        out.append({"move": board.san(mv), "mv": mv, "score": round(score, 2),
                    "acts": acts, "sound": sound})
    return sorted(out, key=lambda c: -c["score"])[:top]


def select_move(board: chess.Board, us: str, plan: str) -> tuple[chess.Move | None, dict]:
    """The recognized plan's MOVE to play: the top-scored SOUND candidate (floor-gated), falling back to
    the top overall, then any legal move. Returns (move, info) — info carries the score/acts for the log."""
    cands = score_candidates(board, us, plan, top=len(list(board.legal_moves)) or 1)
    if not cands:
        return None, {}
    pick = next((c for c in cands if c["sound"]), cands[0])
    return pick["mv"], pick


def render_live(board: chess.Board, us: str, them: str, us_is_white: bool) -> str:
    """The game-so-far as a story, from a LIVE board's move stack (the recognition input in a real game)."""
    g = chess.pgn.Game()
    node = g
    for mv in board.move_stack:
        node = node.add_variation(mv)
    white, black = (us, them) if us_is_white else (them, us)
    return " ".join(render_game(g, white=white, black=black))


def _actual_move(game: chess.pgn.Game, upto: int, board: chess.Board) -> str:
    """The move the master ACTUALLY played at this decision point — the ground truth to check against."""
    moves = list(game.mainline_moves())
    return board.san(moves[upto]) if upto < len(moves) else "(game ended)"


def main() -> None:
    # Flags: --read <understand.json> automates recognition (derive the plan); else emit the story to fire.
    argv = sys.argv[1:]
    read_path = None
    if "--read" in argv:
        i = argv.index("--read")
        read_path = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    game_path, upto = argv[0], int(argv[1])
    us = argv[2] if len(argv) > 2 else "Yami"
    plan_override = argv[3] if len(argv) > 3 else None      # legacy manual plan (else derived from --read)
    them = "Opponent"

    g = chess.pgn.read_game(open(game_path))
    if g is None:
        sys.exit(f"no game in {game_path}")
    story, board = render_history(g, upto, us, them)

    if read_path is None and plan_override is None:
        print(f"=== GAME-SO-FAR ({upto} plies) — the story we are IN (fire this for recognition) ===")
        print(story)
        print(f"\n=== decision point: {us} to move ({len(list(board.legal_moves))} legal) ===")
        print("(fire the story above through Regenesis `understand`, then re-run with --read <read.json>)")
        return

    if read_path is not None:                              # AUTOMATED: derive the plan from the read
        with open(read_path) as fh:
            read = json.load(fh)
        plan, why = recognized_plan(read, us)
        print(f"=== decision point: {us} to move ({len(list(board.legal_moves))} legal) ===")
        print(f"recognized plan: {plan}   ({why})")
    else:                                                  # legacy: plan passed explicitly
        plan, why = plan_override, "manual override"
        print(f"=== decision point: {us} to move ({len(list(board.legal_moves))} legal) ===")
        print(f"plan (manual): {plan}")

    for i, c in enumerate(score_candidates(board, us, str(plan)), 1):
        acts = (" via " + "+".join(c["acts"])) if c["acts"] else ""
        print(f"  {i}. {c['move']:7s} score={c['score']:5.2f}{acts}{'' if c['sound'] else ' [UNSOUND]'}")
    print(f"\nground truth — {us} actually played: {_actual_move(g, upto, board)}")


if __name__ == "__main__":
    main()
