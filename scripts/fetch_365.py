"""365chess game fetcher — Scale-0 symbolic extraction (STRUCTURED_WEB_KNOWLEDGE_EXTRACTION §7d).

Games are structured (SAN + player names in HTML), so pure regex, zero LLM. Crash-safe (cache per gid,
resume), rate-limited (polite delay), provenance (gid). Metadata from the search table, moves from the
show_game AJAX endpoint (game.php is 403 to curl; show_game.php is 200).

Extends the original either-color fetcher with RESULT-FILTERED, paginated collection (the negative lens
needs each master's LOSSES; the equilibrium lens will need their DRAWS). The 365chess search exposes
`res` (1=`1-0`, 2=`0-1`, 3=`½-½`) as the ABSOLUTE result and `&start=` for pagination; a target LOSS is
`1-0` when the target is Black or `0-1` when White, so we filter per-game by (target colour, result)."""
import html
import os
import re
import time
import urllib.parse
import urllib.request
from collections import Counter

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
BASE = "https://www.365chess.com"
CACHE = os.path.join(os.path.dirname(__file__), "..", "data", "genesis_roster", "365cache")
CACHE = os.path.abspath(CACHE)
os.makedirs(CACHE, exist_ok=True)

_RES_CODE = {"loss_black": "1", "loss_white": "2", "draw": "3"}   # 365chess `res` param → absolute result


def _get(url, referer=None):
    hdr = {"User-Agent": UA, "X-Requested-With": "XMLHttpRequest"}
    if referer:
        hdr["Referer"] = referer
    return urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=30).read().decode("utf-8", "replace")


def _canon(res: str) -> str:
    """Display result ('1-0' / '0-1' / '½-½') → canonical PGN result."""
    r = html.unescape(res).replace("½", "1/2").strip()
    return {"1-0": "1-0", "0-1": "0-1"}.get(r, "1/2-1/2")


def _parse_page(h: str) -> list[dict]:
    """One search page → [{gid, w_slug, b_slug, result}]. Players list W,B,W,B…; zip positionally."""
    players = re.findall(r'href="/players/([^"]+)">([^<]+)</a>', h)      # (slug, display)
    gidres = re.findall(r'href="/game\.php\?gid=(\d+)">([^<]+)</a>', h)  # (gid, result-as-link-text)
    out = []
    for i, (gid, res) in enumerate(gidres):
        if 2 * i + 1 < len(players):
            out.append({"gid": gid, "w_slug": players[2 * i][0], "b_slug": players[2 * i + 1][0],
                        "result": _canon(res)})
    return out


def _search(name: str, res: str | None = None, start: int = 0) -> list[dict]:
    """One page of the target's games (either colour), optionally filtered to an absolute `res` code."""
    params = {"submit_search": "1", "wlname": name, "nocolor": "on"}
    if res:
        params["res"] = res
    if start:
        params["start"] = str(start)
    return _parse_page(_get(f"{BASE}/chess-games.php?{urllib.parse.urlencode(params)}"))


def _anchor_target(name: str) -> str | None:
    """The target slug = the one player present in every game of a name search (most common)."""
    games = _search(name)
    slugs = [g["w_slug"] for g in games] + [g["b_slug"] for g in games]
    return Counter(slugs).most_common(1)[0][0] if slugs else None


def _target_lost(g: dict, target: str) -> bool:
    return (g["b_slug"] == target and g["result"] == "1-0") or (g["w_slug"] == target and g["result"] == "0-1")


def collect(name: str, want: str, target_n: int, delay: float = 0.7, max_pages: int = 10) -> list[dict]:
    """Gather up to target_n of the target's games matching `want` in {'loss','draw'}, paginated + deduped.

    Losses stream res=1 (`1-0`, keep target-Black) and res=2 (`0-1`, keep target-White); draws stream res=3
    (every game is the target's draw). Stops at target_n, page exhaustion (<100 rows), or max_pages/stream."""
    target = _anchor_target(name)
    if not target:
        return []
    streams = {"loss": [_RES_CODE["loss_black"], _RES_CODE["loss_white"]], "draw": [_RES_CODE["draw"]]}[want]
    keep = (lambda g: _target_lost(g, target)) if want == "loss" else (lambda g: g["result"] == "1/2-1/2")
    out: dict[str, dict] = {}                                   # gid → game (dedupe)
    for res in streams:
        for page in range(max_pages):
            if len(out) >= target_n:
                break
            rows = _search(name, res=res, start=page * 100)
            time.sleep(delay)                                  # rate-limit the search too (be polite)
            for g in rows:
                if keep(g):
                    g["target"] = target
                    out[g["gid"]] = g
            if len(rows) < 100:                                # last page of this stream
                break
    return list(out.values())[:target_n]


def fetch_moves(gid: str, delay: float = 0.7) -> str | None:
    """Moves for one game (cached). show_game.php holds the SAN in chess_game_popup.Init."""
    cf = f"{CACHE}/{gid}.txt"
    if os.path.exists(cf):
        return open(cf).read() or None
    time.sleep(delay)
    h = _get(f"{BASE}/show_game.php?g={gid}", referer=f"{BASE}/chess-games.php")
    m = re.search(r"pgn:\s*'([^']*)'", h)
    moves = re.sub(r"\s+", " ", html.unescape(m.group(1))).strip() if m else ""
    moves = re.sub(r"\s*(1-0|0-1|1/2-1/2|\*)\s*$", "", moves).strip()   # drop trailing result token
    open(cf, "w").write(moves)
    return moves or None


def get_player_pgns(name: str, short: str, want: str, target_n: int = 50, delay: float = 0.7) -> tuple[list[str], str | None]:
    """Target's `want` games ('loss'/'draw') as PGN, target labelled `short`, opponent 'Opponent'."""
    games = collect(name, want, target_n, delay)
    if not games:
        return [], None
    target = games[0]["target"]
    pgns = []
    for g in games:
        mv = fetch_moves(g["gid"], delay)
        if not mv:
            continue
        white = short if g["w_slug"] == target else "Opponent"
        black = short if g["b_slug"] == target else "Opponent"
        pgns.append(f'[White "{white}"]\n[Black "{black}"]\n[Result "{g["result"]}"]\n[Source "365chess:{g["gid"]}"]\n\n{mv} {g["result"]}')
    return pgns, target
