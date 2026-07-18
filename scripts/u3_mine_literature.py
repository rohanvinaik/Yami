"""U3 — literature-mining pass (top-down, intensional). Search→Extract→Structure→Store, per
`STRUCTURED_WEB_KNOWLEDGE_EXTRACTION` (LLM-as-parser-not-source). Local-only: MediaWiki curl +
local Ollama. Every atom traces to a URL + a VERBATIM source span or it is dropped (the cardinal rule).

What U3 actually harvests: NOT the canonical frames (Yami already transcribed prophylaxis/blockade/
outpost/restraint into chess_positional.rules) — it harvests their BREAK-CONDITIONS: *when to abandon
the rule*. That "restrain … THEN strike" transition is exactly what the composition thermometer
(scripts/style_composition_probe.py) showed the current 18-frame vocabulary cannot encode, and what the
precedence operator is waiting to detect. The break-condition is the novel, load-bearing field here.

Output: data/genesis_roster/u3_atoms.jsonl — one grounded atom per line (crash-safe; resume skips
concepts already in the ledger). Increment 1 = knowledge acquisition. Wiring the atoms into the renderer
as signal-detectors (so games EMIT them) is increment 2 — it touches the "renderer makes zero chess
judgments" discipline and is a separate, deliberate step.
"""
from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

LEDGER = Path(__file__).resolve().parent.parent / "data" / "genesis_roster" / "u3_atoms.jsonl"
UA = "YamiChessResearch/0.1 (rohanpvinaik@gmail.com)"
MODEL = "qwen3.5:35b"
WIKI = "https://en.wikipedia.org/w/api.php"

# Canonical My System / Nimzowitsch frame set — the article TITLES to fetch (search-target selection,
# not knowledge generation; the atom CONTENT still comes only from the fetched text).
CONCEPTS = [
    "Prophylaxis (chess)", "Overprotection (chess)", "Blockade (chess)", "Outpost (chess)",
    "Isolated pawn", "Pawn structure", "Passed pawn", "Minority attack",
    "Weak square", "Open file", "Undermining (chess)", "Fianchetto",
]

# Existing Genesis vocabulary (from chess_positional.rules) — the model is STEERED to reuse it so the
# atoms compose with what exists; it may coin a new single lowercase token when nothing fits (growth).
VERBS = ["maneuvers", "develops", "restricts", "restrains", "cramps", "occupies", "controls",
         "advances", "weakens", "isolates", "doubles", "exposes", "blockades", "prevents",
         "infiltrates", "penetrates", "overprotects", "undermines", "strikes"]
STATES = ["active", "passive", "cramped", "immobilized", "restrained", "isolated", "doubled",
          "exposed", "safe", "weak", "strong", "controlled"]

SCHEMA = {
    "type": "object",
    "properties": {
        "action_verb": {"type": "string"},
        "consequent_state": {"type": "string"},
        "break_condition": {"type": ["string", "null"]},
        "break_span": {"type": ["string", "null"]},
        "core_span": {"type": "string"},
    },
    "required": ["action_verb", "consequent_state", "core_span"],
}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


# The model sometimes writes its "no exception" signal as prose in the break FIELD instead of JSON null.
# Normalize those to a real None so the ledger never carries a fake break-condition.
_NULL_BREAKS = {"null", "none", "n/a", "na", "none stated", "none stated in text",
                "no exception", "no exceptions", "not stated", "not mentioned"}


def _clean_break(bc: str | None) -> str | None:
    return None if not bc or _norm(bc).strip(".") in _NULL_BREAKS else bc


def _focus_window(text: str, concept: str, span: int = 1600) -> str:
    """Relevance-scored window (framework §6c): when the article is long (e.g. a redirect landed on the
    122K Glossary of chess), center the extraction window on the concept term's definition rather than
    feeding the alphabetical top — otherwise the term never enters the LLM's window."""
    if len(text) <= 6000:
        return text[:6000]
    term = re.sub(r"\s*\(chess\)\s*", "", concept).strip().lower()
    i = text.lower().find(term)
    if i == -1:
        return text[:6000]
    return text[max(0, i - 200): i - 200 + span]


def _grounded(span: str, ntext: str) -> bool:
    """Grounding gate with paraphrase tolerance: the WHOLE normalized span in the text (fast path), or —
    for longer spans the model may have trimmed at the edges — at least one contiguous 40-char verbatim
    chunk present. A real quote must exist; only edge-paraphrase is forgiven, never invention."""
    ns = _norm(span)
    if len(ns) < 10:
        return False
    if ns in ntext:
        return True
    return len(ns) >= 40 and any(ns[i:i + 40] in ntext for i in range(0, len(ns) - 40, 8))


def _wiki_query(title: str) -> tuple[str, str]:
    params = urllib.parse.urlencode({"action": "query", "titles": title, "prop": "extracts",
                                     "explaintext": "true", "redirects": "1", "format": "json"})
    req = urllib.request.Request(f"{WIKI}?{params}", headers={"User-Agent": UA})
    data = json.loads(urllib.request.urlopen(req, timeout=20).read())
    page = list(data["query"]["pages"].values())[0]
    canonical = urllib.parse.quote(page.get("title", title).replace(" ", "_"))
    return page.get("extract", ""), f"https://en.wikipedia.org/wiki/{canonical}"


def _wiki_search_title(query: str) -> str | None:
    params = urllib.parse.urlencode({"action": "query", "list": "search", "srsearch": query,
                                     "srlimit": "1", "format": "json"})
    req = urllib.request.Request(f"{WIKI}?{params}", headers={"User-Agent": UA})
    hits = json.loads(urllib.request.urlopen(req, timeout=20).read())["query"]["search"]
    return hits[0]["title"] if hits else None


def wiki_extract(title: str) -> tuple[str, str]:
    """MediaWiki extract, following redirects; if the exact title is thin, fall back to a search hit."""
    text, url = _wiki_query(title)
    if len(text) < 200:                                   # redirect/disambig miss → search fallback
        alt = _wiki_search_title(title if "chess" in title.lower() else f"{title} chess")
        if alt and alt.lower() != title.lower():
            text, url = _wiki_query(alt)
    return text, url


def ollama_parse(concept: str, text: str) -> dict | None:
    prompt = f"""You are a PARSER, not a chess author — use ONLY the text below, never outside knowledge.
From this text about the chess concept "{concept}", extract its intensional rule AND its exception.

Output ONLY a single JSON object, nothing else, with EXACTLY these keys:
{{"action_verb": <str>, "consequent_state": <str>, "break_condition": <str or null>,
  "break_span": <str or null>, "core_span": <str>}}

- action_verb: one lowercase verb for what the player DOES to apply this concept. Prefer one of {VERBS};
  coin a new single lowercase verb ONLY if none fit.
- consequent_state: the resulting one-word lowercase state. Prefer one of {STATES}.
- break_condition: a SHORT phrase (<=12 words) for WHEN this concept should be ABANDONED or FAILS — the
  exception, the "however / but / unless / too slow / drawback" case. null ONLY if the text states none.
- break_span: a VERBATIM substring from the text supporting the break_condition (or null).
- core_span: a VERBATIM substring (10-200 chars) from the text supporting the core rule. Copy exactly.

TEXT:
{_focus_window(text, concept)}"""
    payload = json.dumps({
        "model": MODEL, "messages": [{"role": "user", "content": prompt}],
        # format:"json" (the STRING) is enforced by this Ollama; the schema DICT was silently ignored.
        "stream": False, "format": "json", "think": False,
        "options": {"temperature": 0.1, "num_predict": 500, "num_ctx": 8192},
    }).encode()
    req = urllib.request.Request("http://localhost:11434/api/chat", data=payload,
                                 headers={"Content-Type": "application/json"})
    resp = json.loads(urllib.request.urlopen(req, timeout=240).read())
    raw = (resp.get("message", {}).get("content", "") or "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)          # tolerant: pull the JSON object out of any wrapper
        return json.loads(m.group(0)) if m else None


def already_done() -> set[str]:
    if not LEDGER.exists():
        return set()
    return {json.loads(ln)["concept"] for ln in LEDGER.read_text().splitlines() if ln.strip()}


def main() -> None:
    done = already_done()
    with LEDGER.open("a") as ledger:
        for concept in CONCEPTS:
            if concept in done:
                print(f"  skip (done): {concept}")
                continue
            try:
                text, url = wiki_extract(concept)
            except Exception as e:  # noqa: BLE001 — network; log and continue (crash-safe)
                print(f"  FETCH FAIL {concept}: {e}")
                continue
            if len(text) < 200:
                print(f"  thin article, skip: {concept} ({len(text)} chars)")
                continue
            t0 = time.time()
            try:
                atom = ollama_parse(concept, text)
            except Exception as e:  # noqa: BLE001
                print(f"  PARSE FAIL {concept}: {e}")
                continue
            if not atom:
                print(f"  empty parse: {concept}")
                continue
            # GROUNDING GATE (cardinal rule): a real verbatim quote must exist in the text.
            ntext = _norm(text)
            break_grounded = (not atom.get("break_span")) or _grounded(atom["break_span"], ntext)
            if not _grounded(atom.get("core_span") or "", ntext):
                print(f"  UNGROUNDED core span, DROP: {concept}")
                continue
            if not break_grounded:  # keep the atom but null out an unsupported break-condition
                atom["break_condition"] = atom["break_span"] = None
            rec = {"concept": concept, "url": url, "model": MODEL,
                   "action_verb": atom["action_verb"], "consequent_state": atom["consequent_state"],
                   "break_condition": _clean_break(atom.get("break_condition")),
                   "break_span": atom.get("break_span"), "core_span": atom["core_span"]}
            ledger.write(json.dumps(rec) + "\n")
            ledger.flush()
            bc = rec["break_condition"]
            print(f"  [{time.time()-t0:4.0f}s] {concept}: {rec['action_verb']} -> {rec['consequent_state']}"
                  f"  | break: {bc!r}")
    print("\nledger:", LEDGER)


if __name__ == "__main__":
    main()
