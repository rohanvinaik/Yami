"""U3 increment 2b — PROOF OF MECHANISM (one story, before any renderer change or roster fire).

Does a literature-mined break-condition, encoded as an EXPECTATION + `cannot` censor in the chess_breaks
universe, actually produce a SURPRISE (a retraction) when fired through Genesis and read by retraction.py?
The story below commits the passer expectation ("advances passer" -> passer strong) and then violates it
("Opponent blocks passer" — the Tarrasch break-condition). We expect the derived `passer become strong`
to be RETRACTED, and retraction.py to report it as a surprise. The surprise's MAGNITUDE = the size of its
retraction cascade in the DAG (how much downstream belief the violation overturns).
"""
import sys

sys.path.insert(0, "/Users/rohanvinaik/Genesis/mcp")
sys.path.insert(0, "/Users/rohanvinaik/Projects/Regenesis")

import genesis_access as ga  # noqa: E402
from regenesis.retraction import build_dag, recompute_retraction  # noqa: E402

STORY = "Rubinstein promotes pawn. Opponent blocks pawn."

srv = ga._server_module()
path = srv._persist_story(STORY, srv._slug(STORY))
jvm = ga.Jvm(universe=None, universe_only=False, dual_projection=False)
o = jvm.call({"cmd": "understand_file", "path": str(path)}, timeout=120.0)
if hasattr(jvm, "close"):
    jvm.close()

art = dict(o.get("artifact") or {})
trace = art.get("trace", [])

print("=== derive / retracted events ===")
for e in trace:
    if e.get("kind") == "derive":
        print(f"  derive    {e.get('fact')}")
    elif e.get("kind") == "retracted":
        print(f"  RETRACTED {e.get('fact')}  <= censor: {e.get('censor_premise')}")

retracted = recompute_retraction(trace)
print(f"\n=== retraction.py surprises: {len(retracted)} ===")
for f in retracted:
    print(f"  surprise: {f}")

# Magnitude = cascade size: the seed + every DAG-descendant that loses all support once it is gone.
dag = build_dag(trace)
print(f"\nDAG facts: {len(dag)} | surprise magnitude (retracted cascade): {len(retracted)}")
print("MECHANISM PROVEN" if retracted else "NO SURPRISE FIRED — expectation/censor not triggered")
