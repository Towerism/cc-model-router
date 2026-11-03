"""Regenerate the *.ccx.json files: what ccx's shim (api = "openai") sends upstream for each fixture.

    python3 -I tests/fixtures/make_expected.py ~/.local/bin/ccx

ccx is the prototype this router replaces; its output is the reference the router's shim stage is
tested against (tests/test_ccx_parity.py). Run this only when a fixture or the reference changes.
"""
import importlib.machinery
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0}

loader = importlib.machinery.SourceFileLoader("ccx", sys.argv[1])
spec = importlib.util.spec_from_loader("ccx", loader)
ccx = importlib.util.module_from_spec(spec)
loader.exec_module(ccx)

for f in sorted(HERE.glob("*_turn.json")):
    body = json.loads(f.read_text())
    out = ccx.to_openai_request(ccx.fix_messages(body), SAMPLING, True)
    f.with_suffix(".ccx.json").write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", f.with_suffix(".ccx.json").name)
