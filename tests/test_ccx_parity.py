"""The open-weight route sends the backend what ccx's shim sent, for recorded Claude Code requests.

The fixtures are Claude Code requests recorded with an empty config and working directory, with every
free text replaced. tests/fixtures/make_expected.py produced the *.ccx.json references from ccx.
"""
import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def text_of(content):
    if isinstance(content, list):
        return "\n\n".join(p["text"] for p in content if p.get("type") == "text")
    return content


def normalize(req):
    """Erase differences that don't change what the model sees: text split into parts vs. joined, JSON spacing."""
    out = {k: v for k, v in req.items() if k != "messages"}
    out["messages"] = []
    for m in req["messages"]:
        m = {**m, "content": text_of(m.get("content"))}
        if "tool_calls" in m:
            m["tool_calls"] = [{**c, "function": {**c["function"], "arguments": json.loads(c["function"]["arguments"])}}
                               for c in m["tool_calls"]]
        out["messages"].append(m)
    out["tools"] = sorted(req.get("tools", []), key=lambda t: t["function"]["name"])
    return out


def with_accepted_differences(ccx):
    """ccx's request, adjusted where the router differs on purpose or LiteLLM differs harmlessly."""
    # ccx left max_tokens to CLAUDE_CODE_MAX_OUTPUT_TOKENS, which would also cap Claude; the router caps per model.
    ccx = {**ccx, "max_tokens": 16384}
    # LiteLLM drops `additionalProperties: false` from tool schemas; it doesn't change unconstrained generation.
    for t in ccx["tools"]:
        t["function"]["parameters"].pop("additionalProperties", None)
    return ccx


@pytest.mark.parametrize("name", sorted(p.stem for p in FIXTURES.glob("*_turn.json")))
def test_backend_request_matches_ccx(router, upstream, name):
    router.post("/v1/messages?beta=true", content=(FIXTURES / f"{name}.json").read_bytes())

    ours = normalize(json.loads(upstream.last("/v1/chat/completions")["raw"]))
    ccx = normalize(with_accepted_differences(json.loads((FIXTURES / f"{name}.ccx.json").read_text())))
    assert ours == ccx
