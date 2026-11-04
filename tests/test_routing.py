import json

from starlette.testclient import TestClient

from cc_model_router import create_app
from conftest import MODEL

CLAUDE_HEADERS = {"authorization": "Bearer sk-ant-oat01-subscription", "anthropic-beta": "oauth-2025-04-20,x",
                  "anthropic-version": "2023-06-01", "x-claude-code-session-id": "s1"}


def events(text):
    return [line[6:].strip() for line in text.splitlines() if line.startswith("event:")]


def test_claude_request_reaches_anthropic_unchanged(router, upstream):
    raw = b'{"model": "claude-opus-5-5",  "max_tokens": 5, "messages": [{"role": "user", "content": "hi"}]}'
    resp = router.post("/v1/messages?beta=true", content=raw, headers=CLAUDE_HEADERS)

    sent = upstream.last("/v1/messages")
    assert sent["path"] == "/v1/messages?beta=true"
    assert sent["raw"] == raw
    assert {k: sent["headers"][k] for k in CLAUDE_HEADERS} == CLAUDE_HEADERS
    assert resp.status_code == 200
    assert resp.json() == {"from": "anthropic"}
    assert resp.headers["x-upstream"] == "yes"


def test_anthropic_cookies_are_not_replayed_on_later_requests(router, upstream):
    raw = b'{"model": "claude-opus-5-5", "max_tokens": 5, "messages": []}'
    router.post("/v1/messages", content=raw, headers=CLAUDE_HEADERS)
    router.cookies.clear()  # the test client keeps cookies itself; only the router's own jar is under test
    router.post("/v1/messages", content=raw, headers=CLAUDE_HEADERS)

    assert "cookie" not in upstream.last("/v1/messages")["headers"]


def test_anthropic_unreachable_is_an_anthropic_error():
    app = create_app({"models": {}}, anthropic_url="http://127.0.0.1:1")
    with TestClient(app) as client:
        resp = client.post("/v1/messages", json={"model": "claude-opus-5-5", "messages": []})

    assert resp.status_code == 502
    assert resp.json()["error"]["type"] == "api_error"


def test_unsigned_thinking_is_removed_before_anthropic(router, upstream):
    body = {"model": "claude-opus-5-5", "max_tokens": 5, "messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "from qwen", "signature": ""},
                                          {"type": "text", "text": "hello"}]},
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "qwen only", "signature": ""}]},
        {"role": "user", "content": "more"},
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "from claude", "signature": "EqQB"},
                                          {"type": "text", "text": "sure"}]},
        {"role": "user", "content": "again"}]}
    router.post("/v1/messages", json=body, headers=CLAUDE_HEADERS)

    sent = json.loads(upstream.last("/v1/messages")["raw"])
    assert sent["messages"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
        {"role": "user", "content": "more"},
        body["messages"][4],
        {"role": "user", "content": "again"}]


def test_open_weight_request_carries_backend_token_and_model_settings(router, upstream):
    body = {"model": MODEL, "max_tokens": 32000, "temperature": 1, "messages": [{"role": "user", "content": "hi"}],
            "thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}, "metadata": {"user_id": "u"}}
    resp = router.post("/v1/messages?beta=true", json=body, headers=CLAUDE_HEADERS)

    sent = upstream.last("/v1/chat/completions")
    assert sent["headers"]["authorization"] == "Bearer backend-token"
    assert not {"anthropic-beta", "anthropic-version", "x-claude-code-session-id"} & set(sent["headers"])
    assert "sk-ant" not in json.dumps(sent["headers"]) and b"sk-ant" not in sent["raw"]
    out = json.loads(sent["raw"])
    assert out["model"] == MODEL
    assert out["max_tokens"] == 16384
    assert {k: out[k] for k in ("temperature", "top_p", "top_k", "min_p", "repetition_penalty")} == {
        "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0}
    assert out["chat_template_kwargs"] == {"preserve_thinking": True}
    assert out["reasoning_effort"]  # Claude Code's thinking settings reach LiteLLM
    assert resp.status_code == 200
    assert resp.json()["content"] == [{"type": "text", "text": "ok"}]


def backend_request(router, upstream, body):
    router.post("/v1/messages?beta=true", json={"model": MODEL, "max_tokens": 32000, **body}, headers=CLAUDE_HEADERS)
    return json.loads(upstream.last("/v1/chat/completions")["raw"])


def test_open_weight_request_drops_only_the_billing_header_block(router, upstream):
    out = backend_request(router, upstream, {"messages": [{"role": "user", "content": "hi"}], "system": [
        {"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.310; cch=37a;"},
        {"type": "text", "text": "You are Claude Code."},
        {"type": "text", "text": "Project instructions."}]})

    system = json.dumps([m for m in out["messages"] if m["role"] == "system"])
    assert "You are Claude Code." in system and "Project instructions." in system
    assert "x-anthropic-billing-header" not in json.dumps(out)


def test_open_weight_request_keeps_what_claude_code_sent(router, upstream):
    tool = {"name": "Read", "description": "Read a file", "input_schema": {
        "$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object",
        "properties": {"file_path": {"type": "string"}}}}
    out = backend_request(router, upstream, {"tools": [tool], "messages": [
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "MID-CONVERSATION NOTE"},
        {"role": "user", "content": "more"}]})

    assert len(out["messages"]) == 3  # LiteLLM's own conversion: one turn each, nothing merged
    assert "MID-CONVERSATION NOTE" in json.dumps(out["messages"])
    assert out["tools"][0]["function"]["parameters"]["$schema"] == tool["input_schema"]["$schema"]


def test_open_weight_thinking_history_reaches_the_backend(make_router, upstream):
    out = backend_request(make_router(), upstream, {"messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "EARLIER REASONING", "signature": ""},
                                          {"type": "text", "text": "hello"}]},
        {"role": "user", "content": "more"}]})

    assert "EARLIER REASONING" in json.dumps(out["messages"])


def test_open_weight_request_without_model_settings_is_left_alone(make_router, upstream):
    out = backend_request(make_router(), upstream, {"temperature": 1, "messages": [{"role": "user", "content": "hi"}]})

    assert out["max_tokens"] == 32000
    assert out["temperature"] == 1
    assert not {"top_p", "top_k", "min_p", "chat_template_kwargs"} & set(out)


def test_open_weight_max_tokens_below_the_cap_is_kept(router, upstream):
    out = backend_request(router, upstream, {"max_tokens": 100, "messages": [{"role": "user", "content": "hi"}]})

    assert out["max_tokens"] == 100


def test_open_weight_route_ignores_proxy_variables(router, upstream, monkeypatch):
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.setenv(var, "http://127.0.0.1:1")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    resp = router.post("/v1/messages", json={"model": MODEL, "max_tokens": 5, "messages": [{"role": "user", "content": "hi"}]})

    assert resp.status_code == 200
    assert upstream.last("/v1/chat/completions")


def test_open_weight_stream_is_anthropic_events(router):
    body = {"model": MODEL, "max_tokens": 50, "stream": True, "messages": [{"role": "user", "content": "hi"}]}
    resp = router.post("/v1/messages", json=body)

    assert resp.headers["content-type"].startswith("text/event-stream")
    assert events(resp.text)[0] == "message_start" and events(resp.text)[-1] == "message_stop"
    assert '"text_delta"' in resp.text


def test_open_weight_tool_call_streams_as_tool_use(router):
    body = {"model": MODEL, "max_tokens": 50, "stream": True,
            "messages": [{"role": "user", "content": "USE_TOOL please"}],
            "tools": [{"name": "Read", "description": "Read a file",
                       "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}}}]}
    resp = router.post("/v1/messages", json=body)

    starts = [json.loads(line[5:]) for line in resp.text.splitlines()
              if line.startswith("data:") and '"content_block_start"' in line]
    assert starts[0]["content_block"]["type"] == "tool_use"
    assert starts[0]["content_block"]["name"] == "Read"
    assert '"stop_reason": "tool_use"' in resp.text or '"stop_reason":"tool_use"' in resp.text


def test_open_weight_backend_error_becomes_anthropic_error(router):
    body = {"model": MODEL, "max_tokens": 50, "messages": [{"role": "user", "content": "FAIL400"}]}
    resp = router.post("/v1/messages", json=body)

    assert resp.status_code == 400
    assert resp.json()["type"] == "error"
    assert resp.json()["error"]["type"] == "invalid_request_error"
    assert "prompt is too long" in resp.json()["error"]["message"]


def test_count_tokens_for_open_weight_model_is_answered_locally(router, upstream):
    resp = router.post("/v1/messages/count_tokens", json={"model": MODEL, "messages": [{"role": "user", "content": "x" * 400}]})

    assert resp.status_code == 200
    assert resp.json()["input_tokens"] > 100
    assert upstream.requests == []


def test_count_tokens_for_claude_goes_to_anthropic(router, upstream):
    router.post("/v1/messages/count_tokens", json={"model": "claude-opus-5-5", "messages": []}, headers=CLAUDE_HEADERS)

    assert upstream.last("/v1/messages/count_tokens")["headers"]["authorization"] == CLAUDE_HEADERS["authorization"]


def test_other_paths_go_to_anthropic(router, upstream):
    assert router.head("/api/hello").status_code == 200
    assert router.get("/v1/models?limit=1000").json() == {"from": "anthropic"}
    assert [r["path"] for r in upstream.requests] == ["/api/hello", "/v1/models?limit=1000"]


WEB_SEARCH_BODY = {"model": MODEL, "max_tokens": 32000, "stream": True,
                   "system": [{"type": "text", "text": "You run a web search for another Claude model."}],
                   "messages": [{"role": "user", "content": "Perform a web search for the query: uv 0.13"}],
                   "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 8}],
                   "tool_choice": {"type": "tool", "name": "web_search"}}


def test_open_weight_web_search_goes_to_anthropic_with_a_claude_model(router, upstream):
    resp = router.post("/v1/messages?beta=true", json=WEB_SEARCH_BODY, headers=CLAUDE_HEADERS)

    sent = upstream.last("/v1/messages")
    assert sent["path"] == "/v1/messages?beta=true"
    assert {k: sent["headers"][k] for k in CLAUDE_HEADERS} == CLAUDE_HEADERS
    assert json.loads(sent["raw"]) == {**WEB_SEARCH_BODY, "model": "claude-haiku-5-5"}
    assert not [r for r in upstream.requests if r["path"] == "/v1/chat/completions"]
    assert resp.json() == {"from": "anthropic"}


def test_web_search_model_comes_from_the_config(make_router, upstream):
    router = make_router(top='web_search_model = "claude-sonnet-5-5"')
    router.post("/v1/messages", json=WEB_SEARCH_BODY, headers=CLAUDE_HEADERS)

    assert json.loads(upstream.last("/v1/messages")["raw"])["model"] == "claude-sonnet-5-5"


def test_open_weight_turn_that_only_lists_web_search_stays_open_weight(router, upstream):
    body = {k: v for k, v in WEB_SEARCH_BODY.items() if k != "tool_choice"}
    router.post("/v1/messages", json={**body, "stream": False}, headers=CLAUDE_HEADERS)

    assert upstream.last("/v1/chat/completions")
    assert not [r for r in upstream.requests if r["path"] == "/v1/messages"]
