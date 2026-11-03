import json

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


def test_open_weight_request_carries_backend_token_and_settings_only(router, upstream):
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
    assert not {"reasoning_effort", "thinking", "metadata", "user"} & set(out)
    assert resp.status_code == 200
    assert resp.json()["content"] == [{"type": "text", "text": "ok"}]


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
