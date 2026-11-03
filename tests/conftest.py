import http.server
import json
import threading

import pytest
from starlette.testclient import TestClient

from cc_model_router import create_app, load_config

MODEL = "qwen3.8-27b"  # the model ID in the recorded fixtures


def sse(events):
    return b"".join(f"data: {json.dumps(e)}\n\n".encode() for e in events) + b"data: [DONE]\n\n"


def chat_chunks(body):
    """An OpenAI streamed reply: a tool call when the last message asks for one, otherwise the text "ok"."""
    base = {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": body["model"]}
    wants_tool = "USE_TOOL" in json.dumps(body["messages"][-1])
    delta = ({"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                              "function": {"name": "Read", "arguments": '{"file_path": "/a"}'}}]}
             if wants_tool else {"content": "ok"})
    return [{**base, "choices": [{"index": 0, "delta": {"role": "assistant", **delta}, "finish_reason": None}]},
            {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if wants_tool else "stop"}]},
            {**base, "choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}}]


class Upstream:
    """A real HTTP server standing in for both Anthropic and the open-weight backend; records each request."""

    def __init__(self):
        self.requests = []
        upstream = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def handle_one(self):
                raw = self.rfile.read(int(self.headers.get("content-length") or 0))
                upstream.requests.append({"method": self.command, "path": self.path,
                                          "headers": {k.lower(): v for k, v in self.headers.items()}, "raw": raw})
                path = self.path.split("?")[0]
                body = json.loads(raw) if raw else {}
                if path == "/v1/chat/completions" and "FAIL400" in json.dumps(body.get("messages")):
                    return self.reply(400, json.dumps({"error": {"message": "prompt is too long"}}).encode())
                if path == "/v1/chat/completions" and body.get("stream"):
                    return self.reply(200, sse(chat_chunks(body)), "text/event-stream")
                if path == "/v1/chat/completions":
                    return self.reply(200, json.dumps({
                        "id": "c1", "object": "chat.completion", "created": 0, "model": body["model"],
                        "choices": [{"index": 0, "finish_reason": "stop",
                                     "message": {"role": "assistant", "content": "ok"}}],
                        "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}}).encode())
                self.reply(200, b'{"from": "anthropic"}')

            def reply(self, status, data, ctype="application/json"):
                self.send_response(status)
                self.send_header("content-type", ctype)
                self.send_header("x-upstream", "yes")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(data)

            do_GET = do_POST = do_HEAD = handle_one

            def log_message(self, *a):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def last(self, path):
        return [r for r in self.requests if r["path"].split("?")[0] == path][-1]


@pytest.fixture
def upstream():
    u = Upstream()
    yield u
    u.server.shutdown()


@pytest.fixture
def router(upstream, tmp_path):
    token = tmp_path / "api-key"
    token.write_text("backend-token\n")
    config = tmp_path / "config.toml"
    config.write_text(f"""
listen = "127.0.0.1:9940"
[models."{MODEL}"]
base_url = "{upstream.url}"
token_file = "{token}"
max_tokens = 16384
preserve_thinking = true
sampling = {{ temperature = 0.6, top_p = 0.95, top_k = 20, min_p = 0.0, repetition_penalty = 1.0 }}
""")
    with TestClient(create_app(load_config(config), anthropic_url=upstream.url)) as client:
        yield client
