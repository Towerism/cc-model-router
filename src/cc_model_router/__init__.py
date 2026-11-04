"""Route Claude Code's model requests: Claude models to Anthropic untouched, configured open-weight models
to an OpenAI-compatible server through LiteLLM.

Data may only leave the machine for Anthropic and the configured servers, so LiteLLM's catalog downloads
and its use of proxy variables are switched off before it is imported, and main() refuses every other
outbound connection.
"""
import os

for _flag in ("LITELLM_OFFLINE_MODE", "LITELLM_LOCAL_MODEL_COST_MAP", "LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS",
              "LITELLM_DISABLE_LATEST_RELEASE_CHECK", "DISABLE_AIOHTTP_TRUST_ENV"):
    os.environ.setdefault(_flag, "true")

import argparse
import contextlib
import ipaddress
import json
import logging
import socket
import sys
import tomllib
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import litellm
import uvicorn
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

ANTHROPIC = "https://api.anthropic.com"
# Anthropic's published inbound ranges (https://platform.claude.com/docs/en/api/ip-addresses).
ANTHROPIC_NETS = [ipaddress.ip_network(n) for n in ("160.79.104.0/23", "153.61.224.0/22", "2607:6bc0::/48")]
DEFAULT_CONFIG = "~/.config/cc-model-router/config.toml"
DEFAULT_WEB_SEARCH_MODEL = "claude-haiku-5-5"
HOP_BY_HOP = {"host", "content-length", "connection", "keep-alive", "transfer-encoding"}
TOP_LEVEL_SAMPLING = ("temperature", "top_p", "top_k")  # the rest go to the server as extra body fields
ERROR_TYPES = {400: "invalid_request_error", 401: "authentication_error", 403: "permission_error",
               404: "not_found_error", 413: "request_too_large", 429: "rate_limit_error"}

log = logging.getLogger("cc-model-router")
litellm.suppress_debug_info = True


def load_config(path):
    cfg = tomllib.loads(Path(path).expanduser().read_text())
    models = {model_id: {**m, "token": Path(m["token_file"]).expanduser().read_text().strip()}
              for model_id, m in (cfg.get("models") or {}).items()}
    web_search_model = cfg.get("web_search_model", DEFAULT_WEB_SEARCH_MODEL)
    if web_search_model in models:
        raise SystemExit(f"cc-model-router: web_search_model = {web_search_model!r} is an open-weight model; "
                         "it must be a Claude model, because only Anthropic can run WebSearch")
    return {"listen": cfg.get("listen", "127.0.0.1:9940"), "models": models, "web_search_model": web_search_model}


def allow_only(hosts):
    """Refuse outbound connections except to loopback, Anthropic and `hosts`. Lasts for the process."""
    allowed = set()
    for host in hosts:
        try:
            allowed |= {ipaddress.ip_address(info[4][0]) for info in socket.getaddrinfo(host, None)}
        except socket.gaierror:
            log.warning("cannot resolve %s; requests for it will fail until the router restarts", host)

    def hook(event, args):
        if event != "socket.connect" or args[0].family not in (socket.AF_INET, socket.AF_INET6):
            return
        ip = ipaddress.ip_address(args[1][0])
        if not (ip.is_loopback or ip in allowed or any(ip in net for net in ANTHROPIC_NETS)):
            raise PermissionError(f"cc-model-router: outbound connection to {ip} is not allowed")

    sys.addaudithook(hook)


def without_unsigned_thinking(body):
    """The body minus thinking blocks with no signature (the open-weight model's), or None if it has none.

    Anthropic rejects a thinking block it didn't sign.
    """
    messages, changed = [], False
    for m in body.get("messages", []):
        if isinstance(m.get("content"), list):
            kept = [b for b in m["content"] if not (b.get("type") == "thinking" and not b.get("signature"))]
            if len(kept) != len(m["content"]):
                changed = True
                if not kept:
                    continue
                m = {**m, "content": kept}
        messages.append(m)
    return {**body, "messages": messages} if changed else None


def is_web_search(body):
    """True for Claude Code's WebSearch side request, which needs Anthropic's server-side web_search tool.

    That request offers web_search and nothing else (tool_choice is auto); a turn that also has other tools
    stays on its model.
    """
    tools = body.get("tools")
    return isinstance(tools, list) and bool(tools) and all(
        isinstance(t, dict) and str(t.get("type", "")).startswith("web_search_") for t in tools)


def open_weight_request(body, model):
    """LiteLLM arguments for an open-weight model: Claude Code's body minus the billing-header system block,
    plus the model's settings.

    The billing header differs per conversation, so keeping it would stop a new session reusing the server's
    prompt cache. Anything else the server rejects shows up as its error; add an edit here only then.
    """
    req = dict(body)
    if isinstance(req.get("system"), list):
        req["system"] = [b for b in req["system"] if not b.get("text", "").startswith("x-anthropic-billing-header")]
    if "max_tokens" in model:
        req["max_tokens"] = min(req.get("max_tokens", model["max_tokens"]), model["max_tokens"])
    extra = {}
    for k, v in model.get("sampling", {}).items():
        (req if k in TOP_LEVEL_SAMPLING else extra)[k] = v
    if "preserve_thinking" in model:
        extra["chat_template_kwargs"] = {"preserve_thinking": model["preserve_thinking"]}
    return {**req, "model": f"hosted_vllm/{body['model']}", "api_base": model["base_url"].rstrip("/") + "/v1",
            "api_key": model["token"], "extra_body": extra}


def create_app(config, anthropic_url=ANTHROPIC):
    models = config["models"]
    web_search_model = config.get("web_search_model", DEFAULT_WEB_SEARCH_MODEL)
    client = httpx.AsyncClient(timeout=None, trust_env=False)

    async def to_anthropic(request, content):
        url = anthropic_url + request.url.path + (f"?{request.url.query}" if request.url.query else "")
        headers = [(k, v) for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP]
        try:  # a bare Request, so the client's cookie jar and default headers stay out of it
            resp = await client.send(httpx.Request(request.method, url, headers=headers, content=content), stream=True)
        except (httpx.HTTPError, OSError) as e:
            log.warning("cannot reach Anthropic: %s", e)
            return JSONResponse({"type": "error", "error": {"type": "api_error",
                                                            "message": f"cc-model-router cannot reach Anthropic: {e}"}},
                                status_code=502)
        return StreamingResponse(resp.aiter_raw(), status_code=resp.status_code,
                                 headers={k: v for k, v in resp.headers.items() if k.lower() not in HOP_BY_HOP},
                                 background=BackgroundTask(resp.aclose))

    async def to_open_weight(body, model):
        try:
            result = await litellm.anthropic.messages.acreate(**open_weight_request(body, model))
        except Exception as e:  # LiteLLM wraps upstream errors in many exception types
            status = e.status_code if isinstance(getattr(e, "status_code", None), int) else 502
            log.warning("open-weight error %s for %s", status, body.get("model"))
            return JSONResponse({"type": "error", "error": {"type": ERROR_TYPES.get(status, "api_error"),
                                                            "message": str(e)[:1000]}}, status_code=status)
        if body.get("stream"):
            return StreamingResponse(result, media_type="text/event-stream")
        return JSONResponse(result)

    def parse(raw):
        try:
            body = json.loads(raw)
        except ValueError:
            body = None
        return body if isinstance(body, dict) else {}  # let Anthropic report a bad body

    async def messages(request):
        raw = await request.body()
        body = parse(raw)
        model = models.get(body.get("model"))
        if model and is_web_search(body):
            # Claude Code sends WebSearch with the main-loop model; only Anthropic can run the search.
            log.info("%s -> anthropic (web search via %s)", body.get("model"), web_search_model)
            body = {**body, "model": web_search_model}
            raw = json.dumps(body).encode()
            model = None
        else:
            log.info("%s -> %s", body.get("model"), "open-weight" if model else "anthropic")
        if model:
            return await to_open_weight(body, model)
        if (cleaned := without_unsigned_thinking(body)) is not None:
            raw = json.dumps(cleaned).encode()
        return await to_anthropic(request, raw)

    async def count_tokens(request):
        raw = await request.body()
        if parse(raw).get("model") in models:
            return JSONResponse({"input_tokens": max(1, len(raw) // 4)})
        return await to_anthropic(request, raw)

    async def passthrough(request):
        return await to_anthropic(request, await request.body())

    @contextlib.asynccontextmanager
    async def lifespan(app):
        yield
        await client.aclose()

    methods = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]
    return Starlette(routes=[Route("/v1/messages", messages, methods=["POST"]),
                             Route("/v1/messages/count_tokens", count_tokens, methods=["POST"]),
                             Route("/{path:path}", passthrough, methods=methods)],
                     lifespan=lifespan)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=DEFAULT_CONFIG, help=f"TOML config (default {DEFAULT_CONFIG})")
    config = load_config(parser.parse_args().config)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    allow_only([urlsplit(ANTHROPIC).hostname] + [urlsplit(m["base_url"]).hostname for m in config["models"].values()])
    host, port = config["listen"].rsplit(":", 1)
    # asyncio, not uvloop: uvloop's connections bypass the audit hook allow_only() relies on.
    uvicorn.run(create_app(config), host=host, port=int(port), log_level="warning", loop="asyncio")
