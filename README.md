# cc-model-router

A local HTTP router that lets Claude Code's `/model` picker offer an OpenAI-compatible open-weight model next to every Anthropic model, and switch between them inside one session.

```
Claude Code ── ANTHROPIC_BASE_URL=http://127.0.0.1:9940 ──► cc-model-router
   model listed in the config  → LiteLLM → <base_url>/v1/chat/completions   (open-weight server)
   any other model             → api.anthropic.com, unchanged               (your Claude login)
```

- **Claude models** keep your Claude login (subscription OAuth or API key). Requests go to `api.anthropic.com` byte for byte; the one edit is removing thinking blocks with no signature, which the open-weight model produced earlier in the session and which Anthropic would reject.
- **Open-weight models** get the server's own bearer token instead of your Claude credential. LiteLLM translates to and from OpenAI Chat Completions, including streaming, tool calls and `reasoning_content`. Before that, the router removes only the billing-header system block, which differs per conversation and would stop a new session reusing the server's prompt cache. It then applies the model's optional settings (`max_tokens` cap, sampling, `preserve_thinking`). Everything else Claude Code sends goes to LiteLLM as-is.
- **The router's dependencies don't phone home.** LiteLLM's catalog downloads, release check and use of proxy variables are switched off, and the router refuses any outbound connection except to loopback, `api.anthropic.com` (and Anthropic's published IP ranges) and the configured servers. Claude Code itself is not restricted: it runs as a normal session.

## Install

Runs on macOS and Linux. Requires [uv](https://docs.astral.sh/uv/) and Python 3.14.

```bash
git clone https://github.com/Towerism/cc-model-router && cd cc-model-router
uv tool install --python 3.14 .
mkdir -p ~/.config/cc-model-router
```

Write `~/.config/cc-model-router/config.toml`:

```toml
listen = "127.0.0.1:9940"
web_search_model = "claude-haiku-5-5"  # optional: the Claude model that runs WebSearch for open-weight models

[models."qwen3.8-27b"]                 # the model ID exactly as Claude Code will send it
base_url = "http://127.0.0.1:9931"     # the server's root; the router calls <base_url>/v1/chat/completions
token_file = "~/.config/qlm/api-key"   # file holding the server's bearer token
max_tokens = 16384                     # optional: cap on the output tokens Claude Code asks for
preserve_thinking = true               # optional: sent as chat_template_kwargs.preserve_thinking
# optional: overrides Claude Code's values; keys other than temperature, top_p and top_k go in the extra body
sampling = { temperature = 0.6, top_p = 0.95, top_k = 20, min_p = 0.0, repetition_penalty = 1.0 }
```

Add another `[models."<id>"]` table for each further model. Then install the router as a user service that starts at login and restarts after a crash: a launchd agent on macOS, a systemd user unit on Linux.

```bash
contrib/install-service.sh
```

Run the script again after editing the config file; it restarts the router. The log has one line per request, with the model and route and never headers:

```bash
tail -f ~/Library/Logs/cc-model-router.log      # macOS
journalctl --user -u cc-model-router -f         # Linux
```

## Configure Claude Code

Merge [`contrib/claude-settings.example.json`](contrib/claude-settings.example.json) into `~/.claude/settings.json`:

- `env.ANTHROPIC_BASE_URL` points Claude Code at the router. It belongs in the settings file, not a shell export, so sessions started from agent view get it too. Every Claude Code session on the machine then needs the router running.
- `env.CLAUDE_CODE_MAX_CONTEXT_TOKENS` is the open-weight server's context size. It applies only to model IDs Claude Code doesn't recognize, so Claude models keep their own windows.
- `modelPicker.options` adds one row per open-weight model, using the bare model ID. Leave `replaceBuiltInOptions` unset to keep the Anthropic rows.
- Don't set `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_API_KEY` or `apiKeyHelper`: Claude requests should keep using your Claude login.

## Limits

- **Switching into a smaller context window.** A conversation that grew on a Claude model can be larger than the open-weight server accepts. The server's error reaches Claude Code; run `/compact`, then retry.
- **Fields the server rejects.** The router doesn't filter Claude Code's request for the open-weight route. If a server rejects a field, its error reaches Claude Code, and the edit for that field is added to the router only then. On llama-server nothing besides the billing header needed removing. Not yet checked: servers such as vLLM, which reject a prompt plus `max_tokens` larger than their context. Set `max_tokens` for those models.
- **Web search runs on Anthropic.** Claude Code's WebSearch sends its own request with the current model and Anthropic's server-side `web_search` tool. For an open-weight model, the router sends that request to Anthropic with `web_search_model` (default `claude-haiku-5-5`) and your Claude login, so the search query goes to Anthropic.
- **Other Anthropic-only tools** (web fetch, the advisor) don't exist on the open-weight route.
- Anthropic doesn't support routing Claude Code to non-Claude models; problems on the open-weight route are this project's to fix.

## Develop

```bash
uv run pytest
```

The tests run the router against a local stand-in for both Anthropic and the open-weight server.
