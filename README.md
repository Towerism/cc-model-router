# cc-model-router

A local HTTP router that lets Claude Code's `/model` picker offer an OpenAI-compatible open-weight model next to every Anthropic model, and switch between them inside one session.

```
Claude Code ── ANTHROPIC_BASE_URL=http://127.0.0.1:9940 ──► cc-model-router
   model listed in the config  → LiteLLM → <base_url>/v1/chat/completions   (open-weight server)
   any other model             → api.anthropic.com, unchanged               (your Claude login)
```

- **Claude models** keep your Claude login (subscription OAuth or API key). Requests go to `api.anthropic.com` byte for byte; the one edit is removing thinking blocks with no signature, which the open-weight model produced earlier in the session and which Anthropic would reject.
- **Open-weight models** get the server's own bearer token instead of your Claude credential. LiteLLM translates to and from OpenAI Chat Completions, including streaming, tool calls and `reasoning_content`. Before that, the router turns Claude Code's mid-conversation system messages into user turns, drops the billing-header system block, strips `$schema` from tool schemas, drops server-side tools (web search), caps `max_tokens` and applies your sampling settings.
- **Outbound connections** are refused except to loopback, `api.anthropic.com` (and Anthropic's published IP ranges) and the configured servers. LiteLLM's catalog downloads are switched off.

## Install

Requires [uv](https://docs.astral.sh/uv/) and Python 3.14.

```bash
git clone https://github.com/Towerism/cc-model-router && cd cc-model-router
uv tool install .
mkdir -p ~/.config/cc-model-router ~/.config/systemd/user
cp contrib/cc-model-router.service ~/.config/systemd/user/
```

Write `~/.config/cc-model-router/config.toml`:

```toml
listen = "127.0.0.1:9940"

[models."qwen3.8-27b"]                 # the model ID exactly as Claude Code will send it
base_url = "http://127.0.0.1:9931"     # the server's root; the router calls <base_url>/v1/chat/completions
token_file = "~/.config/qlm/api-key"   # file holding the server's bearer token
max_tokens = 16384                     # cap on the output tokens Claude Code asks for
preserve_thinking = true               # send earlier reasoning back (chat_template_kwargs.preserve_thinking)
sampling = { temperature = 0.6, top_p = 0.95, top_k = 20, min_p = 0.0, repetition_penalty = 1.0 }
```

Add another `[models."<id>"]` table for each further model. Restart the router after editing the file:

```bash
systemctl --user enable --now cc-model-router
journalctl --user -u cc-model-router -f      # one line per request: model and route, never headers
```

## Configure Claude Code

Merge [`contrib/claude-settings.example.json`](contrib/claude-settings.example.json) into `~/.claude/settings.json`:

- `env.ANTHROPIC_BASE_URL` points Claude Code at the router. It belongs in the settings file, not a shell export, so sessions started from agent view get it too. Every Claude Code session on the machine then needs the router running.
- `env.CLAUDE_CODE_MAX_CONTEXT_TOKENS` is the open-weight server's context size. It applies only to model IDs Claude Code doesn't recognize, so Claude models keep their own windows.
- `modelPicker.options` adds one row per open-weight model, using the bare model ID. Leave `replaceBuiltInOptions` unset to keep the Anthropic rows.
- Don't set `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_API_KEY` or `apiKeyHelper`: Claude requests should keep using your Claude login.

### Keep data on the two destinations

The router controls only its own connections. If data may go only to Anthropic and the open-weight server, Claude Code needs the rest of the example file too:

- `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC`, `DISABLE_TELEMETRY` and `DISABLE_ERROR_REPORTING` stop version checks, telemetry and error reports, some of which go to third parties.
- `WebFetch` and `WebSearch` are denied: WebFetch fetches arbitrary sites from your machine, and WebSearch sends your query to a search provider.
- The Bash sandbox has no allowed network domains. On Linux it needs `bubblewrap` and `socat`.
- MCP servers, claude.ai connectors, plugins and hooks run outside the sandbox. Turn off any that send data to a third party.

## Limits

- **Switching into a smaller context window.** A conversation that grew on a Claude model can be larger than the open-weight server accepts. The server's error reaches Claude Code; run `/compact`, then retry.
- **Anthropic-only tools** (web search, web fetch, the advisor) don't exist on the open-weight route.
- Anthropic doesn't support routing Claude Code to non-Claude models; problems on the open-weight route are this project's to fix.

## Develop

```bash
uv run pytest
```

`tests/test_ccx_parity.py` checks the open-weight route against the request shim of `ccx`, the prototype this replaces, using recorded Claude Code requests in `tests/fixtures/` with every private text replaced. `tests/fixtures/make_expected.py` regenerates the reference files.
