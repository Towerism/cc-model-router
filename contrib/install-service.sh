#!/bin/sh
# Install cc-model-router as a user service and (re)start it: a launchd agent on macOS,
# a systemd user unit on Linux. Run it again after editing config.toml to restart the router.
set -eu

label=com.github.towerism.cc-model-router
contrib=$(cd "$(dirname "$0")" && pwd)
bin="$HOME/.local/bin/cc-model-router"

if [ ! -x "$bin" ]; then
    echo "install-service.sh: $bin not found; run 'uv tool install .' first" >&2
    exit 1
fi

case $(uname -s) in
Darwin)
    domain="gui/$(id -u)"
    plist="$HOME/Library/LaunchAgents/$label.plist"
    mkdir -p "$HOME/Library/LaunchAgents" "$HOME/Library/Logs"
    home=$(printf '%s\n' "$HOME" | sed 's/[&|\\]/\\&/g')
    sed "s|@HOME@|$home|g" "$contrib/cc-model-router.plist" > "$plist"
    plutil -lint "$plist" > /dev/null
    # bootout returns before the agent is gone, and bootstrap fails while it is still loaded.
    if launchctl bootout "$domain/$label" 2> /dev/null; then
        i=0
        while launchctl print "$domain/$label" > /dev/null 2>&1; do
            i=$((i + 1))
            if [ "$i" -gt 50 ]; then
                echo "install-service.sh: $label did not stop" >&2
                exit 1
            fi
            sleep 0.2
        done
    fi
    launchctl bootstrap "$domain" "$plist"
    echo "cc-model-router started; log: ~/Library/Logs/cc-model-router.log"
    ;;
Linux)
    mkdir -p "$HOME/.config/systemd/user"
    cp "$contrib/cc-model-router.service" "$HOME/.config/systemd/user/"
    systemctl --user daemon-reload
    systemctl --user enable cc-model-router
    systemctl --user restart cc-model-router
    echo "cc-model-router started; log: journalctl --user -u cc-model-router -f"
    ;;
*)
    echo "install-service.sh: unsupported OS $(uname -s); only macOS and Linux are supported" >&2
    exit 1
    ;;
esac
