"""allow_only() refuses outbound connections outside its allowlist. Run in a child process: the hook is permanent."""
import subprocess
import sys

CHILD = """
import socket, sys
from cc_model_router import allow_only

allow_only(["localhost"])
srv = socket.create_server(("127.0.0.1", 0))
socket.create_connection(srv.getsockname(), timeout=2).close()        # loopback: allowed
try:
    socket.create_connection(("192.0.2.1", 443), timeout=2)            # a documentation address: refused
except PermissionError:
    print("refused 192.0.2.1")
else:
    sys.exit("connected to 192.0.2.1")
"""


def test_connections_outside_the_allowlist_are_refused():
    result = subprocess.run([sys.executable, "-c", CHILD], capture_output=True, text=True, timeout=60)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "refused 192.0.2.1"


def test_anthropic_addresses_are_allowed():
    child = ("import ipaddress\nfrom cc_model_router import ANTHROPIC_NETS\n"
             "print(any(ipaddress.ip_address('160.79.104.10') in n for n in ANTHROPIC_NETS))")
    result = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True, timeout=60)

    assert result.stdout.strip() == "True"
