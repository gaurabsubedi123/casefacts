"""Start the portal: make sure Ollama is up, serve the page, open the browser.

This is also the .exe's entry point, so it has to work with nobody at a
terminal: it picks a free port if the usual one is taken, starts Ollama if it
is installed but not running, and opens the browser itself. The console window
that stays open is the off switch — closing it stops the portal.
"""

from __future__ import annotations

import argparse
import logging
import socket
import sys
import threading
import webbrowser

from . import __version__, hardware
from .ollama import DOWNLOAD_URL, Ollama, find_binary

DEFAULT_PORT = 5050


def free_port(host: str, preferred: int) -> int:
    for port in [preferred, *range(preferred + 1, preferred + 50)]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind((host, port))
                return port
            except OSError:
                continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((host, 0))
        return probe.getsockname()[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="modelportal", description="Local model portal")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"default {DEFAULT_PORT}, or the next free one")
    parser.add_argument("--no-browser", action="store_true", help="do not open the browser")
    parser.add_argument("--version", action="version", version=__version__)
    args = parser.parse_args(argv)
    host = "127.0.0.1"
    try:  # show each line as it happens, even when the output is not a console
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

    print(f"Model Portal {__version__}")
    machine = hardware.detect()
    print(f"  This computer: {machine.summary()}")

    client = Ollama()
    if client.is_up():
        print(f"  Ollama {client.version()} is running.")
    elif find_binary():
        print("  Starting Ollama…")
        ok, message = client.start()
        print(f"  {message}")
    else:
        print(f"  Ollama is not installed. Install it from {DOWNLOAD_URL}, then reopen this.")
        print("  (The portal still opens, and will tell you the same thing.)")

    from werkzeug.serving import make_server

    from .web import create_app

    port = free_port(host, args.port)
    url = f"http://{host}:{port}"
    server = make_server(host, port, create_app(client), threaded=True)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    print()
    print(f"  Open {url} in your browser.")
    print("  Keep this window open while you use it; close it to stop the portal.")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
