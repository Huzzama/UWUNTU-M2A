# src/networking/p2p_session.py
"""
P2PSession — async WebSocket client that runs on a background thread.

Exposes a thread-safe send() and an on_message callback so Tkinter code
(which runs on the main thread) can use it without blocking the GUI.
"""
from __future__ import annotations

import asyncio
import json
import threading
from typing import Callable, Optional


class P2PSession:
    """
    Runs an asyncio event loop on a daemon background thread.

    Usage
    -----
        def handle(msg: dict): ...

        p2p = P2PSession("ws://192.168.1.10:8765", on_message=handle)
        p2p.start()
        p2p.send({"type": "join", "name": "Alice", "password": "s3cr3t"})
        # ... later ...
        p2p.stop()
    """

    def __init__(self, server_url: str, on_message: Callable[[dict], None]):
        self._url = server_url
        self.on_message: Callable[[dict], None] = on_message

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ws = None
        self._connected = False
        self._stopped = False

        self._thread = threading.Thread(target=self._run, daemon=True)

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background thread that owns the asyncio loop."""
        self._thread.start()

    def send(self, msg: dict) -> None:
        """Thread-safe: enqueue a JSON message for sending."""
        if self._loop and self._connected and not self._stopped:
            asyncio.run_coroutine_threadsafe(
                self._async_send(json.dumps(msg)), self._loop
            )

    def stop(self) -> None:
        """Gracefully shut down the WebSocket connection and background thread."""
        self._stopped = True
        self._connected = False
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

    @property
    def connected(self) -> bool:
        return self._connected

    # ── Internals ──────────────────────────────────────────────────────────────

    def _run(self) -> None:
        """Entry point for the background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._connect())
        except Exception as exc:
            print(f"[P2PSession] Connection error: {exc}")
        finally:
            self._connected = False

    async def _connect(self) -> None:
        import websockets  # imported here so the module loads even without it
        try:
            async with websockets.connect(self._url, max_size=100 * 1024 * 1024) as ws:
                self._ws = ws
                self._connected = True
                async for raw in ws:
                    if self._stopped:
                        break
                    try:
                        msg = json.loads(raw)
                        self.on_message(msg)
                    except json.JSONDecodeError:
                        print(f"[P2PSession] Bad JSON: {raw[:80]}")
        except Exception as exc:
            self._connected = False
            # Deliver a synthetic error message so the GUI can react
            try:
                self.on_message({
                    "type": "connection_error",
                    "message": str(exc)
                })
            except Exception:
                pass

    async def _async_send(self, data: str) -> None:
        if self._ws and not self._stopped:
            try:
                await self._ws.send(data)
            except Exception as exc:
                print(f"[P2PSession] Send error: {exc}")