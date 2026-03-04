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

    Key design decisions
    --------------------
    - send() uses call_soon_threadsafe + an asyncio.Queue (created inside the
      loop) instead of run_coroutine_threadsafe.  This avoids the
      "Event loop stopped before Future completed" crash: call_soon_threadsafe
      is fire-and-forget and never raises even if the loop is winding down.
    - Receive and send run as two concurrent coroutines under asyncio.gather so
      neither blocks the other.
    - ping_interval keeps the connection alive during long renders so the server
      does not close it while the host is busy processing Pixel Art.
    """

    def __init__(self, server_url: str, on_message: Callable[[dict], None]):
        self._url = server_url
        self.on_message: Callable[[dict], None] = on_message

        self._loop:      Optional[asyncio.AbstractEventLoop] = None
        self._ws         = None
        self._connected  = False
        self._stopped    = False

        # Async queue — created inside the event loop in _connect()
        self._out_queue: Optional[asyncio.Queue] = None

        self._thread = threading.Thread(target=self._run, daemon=True)

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background thread that owns the asyncio loop."""
        self._thread.start()

    def send(self, msg: dict) -> None:
        """
        Thread-safe: enqueue a JSON message for sending.

        Uses call_soon_threadsafe so it NEVER raises even if the loop is
        stopping — the _enqueue callback simply discards the message when
        the queue does not exist yet or is full.
        """
        if self._stopped:
            return
        if not self._loop or not self._loop.is_running():
            return
        try:
            data = json.dumps(msg)
            self._loop.call_soon_threadsafe(self._enqueue, data)
        except Exception as exc:
            print(f"[P2PSession] send() scheduling error: {exc}")

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

    def _enqueue(self, data: str) -> None:
        """Called from within the event loop via call_soon_threadsafe."""
        if self._out_queue is not None and not self._stopped:
            try:
                self._out_queue.put_nowait(data)
            except asyncio.QueueFull:
                print("[P2PSession] Outbound queue full — dropping oldest message")
                try:
                    self._out_queue.get_nowait()   # drop oldest
                    self._out_queue.put_nowait(data)
                except Exception:
                    pass

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
            # Cancel any remaining tasks before closing
            try:
                pending = asyncio.all_tasks(self._loop)
                for task in pending:
                    task.cancel()
                if pending:
                    self._loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
            except Exception:
                pass
            try:
                self._loop.close()
            except Exception:
                pass

    async def _connect(self) -> None:
        import websockets

        # Create the outbound queue inside the event loop
        self._out_queue = asyncio.Queue(maxsize=512)

        try:
            async with websockets.connect(
                self._url,
                max_size=100 * 1024 * 1024,
                ping_interval=20,   # keep-alive ping every 20s
                ping_timeout=60,    # allow up to 60s for pong (host may be busy rendering)
            ) as ws:
                self._ws = ws
                self._connected = True

                # Run receive and send loops concurrently;
                # either finishing (or raising) will cancel the other.
                await asyncio.gather(
                    self._recv_loop(ws),
                    self._send_loop(ws),
                    return_exceptions=False,
                )

        except Exception as exc:
            self._connected = False
            if not self._stopped:
                try:
                    self.on_message({
                        "type":    "connection_error",
                        "message": str(exc),
                    })
                except Exception:
                    pass
        finally:
            self._connected = False
            self._ws = None

    async def _recv_loop(self, ws) -> None:
        """Continuously receive messages from the server."""
        try:
            async for raw in ws:
                if self._stopped:
                    break
                try:
                    msg = json.loads(raw)
                    self.on_message(msg)
                except json.JSONDecodeError:
                    print(f"[P2PSession] Bad JSON: {raw[:80]}")
        except Exception as exc:
            if not self._stopped:
                print(f"[P2PSession] Receive error: {exc}")
            raise   # propagate so gather() cancels _send_loop too

    async def _send_loop(self, ws) -> None:
        """Drain the outbound queue and forward messages to the server."""
        try:
            while not self._stopped:
                try:
                    # Poll with timeout so we can notice _stopped quickly
                    data = await asyncio.wait_for(
                        self._out_queue.get(), timeout=0.5
                    )
                    await ws.send(data)
                    self._out_queue.task_done()
                except asyncio.TimeoutError:
                    continue   # no message ready; loop back and check _stopped
                except Exception as exc:
                    if not self._stopped:
                        print(f"[P2PSession] Send error: {exc}")
                    raise      # propagate so gather() cancels _recv_loop too
        except Exception:
            raise