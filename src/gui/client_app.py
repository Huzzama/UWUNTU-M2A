# src/gui/client_app.py
"""
ClientApp — window shown after joining a room as a Client.

Looks and feels identical to the standalone ASCII app (same controls,
same layout, same dark theme) with two differences:

  1. A "Connected to: <host>" badge is injected into the top title bar.
  2. The processing pipeline is NOT run locally — instead:
       • File loading chunks the bytes and sends them to the host.
       • Slider / effect changes send a settings_update to the host.
       • Incoming preview_frame messages display the result directly.
"""
from __future__ import annotations

import base64
import threading
from io import BytesIO
from pathlib import Path
from typing import Optional
import tkinter as tk
from tkinter import filedialog, messagebox

# ── Path setup ────────────────────────────────────────────────────────────────
import sys
_HERE = Path(__file__).resolve().parent
_SRC  = _HERE.parent
_ROOT = _SRC.parent
for _p in [str(_SRC), str(_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gui.gui_app import App, C                  # inherit the full ASCII GUI
from networking.p2p_session import P2PSession


_CHUNK_SIZE = 48 * 1024   # 48 KB per media chunk


class ClientApp(App):
    """
    Subclass of the standalone App that delegates all processing to the host.

    Extra constructor parameters
    ----------------------------
    name       : this client's display name
    room_code  : the room code (shown in the badge)
    host_name  : the host's display name (shown in the badge)
    p2p        : a started P2PSession
    client_id  : assigned by the server (from room_assigned message)
    """

    def __init__(
        self,
        root: tk.Tk,
        name: str,
        room_code: str,
        host_name: str,
        p2p: P2PSession,
        client_id: str = "",
        host_id: str = "",
    ):
        self._p2p_name   = name
        self._room_code  = room_code
        self._host_name  = host_name
        self._p2p        = p2p
        self._client_id  = client_id
        self._host_id    = host_id
        self._host_disconnected = False   # True once the host leaves; blocks further status updates
        self._file_sent      = False   # True once the first media chunk was sent
        self._sending        = False   # guard: don't send two files at once
        self._received_frames: list    = []   # PNG bytes from host, one per frame
        self._source_filename: str     = ""   # original filename for export naming
        self._pending_export: dict     = {}   # {fmt, out_path, chunks: [], total}

        # Wire message handler before calling super().__init__ so any early
        # messages during layout construction are handled
        self._p2p.on_message = self._on_p2p_message

        # Build the full standard GUI
        super().__init__(root)

        # Inject the connection badge into the title bar after it was built
        self._inject_connection_badge()

        root.title(f"UWUNTU — Client  ·  Room {room_code}")
        root.resizable(True, True)
        self._fullscreen = False
        root.bind("<F11>", self._toggle_fullscreen)
        root.bind("<Escape>", lambda e: self._exit_fullscreen())
        root.protocol("WM_DELETE_WINDOW", self._on_client_close)

    # ── Badge injection ───────────────────────────────────────────────────────

    def _inject_connection_badge(self):
        """Add a connection status badge to the right side of the existing title bar."""
        children = self.root.winfo_children()
        title_bar = next(
            (w for w in children if isinstance(w, tk.Frame) and
             w.cget("height") == 48),
            None,
        )
        if title_bar is None:
            return

        badge = tk.Frame(title_bar, bg=C["bg"])
        badge.pack(side="right", fill="y", padx=16)

        # Green connected dot
        tk.Label(
            badge, text="●",
            bg=C["bg"], fg=C["green"],
            font=("Segoe UI", 13),
        ).pack(side="left")

        # Client's own name
        tk.Label(
            badge,
            text=f"  {self._p2p_name}",
            bg=C["bg"], fg=C["text_bright"],
            font=("Segoe UI", 13, "bold"),
        ).pack(side="left")

        # Separator
        tk.Label(
            badge, text="  ·",
            bg=C["bg"], fg=C["text_dim"],
            font=("Segoe UI", 13),
        ).pack(side="left")

        # Host name
        tk.Label(
            badge,
            text=f"  Connected to: {self._host_name}",
            bg=C["bg"], fg=C["text_dim"],
            font=("Segoe UI", 13),
        ).pack(side="left")

        # Separator
        tk.Label(
            badge, text="  ·",
            bg=C["bg"], fg=C["text_dim"],
            font=("Segoe UI", 13),
        ).pack(side="left")

        # Room code
        tk.Label(
            badge,
            text=f"  Room: {self._room_code}",
            bg=C["bg"], fg=C["accent"],
            font=("Segoe UI", 13, "bold"),
        ).pack(side="left")

    # ── Override: schedule_preview ────────────────────────────────────────────

    def schedule_preview(self, _=None):
        """
        Instead of running the pipeline locally, debounce and send a
        settings_update to the host so it can reprocess.
        """
        if self.canvas is None or self.settings is None:
            return
        if self._preview_job:
            self.root.after_cancel(self._preview_job)
        self._preview_job = self.root.after(300, self._send_settings_update)

    def _send_settings_update(self):
        if not self._file_sent:
            return   # no media loaded yet; nothing to reprocess
        payload = self.settings.build_settings_dict()
        payload["effect"] = self.sidebar.get_selected_effect() if self.sidebar else "ASCII"
        self._p2p.send({
            "type":        "settings_update",
            "room":        self._room_code,
            "from_id":     self._client_id,
            "client_name": self._p2p_name,
            "payload":     payload,
        })

    # ── Override: load_file ───────────────────────────────────────────────────

    def load_file(self, path: str):
        """
        Instead of processing locally, read the bytes and send them to the host
        in chunks, followed by the current settings.
        """
        if self._host_disconnected:
            self._set_host_disconnected()  # re-pin the message in case something overwrote it
            return
        # Sanitize path (same as parent)
        path = path.strip()
        if path.startswith("{") and path.endswith("}"):
            path = path[1:-1]
        path = path.strip("'\""  )

        ext = Path(path).suffix.lower()
        allowed = {".png", ".jpg", ".jpeg", ".gif", ".mp4", ".webm", ".mov", ".avi", ".mkv"}
        if ext not in allowed:
            messagebox.showwarning(
                "Unsupported file",
                f"File type '{ext}' is not supported.\n"
                "Accepted: PNG, JPG, JPEG, GIF, MP4, WEBM, MOV, AVI, MKV"
            )
            return

        if self._sending:
            return  # ignore while a previous upload is in progress

        self._sending = True
        self._file_sent = False
        self._received_frames = []     # clear previous frames
        self._source_filename = Path(path).name

        if self.sidebar:
            self.sidebar.set_status(f"Sending to host…", C["yellow"])
        if self.statusbar:
            self.statusbar.set_message(f"  Uploading {Path(path).name}…", C["text"])
        if self.canvas:
            self.canvas.show_loading()

        threading.Thread(
            target=self._send_media_worker,
            args=(path,),
            daemon=True,
        ).start()

    def _send_media_worker(self, path: str):
        try:
            data     = Path(path).read_bytes()
            filename = Path(path).name
            chunks   = [
                data[i: i + _CHUNK_SIZE]
                for i in range(0, len(data), _CHUNK_SIZE)
            ]

            # 1. Send metadata
            self._p2p.send({
                "type":        "media_metadata",
                "room":        self._room_code,
                "from_id":     self._client_id,
                "client_name": self._p2p_name,
                "payload": {
                    "filename":    filename,
                    "total_bytes": len(data),
                    "chunk_count": len(chunks),
                },
            })

            # 2. Send chunks
            for i, chunk in enumerate(chunks):
                self._p2p.send({
                    "type":        "media_chunk",
                    "room":        self._room_code,
                    "from_id":     self._client_id,
                    "chunk_index": i,
                    "data":        base64.b64encode(chunk).decode(),
                })
                pct = int((i + 1) / len(chunks) * 100)
                self.root.after(
                    0, lambda p=pct: self.sidebar.set_status(
                        f"Uploading: {p}%", C["yellow"]
                    ) if self.sidebar else None
                )

            # 3. Flag that file is sent, then push current settings
            self._file_sent = True
            self.root.after(0, self._send_settings_update)
            self.root.after(
                0, lambda: self.sidebar.set_status("Sent — waiting for host…", C["text_dim"])
                if self.sidebar else None
            )

        except Exception as exc:
            self.root.after(
                0, lambda e=str(exc): messagebox.showerror("Upload error", e)
            )
            self.root.after(
                0, lambda: self.sidebar.set_status("Upload failed", C["red"])
                if self.sidebar else None
            )
        finally:
            self._sending = False

    # ── P2P message handling ──────────────────────────────────────────────────

    def _on_p2p_message(self, msg: dict):
        """Called on the P2P background thread."""
        t = msg.get("type", "")

        # Once the host disconnects, ignore all incoming messages except
        # connection_error (which could be the WebSocket closing cleanly).
        if self._host_disconnected and t != "connection_error":
            return

        if t == "preview_frame":
            p   = msg.get("payload", {})
            png = base64.b64decode(p["png_b64"])
            self.root.after(0, self._show_preview_frame, png, p.get("is_final", False))

        elif t == "processing_start":
            p = msg.get("payload", {})
            self.root.after(
                0, lambda: self.sidebar.set_status(
                    f"Host rendering ({p.get('total_frames', '?')} frames)…",
                    C["yellow"],
                ) if self.sidebar else None
            )

        elif t == "processing_progress":
            p = msg.get("payload", {})
            pct = p.get("percent", 0)
            self.root.after(
                0, lambda: self.sidebar.set_status(
                    f"Host rendering: {pct}%", C["yellow"]
                ) if self.sidebar else None
            )

        elif t == "processing_done":
            self.root.after(
                0, lambda: self.sidebar.set_status("Done ✓", C["green"])
                if self.sidebar else None
            )

        elif t == "error_message":
            err = msg.get("payload", {}).get("message", "Unknown error")
            self.root.after(0, messagebox.showerror, "Host Error", err)
            self.root.after(
                0, lambda: self.sidebar.set_status("Error from host", C["red"])
                if self.sidebar else None
            )

        elif t == "peer_left":
            departed_id = msg.get("client_id", "")
            if departed_id == self._host_id:
                self.root.after(0, self._set_host_disconnected)
            # Si se fue otro cliente, no mostrar ningún mensaje

        elif t == "export_chunk":
            if self._pending_export:
                p = msg.get("payload", {})
                self._pending_export["chunks"][p["index"]] = base64.b64decode(p["data"])
                self._pending_export["total"] = p["total"]
                recv  = len(self._pending_export["chunks"])
                total = p["total"]
                self.root.after(
                    0, lambda r=recv, t2=total: self.sidebar.set_status(
                        f"Receiving {self._pending_export.get('fmt','').upper()}: {r}/{t2}", C["yellow"]
                    ) if self.sidebar else None
                )
                if recv >= total:
                    data = b"".join(
                        self._pending_export["chunks"][i] for i in range(total)
                    )
                    fmt      = self._pending_export["fmt"]
                    out_path = self._pending_export["out_path"]
                    self._pending_export = {}
                    self.root.after(0, self._save_host_export_result, data, fmt, out_path)

        elif t == "connection_error":
            err = msg.get("message", "Unknown error")
            self.root.after(
                0, messagebox.showerror, "Connection Error", err
            )

    def _set_host_disconnected(self):
        """Mark host as gone and pin a permanent status message that can't be overwritten."""
        self._host_disconnected = True
        self._sending = False  # unblock any stuck upload guard
        if self.sidebar:
            self.sidebar.set_status("Host disconnected", C["red"])

    def _show_preview_frame(self, png: bytes, is_final: bool):
        """Display a PNG frame received from the host and accumulate for export."""
        self._received_frames.append(png)
        if self.canvas:
            self.canvas.show_image(png)
        if is_final and self.sidebar:
            self.sidebar.set_status("Done ✓", C["green"])

    # ── Host-assisted TXT / HTML export ─────────────────────────────────────────

    def _request_host_export(self, fmt: str, out_path: str):
        """Ask the host to render and send back a TXT or HTML file."""
        if not self._file_sent:
            messagebox.showwarning("Not ready", "No file processed yet.")
            return
        self._pending_export = {"fmt": fmt, "out_path": out_path, "chunks": {}, "total": 0}
        payload = self.settings.build_settings_dict() if self.settings else {}
        payload["effect"] = self.sidebar.get_selected_effect() if self.sidebar else "ASCII"
        self._p2p.send({
            "type":    "export_request",
            "room":    self._room_code,
            "from_id": self._client_id,
            "payload": {"fmt": fmt, "settings": payload},
        })
        if self.sidebar:
            self.sidebar.set_status(f"Requesting {fmt.upper()} from host…", C["yellow"])

    def _save_host_export_result(self, data: bytes, fmt: str, out_path: str):
        """Called on main thread when all export chunks have arrived."""
        try:
            Path(out_path).write_bytes(data)
            name = Path(out_path).name
            if self.sidebar:
                self.sidebar.set_status("Done ✓", C["green"])
            messagebox.showinfo("Export complete", f"Saved: {name}")
        except Exception as exc:
            messagebox.showerror("Export error", str(exc))
            if self.sidebar:
                self.sidebar.set_status("Export failed", C["red"])

    # ── Fullscreen ────────────────────────────────────────────────────────────

    def _toggle_fullscreen(self, event=None):
        self._fullscreen = not self._fullscreen
        self.root.attributes("-fullscreen", self._fullscreen)

    def _exit_fullscreen(self):
        if self._fullscreen:
            self._fullscreen = False
            self.root.attributes("-fullscreen", False)

    # ── Close ─────────────────────────────────────────────────────────────────

    def _on_client_close(self):
        self._stop_all_animations()
        self._p2p.stop()
        self.root.destroy()

    # ── Disabled local actions ────────────────────────────────────────────────

    def _run_preview_thread(self):
        """Do not run local preview; the host handles everything."""
        pass

    def export(self):
        """Export the frames received from the host."""
        if self._host_disconnected:
            self._set_host_disconnected()  # re-pin the message
            return
        if not self._received_frames:
            messagebox.showwarning(
                "Nothing to export",
                "No rendered frames yet.\nLoad a file and wait for the host to finish processing."
            )
            return

        if not self.settings:
            return

        formats = self.settings.get_selected_formats()
        fmt = formats[0] if formats else "png"

        ext_map = {"png": ".png", "jpeg": ".jpg", "gif": ".gif", "mp4": ".mp4",
                   "txt": ".txt", "html": ".html"}
        default_ext = ext_map.get(fmt, ".png")
        base = Path(self._source_filename).stem if self._source_filename else "output"

        out_path = filedialog.asksaveasfilename(
            title="Save as",
            defaultextension=default_ext,
            initialfile=f"{base}{default_ext}",
            filetypes=[(fmt.upper(), f"*{default_ext}"), ("All files", "*.*")],
        )
        if not out_path:
            return

        if self.sidebar:
            self.sidebar.set_status("Exporting…", C["yellow"])

        threading.Thread(
            target=self._client_export_worker,
            args=(out_path, fmt),
            daemon=True,
        ).start()

    def _client_export_worker(self, out_path: str, fmt: str):
        """Write the received PNG frames to the requested format."""
        try:
            from PIL import Image

            frames_pil = [Image.open(BytesIO(png)).convert("RGB")
                          for png in self._received_frames]

            if fmt == "png":
                # Single image: save last (or only) frame
                frames_pil[-1].save(out_path)

            elif fmt == "jpeg":
                frames_pil[-1].save(out_path, format="JPEG", quality=92)

            elif fmt == "gif":
                fps = 12
                duration_ms = int(1000 / fps)
                frames_pil[0].save(
                    out_path,
                    format="GIF",
                    save_all=True,
                    append_images=frames_pil[1:],
                    duration=duration_ms,
                    loop=0,
                    optimize=False,
                )

            elif fmt == "mp4":
                try:
                    import cv2
                    import numpy as np
                    h, w = frames_pil[0].height, frames_pil[0].width
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    vw = cv2.VideoWriter(out_path, fourcc, 12.0, (w, h))
                    for frame in frames_pil:
                        vw.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
                    vw.release()
                except ImportError:
                    # Fallback to GIF if cv2 not available
                    fallback = Path(out_path).with_suffix(".gif")
                    duration_ms = int(1000 / 12)
                    frames_pil[0].save(
                        str(fallback), format="GIF", save_all=True,
                        append_images=frames_pil[1:], duration=duration_ms, loop=0
                    )
                    out_path = str(fallback)
                    self.root.after(0, messagebox.showinfo, "Export note",
                        f"MP4 encoder not available — saved as GIF:\n{fallback.name}")

            elif fmt in ("txt", "html"):
                # Ask the host to generate this format, then save when received
                self.root.after(0, self._request_host_export, fmt, out_path)
                return   # worker exits; _save_host_export_result handles status

            name = Path(out_path).name
            self.root.after(0, lambda: self.sidebar.set_status("Done ✓", C["green"]) if self.sidebar else None)
            self.root.after(0, messagebox.showinfo, "Export complete", f"Saved: {name}")

        except Exception as exc:
            err = str(exc)
            self.root.after(0, messagebox.showerror, "Export error", err)
            self.root.after(0, lambda: self.sidebar.set_status("Export failed", C["red"]) if self.sidebar else None)
