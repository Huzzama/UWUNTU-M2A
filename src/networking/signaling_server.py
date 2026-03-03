# src/networking/signaling_server.py
"""
Signaling server for the Grainrad P2P ASCII art app.

Changes from original:
  - Binds to 0.0.0.0 so LAN clients can connect (not just localhost).
  - Stamps every incoming message with from_id (str(id(websocket))).
  - Generic relay: messages with a "to_id" are forwarded only to that
    specific client; messages without to_id go to every other peer in the room.
  - Sends peer_left notification when a client disconnects.
  - Prints the server's LAN IP at startup for convenience.

Start with:
    python -m src.networking.signaling_server
"""
from __future__ import annotations

import asyncio
import json
import socket

import websockets

from src.networking.utils import generate_room_code


# rooms[room_code] = {
#     "password":  str,
#     "host_name": str,
#     "clients":   list[websocket],
# }
rooms: dict = {}


def _get_local_ip() -> str:
    """Return the machine's LAN IP (best-effort)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


async def handler(websocket):
    # Unique, stable ID for this connection (used as client_id throughout)
    client_id = str(id(websocket))
    room_code: str | None = None

    try:
        async for message in websocket:
            data = json.loads(message)

            # Stamp every message with the sender's ID so recipients know who sent it
            data["from_id"] = client_id

            msg_type = data.get("type")

            # ── JOIN / CREATE ──────────────────────────────────────────────────
            if msg_type == "join":
                room_code = data.get("room")
                client_name = data.get("name", "Anonymous")
                password = data.get("password", "")

                # HOST: no room code → create a new room
                if not room_code:
                    if not password:
                        await websocket.send(json.dumps({
                            "type": "error",
                            "message": "Host must set a room password."
                        }))
                        return

                    room_code = generate_room_code()
                    rooms[room_code] = {
                        "password":  password,
                        "host_name": client_name,
                        "host_id":   client_id,
                        "clients":   [websocket],
                    }
                    print(f"[+] Room created: {room_code}  |  host: '{client_name}'  |  id: {client_id}")

                    await websocket.send(json.dumps({
                        "type":        "room_assigned",
                        "room":        room_code,
                        "host_name":   client_name,
                        "client_id":   client_id,
                    }))

                # CLIENT: has room code → join existing room
                else:
                    room_code = room_code.upper()

                    if room_code not in rooms:
                        await websocket.send(json.dumps({
                            "type":    "error",
                            "message": "Room does not exist."
                        }))
                        return

                    if rooms[room_code]["password"] != password:
                        await websocket.send(json.dumps({
                            "type":    "error",
                            "message": "Incorrect password."
                        }))
                        print(f"[-] Failed join attempt on {room_code} by '{client_name}'")
                        return

                    rooms[room_code]["clients"].append(websocket)
                    host_name = rooms[room_code]["host_name"]
                    host_id   = rooms[room_code]["host_id"]

                    print(f"[+] '{client_name}' joined room {room_code}  |  host: '{host_name}'  |  id: {client_id}")

                    await websocket.send(json.dumps({
                        "type":        "room_assigned",
                        "room":        room_code,
                        "host_name":   host_name,
                        "host_id":     host_id,
                        "client_name": client_name,
                        "client_id":   client_id,
                    }))

                    # Notify every other peer (host + earlier clients)
                    for peer in rooms[room_code]["clients"]:
                        if peer is not websocket:
                            await peer.send(json.dumps({
                                "type":      "peer_joined",
                                "name":      client_name,
                                "client_id": client_id,
                            }))

            # ── LEGACY SIGNAL (WebRTC passthrough, kept for compatibility) ─────
            elif msg_type == "signal":
                for peer in rooms.get(room_code, {}).get("clients", []):
                    if peer is not websocket:
                        await peer.send(json.dumps(data))

            # ── GENERIC RELAY ─────────────────────────────────────────────────
            # Messages with "to_id" → forward only to that specific peer.
            # Messages without "to_id" → broadcast to all other peers in the room.
            else:
                peers = rooms.get(room_code, {}).get("clients", [])
                to_id: str | None = data.get("to_id")

                for peer in peers:
                    if peer is websocket:
                        continue  # never echo back to sender

                    peer_id = str(id(peer))
                    if to_id is None or peer_id == to_id:
                        try:
                            await peer.send(json.dumps(data))
                        except Exception as send_err:
                            print(f"[!] Relay send error to {peer_id}: {send_err}")

    except Exception as exc:
        print(f"[!] Handler error for {client_id}: {exc}")

    finally:
        # ── Clean up when the connection closes ────────────────────────────────
        if room_code and room_code in rooms:
            clients = rooms[room_code]["clients"]
            if websocket in clients:
                clients.remove(websocket)
                print(f"[-] Client {client_id} left room {room_code}  ({len(clients)} remaining)")

            # Notify remaining peers that this client has left
            for peer in clients:
                try:
                    await peer.send(json.dumps({
                        "type":      "peer_left",
                        "client_id": client_id,
                    }))
                except Exception:
                    pass

            # Remove empty rooms
            if not clients:
                del rooms[room_code]
                print(f"[x] Room {room_code} removed (empty)")


async def main():
    host = "0.0.0.0"   # ← binds on all interfaces so LAN clients can connect
    port = 8765
    lan_ip = _get_local_ip()

    async with websockets.serve(handler, host, port):
        print(f"[Grainrad] Signaling server running")
        print(f"  Local:   ws://127.0.0.1:{port}")
        print(f"  LAN:     ws://{lan_ip}:{port}  ← share this with clients on the same network")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
