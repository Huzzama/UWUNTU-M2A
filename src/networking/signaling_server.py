import asyncio
import websockets
import json

from src.networking.utils import generate_room_code

# rooms[room_code] = {
#     "password": str,
#     "host_name": str,
#     "clients": [websocket, ...]
# }
rooms = {}


async def handler(websocket):
    room_code = None

    try:
        async for message in websocket:
            data = json.loads(message)

            # ── Cliente crea o se une a sala ──────────────────────
            if data["type"] == "join":
                room_code = data.get("room")
                client_name = data.get("name", "Anónimo")
                password = data.get("password", "")

                # ── HOST: no manda código → crea sala nueva ───────
                if not room_code:
                    if not password:
                        await websocket.send(json.dumps({
                            "type": "error",
                            "message": "El host debe establecer una contraseña."
                        }))
                        return

                    room_code = generate_room_code()
                    rooms[room_code] = {
                        "password": password,
                        "host_name": client_name,
                        "clients": [websocket]
                    }

                    print(f"[+] Room creado: {room_code}  |  host: '{client_name}'")

                    await websocket.send(json.dumps({
                        "type": "room_assigned",
                        "room": room_code,
                        "host_name": client_name
                    }))

                # ── CLIENTE: manda código → se une ────────────────
                else:
                    if room_code not in rooms:
                        await websocket.send(json.dumps({
                            "type": "error",
                            "message": "El room no existe."
                        }))
                        return

                    if rooms[room_code]["password"] != password:
                        await websocket.send(json.dumps({
                            "type": "error",
                            "message": "Contraseña incorrecta."
                        }))
                        print(f"[-] Intento fallido en room {room_code} por '{client_name}'")
                        return

                    rooms[room_code]["clients"].append(websocket)
                    host_name = rooms[room_code]["host_name"]

                    print(f"[+] '{client_name}' se unió a room {room_code}  |  host: '{host_name}'")

                    await websocket.send(json.dumps({
                        "type": "room_assigned",
                        "room": room_code,
                        "host_name": host_name,
                        "client_name": client_name
                    }))

                    # Notificar al host que alguien se unió
                    for client in rooms[room_code]["clients"]:
                        if client != websocket:
                            await client.send(json.dumps({
                                "type": "peer_joined",
                                "name": client_name
                            }))

            # ── Mensajes de señalización (WebRTC) ─────────────────
            elif data["type"] == "signal":
                for client in rooms.get(room_code, {}).get("clients", []):
                    if client != websocket:
                        await client.send(json.dumps(data))

    except Exception as e:
        print("Error:", e)

    finally:
        if room_code and room_code in rooms:
            try:
                rooms[room_code]["clients"].remove(websocket)
                print(f"[-] Cliente salió de room {room_code}")
            except ValueError:
                pass  # El websocket ya no estaba en la lista

            if not rooms[room_code]["clients"]:
                del rooms[room_code]
                print(f"[x] Room {room_code} eliminado (vacío)")


async def main():
    async with websockets.serve(handler, "localhost", 8765):
        print("Servidor de signaling corriendo en ws://localhost:8765")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())