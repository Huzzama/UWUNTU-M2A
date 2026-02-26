import tkinter as tk
from tkinter import messagebox
import asyncio
import websockets
import json
import threading

SERVER_URL = "ws://localhost:8765"


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("ASCII P2P")
        self.root.geometry("420x420")
        self.root.resizable(False, False)
        self._running = True
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # ── Título ──────────────────────────────────────────────
        tk.Label(root, text="ASCII P2P", font=("Courier", 16, "bold")).pack(pady=(14, 4))

        # ── Campo: Nombre ────────────────────────────────────────
        tk.Label(root, text="Tu nombre:").pack()
        self.name_entry = tk.Entry(root, width=30)
        self.name_entry.pack(pady=(2, 10))

        # ── Separador HOST ───────────────────────────────────────
        tk.Label(root, text="─── Host ───────────────────────────",
                 fg="gray").pack()

        tk.Label(root, text="Contraseña del room:").pack()
        self.host_password_entry = tk.Entry(root, width=30, show="*")
        self.host_password_entry.pack(pady=(2, 6))

        self.host_btn = tk.Button(root, text="Crear Room", width=20,
                                  command=self.host)
        self.host_btn.pack(pady=(0, 12))

        # ── Separador JOIN ───────────────────────────────────────
        tk.Label(root, text="─── Unirse ─────────────────────────",
                 fg="gray").pack()

        tk.Label(root, text="Código del room:").pack()
        self.room_entry = tk.Entry(root, width=30)
        self.room_entry.pack(pady=(2, 4))

        tk.Label(root, text="Contraseña del room:").pack()
        self.join_password_entry = tk.Entry(root, width=30, show="*")
        self.join_password_entry.pack(pady=(2, 6))

        self.join_btn = tk.Button(root, text="Unirse al Room", width=20,
                                  command=self.join)
        self.join_btn.pack(pady=(0, 12))

        # ── Estado ───────────────────────────────────────────────
        self.status_label = tk.Label(root, text="", fg="darkgreen",
                                     font=("Courier", 10))
        self.status_label.pack(pady=4)

    # ─────────────────────────────────────────────────────────────
    # HOST
    # ─────────────────────────────────────────────────────────────
    def host(self):
        name = self.name_entry.get().strip()
        password = self.host_password_entry.get().strip()

        if not name:
            messagebox.showwarning("Campo vacío", "Por favor ingresa tu nombre.")
            return
        if not password:
            messagebox.showwarning("Campo vacío",
                                   "El host debe definir una contraseña para el room.")
            return

        self.set_status("Creando room…")
        threading.Thread(target=self.run_host,
                         args=(name, password), daemon=True).start()

    def run_host(self, name, password):
        asyncio.run(self.host_async(name, password))

    async def host_async(self, name, password):
        try:
            async with websockets.connect(SERVER_URL) as ws:
                await ws.send(json.dumps({
                    "type": "join",
                    "name": name,
                    "password": password
                }))

                response = await ws.recv()
                data = json.loads(response)

                if data["type"] == "error":
                    self.set_status(f"Error: {data.get('message', 'desconocido')}")
                    return

                if data["type"] == "room_assigned":
                    room = data["room"]
                    self.set_status(
                        f"Room creado: {room}\n"
                        f"Esperando jugador como '{name}'…"
                    )

                # ✅ Mantener conexión viva hasta que se cierre la ventana
                async for message in ws:
                    if not self._running:
                        break
                    data = json.loads(message)
                    if data["type"] == "peer_joined":
                        self.set_status(
                            f"Room: {room}\n"
                            f"'{data['name']}' se unió. ¡Conexión establecida!"
                        )
                    elif data["type"] == "signal":
                        # Aquí se procesarán las señales WebRTC en el futuro
                        pass

        except Exception as e:
            self.set_status(f"Error de conexión: {e}")

    # ─────────────────────────────────────────────────────────────
    # JOIN
    # ─────────────────────────────────────────────────────────────
    def join(self):
        name = self.name_entry.get().strip()
        room_code = self.room_entry.get().strip()
        password = self.join_password_entry.get().strip()

        if not name:
            messagebox.showwarning("Campo vacío", "Por favor ingresa tu nombre.")
            return
        if not room_code:
            messagebox.showwarning("Campo vacío",
                                   "Ingresa el código del room al que quieres unirte.")
            return
        if not password:
            messagebox.showwarning("Campo vacío",
                                   "Ingresa la contraseña del room.")
            return

        self.set_status("Uniéndose al room…")
        threading.Thread(target=self.run_join,
                         args=(name, room_code, password), daemon=True).start()

    def run_join(self, name, room_code, password):
        asyncio.run(self.join_async(name, room_code, password))

    async def join_async(self, name, room_code, password):
        try:
            async with websockets.connect(SERVER_URL) as ws:
                await ws.send(json.dumps({
                    "type": "join",
                    "room": room_code,
                    "name": name,
                    "password": password
                }))

                response = await ws.recv()
                data = json.loads(response)

                if data["type"] == "error":
                    self.set_status(f"Error: {data.get('message', 'desconocido')}")
                    return

                if data["type"] == "room_assigned":
                    host_name = data.get("host_name", "el host")
                    self.set_status(
                        f"¡Unido al room {room_code}!\n"
                        f"Conectado con '{host_name}' como '{name}'"
                    )

                # ✅ Mantener conexión viva para recibir señales WebRTC
                async for message in ws:
                    if not self._running:
                        break
                    data = json.loads(message)
                    if data["type"] == "signal":
                        # Aquí se procesarán las señales WebRTC en el futuro
                        pass

        except Exception as e:
            self.set_status(f"Error de conexión: {e}")

    # ─────────────────────────────────────────────────────────────
    # UI helpers
    # ─────────────────────────────────────────────────────────────
    def on_closing(self):
        self._running = False
        self.root.destroy()

    def set_status(self, text):
        self.root.after(0, lambda: self.status_label.config(text=text))


def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()